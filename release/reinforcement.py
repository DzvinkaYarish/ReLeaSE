"""
This class implements simple policy gradient algorithm for
biasing the generation of molecules towards desired values of
properties aka Reinforcement Learninf for Structural Evolution (ReLeaSE)
as described in
Popova, M., Isayev, O., & Tropsha, A. (2018).
Deep reinforcement learning for de novo drug design.
Science advances, 4(7), eaap7885.
"""

import torch
import torch.nn.functional as F
import numpy as np
import collections
from rdkit import Chem


from utils import get_ECFP, normalize_fp, mol2image


class Reinforcement(object):
    def __init__(self, args, generator, predictor, get_reward, get_end_of_batch_reward=None):
        """
        Constructor for the Reinforcement object.

        Parameters
        ----------
        args: object used to store parameters of the model

        generator: object of type StackAugmentedRNN
            generative model that produces string of characters (trajectories)

        predictor: object of any predictive model type
            predictor accepts a trajectory and returns a numerical
            prediction of desired property for the given trajectory

        get_reward: function
            custom reward function that accepts a trajectory, predictor and
            any number of positional arguments and returns a single value of
            the reward for the given trajectory
            Example:
            reward = get_reward(trajectory=my_traj, predictor=my_predictor,
                                custom_parameter=0.97)

        Returns
        -------
        object of type Reinforcement used for biasing the properties estimated
        by the predictor of trajectories produced by the generator to maximize
        the custom reward function get_reward.
        """

        super(Reinforcement, self).__init__()
        self.args = args
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward
        self.get_end_of_batch_reward = get_end_of_batch_reward

        self.trajectories = collections.deque(maxlen=500)
        self.similarity_fingerprint = np.zeros((2048, ))

        self.init_experience_buffer()


    def policy_gradient(self, data, n_batch=10, gamma=0.97,
                        std_smiles=False, grad_clipping=None, **kwargs):
        """
        Implementation of the policy gradient algorithm.

        Parameters:
        -----------

        data: object of type GeneratorData
            stores information about the generator data format such alphabet, etc

        n_batch: int (default 10)
            number of trajectories to sample per batch. When training on GPU
            setting this parameter to to some relatively big numbers can result
            in out of memory error. If you encountered such an error, reduce
            n_batch.

        gamma: float (default 0.97)
            factor by which rewards will be discounted within one trajectory.
            Usually this number will be somewhat close to 1.0.


        std_smiles: bool (default False)
            boolean parameter defining whether the generated trajectories will
            be converted to standardized SMILES before running policy gradient.
            Leave this parameter to the default value if your trajectories are
            not SMILES.

        grad_clipping: float (default None)
            value of the maximum norm of the gradients. If not specified,
            the gradients will not be clipped.

        kwargs: any number of other positional arguments required by the
            get_reward function.

        Returns
        -------
        total_reward: float
            value of the reward averaged through n_batch sampled trajectories

        rl_loss: float
            value for the policy_gradient loss averaged through n_batch sampled
            trajectories

        """

        total_reward = 0

        trajectories = []


        while len(trajectories) < n_batch:
            batch_trajectories = self.generator.evaluate(data, batch_size=self.args.batch_size_for_generate)
            for tr in batch_trajectories:
                mol = Chem.MolFromSmiles(tr[1:-1])
                if mol:
                    trajectories.append(tr)
                if len(trajectories) == n_batch:
                    break

        batch_rewards, batch_distinct_rewards = self.get_reward(self.args, [tr[1:-1] for tr in trajectories],
                                                                self.predictor, **kwargs)
        n_to_sample = 0
        if self.experience_buffer: #replace the most inactive to jak2 molecules in the batch with known active
            n_to_sample = max(1, int(n_batch * ((np.sum(batch_distinct_rewards[:, 0] < 2.)) / n_batch) - 3.))
            if n_to_sample > 0:
                samples = [self.experience_buffer[i] for i in np.random.randint(0, len(self.experience_buffer), n_to_sample)]
                sample_rewards, sample_distinct_rewards = self.get_reward(self.args, samples,
                                                                          self.predictor, **kwargs)
                indx_to_replace = batch_distinct_rewards[:, 0].argsort()[:n_to_sample]

                for i, indx in enumerate(indx_to_replace):
                    trajectories[indx] = '<' + samples[i] + '>'
                    batch_rewards[indx] = sample_rewards[i]
                    batch_distinct_rewards[indx] = sample_distinct_rewards[i]


        end_of_batch_rewards = np.zeros((n_batch,))

        if self.get_end_of_batch_reward:
            fngps = [mol2image(Chem.MolFromSmiles(tr[1:-1])) for tr in trajectories]

            end_of_batch_rewards = self.get_end_of_batch_reward(fngps)


        # Converting string of characters into tensor
        trajectory_input, _ = data.pad_sequences(trajectories, pad_symbol=data.pad_symbol)
        trajectory_input, _ = data.seq2tensor(trajectory_input, data.tokens, flip=False)
        trajectory_input = torch.tensor(trajectory_input).long()
        if self.args.use_cuda:
            trajectory_input = trajectory_input.cuda()

        discounted_reward = torch.Tensor(batch_rewards + end_of_batch_rewards).long()
        if self.args.normalize_rewards:
            discounted_reward = discounted_reward - np.mean(discounted_reward)
        if self.args.use_cuda:
            discounted_reward = discounted_reward.cuda()

        total_reward += np.sum(batch_rewards)
        total_reward += np.sum(end_of_batch_rewards)

        old_log_probs = []
        clip_fraction = 0

        for upd_step in self.args.n_update_steps:
            # Initializing the generator's hidden state

            hidden = self.generator.init_hidden(batch_size=n_batch)
            if self.generator.has_cell:
                cell = self.generator.init_cell(batch_size=n_batch)
                hidden = (hidden, cell)
            if self.generator.has_stack:
                stack = self.generator.init_stack(batch_size=n_batch)
            else:
                stack = None

            rl_loss = torch.zeros((n_batch,))
            if self.args.use_cuda:
                rl_loss = rl_loss.cuda()

            # "Following" the trajectory and accumulating the loss
            for p in range(trajectory_input.shape[1] - 1):
                output, hidden, stack = self.generator(trajectory_input[:, p],
                                                       hidden,
                                                       stack)

                indx_terminal = (trajectory_input[:, p] == data.char2idx[data.end_token]) + (trajectory_input[:, p] == data.char2idx[data.pad_symbol])
                discounted_reward[indx_terminal] = 0.

                log_probs = F.log_softmax(output, dim=1)
                top_i = trajectory_input[:, p + 1].detach().cpu().numpy()
                actual_log_probs = log_probs[np.arange(0, n_batch), top_i]
                if upd_step == 0:
                    old_log_probs.append(actual_log_probs.detach())

                ratios = torch.exp(actual_log_probs - old_log_probs[p])
                surr1 = ratios * discounted_reward
                surr2 = torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * discounted_reward
                rl_loss -= torch.min(surr1, surr2)

                if upd_step == 0:
                    discounted_reward = discounted_reward * gamma

                clipped = ratios.gt(1 + self.args.eps_clip) | ratios.lt(1 - self.args.eps_clip)
                clip_fraction += torch.as_tensor(clipped, dtype=torch.float32).mean().item()


            # useful things to log
            # approx_kl = (logprobs.detach() - old_logprobs.detach()).mean().item()


            # Doing backward pass and parameters update
            self.generator.optimizer.zero_grad()

            rl_loss = rl_loss.mean()

            rl_loss.backward()

            self.generator.optimizer.step()

        total_reward = total_reward / n_batch

        batch_distinct_rewards = np.concatenate((batch_distinct_rewards, np.expand_dims(end_of_batch_rewards, axis=1)), axis=1)

        batch_distinct_rewards = np.mean(batch_distinct_rewards, axis=0)

        return total_reward, rl_loss.item(), batch_distinct_rewards, n_to_sample / n_batch, clip_fraction / self.args.n_update_steps


    def update_trajectories(self, smiles):
        self.trajectories.extend(smiles)
        self.similarity_fingerprint = self.get_sim_fngp()

    def get_sim_fngp(self):
        fp_all = np.zeros(2048)
        fps = {}
        for sm in self.trajectories:
            fps[sm] = get_ECFP(sm)
            fp_all += fps[sm]
        fp_all_norm = normalize_fp(fp_all)
        return fp_all_norm

    def init_experience_buffer(self):
        if self.args.experience_buffer_path:
            with open(self.args.experience_buffer_path, 'r') as f:
                self.experience_buffer = [l.strip('\n').strip(',') for l in f.readlines()]
                self.experience_buffer.pop(0)
        else:
            self.experience_buffer = []



