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
from copy import  deepcopy

torch.autograd.set_detect_anomaly(True)

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
        # trajectories = ['<CC(=O)C(=CNc1ccc(F)cc1)c1cc(C)cc(C)c1>',
        # '<CC(=O)N1CC(C)C(C)C2SCC(C)(C)OC2CC1CO>',
        #    '<CC(=O)N1CCN(CC(=O)N2CCN(C(=O)OC(C)(C)C)CC2)CC1>',
        #    '<CC(=O)NC1C#CC(O)=Nc2cc(-c3ccc(Cl)c(Cl)c3)ccc21>']

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


        discounted_reward = torch.Tensor(batch_rewards + end_of_batch_rewards).float()
        if self.args.normalize_rewards:
            discounted_reward = (discounted_reward - torch.mean(discounted_reward)) / torch.std(discounted_reward)
        discounted_rewards = torch.zeros((n_batch, trajectory_input.shape[1]))
        discounted_rewards[:, 0] = discounted_reward
        for d in range(1, trajectory_input.shape[1]):
            discounted_rewards[:, d] = discounted_rewards[:, d-1] * gamma
            indx_terminal = (trajectory_input[:, d] == data.char2idx[data.end_token]) + (
                    trajectory_input[:, d] == data.char2idx[data.pad_symbol])

            discounted_rewards[indx_terminal, d] = 0.

        if self.args.use_cuda:
            discounted_rewards = discounted_rewards.cuda()

        total_reward += np.sum(batch_rewards)
        total_reward += np.sum(end_of_batch_rewards)

        old_log_probs = torch.zeros((n_batch, trajectory_input.shape[1]))
        if self.args.use_cuda:
            old_log_probs = old_log_probs.cuda()

        clip_fraction = 0
        for upd_step in range(self.args.n_update_steps):

            self.generator.optimizer.zero_grad()

            for b in range(0, n_batch, self.args.batch_size_for_generate):
                rl_loss = torch.zeros((min(self.args.batch_size_for_generate, trajectory_input.shape[0] - b),))
                if self.args.use_cuda:
                    rl_loss = rl_loss.cuda()
                end = min(trajectory_input.shape[0], b + self.args.batch_size_for_generate)

                hidden = self.generator.init_hidden(batch_size=min(self.args.batch_size_for_generate, trajectory_input.shape[0] - b))

                if self.generator.has_cell:
                    cell = self.generator.init_cell(batch_size=min(self.args.batch_size_for_generate, trajectory_input.shape[0] - b))
                    hidden = (hidden, cell)
                if self.generator.has_stack:
                    stack = self.generator.init_stack(batch_size=min(self.args.batch_size_for_generate, trajectory_input.shape[0] - b))
                else:
                    stack = None

                trajectory_input_batch = torch.tensor(trajectory_input[b:end]).long()
                if self.args.use_cuda:
                    trajectory_input_batch = trajectory_input_batch.cuda()

                for p in range(trajectory_input.shape[1] - 1):

                    output, hidden, stack = self.generator(trajectory_input_batch[:, p],
                                                           hidden,
                                                           stack)


                    log_probs = F.log_softmax(output, dim=1)
                    top_i = trajectory_input_batch[:, p + 1].detach().cpu().numpy()
                    actual_log_probs = log_probs[np.arange(0, len(log_probs)), top_i]
                    if upd_step == 0:
                        old_log_probs[b:end, p] = actual_log_probs.detach()

                    ratios = torch.exp(actual_log_probs - old_log_probs[b:end, p])
                    surr1 = ratios * discounted_rewards[b:end, p]
                    surr2 = torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * \
                            discounted_rewards[b:end, p]

                    rl_loss -= torch.min(surr1, surr2)

                    clipped = ratios.gt(1 + self.args.eps_clip) | ratios.lt(1 - self.args.eps_clip)
                    clip_fraction += torch.as_tensor(clipped, dtype=torch.float32).mean().item()


            # useful things to log
            # approx_kl = (logprobs.detach() - old_logprobs.detach()).mean().item()


            # Doing backward pass and parameters update


                rl_loss = rl_loss.mean()
                rl_loss = rl_loss / int(np.ceil(n_batch / self.args.batch_size_for_generate))

                rl_loss.backward()

            self.generator.optimizer.step()

        total_reward = total_reward / n_batch

        batch_distinct_rewards = np.concatenate((batch_distinct_rewards, np.expand_dims(end_of_batch_rewards, axis=1)), axis=1)

        batch_distinct_rewards = np.mean(batch_distinct_rewards, axis=0)

        return total_reward, rl_loss.item(), batch_distinct_rewards, n_to_sample / n_batch, \
        clip_fraction / (self.args.n_update_steps * np.ceil(n_batch / self.args.batch_size_for_generate) * trajectory_input.shape[1] - 1)


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

    def finetune(self, gen_data):
        self.generator.fit(gen_data, self.args.batch_size_for_generate, self.args.n_finetune, 10000, 10000)




