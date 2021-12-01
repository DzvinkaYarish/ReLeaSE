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
        self.generator.optimizer.zero_grad()
        total_reward = 0

        trajectories = []
        fngps = []
        # for _ in range(n_batch):
            #
            # # Sampling new trajectory
            # reward = 0
            # trajectory = '<>'
            # while reward == 0:
            #     trajectory = self.generator.evaluate(data)
            #     if std_smiles:
            #             mol = Chem.MolFromSmiles(trajectory[1:-1])
            #             if mol:
            #                 fngp = mol2image(mol)
            #                 fngps.append(fngp)
            #                 trajectory = '<' + Chem.MolToSmiles(mol) + '>'
            #                 if isinstance(self.predictor, list):
            #                     reward, distinct_rwds = self.get_reward(self, self.args, [mol], np.expand_dims(fngp, axis=0),
            #                                                             self.predictor,
            #                                                             **kwargs)
            #                 else:
            #                     reward = self.get_reward(self.args, trajectory[1:-1],
            #                                              self.predictor,
            #                                              **kwargs)
            #                     distinct_rwds = []
            #
            #             else:
            #                 reward = 0
            #     else:
            #         if isinstance(self.predictor, list):
            #
            #             reward, distinct_rwds = self.get_reward(self.args, trajectory[1:-1],
            #                                                     self.predictor,
            #                                                     **kwargs)
            #         else:
            #             reward = self.get_reward(self.args, trajectory[1:-1],
            #                                      self.predictor,
            #                                      **kwargs)
            #             distinct_rwds = []
            #
            # batch_rewards.append(reward)
            # if distinct_rwds:
            #     batch_distinct_rewards.append(distinct_rwds)
            #
            # trajectories.append(trajectory)

        while len(trajectories) < n_batch:
            batch_trajectories = self.generator.evaluate(data)
            for tr in batch_trajectories:
                mol = Chem.MolFromSmiles(tr[1:-1])
                if mol:
                    trajectories.append(tr)
                if len(trajectories) == n_batch:
                    break




        end_of_batch_rewards = np.zeros((n_batch,))

        if self.get_end_of_batch_reward:
            fngps = [mol2image(Chem.MolFromSmiles(tr[1:-1])) for tr in trajectories]

            end_of_batch_rewards = self.get_end_of_batch_reward(fngps)
        batch_rewards, batch_distinct_rewards = self.get_reward(self.args, [tr[1:-1] for tr in trajectories], self.predictor, **kwargs)




        # Converting string of characters into tensor
        trajectory_input, _ = data.pad_sequences(trajectories, pad_symbol=data.pad_symbol)
        trajectory_input, _ = data.seq2tensor(trajectory_input, data.tokens, flip=False)
        trajectory_input = torch.tensor(trajectory_input).long()
        if self.args.use_cuda:
            trajectory_input = trajectory_input.cuda()

        discounted_reward = torch.Tensor(batch_rewards + end_of_batch_rewards).long()
        if self.args.use_cuda:
            discounted_reward = discounted_reward.cuda()

        total_reward += np.sum(batch_rewards)
        total_reward += np.sum(end_of_batch_rewards)


        # Initializing the generator's hidden state
        hidden = self.generator.init_hidden(batch_size=n_batch)
        if self.generator.has_cell:
            cell = self.generator.init_cell(batch_size=n_batch)
            hidden = (hidden, cell)
        if self.generator.has_stack:
            stack = self.generator.init_stack(batch_size=n_batch)
        else:
            stack = None

        # "Following" the trajectory and accumulating the loss
        rl_loss = torch.zeros((n_batch,))
        if self.args.use_cuda:
            rl_loss = rl_loss.cuda()

        for p in range(trajectory_input.shape[1] - 1):
            output, hidden, stack = self.generator(trajectory_input[:, p],
                                                   hidden,
                                                   stack)
            indx_terminal = (trajectory_input[:, p] == data.char2idx[data.end_token]) + (trajectory_input[:, p] == data.char2idx[data.pad_symbol])
            log_probs = F.log_softmax(output, dim=1)
            top_i = trajectory_input[:, p + 1].detach().numpy()


            discounted_reward[indx_terminal] = 0.
            actual_log_probs = log_probs[np.arange(0, n_batch), top_i]

            rl_loss -= (actual_log_probs * discounted_reward)
            discounted_reward = discounted_reward * gamma

        # Doing backward pass and parameters update
        rl_loss = rl_loss.mean()

        rl_loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
                                           grad_clipping)
        self.generator.optimizer.step()

        total_reward = total_reward / n_batch

        batch_distinct_rewards = np.concatenate((batch_distinct_rewards, np.expand_dims(end_of_batch_rewards, axis=1)), axis=1)

        batch_distinct_rewards = np.mean(batch_distinct_rewards, axis=0)

        return total_reward, rl_loss.item(), batch_distinct_rewards


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



