import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque
import random

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 state_size, num_training_per_episode):
        self.observations = torch.zeros(
            num_steps+1, num_training_per_episode, num_processes, *obs_shape)
        self.states = torch.zeros(
            num_steps+1, num_training_per_episode, num_processes, state_size)
        self.rewards = torch.zeros(
            num_steps, num_training_per_episode, num_processes, 1)
        self.value_preds = torch.zeros(
            num_steps+1, num_training_per_episode, num_processes, 1)
        self.returns = torch.zeros(
            num_steps+1, num_training_per_episode, num_processes, 1)
        self.action_log_probs = torch.zeros(
            num_steps, num_training_per_episode, num_processes, 1)

        action_shape = 1
        self.actions = torch.zeros(num_steps, num_training_per_episode,
                                   num_processes, action_shape).long()
        self.masks = torch.ones(num_steps+1, num_training_per_episode,
                                num_processes, 1)
        self.dagger_probs_distr = torch.zeros(
            num_steps, num_training_per_episode, num_processes, action_space.n)
        self.action_log_probs_distr = torch.zeros(
            num_steps, num_training_per_episode, num_processes, action_space.n)

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.action_log_probs_distr = self.action_log_probs_distr.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.dagger_probs_distr = self.dagger_probs_distr.cuda()

    def insert(self, step, current_obs, state, action, action_log_prob,
               value_pred, reward, mask, action_log_probs_distr,
               dagger_probs_distr):
        self.observations[step+1].copy_(current_obs)
        self.states[step+1].copy_(state)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step+1].copy_(mask)

        if dagger_probs_distr is not None:
            self.dagger_probs_distr[step].copy_(dagger_probs_distr)
        if action_log_probs_distr is not None:
            self.action_log_probs_distr[step].copy_(action_log_probs_distr)

    def after_epoch(self):
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau, num_agent=0):
        if use_gae:
            self.value_preds[-1, num_agent] = next_value
            masks = self.masks
            value_preds = self.value_preds
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                mask = value_preds[step+1, num_agent]
                mask *= masks[step+1, num_agent]
                delta = self.rewards[step, num_agent]
                delta += gamma * mask - value_preds[step, num_agent]
                gae = delta + gamma * tau * gae * masks[step+1, num_agent]
                self.returns[step, num_agent] = value_preds[step, num_agent]
                self.returns[step, num_agent] += gae
        else:
            self.returns[-1, num_agent] = next_value
            for step in reversed(range(self.rewards.size(0))):
                rewards = self.rewards[step, num_agent]
                masks = self.masks[step+1, num_agent]
                next_returns = self.returns[step+1, num_agent]
                self.returns[step, num_agent] = next_returns * gamma * masks
                self.returns[step, num_agent] += rewards

    def compute_advantages(self):
        return self.returns[:-1] - self.value_preds[:-1]

    def feed_forward_generator(self, advantages, num_mini_batch, batch_size,
                               num_steps, kl_factor):
        # TODO: Consider excluding from the indices the rollouts where the
        # agent died before this rollout. They're signature is that every step
        # is masked out.
        advantages = advantages.view([-1, 1])
        num_steps = self.rewards.size(0)
        num_training_per_episode = self.rewards.size(1)
        num_processes = self.rewards.size(2)
        num_total = num_training_per_episode * num_processes
        obs_shape = self.observations.shape[3:]
        action_shape = self.actions.shape[3:]
        state_size = self.states.size(3)

        total_steps = num_training_per_episode * num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(total_steps)),
                               mini_batch_size, drop_last=True)

        # Reshape so that trajectories per agent look like new processes.
        observations = self.observations.view([
            num_steps+1, num_total, *obs_shape])
        states = self.states.view([num_steps+1, num_total, state_size])
        rewards = self.rewards.view([num_steps, num_total, 1])
        value_preds = self.value_preds.view([num_steps+1, num_total, 1])
        returns = self.returns.view([num_steps+1, num_total, 1])
        actions = self.actions.view([num_steps, num_total, *action_shape])
        action_log_probs = self.action_log_probs.view([
            num_steps, num_total, 1])
        masks = self.masks.view([num_steps+1, num_total, 1])

        if kl_factor > 0:
            distr_shape = self.dagger_probs_distr.shape[3:]
            dagger_probs_distr = self.dagger_probs_distr.view(
                [num_steps, num_total, *distr_shape])
            action_log_probs_distr = self.action_log_probs_distr.view(
                [num_steps, num_total, *distr_shape])

        counter = 0
        for indices in sampler:
            if counter > 5:
                break
            counter += 1
            
            indices = torch.LongTensor(indices)

            if advantages.is_cuda:
                indices = indices.cuda()

            observations_batch = observations[:-1] \
                                 .contiguous() \
                                 .view((num_steps*num_total),
                                       *observations.size()[2:])[indices]
            states_batch = states[:-1] \
                           .contiguous() \
                           .view((num_steps*num_total), 1)[indices]
            actions_batch = actions \
                            .contiguous() \
                            .view((num_steps*num_total), 1)[indices]
            return_batch = returns[:-1] \
                           .contiguous() \
                           .view((num_steps*num_total), 1)[indices]
            masks_batch = masks[:-1] \
                          .contiguous() \
                          .view((num_steps*num_total), 1)[indices]
            old_action_log_probs_batch = \
                                action_log_probs \
                                .contiguous() \
                                .view((num_steps*num_total), 1)[indices]
            adv_targ = advantages.contiguous().view(-1, 1)[indices]

            if kl_factor > 0:
                # TODO: Change the hard-coded 6.
                dagger_probs_distr_batch = dagger_probs_distr \
                                           .contiguous() \
                                           .view((num_steps*num_total), 6)
                dagger_probs_distr_batch = dagger_probs_distr_batch[indices]
                action_log_probs_distr_batch = action_log_probs_distr \
                                               .contiguous() \
                                               .view((num_steps*num_total), 6)
                action_log_probs_distr_batch = action_log_probs_distr_batch[indices]
            else:
                dagger_probs_distr_batch = None
                action_log_probs_distr_batch = None

            yield observations_batch, states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ, action_log_probs_distr_batch, \
                dagger_probs_distr_batch

    def recurrent_generator(self, advantages, num_mini_batch, batch_size,
                            num_steps, kl_factor):
        advantages = advantages.view([-1, 1])
        num_steps = self.rewards.size(0)
        num_training_per_episode = self.rewards.size(1)
        num_processes = self.rewards.size(2)
        num_total = num_training_per_episode * num_processes
        obs_shape = self.observations.shape[3:]
        action_shape = self.actions.shape[3:]
        state_size = self.states.size(3)
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)

        # Reshape so that trajectories per agent look like new processes.
        observations = self.observations.view([
            num_steps+1, num_total, *obs_shape])
        states = self.states.view([num_steps+1, num_total, state_size])
        rewards = self.rewards.view([num_steps, num_total, 1])
        value_preds = self.value_preds.view([num_steps+1, num_total, 1])
        returns = self.returns.view([num_steps+1, num_total, 1])
        actions = self.actions.view([num_steps, num_total, *action_shape])
        action_log_probs = self.action_log_probs.view([
            num_steps, num_total, 1])
        masks = self.masks.view([num_steps+1, num_total, 1])

        if kl_factor > 0:
            distr_shape = self.dagger_probs_distr.shape[3:]
            dagger_probs_distr = self.dagger_probs_distr.view(
                [num_steps, num_total, *distr_shape])
            action_log_probs_distr = self.action_log_probs_distr.view(
                [num_steps, num_total, *distr_shape])

        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = []
            states_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            dagger_probs_distr_batch = []
            action_log_probs_distr_batch = []

            for offset in range(num_envs_per_batch):
                indices = perm[start_ind + offset]
                observations_batch.append(observations[:-1] \
                                     .contiguous() \
                                     .view((num_steps*num_total), \
                                           *observations.size()[2:])[indices])
                states_batch.append(states[:-1] \
                               .contiguous() \
                               .view((num_steps*num_total), states.size()[2])[indices])
                actions_batch.append(actions \
                                .contiguous() \
                                .view((num_steps*num_total), 1)[indices])
                return_batch.append(returns[:-1] \
                               .contiguous() \
                               .view((num_steps*num_total), 1)[indices])
                masks_batch.append(masks[:-1] \
                              .contiguous() \
                              .view((num_steps*num_total), 1)[indices])
                old_action_log_probs_batch.append(action_log_probs \
                                    .contiguous() \
                                    .view((num_steps*num_total), 1)[indices])
                adv_targ.append(advantages.contiguous().view(-1, 1)[indices])

                if kl_factor > 0:
                    # TODO: Change the hard-coded 6.
                    dagger_probs_distr_batch.append(dagger_probs_distr \
                                               .contiguous() \
                                               .view((num_steps*num_total), 6)[indices])
                    action_log_probs_distr_batch.append(action_log_probs_distr \
                                                   .contiguous() \
                                                   .view((num_steps*num_total), 6)[indices])

            observations_batch = torch.cat(observations_batch, 0) \
                                 .view(num_envs_per_batch, *observations.size()[2:])
            states_batch = torch.cat(states_batch, 0) \
                           .view(num_envs_per_batch, states.size()[2])
            actions_batch = torch.cat(actions_batch, 0) \
                            .view(num_envs_per_batch, 1)
            return_batch = torch.cat(return_batch, 0) \
                           .view(num_envs_per_batch, 1)
            masks_batch = torch.cat(masks_batch, 0) \
                          .view(num_envs_per_batch, 1)
            old_action_log_probs_batch = torch.cat(old_action_log_probs_batch, 0) \
                                         .view(num_envs_per_batch, 1)
            adv_targ = torch.cat(adv_targ, 0) \
                       .view(num_envs_per_batch, 1)

            if kl_factor > 0:
                dagger_probs_distr_batch = torch.cat(dagger_probs_distr_batch, 0) \
                                           .view(num_envs_per_batch, 6)
                action_log_probs_distr_batch = torch.cat(action_log_probs_distr_batch, 0) \
                                               .view(num_envs_per_batch, 6)
            else:
                dagger_probs_distr_batch = None
                action_log_probs_distr_batch = None

            yield observations_batch, states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ, action_log_probs_distr_batch, dagger_probs_distr_batch


class CPUReplayBuffer:
    """
    This buffer stores episode rollouts. Each call to `extend` must contain a list of
    episodes and will be sampled by using a call to `sample`. Note that the `sample`
    function is a generator and should be looped over using iterators like `for`. This
    should (possibly?) save memory copying all at once. Each iterate will have whatever
    list sent in to the `extend` call.
    """
    def __init__(self, size=5000):
        self.buffer = deque(maxlen=size)

    def append(self, history_item):
        self.buffer.append(history_item)

    def extend(self, history):
        assert isinstance(history, list), 'The argument should be a list of episode rollouts'
        self.buffer.extend(history)

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        assert batch_size <= self.__len__(), \
            'Unable to sample {} items, current buffer size {}'.format(
                batch_size, self.__len__())

        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
