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

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_training_per_episode,
                                   num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps+1, num_training_per_episode,
                                num_processes, 1)
        self.dagger_actions = torch.zeros(num_steps, num_training_per_episode,
                                   num_processes, action_shape)

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.dagger_actions = self.dagger_actions.cuda()

    def insert(self, step, current_obs, state, action, action_log_prob,
               value_pred, reward, mask, dagger_action):
        self.observations[step+1].copy_(current_obs)
        self.states[step+1].copy_(state)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step+1].copy_(mask)

        if dagger_action is not None:
            self.dagger_actions[step].copy_(dagger_action)


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

    def feed_forward_generator(self, advantages, num_mini_batch, num_steps, kl_factor):
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

        batch_size = num_total * num_steps
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=False)

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
        dagger_actions_batch = None
        if kl_factor > 0:
            dagger_actions = self.dagger_actions.view([num_steps, num_total, *action_shape])

        for indices in sampler:
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
                dagger_actions_batch = dagger_actions \
                                .contiguous() \
                                .view((num_steps*num_total), 1)[indices]

            yield observations_batch, states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ, dagger_actions_batch

class ReplayBuffer:
    """This class implements a GPU-ready replay buffer."""

    def __init__(self, state_shape, action_shape, size=100000):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.buffer_size = size

        self.state_buffer = torch.zeros(0, *state_shape)
        self.action_buffer = torch.zeros(0, *action_shape).long()
        self.reward_buffer = torch.zeros(0, 1)
        self.next_state_buffer = torch.zeros(0, *state_shape)
        self.done_buffer = torch.zeros(0, 1).long()

        self._size = 0

    def __len__(self):
        return len(self.state_buffer)

    def cuda(self):
        self.state_buffer.cuda()
        self.action_buffer.cuda()
        self.reward_buffer.cuda()
        self.next_state_buffer.cuda()
        self.done_buffer.cuda()

    def push(self, state, action, reward, next_state, done, *args, **kwargs):
        incoming_size = state.shape[0]

        assert all([incoming_size == action.shape[0],
                    incoming_size == reward.shape[0],
                    incoming_size == next_state.shape[0],
                    incoming_size == done.shape[0]]), \
                    'Input tensors shape mismatch in dimension 0.'


        self._size += incoming_size

        if self._size > self.buffer_size:
            overflow = self._size - self.buffer_size

            self.state_buffer = self.state_buffer[overflow:]
            self.action_buffer = self.action_buffer[overflow:]
            self.reward_buffer = self.reward_buffer[overflow:]
            self.next_state_buffer = self.next_state_buffer[overflow:]
            self.done_buffer = self.done_buffer[overflow:]

            self._size = self.buffer_size

        self.state_buffer = torch.cat([self.state_buffer, state], dim=0)
        self.action_buffer = torch.cat([self.action_buffer, action], dim=0)
        self.reward_buffer = torch.cat([self.reward_buffer, reward], dim=0)
        self.next_state_buffer = torch.cat(
            [self.next_state_buffer, next_state], dim=0)
        self.done_buffer = torch.cat([self.done_buffer, done], dim=0)

    def sample(self, batch_size):
        assert batch_size <= self._size, \
            'Unable to sample {} items, current buffer size {}'.format(
                batch_size, self._size)

        batch_index = (torch.rand(batch_size) * self._size).long()

        state_batch = self.state_buffer.index_select(0, batch_index)
        action_batch = self.action_buffer.index_select(0, batch_index)
        reward_batch = self.reward_buffer.index_select(0, batch_index)
        next_state_batch = self.next_state_buffer.index_select(0, batch_index)
        done_batch = self.done_buffer.index_select(0, batch_index)

        return state_batch, action_batch, reward_batch, next_state_batch, \
            done_batch

    def __len__(self):
        return self._size


class EpisodeBuffer:
    def __init__(self, size=5000):
        self.buffer = deque(maxlen=size)

    def extend(self, history):
        self.buffer.extend(history)

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        assert batch_size <= self.__len__(), \
            'Unable to sample {} items, current buffer size {}'.format(
                batch_size, self.__len__())

        batch_index = random.sample(range(self.__len__()), batch_size)
        for index in batch_index:
            yield self.buffer[index]

    def __len__(self):
        return len(self.buffer)
