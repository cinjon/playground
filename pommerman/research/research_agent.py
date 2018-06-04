from collections import deque

import numpy as np
from pommerman import constants
from pommerman import characters
from pommerman.agents import BaseAgent
import torch
from torch.autograd import Variable

import networks


class ResearchAgent(BaseAgent):

    def __init__(self, character=characters.Bomber,
                 **kwargs):
        super(ResearchAgent, self).__init__(character)
        # NOTE: This is assuming that our num_stack size is 2.
        self._recurrent_policy = kwargs.get('recurrent_policy')
        print("self recurrent policy: ", self._recurrent_policy)
        if self._recurrent_policy:
            self._num_stack = kwargs.get('num_stack', 1)
        else:
            self._num_stack = kwargs.get('num_stack', 2)
        print("self num stack ", self._num_stack)
        print("")
        # NOTE: This doesn't work in eval simple with more than one process
        # because it's getting passed args.num_processes.
        self._num_processes = kwargs.get('num_processes', 1)
        self._obs_stacks = [deque([], maxlen=self._num_stack)
                            for _ in range(self._num_processes)]
        self._cuda = kwargs.get('cuda', False)
        if self._actor_critic is not None:
            # This caveat is so that it works with both train and eval.
            self._states = torch.zeros(1, self._actor_critic.state_size)
        self._masks = torch.ones(1, 1)
        if self._cuda:
            self._masks = self._masks.cuda()

    @staticmethod
    def _featurize_obs(obs):
        if type(obs) != list:
            obs = [obs]

        obs = np.array([networks.featurize3D(o) for o in obs])
        obs = torch.from_numpy(obs)
        return obs.float()

    def clear_obs_stack(self, num_stack=None):
        if num_stack is None:
            [obs_stack.clear() for obs_stack in self._obs_stacks]
        else:
            self._obs_stacks[num_stack].clear()

    def act(self, obs, action_space):
        if type(obs) != list:
            obs = [obs]

        # NOTE: If this errors, did you set num_processes correctly?
        for num, o in enumerate(obs):
            self._obs_stacks[num].append(self._featurize_obs(o))

        lst_stacked_obs = []
        for num, obs_stack in enumerate(self._obs_stacks):
            stacked_obs = list(obs_stack)
            if len(stacked_obs) < self._num_stack:
                prepend = [stacked_obs[0]]*(self._num_stack - len(stacked_obs))
                stacked_obs = prepend + stacked_obs
            stacked_obs = torch.cat(stacked_obs, 1)
            lst_stacked_obs.append(stacked_obs)

        stacked_obs = torch.cat(lst_stacked_obs, 0)
        if self._cuda:
            stacked_obs = stacked_obs.cuda()
            self._states = self._states.cuda()

        _, action, _, states, _, _ = self._actor_critic.act(
            Variable(stacked_obs, volatile=True),
            Variable(self._states, volatile=True),
            Variable(self._masks, volatile=True),
            deterministic=True)
        self._states = states.data

        action = action.data.squeeze(1).cpu().numpy()
        if len(action) == 1:
            # NOTE: This is a special case to return a singular action if that
            # is all that was passed through.
            action = action[0]
        return action

    def act_on_data(self, observations, states, masks, deterministic=False):
        return self._actor_critic.act(observations, states, masks,
                                      deterministic)
