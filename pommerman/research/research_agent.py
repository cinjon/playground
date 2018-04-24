from collections import deque

import numpy as np
import torch
from torch.autograd import Variable

from pommerman import constants
from pommerman import characters
from pommerman.agents import BaseAgent

import networks


class ResearchAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""
    def __init__(self, character=characters.Bomber, **kwargs):
        super(ResearchAgent, self).__init__(character)
        # NOTE: This is assuming that our num_stack size is 2.
        self._num_stack = kwargs.get('num_stack', 2)
        self._obs_stack = deque([], maxlen=self._num_stack)
        self._cuda = kwargs.get('cuda', False)
        if self._actor_critic is not None:
            # This caveat is so that it works with both train and eval.
            self._states = torch.zeros(1, self._actor_critic.state_size)

    def act(self, obs, action_space):
        obs = networks.featurize3D(obs)
        obs = torch.from_numpy(obs)
        if self._cuda:
            obs = obs.cuda()
        self._obs_stack.append(obs)
        stacked_obs = list(self._obs_stack)
        if len(stacked_obs) < self._num_stack:
            prepend = [stacked_obs[0]]*(self._num_stack - len(stacked_obs))
            stacked_obs = prepend + stacked_obs
        stacked_obs = torch.cat(stacked_obs).unsqueeze(0).float()
        masks = torch.ones(1, 1)
        _, action, _, states, _, _ = self._actor_critic.act(
            Variable(stacked_obs, volatile=True),
            Variable(self._states, volatile=True),
            Variable(masks, volatile=True),
            deterministic=True)
        self._states = states.data
        action = action.data.squeeze(1).cpu().numpy()[0]
        return action

    def act_on_data(self, observations, states, masks, deterministic=False):
        return self._actor_critic.act(observations, states, masks,
                                      deterministic)
