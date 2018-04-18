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
        self._num_stack = kwargs.get('num_stack', 1)
        self._obs_stack = deque([], maxlen=self._num_stack)

    def act(self, obs, action_space):
        obs = networks.featurize3D(obs)
        obs = torch.from_numpy(obs) # 18,13,13
        self._obs_stack.append(obs)
        stacked_obs = list(self._obs_stack) # [(18,13,13)] --> [(18,13,13)]*2
        if len(stacked_obs) < self._num_stack:
            prepend = [stacked_obs[0]]*(self._num_stack - len(stacked_obs))
            stacked_obs = prepend + stacked_obs
        stacked_obs = torch.cat(stacked_obs).unsqueeze(0).float() # 1,36,13,13
        masks = torch.ones(1, 1)
        value, action, _, states = self._actor_critic.act(
            Variable(stacked_obs, volatile=True),
            Variable(self._states, volatile=True),
            Variable(masks, volatile=True),
            deterministic=True)
        self._states = states.data
        action = action.data.squeeze(1).cpu().numpy()[0]
        return action
