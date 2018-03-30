"""The baseline Pommerman environment.

This evironment acts as game manager for Pommerman. Further environments, such as in v1.py, will inherit from this.
"""
import json
import os

import numpy as np
from scipy.misc import imresize as resize
import time
from gym import spaces
from gym.utils import seeding
import gym

from v0 import Pomme as PommeV0

class Pomme(PommeV0):
    def _get_done(self):
        return self.model.get_done(
            self._agents, self._step_count, self._max_steps, self._game_type,
            self.training_agents, all_agents=True)
            
    
