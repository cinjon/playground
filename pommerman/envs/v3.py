"""Environment used in training that inherits from PommeV0.

The differences here are that this environment:
1. Uses a get_done that is agent aware.
2. Uses dense rewards:
  a. Killing another agent: 0.2
  b. Laying a bomb in proximity to an enemy: 0.1 / distance
  c. Picking up a good item: 0.1
  d. When another agent dies (that this agent didn't kill) 0.1
  e. Picking up a bad item: -0.1
"""
from collections import defaultdict
import json
import os

import numpy as np
from scipy.misc import imresize as resize
import time
from gym import spaces
from gym.utils import seeding
import gym

from .v0 import Pomme as PommeV0

class Pomme(PommeV0):
    def step(self, actions):
        """We modify this in order to include the step_info in the rewards."""
        result = self.model.step(actions, self._board, self._agents,
                                 self._bombs, self._items, self._flames)
        self._board, self._agents, self._bombs = result[:3]
        self._items, self._flames = result[3:]

        done = self._get_done()
        obs = self.get_observations()
        reward = self._get_rewards()
        info = self._get_info(done, reward)

        died_agents = []
        step_info = {}
        for id_ in self.training_agents:
            if id_ not in self.model.step_info:
                continue
            step_info[id_] = []
            for result in self.model.step_info[id_]:
                if result == 'died':
                    step_info[id_].append('died:%d' % self._step_count)
                    died_agents.append(id_)
                else:
                    step_info[id_].append(result)
                
        info['step_info'] = step_info

        for agent_id, info_ in step_info.items():
            if agent_id not in died_agents:
                reward[agent_id] += 0.1 * len(died_agents)

            for result in info_:
                if result == 'bad_item':
                    # Picked up a bad item.
                    reward[agent_id] -= 0.1
                elif result == 'good_item':
                    # Picked up a good item.
                    reward[agent_id] += 0.1
                elif result.startswith('bomb'):
                    # Layed a bomb.
                    spaces_to_enemy = int(result.split(':')[1])
                    if spaces_to_enemy < 6:
                        reward[agent_id] += 0.1 / float(spaces_to_enemy)  
                    reward[agent_id] += 0.1 / float(spaces_to_enemy)  

        self._step_count += 1
        return obs, reward, done, info

    def _get_done(self):
        return self.model.get_done(
            self._agents, self._step_count, self._max_steps, self._game_type,
            self.training_agents, all_agents=True)
            
    
