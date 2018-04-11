from collections import deque

import numpy as np
import torch
from torch.autograd import Variable

from pommerman import constants
from pommerman import characters
from pommerman.agents import BaseAgent


class ResearchAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""
    def __init__(self, character=characters.Bomber, **kwargs):
        super(ResearchAgent, self).__init__(character)
        self._num_stack = kwargs.get('num_stack', 1)
        self._obs_stack = deque([], maxlen=self._num_stack)

    def act(self, obs, action_space):
        obs = self._featurize3D(obs)
        obs = torch.from_numpy(obs) # 18,13,13
        self._obs_stack.append(obs)
        stacked_obs = list(self._obs_stack) # [(18,13,13)] --> [(18,13,13)]*2
        if len(stacked_obs) < self._num_stack:
            prepend = [stacked_obs[0]]*(self._num_stack - len(stacked_obs))
            stacked_obs = prepend + stacked_obs
        stacked_obs = torch.cat(stacked_obs).unsqueeze(0).float() # 1,36,13,13
        masks = torch.ones(1, 1)
        print("RAGEN: ", np.where(stacked_obs[0] != 0))
        value, action, _, states = self._actor_critic.act(
            Variable(stacked_obs, volatile=True),
            Variable(self._states, volatile=True),
            Variable(masks, volatile=True),
            deterministic=False)
        self._states = states.data
        action = action.data.squeeze(1).cpu().numpy()[0]
        print("AG ACT: ", action)
        return action

    def _featurize3D(self, obs):
        """Create 3D Feature Maps for Pommerman.

        Args:
          obs: The observation input. Should be for a single agent.

        Returns:
          A 3D Feature Map where each map is bsXbs. The 19 features are:
          - (2) Bomb blast strength and Bomb life.
          - (4) Agent position, ammo, blast strength, can_kick.
          - (1) Whether has teammate.
          - (1 / 0) If teammate, then the teammate's position.
          - (2 / 3) Enemies' positions.
          - (8) Positions for:
                Passage/Rigid/Wood/Flames/ExtraBomb/IncrRange/Kick/Skull
        """
        map_size = len(obs["board"])

        # feature maps with ints for bomb blast strength and life.
        bomb_blast_strength = obs["bomb_blast_strength"] \
                              .astype(np.float32) \
                              .reshape(1, map_size, map_size)
        bomb_life = obs["bomb_life"].astype(np.float32) \
                                    .reshape(1, map_size, map_size)

        # position of self. If the agent is dead, then this is all zeros.
        position = np.zeros((map_size, map_size)).astype(np.float32)
        if obs["is_alive"]:
            position[obs["position"][0], obs["position"][1]] = 1
        position = position.reshape(1, map_size, map_size)

        # ammo of self agent: constant feature map.
        ammo = np.ones((map_size, map_size)).astype(np.float32) * obs["ammo"]
        ammo = ammo.reshape(1, map_size, map_size)

        # blast strength of self agent: constant feature map
        blast_strength = np.ones((map_size, map_size)).astype(np.float32)
        blast_strength *= obs["blast_strength"]
        blast_strength = blast_strength.reshape(1, map_size, map_size)

        # whether the agent can kick: constant feature map of 1 or 0.
        can_kick = np.ones((map_size, map_size)).astype(np.float32)
        can_kick *= float(obs["can_kick"])
        can_kick = can_kick.reshape(1, map_size, map_size)

        if obs["teammate"] == constants.Item.AgentDummy:
            has_teammate = np.zeros((map_size, map_size)) \
                             .astype(np.float32) \
                             .reshape(1, map_size, map_size)
            teammate = None
        else:
            has_teammate = np.ones((map_size, map_size)) \
                             .astype(np.float32) \
                             .reshape(1, map_size, map_size)
            teammate = np.zeros((map_size, map_size)).astype(np.float32)
            teammate[np.where(obs["board"] == obs["teammate"].value)] = 1
            teammate = teammate.reshape(1, map_size, map_size)

        # Enemy feature maps.
        _enemies = obs["enemies"]
        enemies = np.zeros((len(_enemies), map_size, map_size)) \
                    .astype(np.float32)
        for i in range(len(_enemies)):
            enemies[i][np.where(obs["board"] == _enemies[i].value)] = 1

        items = np.zeros((8, map_size, map_size))
        for item_value in [0, 1, 2, 4, 6, 7, 8, 9]:
            items[i][obs["board"] == item_value] = 1

        feature_maps = np.concatenate((
            bomb_blast_strength, bomb_life, position, ammo, blast_strength,
            can_kick, items, has_teammate, enemies
        ))
        if teammate is not None:
            feature_maps = np.concatenate((feature_maps, teammate))

        return feature_maps
