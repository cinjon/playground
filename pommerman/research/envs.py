from collections import deque
import os

from baselines import bench
import gym
from gym import spaces
import numpy as np

import pommerman


def make_env(args, config, rank, training_agents=[]):
    def _thunk():
        game_type = config['game_type']
        agent_type = config['agent']
        env = config['env'](**config["env_kwargs"])
        env.seed(args.seed + rank)

        if args.how_train == 'simple':
            agents = [pommerman.agents.SimpleAgent(agent_type(game_type=game_type))
                      for _ in range(3)]
            training_agent_id = rank % 4
            agents.insert(training_agent_id, training_agents[0])
            for agent_id, agent in enumerate(agents):
                agent.set_agent_id(agent_id)
            env.set_agents(agents)
            env.set_training_agents([training_agent_id])
            env.set_init_game_state(args.game_state_file)
        elif args.how_train == 'homogenous':
            # NOTE: We can't use just one agent character here because it needs to track its own state.
            # We do that by instantiating three more copies. There is probably a better way.
            if rank > -1:
                copies = [
                    training_agents[0].copy(
                        agent_type(agent_id=agent_id, game_type=game_type)
                    )
                    for agent_id in range(4)
                ]
            else:
                copies = training_agents*4
            env.set_agents(copies)
            env.set_training_agents(list(range(4)))
            env.set_init_game_state(args.game_state_file)
        else:
            raise

        env = WrapPomme(env, args.how_train)
        env = MultiAgentFrameStack(env, args.num_stack)
        return env
    return _thunk


class WrapPomme(gym.ObservationWrapper):
    def __init__(self, env=None, how_train='simple'):
        super(WrapPomme, self).__init__(env)
        self._how_train = how_train

        # TODO: make obs_shape an argument.
        obs_shape = (25,13,13)
        self.observation_space = spaces.Box(
            self.observation_space.low[0], self.observation_space.high[0],
            [len(self.env.training_agents), obs_shape[0], obs_shape[1], obs_shape[2]],
            dtype=np.float32
        )
        self.render_fps = env._render_fps

    def _filter(self, arr):
        # TODO: Is arr always an np.array If so, can make this better.
        return np.array([arr[i] for i in self.env.training_agents])

    def observation(self, observation):
        filtered = self._filter(observation)
        # TODO: Consider removing the original featurize3D in favor of v2.
        return np.array([self._featurize3D(obs) for obs in filtered])

    def step(self, actions):
        if self._how_train == 'simple':
            obs = self.env.get_observations()
            all_actions = self.env.act(obs)
            all_actions.insert(self.env.training_agents[0], actions)
        elif self._how_train == 'homogenous':
            all_actions = actions

        observation, reward, done, info = self.env.step(all_actions)
        return self.observation(observation), self._filter(reward), \
            self._filter(done), info

    def reset(self, **kwargs):
        return self.observation(self.env.reset())

    @staticmethod
    def _featurize3D(obs):
        """Create 3D Feature Maps for Pommerman.
        Args:
          obs: The observation input. Should be for a single agent.

        Returns:
          A 3D Feature Map where each map is bsXbs. The 19 features are:
          - (2) Bomb blast strength and Bomb life.
          - (4) Agent position, ammo, blast strength, can_kick.
          - (2) Whether has teammate, teammate's position
          - (3) Enemies's position.
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

        # position of self.
        position = np.zeros((map_size, map_size)).astype(np.float32)
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

        if obs["teammate"] == pommerman.constants.Item.AgentDummy:
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


#######
# The following were graciously taken from baselines because we don't want to install cv2.
#######

class MultiAgentFrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1] * k, shp[2], shp[3]), dtype=np.uint8)
        self.render_fps = env.render_fps

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
