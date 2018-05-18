"""Module to create the environment and apply wrappers."""
from collections import deque
import os
import random

import gym
from gym import spaces
import numpy as np
import pommerman

import networks
from subproc_vec_env import SubprocVecEnv

from pommerman.constants import GameType


def _make_train_env(config, how_train, seed, rank, game_state_file,
                    training_agents, num_stack, do_filter_team=True,
                    state_directory=None, state_directory_distribution=None):
    """Makes an environment callable for multithreading purposes.
    Args:
      config: See the arguments module's config options.
      how_train: Str for the method for training. 'heterogenous' is not
        supported yet.
      seed: The random seed to use.
      rank: The environment count.
      game_state_file: Str location of game state from which to instantiate.
      training_agents: The list of training agents to use.
      num_stack: For stacking frames.
      do_filter_team: Whether we should filter the full team.
      state_directory: A directory of game states to use as inputs. These will
        be loaded randomly by game sub-directory and then by the given
        state_directory_distribution.
      state_directory_distribution: The distribution of how to load a random
        game's state. Options are "uniform" (equally likely to load any of the
        states in the game) and "backloaded" (20% chance of loading each of
        the last three states, 5% chance of loading the next six states,
        uniform chance of remainder).
    Returns a callable to instantiate an environment fit for our PPO training
      purposes.
    """
    def _thunk():
        if how_train == 'dummy':
            agents = [pommerman.agents.SimpleAgent() for _ in range(4)]
            training_agent_ids = []
        elif how_train == 'simple' or how_train == 'dagger':
            training_agent_ids = [rank % 4]
            agents = [pommerman.agents.SimpleAgent() for _ in range(3)]
            agents.insert(training_agent_ids[0], training_agents[0])
        elif how_train == 'homogenous':
            # NOTE: Here we have two training agents on a team together. They
            # are against two other agents that are prior versions of itself.
            training_agent_ids = [[0, 2], [1, 3]][rank % 2]
            agents = [training_agents[0].copy_ex_model()
                      for _ in range(4)]
        elif how_train == 'qmix':
            # randomly pick team [0,2] or [1,3]
            training_agent_ids = [[0, 2], [1, 3]][random.randint(0, 1)]
            agents = [pommerman.agents.SimpleAgent() for _ in range(2)]
            agents.insert(training_agent_ids[0], training_agents[0])
            agents.insert(training_agent_ids[1], training_agents[1])
        else:
            raise

        env = pommerman.make(config, agents, game_state_file,
                             render_mode='rgb_pixel')
        if rank != -1:
            env.seed(seed + rank)
        else:
            env.seed(seed)
        env.rank = rank

        env.set_training_agents(training_agent_ids)
        env.set_state_directory(state_directory, state_directory_distribution)

        if config == 'PommeFFAEasy-v0' or config == 'PommeFFAEasy-v3' or \
            config == 'PommeTeamEasy-v0' or config == 'PommeTeamEasy-v3':
            env = WrapPomme(env, how_train, do_filter_team=do_filter_team)
        else:
            env = WrapPomme(env, how_train, do_filter_team=do_filter_team)

        env = MultiAgentFrameStack(env, num_stack)
        return env
    return _thunk

# NOTE: should we use a different seed + rank (maybe multiplied/increased
# by some large number) so that the eval seeds are different from training seeds?
def _make_eval_env(config, how_train, seed, rank, agents, training_agent_ids,
                   acting_agent_ids, num_stack, state_directory=None,
                   state_directory_distribution=None):

    """Makes an environment callable for multithreading purposes.
    Used in conjunction with eval.py
    Args:
      config: See the arguments module's config options.
      how_train: Str for the method for training. 'heterogenous' is not
        supported yet.
      seed: The random seed to use.
      rank: The environment count.
      agents: The list of agents to use.
      training_agent_ids: The list of training agents to use.
      num_stack: For stacking frames.
    Returns a callable to instantiate an environment.
    """
    def _thunk():
        env = pommerman.make(config, agents, None, render_mode='rgb_pixel')
        env.seed(seed + rank + 1000)
        env.rank = rank
        env.set_training_agents(training_agent_ids)
        env.set_state_directory(state_directory, state_directory_distribution)
        env = WrapPommeEval(env, how_train, acting_agent_ids=acting_agent_ids)
        return env
    return _thunk


def make_train_envs(config, how_train, seed, game_state_file, training_agents,
                    num_stack, num_processes, do_filter_team=True,
                    state_directory=None, state_directory_distribution=None):
    envs = [
        _make_train_env(
            config=config, how_train=how_train, seed=seed, rank=rank,
            game_state_file=game_state_file, training_agents=training_agents,
            num_stack=num_stack, do_filter_team=do_filter_team,
            state_directory=state_directory,
            state_directory_distribution=state_directory_distribution
        )
        for rank in range(num_processes)
    ]
    return SubprocVecEnv(envs)


def make_eval_envs(config, how_train, seed, agents, training_agent_ids,
                   acting_agent_ids, num_stack, num_processes,
                   state_directory=None, state_directory_distribution=None):
    envs = [
        _make_eval_env(
            config=config, how_train=how_train, seed=seed, rank=rank,
            agents=agents, training_agent_ids=training_agent_ids,
            acting_agent_ids=acting_agent_ids, num_stack=num_stack,
            state_directory=state_directory,
            state_directory_distribution=state_directory_distribution)
        for rank in range(num_processes)
    ]
    return SubprocVecEnv(envs)


def get_env_shapes(config, num_stack):
    dummy_env = _make_train_env(config=config, how_train='dummy', seed=None,
                                rank=-1, game_state_file=None,
                                training_agents=[], num_stack=num_stack)()
    envs_shape = dummy_env.observation_space.shape[1:]
    obs_shape = (envs_shape[0], *envs_shape[1:])
    action_space = dummy_env.action_space
    return obs_shape, action_space


class WrapPommeEval(gym.ObservationWrapper):
    def __init__(self, env=None, how_train='simple', acting_agent_ids=None):
        super(WrapPommeEval, self).__init__(env)
        self._how_train = how_train
        self._acting_agent_ids = acting_agent_ids or self.env.training_agents
        self.render_fps = env.render_fps

    def step(self, actions):
        if self._how_train == 'simple' or self._how_train == 'dagger':
            obs = self.env.get_observations()
            all_actions = self.env.act(obs, ex_agent_ids=self._acting_agent_ids)
            training_agents = self.env.training_agents
            if training_agents and type(training_agents[0]) != pommerman.agents.SimpleAgent:
                if type(actions) == list:
                    if len(actions) > 1:
                        raise ValueError
                    else:
                        actions = actions[0]
                all_actions.insert(training_agents[0], actions)
        elif self._how_train == 'homogenous':
            all_actions = actions
        elif self._how_train == 'qmix':
            obs = self.env.get_observations()
            all_actions = self.env.act(obs)
            for id, action in zip(self.env.training_agents, actions):
                all_actions.insert(id, action)

        observation, reward, done, info = self.env.step(all_actions)
        if all(done):
            self.env.clear_agent_obs()

        obs = self.observation(observation)
        rew = reward
        done = done
        return obs, rew, done, info

    def _filter(self, arr):
        return np.array([arr[i] for i in self._acting_agent_ids])

    def _filter_team(self, arr):
        # NOTE: this is only for simple (single agent training) and team config
        acting_id = self._acting_agent_ids[0]
        teammate_id = (acting_id + 2) % 4
        return np.array([arr[acting_id], arr[teammate_id]])

    def observation(self, observation):
        return self._filter(observation)

    def reset(self):
        return self.observation(self.env.reset())

    def record_json(self, directory):
        self.env.record_json(directory)


class WrapPomme(gym.ObservationWrapper):
    def __init__(self, env=None, how_train='simple', acting_agent_ids=None,
                 do_filter_team=True):
        super(WrapPomme, self).__init__(env)
        self._how_train = how_train
        self._do_filter_team = do_filter_team
        self._acting_agent_ids = acting_agent_ids or self.env.training_agents
        board_size = env.spec._kwargs['board_size']
        obs_shape = (19, board_size, board_size)
        extended_shape = [len(self.env.training_agents), obs_shape[0],
                          obs_shape[1], obs_shape[2]]
        self.observation_space = spaces.Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            extended_shape,
            dtype=np.float32
        )
        self.render_fps = env.render_fps

    def _filter(self, arr):
        # TODO: Is arr always an np.array? If so, can do this better.
        acting_agent_ids = self._acting_agent_ids
        return np.array([arr[i] for i in acting_agent_ids])

    def _filter_team(self, arr):
        # NOTE: this is only for simple (single agent training) and team config
        acting_id = self._acting_agent_ids[0]
        teammate_id = (acting_id + 2) % 4
        return np.array([arr[acting_id], arr[teammate_id]])

    def observation(self, observation):
        filtered = self._filter(observation)
        return np.array([networks.featurize3D(obs) for obs in filtered])

    def get_global_obs(self):
        observation = self.env.get_observations()
        return np.array([networks.featurize3D(obs) for obs in observation])

    def get_non_training_obs(self):
        observation = self.env.get_observations()
        return [obs for num, obs in enumerate(observation) \
                if num not in self._acting_agent_ids]

    def set_bomb_penalty_lambda(self, l):
        self.env.set_bomb_penalty_lambda(l)

    def set_uniform_v(self, v):
        self.env.set_uniform_v(v)

    def get_training_ids(self):
        return self.env.training_agents

    def get_expert_obs(self):
        return self._filter(self.env.get_observations())

    def get_expert_actions(self, obs):
        return self.env.get_expert_actions(obs)

    def get_game_type(self):
        return self.env._game_type

    def step(self, actions):
        if self._how_train == 'simple' or self._how_train == 'dagger':
            obs = self.env.get_observations()
            all_actions = self.env.act(obs)
            all_actions.insert(self.env.training_agents[0], actions)
        elif self._how_train == 'homogenous':
            all_actions = actions
        elif self._how_train == 'qmix':
            obs = self.env.get_observations()
            all_actions = self.env.act(obs)
            for id, action in zip(self.env.training_agents, actions):
                all_actions.insert(id, action)

        observation, reward, done, info = self.env.step(all_actions)
        obs = self.observation(observation)

        # return done for the entire training_agent's team and reward
        if all([
                self._do_filter_team,
                self._how_train == 'simple',
                self.env._game_type == GameType.Team
        ]):
            done = self._filter_team(done)
            rew = self._filter_team(reward)
        else:
            done = self._filter(done)
            rew = self._filter(reward)

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.observation(self.env.reset())


#######
# The following were graciously taken from baselines because we don't want to
# install cv2.
#######
class MultiAgentFrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See LazyFrames below
        --------
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1] * k, shp[2], shp[3]),
            dtype=np.uint8)
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

    def __getattr__(self, attr):
        return getattr(self.env, attr)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are
        only stored once. It exists purely to optimize memory usage which can
        be huge for DQN's 1M frames replay buffers.
        This object should only be converted to numpy array before being passed
        to the model. You'd not believe how complex the previous solution was.
        """
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
