"""Module to create the environment and apply wrappers."""
from collections import deque
import os
import random
import time

import gym
from gym import spaces
import numpy as np
import pommerman

import networks
from subproc_vec_env import SubprocVecEnv

from pommerman.constants import GameType


def _make_train_env(config, how_train, seed, rank, game_state_file,
                    training_agents, num_stack, do_filter_team=True,
                    state_directory=None, state_directory_distribution=None,
                    step_loss=0.0, bomb_reward=0.0, item_reward=0.0,
                    use_second_place=False, use_both_places=False,
                    frozen_agent=None, mix_frozen_complex=False,
                    florensa_starts_dir=None):
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
    simple_agent = pommerman.agents.SimpleAgent
    complex_agent = pommerman.agents.ComplexAgent
    astar_agent = pommerman.agents.AstarAgent

    def _thunk():
        ubp = use_both_places
        usp = use_second_place
        is_frozen_complex = False

        frozen_agent_id = None
        if how_train == 'dummy':
            agents = [simple_agent() for _ in range(4)]
            training_agent_ids = []
        elif how_train == 'simple' or how_train == 'dagger' or how_train == 'bc':
            training_agent_ids = [rank % 4]
            # training_agent_ids = [3]
            board_size = 8 if '8' in config else 11
            agents = [complex_agent(board_size=board_size) for _ in range(3)]
            agents.insert(training_agent_ids[0], training_agents[0])
        elif how_train == 'homogenous':
            # NOTE: Here we have two training agents on a team together. They
            # are against two other agents that are prior versions of itself.
            training_agent_ids = [[0, 2], [1, 3]][rank % 2]
            agents = [training_agents[0].copy_ex_model()
                      for _ in range(4)]
        elif how_train == 'backselfplay':
            # Here we have two training agents in an FFA, alongside two complex agents.
            board_size = 8 if '8' in config else 11
            agents = [complex_agent(board_size=board_size) for _ in range(2)]
            training_agent_ids = [rank % 4]
            # TODO: This is a hack because we know that our dataset is limited.
            if "fx-ffacompetition5-s100-complex/train" in state_directory:
                second_id = [3, 0, 0, 1][rank % 4]
            else:
                choices = [i for _ in range(4) if i != training_agent_ids[0]]
                second_id = np.random.choice(choices, 1)[0]
            training_agent_ids.append(second_id)
            training_agent_ids = sorted(training_agent_ids)
            agents.insert(training_agent_ids[0], training_agents[0])
            agents.insert(training_agent_ids[1], training_agents[0].copy_ex_model())
        elif how_train == 'frobackselfplay':
            # Here we have two training agents in an FFA, alongside two complex agents.
            # There is only one agent that is actually though. If the rank is 0-3 mod 8,
            # then we use the rank mod 4. if the rank is 4-7 mod 8, then we use the
            # corresponding second id to the rank mod 4.

            # NOTE: We turn off use_both_places here.
            ubp = False

            board_size = 8 if '8' in config else 11
            agents = [complex_agent(board_size=board_size) for _ in range(2)]

            rank_8 = rank % 8
            if rank_8 in range(4):
                usp = False
                training_agent_ids = [rank % 4]

                if "fx-ffacompetition5-s100-complex/train" in state_directory:
                    frozen_agent_id = [3, 0, 0, 1][rank % 4]
                else:
                    raise
            else:
                usp = True
                if "fx-ffacompetition5-s100-complex/train" in state_directory:
                    training_agent_ids = [[3, 0, 0, 1][rank % 4]]
                    frozen_agent_id = rank % 4
                else:
                    raise

            rank_16 = rank % 16
            is_frozen_complex = rank_16 > 7 and mix_frozen_complex

            # NOTE: This is a test. Remove it afterward.
            # is_frozen_complex = True

            if training_agent_ids[0] > frozen_agent_id:
                if is_frozen_complex:
                    agents.insert(frozen_agent_id, complex_agent(board_size=board_size))
                else:
                    agents.insert(frozen_agent_id, frozen_agent)
                agents.insert(training_agent_ids[0], training_agents[0])
            else:
                agents.insert(training_agent_ids[0], training_agents[0])
                if is_frozen_complex:
                    agents.insert(frozen_agent_id, complex_agent(board_size=board_size))
                else:
                    agents.insert(frozen_agent_id, frozen_agent)
        elif how_train == 'qmix':
            # randomly pick team [0,2] or [1,3]
            training_agent_ids = [[0, 2], [1, 3]][random.randint(0, 1)]
            agents = [simple_agent() for _ in range(2)]
            agents.insert(training_agent_ids[0], training_agents[0])
            agents.insert(training_agent_ids[1], training_agents[1])
        elif how_train == 'astar':
            agents = [astar_agent()]
            training_agent_ids = []
        elif how_train == 'grid' or how_train == 'tree':
            agents = training_agents
            training_agent_ids = [0]
        else:
            raise

        env = pommerman.make(config, agents, game_state_file)
                             # render_mode='rgb_pixel')
        if rank != -1:
            env.seed(seed + rank)
        else:
            env.seed(seed)
        env.rank = rank

        env.set_training_agents(training_agent_ids)
        env.set_state_directory(state_directory,
                                state_directory_distribution,
                                use_second_place=usp,
                                use_both_places=ubp)
        if florensa_starts_dir:
            env.set_florensa_starts_dir(florensa_starts_dir)
        env.set_is_frozen_complex(is_frozen_complex)
        env.set_reward_shaping(step_loss, bomb_reward, item_reward)
        env.frozen_agent_id = frozen_agent_id

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
        # NOTE: commented out to debug Grid-v4
        # env = MultiAgentFrameStack(env, 2)
        return env
    return _thunk


def make_train_envs(config, how_train, seed, game_state_file, training_agents,
                    num_stack, num_processes, do_filter_team=True,
                    state_directory=None, state_directory_distribution=None,
                    step_loss=None, bomb_reward=None, item_reward=None,
                    use_second_place=False, use_both_places=False, frozen_agent=None,
                    mix_frozen_complex=False,
                    florensa_starts_dir=None,
):
    envs = [
        _make_train_env(
            config=config, how_train=how_train, seed=seed, rank=rank,
            game_state_file=game_state_file, training_agents=training_agents,
            num_stack=num_stack, do_filter_team=do_filter_team,
            state_directory=state_directory,
            state_directory_distribution=state_directory_distribution,
            step_loss=step_loss, bomb_reward=bomb_reward, item_reward=item_reward,
            use_second_place=use_second_place, use_both_places=use_both_places,
            frozen_agent=frozen_agent, mix_frozen_complex=mix_frozen_complex,
            florensa_starts_dir=florensa_starts_dir
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
            state_directory_distribution=state_directory_distribution
        )
        for rank in range(num_processes)
    ]
    return SubprocVecEnv(envs)


def get_env_info(config, num_stack):
    if config == GameType.Grid:
        dummy_env = _make_train_env(config=config, how_train='astar', seed=None,
                                    rank=-1, game_state_file=None,
                                    training_agents=[], num_stack=num_stack)()
    else:
        dummy_env = _make_train_env(config=config, how_train='dummy', seed=None,
                                    rank=-1, game_state_file=None,
                                    training_agents=[], num_stack=num_stack)()
    envs_shape = dummy_env.observation_space.shape[1:]
    obs_shape = (envs_shape[0], *envs_shape[1:])
    action_space = dummy_env.action_space
    character = dummy_env.spec._kwargs['character']
    board_size = dummy_env.spec._kwargs['board_size']
    return obs_shape, action_space, character, board_size


class WrapPommeEval(gym.ObservationWrapper):
    def __init__(self, env=None, how_train='simple', acting_agent_ids=None):
        super(WrapPommeEval, self).__init__(env)
        self._how_train = how_train
        self._acting_agent_ids = acting_agent_ids or self.env.training_agents
        self.render_fps = env.render_fps
        # NOTE: commented out for debugging
        # board_size = env.spec._kwargs['board_size']
        # obs_shape = (19, board_size, board_size)
        # extended_shape = [len(self.env.training_agents), obs_shape[0],
        #                   obs_shape[1], obs_shape[2]]
        # self.observation_space = spaces.Box(
        #     self.observation_space.low[0],
        #     self.observation_space.high[0],
        #     extended_shape,
        #     dtype=np.float32
        # )

    def step(self, actions):
        if self._how_train in ['simple', 'dagger', 'astar', 'grid', 'tree', 'bc']:
            obs = self.env.get_observations()
            all_actions = self.env.act(obs, ex_agent_ids=self._acting_agent_ids)
            training_agents = self.env.training_agents
            if training_agents:
                for training_agent, action in zip(training_agents, actions):
                    all_actions.insert(training_agent, action)
        elif self._how_train == 'homogenous':
            all_actions = actions
        elif self._how_train == 'qmix':
            obs = self.env.get_observations()

            all_actions = self.env.act(obs)
            for id, action in zip(self.env.training_agents, actions):
                all_actions.insert(id, action)

        observation, reward, done, info = self.env.step(all_actions)
        if self._how_train == 'astar' and done:
            self.env.clear_agent_obs()
        elif self._how_train != 'astar' and all(done):
            self.env.clear_agent_obs()

        obs = self.observation(observation)
        rew = reward
        done = done
        return obs, rew, done, info

    # NOTE: this is used to record actions for BC
    # NOTE: for Pomme, if we only want to record the actions of one of them we
    # can set the rest as being ex_agent_ids, or self._acting_agent_ids
    def get_actions(self):
        obs = self.env.get_observations()
        all_actions = self.env.act(obs, ex_agent_ids=self._acting_agent_ids)
        return all_actions

    def _filter(self, arr):
        return np.array([arr[i] for i in self._acting_agent_ids])

    def _filter_team(self, arr):
        # NOTE: this is only for simple (single agent training) and team config
        acting_id = self._acting_agent_ids[0]
        teammate_id = (acting_id + 2) % 4
        return np.array([arr[acting_id], arr[teammate_id]])

    def observation(self, observation):
        return self._filter(observation)
        # NOTE: co for debugging
        # filtered = self._filter(observation)
        # return np.array([networks.featurize3D(obs) for obs in filtered])

    def enable_selfbombing(self):
        self.env.enable_selfbombing()

    def reset(self):
        return self.observation(self.env.reset())

    def record_json(self, directory):
        self.env.record_json(directory)

    def record_actions_json(self, directory, actions):
        self.env.record_actions_json(directory, actions)


class WrapPomme(gym.ObservationWrapper):
    def __init__(self, env=None, how_train='simple', acting_agent_ids=None,
                 do_filter_team=True):
        super(WrapPomme, self).__init__(env)
        self._how_train = how_train
        self._do_filter_team = do_filter_team
        self._acting_agent_ids = acting_agent_ids or self.env.training_agents
        board_size = env.spec._kwargs['board_size']
        if env.spec._kwargs['game_type'] == GameType.Grid:
            obs_shape = (5, board_size, board_size)
        elif env.spec._kwargs['game_type'] == GameType.Tree:
            obs_shape = (5, board_size, board_size)
        else:
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

    def get_dead_agents(self):
        agent_ids = [agent.agent_id for agent in self.env._agents \
                     if not agent.is_alive]
        training_agent_indices = [self.env.training_agents.index(id_)
                                  for id_ in agent_ids if id_ in self.env.training_agents]
        return training_agent_indices

    def set_bomb_penalty_lambda(self, l):
        self.env.set_bomb_penalty_lambda(l)

    def set_uniform_v(self, v):
        self.env.set_uniform_v(v)

    def set_florensa_starts_dir(self, d):
        self.env.set_florensa_starts_dir(d)

    def get_training_ids(self):
        return self.env.training_agents

    def enable_selfbombing(self):
        self.env.enable_selfbombing()

    def get_expert_obs(self):
        return self._filter(self.env.get_observations())

    def get_expert_actions(self, data):
        # data consists of obs and string expert.
        return self.env.get_expert_actions(data)

    def get_states_actions_json(self, data):
        return self.env.get_states_actions_json(data)

    def reset_state_file(self, directory):
        return self.env.reset_state_file(directory)

    def get_init_states_json(self, directory):
        return self.env.get_init_states_json(directory)

    def get_game_type(self):
        return self.env._game_type

    def get_json_info(self):
        return self.env.get_json_info()

    def set_json_info(self, game_state=None):
        return self.env.set_json_info(game_state=game_state)

    def step(self, actions):
        if self._how_train in ['simple', 'dagger', 'astar', 'grid', 'backselfplay', 'bc']:
            obs = self.env.get_observations()
            all_actions = self.env.act(obs)
            if type(actions) == list:
                for agent_id, action in zip(self.env.training_agents, actions):
                    all_actions.insert(agent_id, action)
            else:
                all_actions.insert(self.env.training_agents[0], actions)
        elif self._how_train == 'frobackselfplay':
            obs = self.env.get_observations()
            all_actions = self.env.act(obs)
            fid = self.env.frozen_agent_id
            tid = self.env.training_agents[0]
            if tid > fid:
                if not self.env.is_frozen_complex:
                    all_actions.insert(self.env.frozen_agent_id, actions[1])
                all_actions.insert(self.env.training_agents[0], actions[0])
            else:
                all_actions.insert(self.env.training_agents[0], actions[0])
                if not self.env.is_frozen_complex:
                    all_actions.insert(self.env.frozen_agent_id, actions[1])
            # if self.env.rank == 0:
            #     print("AFT FRO ALL ACTS: ", self.env.rank, all_actions, actions, self.env.training_ag
                      # ents[0], self.env.frozen_agent_id, "\n\n")
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
