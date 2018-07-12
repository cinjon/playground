"""The baseline Pommerman environment.
This evironment acts as game manager for Pommerman. Further environments,
such as in v1.py, will inherit from this.
"""
from collections import defaultdict
import logging
import os
import random

import numpy as np
import time
from gym import spaces
from gym.utils import seeding
import gym

from .. import characters
from .. import constants
from .. import forward_model
from .. import graphics
from .. import utility
from ..agents import SimpleAgent
from ..agents import ComplexAgent
from ..agents import AstarAgent
import json


class Pomme(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'rgb_pixel'],
    }

    def __init__(self,
                 render_fps=None,
                 game_type=None,
                 board_size=None,
                 agent_view_size=None,
                 num_rigid=None,
                 num_wood=None,
                 num_items=None,
                 max_steps=1000,
                 is_partially_observable=False,
                 default_bomb_life=None,
                 **kwargs
    ):
        self.render_fps = render_fps
        self._agents = None
        self._game_type = game_type
        self._board_size = board_size
        self._agent_view_size = agent_view_size
        self._num_rigid = num_rigid
        self._num_wood = num_wood
        self._num_items = num_items
        self._max_steps = max_steps
        self._viewer = None
        self._is_partially_observable = is_partially_observable
        self._default_bomb_life = default_bomb_life
        self._bomb_penalty_lambda = 1.0
        self._step_loss = 0.0
        self._bomb_reward = 0.0
        self._item_reward = 0.0
        self._selfbombing = False
        self._optimal_num_steps_directory = {}
        self._optimal_num_steps = None
        self._num_make = []
        self._num_inac = []

        self.training_agents = []
        self.model = forward_model.ForwardModel()

        # This can be changed through set_render_mode
        # or from the cli tool using '--render_mode=MODE_TYPE'
        self._mode = 'human'

        # Observation and Action Spaces. These are both geared towards a single
        # agent even though the environment expects actions and returns
        # observations for all four agents. We do this so that it's clear what
        # the actions and obs are for a single agent. Wrt the observations,
        # they are actually returned as a dict for easier understanding.
        self._set_action_space()
        self._set_observation_space()
        self.simple_expert = SimpleAgent()
        self.complex_expert = ComplexAgent(board_size=board_size)
        self.astar_expert = AstarAgent()

    def _set_action_space(self):
        self.action_space = spaces.Discrete(6)

    def set_render_mode(self, mode):
        self._mode = mode

    def set_bomb_penalty_lambda(self, l):
        self._bomb_penalty_lambda = l

    def enable_selfbombing(self):
        self._selfbombing = True

    def set_reward_shaping(self, step_loss, bomb_reward, item_reward):
        self._step_loss = step_loss
        self._bomb_reward = bomb_reward
        self._item_reward = item_reward

    def _set_observation_space(self):
        """The Observation Space for each agent.
        There are a total of 3*board_size^2+12 observations:
        - all of the board (board_size^2)
        - bomb blast strength (board_size^2).
        - bomb life (board_size^2)
        - agent's position (2)
        - player ammo counts (1)
        - blast strength (1)
        - can_kick (1)
        - teammate (one of {AgentDummy.value, Agent3.value}).
        - enemies (three of {AgentDummy.value, Agent3.value}).
        """
        bss = self._board_size**2
        min_obs = [0] * 3 * bss + [0] * 5 + [constants.Item.AgentDummy.value
                                            ] * 4
        max_obs = [len(constants.Item)] * bss + [self._board_size
                                                ] * bss + [25] * bss
        max_obs += [self._board_size] * 2 + [self._num_items] * 2 + [1]
        max_obs += [constants.Item.Agent3.value] * 4
        self.observation_space = spaces.Box(
            np.array(min_obs), np.array(max_obs))

    def set_agents(self, agents):
        self._agents = agents

    def set_training_agents(self, agent_ids):
        self.training_agents = agent_ids

    def set_uniform_v(self, v):
        self._uniform_v = v

    def set_state_directory(self, directory, distribution,
                            use_second_place=False):
        self._init_game_state_directory = directory
        self._game_state_distribution = distribution
        self._online_backplay = False
        if directory == "online":
            self._online_backplay = True
            return

        self._applicable_games = []
        if self._init_game_state_directory:
            for subdir in os.listdir(self._init_game_state_directory):
                path = os.path.join(self._init_game_state_directory, subdir)
                endgame_file = os.path.join(path, 'endgame.json')
                with open(endgame_file, 'r') as f:
                    endgame = json.loads(f.read())
                    if 'grid' not in path:
                        if use_second_place:
                            players = endgame['second']
                        else:
                            players = endgame['winners']
                        # An agent must be represented in the players.
                        if not any([agent in players
                                    for agent in self.training_agents]):
                            continue

                        # An agent must be alive.
                        alive = endgame.get('alive', self.training_agents)
                        if len(players) == 2 and not any([
                                agent in alive for agent in self.training_agents]):
                            continue

                    step_count = endgame['step_count']
                    self._applicable_games.append((path, step_count))
            # print("PRINT Environment has %d applicable games --> rank %d." % (
            #     len(self._applicable_games), self.rank))
            # if len(self._applicable_games) < 5:
            #     print(self._applicable_games)
            # logging.warn("LOG Environment has %d applicable games --> rank %d" % (
            #              len(self._applicable_games), self.rank))
            # logging.warn(self._applicable_games)

    def set_init_game_state(self, game_state_file):
        """Set the initial game state.
        The expected game_state_file JSON format is:
          - agents: list of agents serialized (agent_id, is_alive, position,
            ammo, blast_strength, can_kick)
          - board: board matrix topology (board_size^2)
          - board_size: board size
          - bombs: list of bombs serialized (position, bomber_id, life,
            blast_strength, moving_direction)
          - flames: list of flames serialized (position, life)
          - items: list of item by position
          - step_count: step count
        Args:
          game_state_file: JSON File input.
        """
        self._init_game_state = None
        if game_state_file:
            with open(game_state_file, 'r') as f:
                self._init_game_state = json.loads(f.read())

    def make_board(self):
        self._board = utility.make_board(self._board_size, self._num_rigid,
                                         self._num_wood)

    def make_items(self):
        self._items = utility.make_items(self._board, self._num_items)

    def clear_agent_obs(self):
        for agent in self._agents:
            if not agent.is_simple_agent:
                agent.clear_obs_stack()

    def act(self, obs, acting_agent_ids=[], ex_agent_ids=None):
        if ex_agent_ids is not None:
            agents = [agent for agent in self._agents \
                      if agent.agent_id not in ex_agent_ids]
        else:
            agents = [agent for agent in self._agents \
                      if agent.agent_id not in self.training_agents]
            # TODO: Replace this hack with something more reasonable.
            agents = [agent for agent in agents if \
                      agent.agent_id not in acting_agent_ids]
        return self.model.act(agents, obs, self.action_space)

    def get_expert_actions(self, data):
        obs, expert = data
        if expert == 'SimpleAgent':
            return self.model.expert_act(self.simple_expert, obs,
                                         self.action_space)
        elif expert == 'ComplexAgent':
            return self.model.expert_act(self.complex_expert, obs,
                                         self.action_space)
        elif expert == 'AstarAgent':
            return self.model.expert_act(self.astar_expert, obs,
                                         self.action_space)

    def get_observations(self):
        self.observations = self.model.get_observations(
            self._board, self._agents, self._bombs,
            self._is_partially_observable, self._agent_view_size,
            self._max_steps, step_count=self._step_count)
        return self.observations

    def _get_rewards(self):
        """If an agent dies in the first 100 steps, then it died from a bomb.
        In that case, multiply the -1 reward by the bomb_penalty_lambda.
        """
        rewards = self.model.get_rewards(self._agents, self._game_type,
                                         self._step_count, self._max_steps)
        if self._step_count < 100:
            rewards = [r * self._bomb_penalty_lambda if r == -1 else r
                       for r in rewards]
        return rewards

    def _get_done(self):
        return self.model.get_done(self._agents, self._step_count,
                                   self._max_steps, self._game_type,
                                   self.training_agents, all_agents=True)

    def _get_info(self, done, rewards):
        ret = self.model.get_info(done, rewards, self._game_type, self._agents,
                                  self.training_agents)
        ret['step_count'] = self._step_count
        if hasattr(self, '_game_state_step_start'):
            ret['game_state_step_start'] = self._game_state_step_start
            ret['game_state_step_start_beg'] = self._game_state_step_start_beg
            ret['game_state_file'] = self._game_state_file
        return ret

    def reset(self):
        assert (self._agents is not None)

        def get_game_state_file(directory, step_count):
            if self._game_state_distribution == 'uniform':
                # Pick a random game state to start from.
                step = random.choice(range(step_count))
            elif self._game_state_distribution == 'uniform21':
                # Pick a game state uniformly over the last 21.
                # NOTE: This is an effort to reduce the effect of the credit
                # assignment problem. If this works well, then we might be able
                # to move a sliding window back across epochs.
                step = random.choice(
                    range(max(0, step_count - 22), step_count - 1)
                )
            elif self._game_state_distribution == 'uniform33':
                # Pick a game state uniformly over the last 33.
                step = random.choice(
                    range(max(0, step_count - 34), step_count - 1)
                )
            elif self._game_state_distribution == 'uniform66':
                # Pick a game state uniformly over the last 66.
                step = random.choice(
                    range(max(0, step_count - 67), step_count - 1)
                )
            elif self._game_state_distribution == 'uniformAdapt':
                step = random.choice(
                    range(max(0, step_count - self._uniform_v), step_count - 1)
                )
            elif self._game_state_distribution.startswith('uniformSchedule'):
                # if uniform_v == 512 and step_count == 290, then this is range(0, 289)
                # ... eh, wtf.
                step = random.choice(
                    range(max(0, step_count - self._uniform_v), step_count - 1)
                )
            elif self._game_state_distribution.startswith('uniformBounds'):
                # (0, 32), (24, 64), (56, 128), (120, 256), (248, 512), (504, 1024), (1016, 2048)
                # --> (504, 1024) --> (0, step_count - 504)
                # --> (1016, 2048) --> (0, 1)
                lb = self._uniform_v
                if self._uniform_v < 40:
                    ub = 1
                else:
                    ub = int(lb / 2) - 8

                # at lb == 512, minrange is either 0 or step_count - 512
                # then maxrange is either 1, step_count - 511, or step_count - 244
                # step_count - 244 > step_count - 511, so it can't be the latter.
                # thus it's either 1 or step_count - 244.
                # step is then either 0, random(0, step_count - 244) if step_count > 244,
                # or random(step_count - 512, step_count - 244) if step_count > 512.
                minrange = max(0, step_count - lb)
                maxrange = max(minrange + 1, step_count - ub)
                step = random.choice(range(minrange, maxrange))
            elif self._game_state_distribution == 'setBoundsTst':
                # (0, 64), (50, 128)
                lb = self._uniform_v
                ub = {
                    64: 1, 128: 50,
                }.get(lb)
                minrange = max(0, step_count - lb)
                maxrange = max(minrange + 1, step_count - ub)
                step = random.choice(range(minrange, maxrange))
            elif self._game_state_distribution in [
                    'setBoundsA', 'setBoundsB', 'setBoundsC'
            ]:
                # (0, 64), (50, 128), (100, 256), (200, 384), (300, 512), (400, 640), (600, 800), (800, 700)
                lb = self._uniform_v
                ub = {
                    64: 1, 128: 50, 256: 100, 384: 200, 512: 300, 640: 400,
                    800: 600, 810: 700, 820: 650
                }.get(lb)
                lb = min(lb, 800)
                minrange = max(0, step_count - lb)
                maxrange = max(minrange + 1, step_count - ub)
                step = random.choice(range(minrange, maxrange))
            elif self._game_state_distribution in ['setBoundsD', 'setBoundsE', 'setBoundsF']:
                # (0, 32), (28, 64), (50, 128), (100, 256), (200, 384), (300, 512), (400, 640), (600, 800)
                lb = self._uniform_v
                ub = {
                    32: 1, 64: 28, 128: 50, 256: 100, 384: 200, 512: 300,
                    640: 400, 750: 550, 800: 600, 810: 700, 820: 650, 830: 620
                }.get(lb)
                lb = min(lb, 800)
                minrange = max(0, step_count - lb)
                maxrange = max(minrange + 1, step_count - ub)
                step = random.choice(range(minrange, maxrange))
            elif utility.is_int(self._game_state_distribution):
                game_state_int = int(self._game_state_distribution)
                step = random.choice(
                    range(max(0, step_count - game_state_int - 5),
                          max(0, step_count - game_state_int) + 5)
                )
            elif self._game_state_distribution == 'genesis':
                step = 0
            elif self._game_state_distribution.startswith('uniformForward'):
                step = random.choice(range(min(self._uniform_v, step_count - 1)))
            else:
                raise
            return os.path.join(directory, '%03d.json' % step), step

        if hasattr(self, '_applicable_games') and self._applicable_games:
            directory, step_count = random.choice(self._applicable_games)
            counter = 0
            while True:
                if counter == 5:
                    raise

                game_state_file, step = get_game_state_file(directory, step_count)
                counter += 1
                try:
                    while not os.path.exists(game_state_file):
                        game_state_file, step = get_game_state_file(directory, step_count)
                    self._game_state_step_start = step_count - step + 1
                    self._game_state_step_start_beg = step
                    self._game_state_file = game_state_file
                    with open(game_state_file, 'r') as f:
                        self.set_json_info(json.loads(f.read()))
                    break
                except json.decoder.JSONDecodeError as e:
                    print("PR --> GSF: %s / sc: %d / step: %d..." % (game_state_file, step_count, step))
                    logging.warn("LOG --> GSF: %s / sc: %d / step: %d..." % (game_state_file, step_count, step))
        elif self._init_game_state is not None:
            self.set_json_info()
        else:
            self._step_count = 0
            self.make_board()
            self.make_items()
            self._bombs = []
            self._flames = []
            self._powerups = []
            for agent_id, agent in enumerate(self._agents):
                pos = np.where(self._board == utility.agent_value(agent_id))
                row = pos[0][0]
                col = pos[1][0]
                agent.set_start_position((row, col))
                agent.reset()

        return self.get_observations()

    def seed(self, seed=None):
        gym.spaces.prng.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        max_blast_strength = self._agent_view_size or 10
        result = self.model.step(actions, self._board, self._agents,
                                 self._bombs, self._items, self._flames,
                                 max_blast_strength = max_blast_strength,
                                 selfbombing=self._selfbombing, do_print=self.rank == 0)
        self._board, self._agents, self._bombs = result[:3]
        self._items, self._flames = result[3:]

        # NOTE: this should be above calling the below functions since they
        # take the step_count to change obs etc., so step_count should be
        # updated before
        self._step_count += 1

        done = self._get_done()
        obs = self.get_observations()
        reward = self._get_rewards()
        info = self._get_info(done, reward)

        step_info = self.model.step_info
        for agent in self._agents:
            if not agent.is_alive:
                continue
            reward[agent.agent_id] -= self._step_loss
            if actions[agent.agent_id] == 5:
                reward[agent.agent_id] += self._bomb_reward
            if step_info[agent.agent_id].get('item'):
                reward[agent.agent_id] += self._item_reward

        return obs, reward, done, info

    def render(self,
               mode=None,
               close=False,
               record_pngs_dir=None,
               record_json_dir=None,
               do_sleep=True
    ):
        if close:
            self.close()
            return

        mode = mode or self._mode or 'human'

        if mode == 'rgb_array':
            rgb_array = graphics.PixelViewer.rgb_array(
                self._board, self._board_size, self._agents,
                self._is_partially_observable, self._agent_view_size)
            return rgb_array[0]

        if self._viewer is None:
            if mode == 'rgb_pixel':
                self._viewer = graphics.PixelViewer(
                    board_size=self._board_size,
                    agents=self._agents,
                    agent_view_size=self._agent_view_size,
                    partially_observable=self._is_partially_observable)
            else:
                self._viewer = graphics.PommeViewer(
                    board_size=self._board_size,
                    agents=self._agents,
                    partially_observable=self._is_partially_observable,
                    agent_view_size=self._agent_view_size,
                    game_type=self._game_type)

            self._viewer.set_board(self._board)
            self._viewer.set_agents(self._agents)
            self._viewer.set_step(self._step_count)
            self._viewer.render()

            # Register all agents which need human input with Pyglet.
            # This needs to be done here as the first `imshow` creates the
            # window. Using `push_handlers` allows for easily creating agents
            # that use other Pyglet inputs such as joystick, for example.
            for agent in self._agents:
                if agent.has_user_input():
                    self._viewer.window.push_handlers(agent)
        else:
            self._viewer.set_board(self._board)
            self._viewer.set_agents(self._agents)
            self._viewer.set_step(self._step_count)
            self._viewer.render()

        if record_pngs_dir:
            self._viewer.save(record_pngs_dir)
        if record_json_dir:
            self.record_json(os.path.join(record_json_dir, suffix))

        time.sleep(1.0 / self.render_fps)

    def record_json(self, directory):
        self.save_json(directory)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

        for agent in self._agents:
            agent.shutdown()

    @staticmethod
    def featurize(obs):
        board = obs["board"].reshape(-1).astype(np.float32)
        bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1) \
                                                        .astype(np.float32)
        bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
        position = utility.make_np_float(obs["position"])
        ammo = utility.make_np_float([obs["ammo"]])
        blast_strength = utility.make_np_float([obs["blast_strength"]])
        can_kick = utility.make_np_float([obs["can_kick"]])

        teammate = utility.make_np_float([obs["teammate"].value])
        enemies = utility.make_np_float([e.value for e in obs["enemies"]])
        return np.concatenate(
            (board, bomb_blast_strength, bomb_life, position, ammo,
             blast_strength, can_kick, teammate, enemies))

    def save_json(self, record_json_dir):
        info = self.get_json_info()
        count = "{0:0=3d}".format(self._step_count)
        suffix = count + '.json'
        path = os.path.join(record_json_dir, suffix)
        with open(path, 'w') as f:
            f.write(json.dumps(info, sort_keys=True, indent=4))

    def get_json_info(self):
        """Returns a json snapshot of the current game state."""
        ret = {
            'board_size': self._board_size,
            'step_count': self._step_count,
            'board': self._board,
            'agents': self._agents,
            'bombs': self._bombs,
            'flames': self._flames,
            'items': [[k, i] for k, i in self._items.items()]
        }
        for key, value in ret.items():
            ret[key] = json.dumps(value, cls=utility.PommermanJSONEncoder)
        return ret

    def set_json_info(self, game_state=None):
        """Sets the game state.

        If no game_state is given, then uses the init_game_state.
        """
        game_state = game_state or self._init_game_state

        self._items = {}
        item_array = json.loads(game_state['items'])
        for position, item_num in item_array:
            if item_num == 9:
                continue
            self._items[tuple(position)] = item_num

        board_size = int(game_state['board_size'])
        self._board_size = board_size
        self._step_count = int(game_state['step_count'])

        board_array = json.loads(game_state['board'])
        self._board = np.ones((board_size, board_size)).astype(np.uint8)
        self._board *= constants.Item.Passage.value
        for x in range(self._board_size):
            for y in range(self._board_size):
                self._board[x, y] = board_array[x][y]

        agent_array = json.loads(game_state['agents'])
        for a in agent_array:
            agent = next(x for x in self._agents \
                         if x.agent_id == a['agent_id'])
            agent.set_start_position((a['position'][0], a['position'][1]))
            agent.reset(
                int(a['ammo']), bool(a['is_alive']), int(a['blast_strength']),
                bool(a['can_kick']))

        self._bombs = []
        bomb_array = json.loads(game_state['bombs'])
        for b in bomb_array:
            bomber = next(x for x in self._agents \
                          if x.agent_id == b['bomber_id'])
            moving_direction = b['moving_direction']
            if moving_direction is not None:
                moving_direction = constants.Action(moving_direction)
            self._bombs.append(characters.Bomb(
                bomber, tuple(b['position']), int(b['life']),
                int(b['blast_strength']), moving_direction)
            )

        self._flames = []
        flameArray = json.loads(game_state['flames'])
        for f in flameArray:
            self._flames.append(
                characters.Flame(tuple(f['position']), f['life']))
