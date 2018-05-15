"""The baseline Pommerman environment.
This evironment acts as game manager for Pommerman. Further environments,
such as in v1.py, will inherit from this.
"""
from collections import defaultdict
import json
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
                 use_skull=True,
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
        self._use_skull = use_skull

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
        self.expert = SimpleAgent()

    def _set_action_space(self):
        self.action_space = spaces.Discrete(6)

    def set_render_mode(self, mode):
        self._mode = mode

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
        min_obs = [0]*3*bss + [0]*5 + [constants.Item.AgentDummy.value]*4
        max_obs = [len(constants.Item)]*bss + [self._board_size]*bss + [25]*bss
        max_obs += [self._board_size]*2 + [self._num_items]*2 + [1]
        max_obs += [constants.Item.Agent3.value]*4
        self.observation_space = spaces.Box(np.array(min_obs),
                                            np.array(max_obs))

    def set_agents(self, agents):
        self._agents = agents

    def set_training_agents(self, agent_ids):
        self.training_agents = agent_ids

    def set_state_directory(self, directory, distribution):
        self._init_game_state_directory = directory
        self._game_state_distribution = distribution
        self._applicable_games = []
        if self._init_game_state_directory:
            for directory in os.listdir(self._init_game_state_directory):
                path = os.path.join(self._init_game_state_directory, directory)
                endgame_file = os.path.join(path, 'endgame.json')
                with open(endgame_file, 'r') as f:
                    endgame = json.loads(f.read())
                    winners = endgame['winners']
                    # An agent must be represented in the winners.
                    if not any([agent in winners
                                for agent in self.training_agents]):
                        continue

                    # An agent must be alive.
                    alive = endgame.get('alive', self.training_agents)
                    if len(winners) == 2 and not any([
                            agent in alive for agent in self.training_agents]):
                        continue

                    step_count = endgame['step_count']
                    # print("%d: " % self.rank, self.training_agents, endgame)
                    self._applicable_games.append((path, step_count))
            print("Environment has %d applicable games." % \
                  len(self._applicable_games))

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
        self._items = utility.make_items(self._board, self._num_items, self._use_skull)

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

    def get_expert_actions(self, obs):
        return self.model.expert_act(self.expert, obs, self.action_space)

    def get_observations(self):
        self.observations = self.model.get_observations(
            self._board, self._agents, self._bombs,
            self._is_partially_observable, self._agent_view_size,
            self._max_steps, step_count=self._step_count)
        return self.observations

    def _get_rewards(self):
        return self.model.get_rewards(self._agents, self._game_type,
                                      self._step_count, self._max_steps)

    def _get_done(self):
        return self.model.get_done(self._agents, self._step_count,
                                   self._max_steps, self._game_type,
                                   self.training_agents, all_agents=True)

    def _get_info(self, done, rewards):
        ret = self.model.get_info(done, rewards, self._game_type, self._agents)
        ret['step_count'] = self._step_count
        return ret

    def reset(self):
        assert(self._agents is not None)

        # TODO: Position the agent in the shoes of the winning agent...
        if hasattr(self, '_applicable_games') and self._applicable_games:
            directory, step_count = random.choice(self._applicable_games)
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
            else:
                raise

            game_state_file = os.path.join(directory, '%d.json' % step)
            with open(game_state_file, 'r') as f:
                # NOTE: The rank is set by envs.py. Remove if causing problems.
                # print("Env %d using game state %s (%d / %d) " % (
                #     self.rank, game_state_file, step, step_count))
                self.set_json_info(json.loads(f.read()))
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
        result = self.model.step(actions, self._board, self._agents,
                                 self._bombs, self._items, self._flames)
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

        # if all(done):
        #     time_avg = defaultdict(float)
        #     time_max = defaultdict(float)
        #     time_cnt = defaultdict(int)
        #     for agent in self._agents:
        #         if type(agent) == SimpleAgent:
        #             for k, v in agent._time_cnt.items():
        #                 time_cnt[k] += v
        #             for k, v in agent._time_avg.items():
        #                 time_avg[k] += v
        #             for k, v in agent._time_max.items():
        #                 time_max[k] = max(time_max[k], v)
        #             agent.reset_times()
            # print("\nEpisode end times:")
            # total = 0.0
            # for key in sorted(time_avg.keys()):
            #     avg = time_avg[key] / 3.0
            #     cnt = time_cnt[key]
            #     mx  = time_max[key]
            #     print("\t%s: %.4f (%d) --> %.4f, %.4f" % (key, avg, cnt,
            #                                               avg * cnt, mx))
            #     total += avg * cnt
            # print("\tTotal: %.4f" % total)

        return obs, reward, done, info

    def render(self, mode=None, close=False, record_pngs_dir=None,
               record_json_dir=None):
        if close:
            self.close()
            return

        mode = mode or self._mode or 'human'

        if mode == 'rgb_array':
            rgb_array = graphics.PixelViewer.rgb_array(
                self._board, self._board_size, self._agents,
                self._is_partially_observable)
            return rgb_array[0]

        if self._viewer is None:
            if mode == 'rgb_pixel':
                self._viewer = graphics.PixelViewer(
                    board_size=self._board_size,
                    agents=self._agents, 
                    partially_observable=self._is_partially_observable)
            else:
                self._viewer = graphics.PommeViewer(
                    board_size=self._board_size,
                    agents=self._agents, 
                    partially_observable=self._is_partially_observable,
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

        suffix = '%d.png' % self._step_count
        if record_pngs_dir:
            self._viewer.save(record_pngs_dir)
        if record_json_dir:
            self.record_json(os.path.join(record_json_dir, suffix))

        time.sleep(1.0 / self.render_fps)

    def record_json(self, directory):
        info = self.get_json_info()
        step_count = self._step_count
        with open(os.path.join(directory, '%d.json' % step_count), 'w') as f:
            f.write(json.dumps(info, sort_keys=True, indent=4))

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
        return np.concatenate((
            board, bomb_blast_strength, bomb_life, position, ammo,
            blast_strength, can_kick, teammate, enemies))

    def get_json_info(self):
        """Returns a json snapshot of the current game state."""
        ret = {
                'board_size': self._board_size,
                'step_count': self._step_count,
                'board': self._board,
                'agents': self._agents,
                'bombs': self._bombs,
                'flames': self._flames,
                'items': [[k, i] for k,i in self._items.items()]
            }
        for key, value in ret.items():
            ret[key] = json.dumps(value, cls=utility.PommermanJSONEncoder)
        return ret

    def set_json_info(self, game_state=None):
        """Sets the game state.

        If no game_state is given, then uses the init_game_state.
        """
        game_state = game_state or self._init_game_state

        board_size = int(game_state['board_size'])
        self._board_size = board_size
        self._step_count = int(game_state['step_count'])

        board_array = json.loads(game_state['board'])
        self._board = np.ones((board_size, board_size)).astype(np.uint8)
        self._board *= constants.Item.Passage.value
        for x in range(self._board_size):
            for y in range(self._board_size):
                self._board[x,y] = board_array[x][y]

        self._items = {}
        item_array = json.loads(game_state['items'])
        for i in item_array:
            self._items[tuple(i[0])] = i[1]

        agent_array = json.loads(game_state['agents'])
        for a in agent_array:
            agent = next(x for x in self._agents \
                         if x.agent_id == a['agent_id'])
            agent.set_start_position((a['position'][0], a['position'][1]))
            agent.reset(int(a['ammo']), bool(a['is_alive']),
                        int(a['blast_strength']), bool(a['can_kick']))

        self._bombs = []
        bomb_array = json.loads(game_state['bombs'])
        for b in bomb_array:
            bomber = next(x for x in self._agents \
                          if x.agent_id == b['bomber_id'])
            self._bombs.append(characters.Bomb(
                bomber, tuple(b['position']), int(b['life']),
                int(b['blast_strength']), b['moving_direction']))


        self._flames = []
        flameArray = json.loads(game_state['flames'])
        for f in flameArray:
            self._flames.append(
                characters.Flame(tuple(f['position']), f['life']))
