"""The baseline Pommerman environment.
This evironment acts as game manager for Pommerman. Further environments,
such as in v1.py, will inherit from this.
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

from .. import characters
from .. import constants
from .. import forward_model
from .. import utility
from ..agents import SimpleAgent


class Pomme(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
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

        self.training_agents = []
        self.model = forward_model.ForwardModel()

        # Observation and Action Spaces. These are both geared towards a single
        # agent even though the environment expects actions and returns
        # observations for all four agents. We do this so that it's clear what
        # the actions and obs are for a single agent. Wrt the observations,
        # they are actually returned as a dict for easier understanding.
        self._set_action_space()
        self._set_observation_space()

    def _set_action_space(self):
        self.action_space = spaces.Discrete(6)

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

    def act(self, obs, acting_agent_ids=[]):
        agents = [agent for agent in self._agents \
                  if agent.agent_id not in self.training_agents]
        # TODO: Replace this hack with something more reasonable.
        agents = [agent for agent in agents if \
                  agent.agent_id not in acting_agent_ids]
        return self.model.act(agents, obs, self.action_space)

    def get_observations(self):
        self.observations = self.model.get_observations(
            self._board, self._agents, self._bombs,
            self._is_partially_observable, self._agent_view_size,
            step_count=self._step_count)
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

        if self._init_game_state is not None:
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

        done = self._get_done()
        obs = self.get_observations()
        reward = self._get_rewards()
        info = self._get_info(done, reward)

        self._step_count += 1
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

    def _render_frames(self):
        agent_view_size = constants.AGENT_VIEW_SIZE
        frames = []

        all_frame = np.zeros((self._board_size, self._board_size, 3))
        num_items = len(constants.Item)
        for row in range(self._board_size):
            for col in range(self._board_size):
                value = self._board[row][col]
                if utility.position_is_agent(self._board, (row, col)):
                    num_agent = value - num_items
                    if self._agents[num_agent].is_alive:
                        all_frame[row][col] = constants.AGENT_COLORS[num_agent]
                else:
                    all_frame[row][col] = constants.ITEM_COLORS[value]

        all_frame = np.array(all_frame)
        frames.append(all_frame)

        fog = constants.Item.Fog.value
        for agent in self._agents:
            row, col = agent.position
            my_frame = all_frame.copy()
            for r in range(self._board_size):
                for c in range(self._board_size):
                    if self._is_partially_observable and not all([
                            row >= r - agent_view_size,
                            row < r + agent_view_size,
                            col >= c - agent_view_size,
                            col < c + agent_view_size
                    ]):
                        my_frame[r, c] = constants.ITEM_COLORS[fog]
            frames.append(my_frame)

        return frames

    def render(self, mode='human', close=False, record_pngs_dir=None,
               record_json_dir=None):
        if close:
            self.close()
            return

        frames = self._render_frames()
        if mode == 'rgb_array':
            return frames[0]

        from PIL import Image
        human_factor = constants.HUMAN_FACTOR

        all_img = resize(frames[0], (self._board_size*human_factor,
                                     self._board_size*human_factor),
                         interp='nearest')
        other_imgs = [
            resize(frame, (int(self._board_size*human_factor/4),
                           int(self._board_size*human_factor/4)),
                   interp='nearest')
            for frame in frames[1:]
        ]

        other_imgs = np.concatenate(other_imgs, 0)
        img = np.concatenate([all_img, other_imgs], 1)

        if self._viewer is None:
            from gym.envs.classic_control import rendering
            self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img)

            # Register all agents which need human input with Pyglet.
            # This needs to be done here as the first `imshow` creates the
            # window. Using `push_handlers` allows for easily creating agents
            # that use other Pyglet inputs such as joystick, for example.
            for agent in self._agents:
                if agent.has_user_input():
                    self._viewer.window.push_handlers(agent)
        else:
            self._viewer.imshow(img)

        if record_pngs_dir:
            Image.fromarray(img).save(
                os.path.join(record_pngs_dir, '%d.png' % self._step_count))

        if record_json_dir:
            info = self.get_json_info()
            with open(os.path.join(record_json_dir,
                                   '%d.json' % self._step_count), 'w') as f:
                f.write(json.dumps(info, sort_keys=True, indent=4))

        time.sleep(1.0 / self._render_fps)

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

    def set_json_info(self):
        """Sets the game state as the init_game_state."""
        self._step_count = int(self._init_game_state['step_count'])
        self._board_size = int(self._init_game_state['board_size'])

        board_array = json.loads(self._init_game_state['board'])
        self._board = np.ones((self._board_size, self._board_size)) \
                        .astype(np.uint8) * constants.Item.Passage.value
        for x in range(self._board_size):
            for y in range(self._board_size):
                self._board[x,y] = board_array[x][y]

        self._items = {}
        item_array = json.loads(self._init_game_state['items'])
        for i in item_array:
            self._items[tuple(i[0])] = i[1]

        agent_array = json.loads(self._init_game_state['agents'])
        for a in agent_array:
            agent = next(x for x in self._agents \
                         if x.agent_id == a['agent_id'])
            agent.set_start_position((a['position'][0], a['position'][1]))
            agent.reset(int(a['ammo']), bool(a['is_alive']),
                        int(a['blast_strength']), bool(a['can_kick']))

        self._bombs = []
        bomb_array = json.loads(self._init_game_state['bombs'])
        for b in bomb_array:
            bomber = next(x for x in self._agents \
                          if x.agent_id == b['bomber_id'])
            self._bombs.append(characters.Bomb(
                bomber, tuple(b['position']), int(b['life']),
                int(b['blast_strength']), b['moving_direction']))


        self._flames = []
        flameArray = json.loads(self._init_game_state['flames'])
        for f in flameArray:
            self._flames.append(
                characters.Flame(tuple(f['position']), f['life']))
