"""Environment used for hyperbolic escape the room.

1. It's a random tree of size N=7. The agent starts at the root and must reach any leaf.
2. The agent can go up (to parent), left, or right. If there is no node where it is trying to reach,
   it is a failed move.
3. The tree is generated randomly.
4. Any leaf of depth size is acceptable for exiting.
5. The agent gets +1 for eaching the goal within max timesteps and -1 for not reaching the goal.
6. Further, it gets -.03 for every step it takes to pressure it to find the goal faster.

Observations should be:
1. The agent's position.
2. The tree's makeup, given in terms of a full tree so as not to make it too easy.

As input, this can be two feature maps (one for agent position, one for where the nodes are)
and num_stack = 1. The agent will not need to have an LSTM. 

Output should be a single discrete action representing [Up, Left, Right].
"""
from collections import defaultdict
import json
import os
import queue
import random
import time

from gym import spaces
import numpy as np

from .. import constants
from .. import utility
from .v0 import Pomme as PommeV0


class Tree(PommeV0):

    def _set_action_space(self):
        # We use 5 just because that includes Up, Down, and Right, and I'm being lazy.
        # The network will just learn to not use Stop or Down.
        self.action_space = spaces.Discrete(5)

    def _set_observation_space(self):
        """The Observation Space for the single agent.

        There are a total of board_size^2 + 2 observations:
        - all of the board (board_size^2)
        - agent's position (2)
        """
        bss = self._board_size**2
        min_obs = [0] * (bss + 1)
        max_obs = [1] * (bss + 1)
        self.observation_space = spaces.Box(
            np.array(min_obs), np.array(max_obs))

    def set_reward_shaping(self, step_loss=0.0, bomb_reward=None,
                           item_reward=None, use_second_place=False):
        self._step_loss = step_loss

    def make_board(self):
        self._board = utility.make_board_tree(self._board_size)

    def get_observations(self):
        self.observations = self.model.get_observations_tree(
            self._board, self._agents, self._max_steps,
            step_count=self._step_count)
        return self.observations

    def _get_rewards(self):
        """
        The agent receives reward +1 for reaching the goal and
        penalty -0.1 for each step it takes in the environment.
        """
        agent_pos = self.observations[0]['position']
        rewards = self.model.get_rewards_tree(agent_pos, self._board_size)
        rewards = [r - self._step_loss for r in rewards]
        return rewards

    def _get_done(self):
        agent_pos = self.observations[0]['position']
        return self.model.get_done_tree(agent_pos, self._board_size, self._max_steps)

    def _get_info(self, done, rewards):
        agent_pos = self.observations[0]['position']
        ret = self.model.get_info_tree(done, agent_pos, self._board_size)
        ret['step_count'] = self._step_count
        ret['optimal_num_steps'] = self._optimal_num_steps
        try:
            ret['game_state_file'] = self._game_state_file
        except AttributeError:
            pass
        if hasattr(self, '_game_state_step_start'):
            ret['game_state_step_start'] = self._game_state_step_start
            ret['game_state_step_start_beg'] = self._game_state_step_start_beg
        return ret

    def reset(self):
        assert (self._agents is not None)

        def get_game_state_step(step_count):
            # TODO: the game_state_distribution types will need
            # some adjustment to the simple grid env
            if self._game_state_distribution == 'uniform':
                # Pick a random game state to start from.
                step = random.choice(range(step_count))
            elif self._game_state_distribution == 'genesis':
                step = 0
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
            elif self._game_state_distribution.startswith('grUniformBounds'):
                # This is the Grid version of uniformBounds. The max value here is 47.
                # (0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (max, max)
                lb = self._uniform_v
                if self._uniform_v < 5:
                    ub = 1
                else:
                    ub = lb // 2

                minrange = max(0, step_count - lb)
                maxrange = max(minrange + 1, step_count - ub)
                step = random.choice(range(minrange, maxrange))
            elif utility.is_int(self._game_state_distribution):
                game_state_int = int(self._game_state_distribution)
                min_range = max(0, step_count - game_state_int - 5)
                max_range = min(max(0, step_count - game_state_int) + 5,
                                step_count)
                step = random.choice(range(min_range, max_range))
            else:
                raise
            return step

        if self._online_backplay:
            # Run a game until completion, saving the states at each step.
            # Then, pick from the right set of steps.
            board = make_board()
            path = self._compute_path_json(board, agent_pos, goal_pos)
            while len(path) < 35:
                board, agent_pos, goal_pos, inaccess_counter = make_board()
                path = self._compute_path_json(board, agent_pos, goal_pos)

            step = get_game_state_step(step_count=len(path))
            self._game_state_step_start = len(path) - step + 1
            self._game_state_step_start_beg = step
            info = path[step]
            self._board = info['board'].astype(np.uint8)
            self._board_size = info['board_size']
            self._step_count = info['step_count']
            agent = self._agents[0]
            agent.set_start_position(info['position'])
            agent.reset(info['step_count'])
            self._optimal_num_steps = len(path)
        elif hasattr(self, '_applicable_games') and self._applicable_games:
            directory, step_count = random.choice(self._applicable_games)
            counter = 0
            while True:
                if counter == 5:
                    raise
                step = get_game_state_step(step_count)
                game_state_file = os.path.join(directory, '%03d.json' % step)
                counter += 1
                try:
                    while not os.path.exists(game_state_file):
                        step = get_game_state_step(step_count)
                        game_state_file = os.path.join(directory, '%03d.json' % step)
                    self._game_state_step_start = step_count - step + 1
                    self._game_state_step_start_beg = step
                    self._game_state_file = game_state_file
                    with open(game_state_file, 'r') as f:
                        self.set_json_info(json.loads(f.read()))
                    break
                except json.decoder.JSONDecodeError as e:
                    print("PR --> GSF: %s / sc: %d / step: %d..." %
                          (game_state_file, step_count, step))
                    logging.warn("LOG --> GSF: %s / sc: %d / step: %d..." %
                                 (game_state_file, step_count, step))
            if directory not in self._optimal_num_steps_directory:
                optimal_num_steps = self._board_size
                self._optimal_num_steps_directory[directory] = optimal_num_steps
            self._optimal_num_steps = self._optimal_num_steps_directory[directory]
        elif self._init_game_state is not None:
            self.set_json_info()
        else:
            self._step_count = 0
            self.make_board()
            for agent_id, agent in enumerate(self._agents):
                pos_agent = self._board.index(constants.GridItem.Agent.value)
                agent.set_start_position(pos_agent)
                agent.reset()
        return self.get_observations()

    def set_json_info(self, game_state=None):
        game_state = game_state or self._init_game_state

        board_size = int(game_state['board_size'])
        self._board_size = board_size
        self._step_count = int(game_state['step_count'])
        self._board = json.loads(game_state['board'])

        # NOTE: We assume there is just one agent.
        agent_array = json.loads(game_state['agents'])
        agent = self._agents[0]
        agent.set_start_position(tuple(agent_array[0]['position']))
        agent.reset(self._step_count)

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
        return self.model.act_tree(agents, obs, self.action_space)

    def step(self, actions):
        results = self.model.step_tree(actions, self._board, self._agents)
        self._board, self._agents = results[:2]

        # NOTE: this should be above calling the below functions since they
        # take the step_count to change obs etc., so step_count should be
        # updated before
        self._step_count += 1

        # NOTE: get_observations needs to be called before
        # the others to change obs state!!
        obs = self.get_observations()
        done = self._get_done()
        reward = self._get_rewards()
        info = self._get_info(done, reward)

        return obs, reward, done, info

    @staticmethod
    def featurize(obs):
        board = obs["board"]
        position = utility.make_np_float(obs["position"])
        ret = np.concatenate(board, agent_position)
        return ret

    def get_json_info(self):
        """Returns a json snapshot of the current game state."""
        ret = {
            'board_size': self._board_size,
            'step_count': self._step_count,
            'board': self._board,
            'agents': self._agents,
        }
        for key, value in ret.items():
            ret[key] = json.dumps(value, cls=utility.PommermanJSONEncoder)
        return ret

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

        print("Step %d / Optimal %d." % (self._step_count,
                                         self._optimal_num_steps))
        print(self._board)
        print("\n")
        time.sleep(1.0 / self.render_fps)

    @staticmethod
    def _compute_path_json(board, start, end):
        seen = set()
        prev = {}
        dist = defaultdict(lambda: 1000000)
        dist[start] = 0
        Q = queue.PriorityQueue()
        Q.put((dist[start], start))

        found = False
        while not Q.empty() and not found:
            d, position = Q.get()
            x, y = position
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (x + row, y + col)
                if new_position in seen:
                    continue
                elif x + row >= len(board) or x + row < 0:
                    continue
                elif y + col >= len(board) or y + col < 0:
                    continue
                elif board[new_position] == 1:
                    continue

                val = d + 1
                if val < dist[new_position]:
                    dist[new_position] = val
                    prev[new_position] = position

                seen.add(new_position)
                Q.put((dist[new_position], new_position))

                if new_position == end:
                    found = True
                    break

        path = []
        curr = end
        while prev.get(curr):
            curr = prev[curr]
            curr_board = board.copy()
            curr_board[start] = constants.GridItem.Passage.value
            curr_board[curr] = constants.GridItem.Agent.value
            path.append({
                'position': curr,
                'board': curr_board,
                'board_size': len(curr_board),
                'step_count': dist[curr]
            })

        path = list(reversed(path))
        return path
