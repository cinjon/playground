"""Environment used in training the Djikstra approach.

This simple environment is very different from the Pomme environments.

1. It's an NxN grid - a single agent starts at A and has to reach a target B.
2. Both A and B are generated randomly. There should be an option to include
   M rigid walls as well (positioned randomly) into which the agent can't move.
3. The "expert" will be Djikstra.
4. The agent gets +1 for reaching the goal within max timesteps (dependent on
   N, the grid size) and -1 for not reaching the goal.

Observations should be:
1. The agent's position.
2. The goal's position.
3. The positions of the rigid walls.

As input, this can be three feature maps and num_stack = 1. The agent will not
need to have an LSTM.

Output should be a single discrete action representing [Up, Down, Left, Right].
"""
from .v0 import Pomme as PommeV0
from gym import spaces
from .. import constants
import numpy as np
from .. import utility


class Grid(PommeV0):
    def _set_action_space(self):
        self.action_space = spaces.Discrete(5)

    def _set_observation_space(self):
        """The Observation Space for the single agent.

        There are a total of board_size^2 + 2 observations:
        - all of the board (board_size^2)
        - agent's position (2)
        """
        bss = self._board_size**2
        min_obs = [0] * bss + [0] * 2
        max_obs = [len(constants.Item)] * bss + [self._board_size] * 2
        self.observation_space = spaces.Box(
            np.array(min_obs), np.array(max_obs))

    def make_board(self):
        # TODO: this requires some changes in make_board
        # use the single_agent_goal flag to create the simple grid (v4)
        # as opposed to single_agent_goal = False for pomme v0 etc.ss
        self._board = utility.make_board(self._board_size, self._num_rigid,
                                        0, single_agent_goal=True)

    def make_items():
        return

    def make_goal(self):
        self._goal = utility.make_goal(self._board)

    def get_observations(self):
        self.observations = self.model.get_observations(
            self._board, self._agents, self._bombs,
            self._is_partially_observable, self._agent_view_size,
            self._max_steps, step_count=self._step_count)
        return self.observations

    def _get_rewards(self):
        """
        The agent receives reward +1 for reaching the goal and
        penalty -0.1 for each step it takes in the environment.
        """
        agent_pos = self.observations['position']
        goal_pos = self.observations['goal_position']
        rewards = self.model.get_rewards(self._agents, self._game_type,
                                         self._step_count, self._max_steps,
                                         agent_pos, goal_pos)
        return rewards

    def _get_done(self):
        agent_pos = self.observations['position']
        goal_pos = self.observations['goal_position']
        return self.model.get_done(self._agents, self._step_count,
                                   self._max_steps, self._game_type,
                                   self.training_agents, all_agents=True,
                                   agent_pos=agent_pos, goal_pos=goal_pos)

    def _get_info(self, done, rewards):
        agent_pos = self.observations['position']
        goal_pos = self.observations['goal_position']
        ret = self.model.get_info(done, rewards, self._game_type, self._agents,
                                  self.training_agents, agent_pos, goal_pos)
        ret['step_count'] = self._step_count
        if hasattr(self, '_game_state_step_start'):
            ret['game_state_step_start'] = self._game_state_step_start
        return ret

    def reset(self):
        assert (self._agents is not None)

        def get_game_state_file(directory, step_count):
            # TODO: the game_state_distribution types will need
            # some adjustment to the simple grid env
            if self._game_state_distribution == 'uniform':
                # Pick a random game state to start from.
                step = random.choice(range(step_count))
            elif self._game_state_distribution == 'uniform3':
                # Pick a game state uniformly over the last 21.
                # NOTE: This is an effort to reduce the effect of the credit
                # assignment problem. If this works well, then we might be able
                # to move a sliding window back across epochs.
                step = random.choice(
                    range(max(0, step_count - 4), step_count - 1)
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
            elif utility.is_int(self._game_state_distribution):
                game_state_int = int(self._game_state_distribution)
                step = random.choice(
                    range(max(0, step_count - game_state_int - 5),
                          max(0, step_count - game_state_int) + 5)
                )
            elif self._game_state_distribution == 'genesis':
                step = 0
            elif self._game_state_distribution.startswith('uniformForward'):
                step = random.choice(
                    range(min(self._uniform_v, step_count - 1)))
            else:
                raise
            return os.path.join(directory, '%d.json' % step), step

        if hasattr(self, '_applicable_games') and self._applicable_games:
            directory, step_count = random.choice(self._applicable_games)
            counter = 0
            while True:
                if counter == 5:
                    raise
                game_state_file, step = get_game_state_file(
                    directory, step_count)
                counter += 1
                try:
                    while not os.path.exists(game_state_file):
                        game_state_file, step = get_game_state_file(
                            directory, step_count)
                    self._game_state_step_start = step_count - step + 1
                    with open(game_state_file, 'r') as f:
                        self.set_json_info(json.loads(f.read()))
                    break
                except json.decoder.JSONDecodeError as e:
                    print("PR --> GSF: %s / sc: %d / step: %d..." %
                          (game_state_file, step_count, step))
                    logging.warn("LOG --> GSF: %s / sc: %d / step: %d..." %
                                 (game_state_file, step_count, step))
        elif self._init_game_state is not None:
            self.set_json_info()

        else: # TODO: where and how
            # to set the initial position of the goal??
            self._step_count = 0
            self.make_board()
            # self.make_items()
            goal_position = self.make_goal()
            agent.set_goal_position(goal_position)
            for agent_id, agent in enumerate(self._agents):
                row = random.randint(range(self._board_size))
                col = random.randint(range(self._board_size))
                agent.set_start_position((row, col))
                agent.reset()

        return self.get_observations()

    def step(self, actions):
        result = self.model.step(actions, self._board, self._agents,
                                 self._bombs, self._items, self._flames,
                                 max_blast_strength=max_blast_strength,
                                 selfbombing=self._selfbombing, do_print=self.rank == 0)
        self._board, self._agents, self._bombs = result[:2]
        self._goal = result[-1]
        # NOTE: this should be above calling the below functions since they
        # take the step_count to change obs etc., so step_count should be
        # updated before
        self._step_count += 1

        done = self._get_done()
        obs = self.get_observations()
        reward = self._get_rewards()
        info = self._get_info(done, reward)

        return obs, reward, done, info

    @staticmethod
    def featurize(obs):
        board = obs["board"].reshape(-1).astype(np.float32)
        position = utility.make_np_float(obs["position"])
        goal_position = utility.make_np_float(obs["goal_position"])

        return np.concatenate(board, agent_position, goal_position)
