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
import random
import json


class Grid(PommeV0):

    def _set_action_space(self):
        self.action_space = spaces.Discrete(5)

    def _set_observation_space(self):
        # TODO: do we need to explicitly have the agent's posiiton?
        # what about the goal's position
        """The Observation Space for the single agent.

        There are a total of board_size^2 + 2 observations:
        - all of the board (board_size^2)
        - agent's position (2)
        """
        bss = self._board_size**2
        min_obs = [0] * bss + [0] * 2
        max_obs = [len(constants.GridItem)] * bss + [self._board_size] * 2
        self.observation_space = spaces.Box(
            np.array(min_obs), np.array(max_obs))

    def make_board(self):
        self._board = utility.make_board_grid(self._board_size, self._num_rigid)

    def make_items(self):
        return

    def get_observations(self):
        self.observations = self.model.get_observations_grid(
            self._board, self._agents, self._max_steps,
            step_count=self._step_count)
        return self.observations

    def _get_rewards(self):
        """
        The agent receives reward +1 for reaching the goal and
        penalty -0.1 for each step it takes in the environment.
        """
        agent_pos = self.observations[0]['position']
        goal_pos = self.observations[0]['goal_position']
        rewards = self.model.get_rewards_grid(agent_pos, goal_pos)

        return rewards

    def _get_done(self):
        agent_pos = self.observations[0]['position']
        goal_pos = self.observations[0]['goal_position']
        return self.model.get_done_grid(agent_pos, goal_pos,
                                        self._step_count, self._max_steps)

    def _get_info(self, done, rewards):
        agent_pos = self.observations[0]['position']
        goal_pos = self.observations[0]['goal_position']
        ret = self.model.get_info_grid(done, agent_pos, goal_pos)
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

        else:
            self._step_count = 0
            self.make_board()
            for agent_id, agent in enumerate(self._agents):
                pos_agent = np.where(self._board == constants.GridItem.Agent.value)
                row_agent = pos_agent[0][0]
                col_agent = pos_agent[1][0]
                agent.set_start_position((row_agent, col_agent))

                pos_goal = np.where(self._board == constants.GridItem.Goal.value)
                row_goal = pos_goal[0][0]
                col_goal = pos_goal[1][0]
                agent.set_goal_position((row_goal, col_goal))

                agent.reset()

        return self.get_observations()

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
        return self.model.act_grid(agents, obs, self.action_space)

    def step(self, actions):
        # print("\n\n \n\n\n ######################## ")
        # print("BEFORE board \n ", self._board)
        # print("actions \n", actions)
        results = self.model.step_grid(actions, self._board, self._agents)
        self._board, self._agents = results[:2]
        # print("AFTER board \n", self._board)

        # NOTE: this should be above calling the below functions since they
        # take the step_count to change obs etc., so step_count should be
        # updated before
        self._step_count += 1

        # NOTE: get_observations needs to be called before
        # the others to change obs state!!
        obs = self.get_observations()
        # print("obs ", obs)
        done = self._get_done()
        reward = self._get_rewards()
        info = self._get_info(done, reward)

        # print("\n\n\n################")
        # print("done ", done)
        # print("obs ", obs)
        # print("reward ", reward)
        # print("info ", info)

        return obs, reward, done, info

    @staticmethod
    def featurize(obs):
        # print("FEATURIZE")
        board = obs["board"].reshape(-1).astype(np.float32)
        position = utility.make_np_float(obs["position"])
        goal_position = utility.make_np_float(obs["goal_position"])

        return np.concatenate(board, agent_position, goal_position)

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
