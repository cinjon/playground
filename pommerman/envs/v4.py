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
        self._board = utility.make_grid(self._board_size, self._num_rigid)

    def make_items():
        return
