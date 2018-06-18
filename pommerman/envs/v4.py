"""Environment used in training that inherits from PommeV0.

The differences here are that this environment:
1. Uses a get_done that is agent aware.
2. Uses dense rewards:
  a. Killing another agent: 0.2
  b. Laying a bomb in proximity to an enemy: 0.1 / distance
  c. Picking up a good item: 0.1
  d. When another agent dies (that this agent didn't kill) 0.1
  e. Picking up a bad item: -0.1
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
