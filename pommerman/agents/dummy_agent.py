from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent
from .. import constants
from .. import utility


class DummyAgent(BaseAgent):
    """This is a baseline agent. After you can beat it, submit your agent to compete."""

    def __init__(self, *args, **kwargs):
        super(DummyAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):

        #return 0
        return random.choice(action_space)

        my_position = obs['position']
        board = obs['board']

        for d in [
                constants.Action.Left, constants.Action.Right,
                constants.Action.Up, constants.Action.Down
        ]:
            new_pos = utility.get_next_position(my_position, d)

            if utility.position_on_board(board, new_pos) and \
               board[new_pos] in [constants.Item.Agent0.value, constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value]:
                return 5

        return random.choice([0, 1, 2, 3, 4])
