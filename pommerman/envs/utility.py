from collections import defaultdict
from enum import Enum
import itertools
import random
import time

import numpy as np


RENDER_FPS = 15
BOARD_SIZE = 13
NUM_RIGID = 50
NUM_WOOD = 50
NUM_ITEMS = int(NUM_WOOD/2)
AGENT_VIEW_SIZE = 4
TIME_LIMIT = 3000
HUMAN_FACTOR = 32
DEFAULT_BLAST_STRENGTH = 3
DEFAULT_BOMB_LIFE = 25
# color for each of the 4 agents
AGENT_COLORS = [[231,76,60], [46,139,87], [65,105,225], [238,130,238]]
# color for each of the items.
ITEM_COLORS = [[240,248,255], [128,128,128], [210,180,140], [255, 153, 51],
               [241, 196, 15], [141, 137, 124]]
ITEM_COLORS += [(153, 153, 255), (153, 204, 204), (97, 169, 169),
                (48, 117, 117)]
# If using collapsing boards, the step at which the board starts to collapse.
FIRST_COLLAPSE = 500 
MAX_STEPS = 2500
RADIO_VOCAB_SIZE = 8
RADIO_NUM_WORDS = 2


class Item(Enum):
    """The Items in the game.

    When picked up:
      - ExtraBomb increments the agent's ammo by 1.
      - IncrRange increments the agent's blast strength by 1.
      - Skull randomly decrements ammo (bounded by 1), decrements
        blast_strength (bounded by 2), or increments blast_strength by 2.

    AgentDummy is used by team games to denote the third enemy and by ffa to
    denote the teammate.
    """
    Passage = 0
    Rigid = 1
    Wood = 2
    Bomb = 3
    Flames = 4
    Fog = 5
    ExtraBomb = 6
    IncrRange = 7
    Kick = 8
    Skull = 9
    AgentDummy = 10
    Agent0 = 11
    Agent1 = 12
    Agent2 = 13
    Agent3 = 14


class GameType(Enum):
    """The Game Types.

    FFA: 1v1v1v1. Submit an agent; it competes against other submitted agents.
    Team: 2v2. Submit an agent; it is matched up randomly with another agent
      and together take on two other similarly matched agents.
    TeamRadio: 2v2. Submit two agents; they are matched up against two other
      agents. Each team passes discrete communications to each other.
    """
    FFA = 1 
    Team = 2
    TeamRadio = 3


class Action(Enum):
    Stop = 0
    Up = 1
    Down = 2
    Left = 3
    Right = 4
    Bomb = 5


class Result(Enum):
    Win = 0
    Loss = 1
    Tie = 2
    Incomplete = 3


class InvalidAction(Exception):
    pass


def make_board(size, num_rigid=0, num_wood=0):
    """Make the random but symmetric board.

    The numbers refer to the Item enum in utility. This is:
     0 - passage
     1 - rigid wall
     2 - wood wall
     3 - bomb
     4 - flames
     5 - fog
     6 - extra bomb item
     7 - extra firepower item
     8 - kick
     9 - skull
     10 - 13: agents

    Args:
      size: The dimension of the board, i.e. it's sizeXsize.
      num_rigid: The number of rigid walls on the board. This should be even.
      num_wood: Similar to above but for wood walls.

    Returns:
      board: The resulting random board.
    """
    def lay_wall(value, num_left, coordinates, board):
        x, y = random.sample(coordinates, 1)[0]
        coordinates.remove((x, y))
        coordinates.remove((y, x))
        board[x, y] = value
        board[y, x] = value
        num_left -= 2
        return num_left

    def make(size, num_rigid, num_wood):
        # Initialize everything as a passage.
        board = np.ones((size, size)).astype(np.uint8) * Item.Passage.value

        # Gather all the possible coordinates to use for walls.
        coordinates = set([
            (x, y) for x, y in \
            itertools.product(range(size), range(size)) \
            if x != y])

        # Set the players down. Exclude them from coordinates.
        board[1, 1] = Item.Agent0.value
        board[size-2, 1] = Item.Agent1.value
        board[size-2, size-2] = Item.Agent2.value
        board[1, size-2] = Item.Agent3.value
        agents = [(1, 1), (size-2, 1), (1, size-2), (size-2, size-2)]
        for position in agents:
            if position in coordinates:
                coordinates.remove(position)

        # Exclude breathing room on either side of the agents.
        for i in range(2, 4):
            coordinates.remove((1, i))
            coordinates.remove((i, 1))
            coordinates.remove((1, size-i-1))
            coordinates.remove((size-i-1, 1))
            coordinates.remove((size-2, size-i-1))
            coordinates.remove((size-i-1, size-2))
            coordinates.remove((i, size-2))
            coordinates.remove((size-2, i))

        # Lay down wooden walls providing guaranteed passage to other agents.
        wood = Item.Wood.value
        for i in range(4, size-4):
            board[1, i] = wood
            board[size-i-1, 1] = wood
            board[size-2, size-i-1] = wood
            board[size-i-1, size-2] = wood
            coordinates.remove((1, i))
            coordinates.remove((size-i-1, 1))
            coordinates.remove((size-2, size-i-1))
            coordinates.remove((size-i-1, size-2))
            num_wood -= 4

        # Lay down the rigid walls.
        while num_rigid > 0:
            num_rigid = lay_wall(Item.Rigid.value, num_rigid, coordinates,
                                 board)

        # Lay down the wooden walls.
        while num_wood > 0:
            num_wood = lay_wall(Item.Wood.value, num_wood, coordinates, board)

        return board, agents

    assert(num_rigid % 2 == 0)
    assert(num_wood % 2 == 0)
    board, agents = make(size, num_rigid, num_wood)

    # Make sure it's possible to reach most of the passages.
    while len(inaccessible_passages(board, agents)) > 4:
        board, agents = make(size, num_rigid, num_wood)

    return board


def make_items(board, num_items):
    item_positions = {}
    while num_items > 0:
        row = random.randint(0, len(board)-1)
        col = random.randint(0, len(board[0])-1)
        if board[row, col] != Item.Wood.value:
            continue
        if (row, col) in item_positions:
            continue

        item_positions[(row, col)] = random.choice([
            Item.ExtraBomb, Item.IncrRange, Item.Kick, Item.Skull
        ]).value
        num_items -= 1
    return item_positions


def inaccessible_passages(board, agent_positions):
    """Return inaccessible passages on this board."""
    seen = set()
    agent_position = agent_positions.pop()
    passage_positions = np.where(board == Item.Passage.value)
    positions = list(zip(passage_positions[0], passage_positions[1]))

    Q = [agent_position]
    while Q:
        row, col = Q.pop()
        for (i, j) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_position = (row+i, col+j)
            if next_position in seen:
                continue
            if not position_on_board(board, next_position):
                continue
            if position_is_rigid(board, next_position):
                continue

            if next_position in positions:
                positions.pop(positions.index(next_position))
                if not len(positions):
                    return []

            seen.add(next_position)
            Q.append(next_position)
    return positions
    

def is_valid_direction(board, position, direction, invalid_values=None):
    row, col = position
    invalid_values = invalid_values or [item.value for item \
                                        in [Item.Rigid, Item.Wood]]

    if Action(direction) == Action.Stop:
        return True

    if Action(direction) == Action.Up:
        return row - 1 >= 0 and board[row-1][col] not in invalid_values

    if Action(direction) == Action.Down:
        return row + 1 < len(board) and board[row+1][col] not in invalid_values

    if Action(direction) == Action.Left:
        return col - 1 >= 0 and board[row][col-1] not in invalid_values

    if Action(direction) == Action.Right:
        return col + 1 < len(board[0]) and \
            board[row][col+1] not in invalid_values

    raise InvalidAction("We did not receive a valid direction: ", direction)


def _position_is_item(board, position, item):
    return board[position] == item.value


def position_is_powerup(board, position):
    powerups = [Item.ExtraBomb, Item.IncrRange, Item.Kick, Item.Skull]
    item_values = [item.value for item in powerups]
    return board[position] in item_values


def position_is_bomb(board, position):
    return _position_is_item(board, position, Item.Bomb)


def position_is_passage(board, position):
    return _position_is_item(board, position, Item.Passage)


def position_is_rigid(board, position):
    return _position_is_item(board, position, Item.Rigid)


def position_is_agent(board, position):
    return board[position] in [
        Item.Agent0.value,
        Item.Agent1.value,
        Item.Agent2.value,
        Item.Agent3.value
    ]


def position_is_enemy(board, position, enemies):
    return Item(board[position]) in enemies


# TODO: Fix this so that it includes the teammate.
def position_is_passable(board, position, enemies):
    return all([
        any([position_is_agent(board, position),
             position_is_powerup(board, position),
             position_is_passage(board, position)]),
        not position_is_enemy(board, position, enemies)
    ])


def position_is_fog(board, position):
    return _position_is_item(board, position, Item.Fog)


def agent_value(id_):
    return getattr(Item, 'Agent%d' % id_).value


def position_in_items(board, position, items):
    return any([_position_is_item(board, position, item)
                for item in items])


def position_on_board(board, position):
    x, y = position
    return all([
        len(board) > x,
        len(board[0]) > y,
        x >= 0,
        y >= 0
    ])


def get_direction(position, next_position):
    """Get the direction such that position --> next_position.

    We assume that they are adjacent.
    """
    x, y = position
    nx, ny = next_position
    if x == nx:
        if y < ny:
            return Action.Right
        else:
            return Action.Left
    elif y == ny:
        if x < nx:
            return Action.Down
        else:
            return Action.Up
    raise InvalidAction("We did not receive a valid position transition.")


def get_next_position(position, direction):
    x, y = position
    if direction == Action.Right:
        return (x, y+1)
    elif direction == Action.Left:
        return (x, y-1)
    elif direction == Action.Down:
        return (x+1, y)
    elif direction == Action.Up:
        return (x-1, y)
    elif direction == Action.Stop:
        return (x, y)
    raise InvalidAction("We did not receive a valid direction.")


def make_np_float(feature):
    return np.array(feature).astype(np.float32)
