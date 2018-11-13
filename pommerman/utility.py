from functools import wraps
import errno
import itertools
import json
from jsonmerge import Merger
import os
import random
import signal
import time

from gym import spaces
import numpy as np

from . import constants


class PommermanJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, constants.Item):
            return obj.value
        elif isinstance(obj, constants.Action):
            return obj.value
        elif isinstance(obj, np.int64):
            return int(obj)
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif isinstance(obj, spaces.Discrete):
            return obj.n
        elif isinstance(obj, spaces.Tuple):
            return [space.n for space in obj.spaces]
        elif isinstance(obs, list):
            return obj
        return json.JSONEncoder.default(self, obj)


def make_board(size, num_rigid=0, num_wood=0):
    """Make the random but symmetric board.
    The numbers refer to the Item enum in constants. This is:
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
        board = np.ones(
            (size, size)).astype(np.uint8) * constants.Item.Passage.value

        # Gather all the possible coordinates to use for walls.
        coordinates = set([
            (x, y) for x, y in \
            itertools.product(range(size), range(size)) \
            if x != y])

        # Set the players down. Exclude them from coordinates.
        # Agent0 is in top left. Agent1 is in bottom left.
        # Agent2 is in bottom right. Agent 3 is in top right.
        board[1, 1] = constants.Item.Agent0.value
        board[size - 2, 1] = constants.Item.Agent1.value
        board[size - 2, size - 2] = constants.Item.Agent2.value
        board[1, size - 2] = constants.Item.Agent3.value
        agents = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        for position in agents:
            if position in coordinates:
                coordinates.remove(position)

        # Exclude breathing room on either side of the agents.
        for i in range(2, 4):
            coordinates.remove((1, i))
            coordinates.remove((i, 1))
            coordinates.remove((1, size - i - 1))
            coordinates.remove((size - i - 1, 1))
            coordinates.remove((size - 2, size - i - 1))
            coordinates.remove((size - i - 1, size - 2))
            coordinates.remove((i, size - 2))
            coordinates.remove((size - 2, i))

        # Lay down wooden walls providing guaranteed passage to other agents.
        wood = constants.Item.Wood.value
        for i in range(4, size - 4):
            board[1, i] = wood
            board[size - i - 1, 1] = wood
            board[size - 2, size - i - 1] = wood
            board[size - i - 1, size - 2] = wood
            coordinates.remove((1, i))
            coordinates.remove((size - i - 1, 1))
            coordinates.remove((size - 2, size - i - 1))
            coordinates.remove((size - i - 1, size - 2))
            num_wood -= 4

        # Lay down the rigid walls.
        while num_rigid > 0:
            num_rigid = lay_wall(constants.Item.Rigid.value, num_rigid,
                                 coordinates, board)

        # Lay down the wooden walls.
        while num_wood > 0:
            num_wood = lay_wall(constants.Item.Wood.value, num_wood,
                                coordinates, board)

        return board, agents

    assert (num_rigid % 2 == 0)
    assert (num_wood % 2 == 0)
    board, agents = make(size, num_rigid, num_wood)

    # Make sure it's possible to reach most of the passages.
    while len(inaccessible_passages(board, agents)) > 4:
        board, agents = make(size, num_rigid, num_wood)

    return board


def make_items(board, num_items):
    item_positions = {}
    while num_items > 0:
        row = random.randint(0, len(board) - 1)
        col = random.randint(0, len(board[0]) - 1)
        if board[row, col] != constants.Item.Wood.value:
            continue
        if (row, col) in item_positions:
            continue

        item_positions[(row, col)] = random.choice([
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]).value
        num_items -= 1
    return item_positions


def make_board_grid(size, num_rigid=0, min_length=1, extra=False):
    # TODO: when num_walls > 0 make sure the agent can reach the goal
    """Make a random board with an agent, a goal and
    a few rigid walls (optional).

    The numbers refer to the GridItem enum in constants. This is:
     0 - passage
     1 - rigid wall
     2 - goal
     3 - agent

    Args:
      size: The dimension of the board, i.e. it's sizeXsize.
      num_rigid: The number of rigid walls on the board.

    Returns:
      board: The resulting random board.
    """
    def lay_wall_grid(value, num_left, coordinates, board):
        for x, y in random.sample(coordinates, num_left):
            board[(x, y)] = value

    def make_grid(size, num_rigid):
        # Initialize everything as a passage.
        board = np.full((size, size), constants.GridItem.Passage.value) \
                  .astype(np.uint8)

        # Gather all the possible coordinates to use for walls.
        coordinates = set([
            (x, y) for x, y in \
            itertools.product(range(size), range(size))])

        # Randomly pick the agent location. Exclude it from coordinates.
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        agent_pos = (x, y)
        board[x, y] = constants.GridItem.Agent.value
        coordinates.remove(agent_pos)

        # Randomly pick the goal location. Exclude it from coordinates
        goal_coordinates = [(gx, gy) for gx, gy in coordinates \
                            if abs(x - gx) + abs(y - gy) >= min_length]
        if not goal_coordinates:
            return None, None, None

        x_g, y_g = random.sample(goal_coordinates, 1)[0]
        goal_pos = (x_g, y_g)
        board[x_g, y_g] = constants.GridItem.Goal.value
        coordinates.remove(goal_pos)

        # Lay down the rigid walls.
        lay_wall_grid(constants.GridItem.Wall.value, num_rigid, coordinates,
                      board)
        return board, agent_pos, goal_pos

    board, agent_pos, goal_pos = make_grid(size, num_rigid)
    counter = 1
    while not agent_pos or not accessible_grid(board, agent_pos, goal_pos):
        counter += 1
        board, agent_pos, goal_pos = make_grid(size, num_rigid)

    if extra:
        return board, agent_pos, goal_pos, counter
    else:
        return board


def make_board_tree(size, num_exits=2):
    """Make a tree of size <size>. It is random which entries are included but
    there is a guaranteed path to a leaf of full depth. In addition, we guarantee
    that there are at least <exits> exits down."""
    num_leaves = 2**(size - 1)

    def recurse_tree(board, root):
        if random.rand() > .8:
            return

        if root + num_leaves >= len(board):
            # We are at a leaf.
            board[root] = constants.GridItem.Goal.value
            return

        board[root] = constants.GridItem.Passage.value
        recurse_tree(board, 2*root + 1)
        recurse_tree(board, 2*root + 2)

    def make_tree():
        # Initialize everything as a passage.
        board = [constants.GridImte.Wall.value]*(2**size - 1)
        print("BEF BOARD")
        recurse_tree(board, 0)
        print(board)
        return board

    # Board is a <2^size - 1> array where the last 2^(size-1) entries are leaves.
    board = make_tree()
    while sum(board[-num_leaves:]) - num_leaves >= num_exits:
        board = make_grid(size)
    board[0] = constants.GridItem.Agent.value

    return board


def accessible_grid(board, agent_pos, goal_pos):
    seen = set()
    passage_positions = np.where(board == constants.GridItem.Passage.value)
    positions = list(zip(passage_positions[0], passage_positions[1]))

    Q = [agent_pos]
    while Q:
        row, col = Q.pop()
        for (i, j) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_position = (row + i, col + j)
            if next_position in seen:
                continue
            if not position_on_board(board, next_position):
                continue
            if position_is_gridrigid(board, next_position):
                continue

            if next_position == goal_pos:
              return True

            if next_position in positions:
                positions.pop(positions.index(next_position))
                if not len(positions):
                    return False

            seen.add(next_position)
            Q.append(next_position)
    return False


def inaccessible_passages(board, agent_positions):
    """Return inaccessible passages on this board."""
    seen = set()
    agent_position = agent_positions.pop()
    passage_positions = np.where(board == constants.Item.Passage.value)
    positions = list(zip(passage_positions[0], passage_positions[1]))

    Q = [agent_position]
    while Q:
        row, col = Q.pop()
        for (i, j) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_position = (row + i, col + j)
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
    if invalid_values is None:
        invalid_values = [item.value for item in \
                          [constants.Item.Rigid, constants.Item.Wood]]

    if constants.Action(direction) == constants.Action.Stop:
        return True

    if constants.Action(direction) == constants.Action.Up:
        return row - 1 >= 0 and board[row - 1][col] not in invalid_values

    if constants.Action(direction) == constants.Action.Down:
        return row + 1 < len(board) and board[row +
                                              1][col] not in invalid_values

    if constants.Action(direction) == constants.Action.Left:
        return col - 1 >= 0 and board[row][col - 1] not in invalid_values

    if constants.Action(direction) == constants.Action.Right:
        return col + 1 < len(board[0]) and \
            board[row][col+1] not in invalid_values

    raise constants.InvalidAction("We did not receive a valid direction: ",
                                  direction)

def is_valid_direction_grid(board, position, direction, invalid_values=None):
    row, col = position
    invalid_values = invalid_values or [constants.GridItem.Wall.value]

    if constants.Action(direction) == constants.Action.Stop:
        return True
    elif constants.Action(direction) == constants.Action.Up:
        return row - 1 >= 0 and board[row - 1][col] not in invalid_values
    elif constants.Action(direction) == constants.Action.Down:
        return row + 1 < len(board) and board[row + 1][col] not in invalid_values
    elif constants.Action(direction) == constants.Action.Left:
        return col - 1 >= 0 and board[row][col - 1] not in invalid_values
    elif constants.Action(direction) == constants.Action.Right:
        return col + 1 < len(board[0]) and \
            board[row][col+1] not in invalid_values
    raise constants.InvalidAction("We did not receive a valid direction: ",
                                  direction)

def is_valid_direction_tree(board, position, direction, invalid_values=None):
    invalid_values = invalid_values or [constants.GridItem.Wall.value]
    if constants.Action(direction) == constants.Action.Up:
        return position != 0
    elif constants.Action(direction) == constants.Action.Left:
        npos = 2*position + 1
        return npos < len(board) and board[npos] not in invalid_values
    elif constants.Action(direction) == constants.Action.Right:
        npos = 2*position + 2
        return npos < len(board) and board[npos] not in invalid_values
    elif constants.Action(direction) in [
            constants.Action.Down, constants.Action.Stop
    ]:
        return False
    raise constants.InvalidAction("We did not receive a valid direction: ",
                                  direction)

def _position_is_item(board, position, item):
    return board[position] == item.value


def position_is_powerup(board, position):
    powerups = [constants.Item.ExtraBomb, constants.Item.IncrRange, constants.Item.Kick]


def position_is_flames(board, position):
    return _position_is_item(board, position, constants.Item.Flames)


def position_is_bomb(bombs, position):
    """Check if a given position is a bomb.

    We don't check the board because that is an unreliable source. An agent
    may be obscuring the bomb on the board.
    """
    for bomb in bombs:
        if position == bomb.position:
            return True
    return False


def position_is_powerup(board, position):
    powerups = [
        constants.Item.ExtraBomb, constants.Item.IncrRange, constants.Item.Kick
    ]
    item_values = [item.value for item in powerups]
    return board[position] in item_values


def position_is_wall(board, position):
    return position_is_rigid(board, position) or \
        position_is_wood(board, position)


def position_is_passage(board, position):
    return _position_is_item(board, position, constants.Item.Passage)


def position_is_rigid(board, position):
    return _position_is_item(board, position, constants.Item.Rigid)


def position_is_gridrigid(board, position):
    return _position_is_item(board, position, constants.GridItem.Wall)


def position_is_wood(board, position):
    return _position_is_item(board, position, constants.Item.Wood)


def position_is_agent(board, position):
    return board[position] in [
        constants.Item.Agent0.value, constants.Item.Agent1.value,
        constants.Item.Agent2.value, constants.Item.Agent3.value
    ]


def position_is_enemy(board, position, enemies):
    return constants.Item(board[position]) in enemies


# TODO: Fix this so that it includes the teammate.
def position_is_passable(board, position, enemies):
    return all([
        any([
            position_is_agent(board, position),
            position_is_powerup(board, position),
            position_is_passage(board, position)
        ]), not position_is_enemy(board, position, enemies)
    ])


def position_is_fog(board, position):
    return _position_is_item(board, position, constants.Item.Fog)


def agent_value(id_):
    return getattr(constants.Item, 'Agent%d' % id_).value

def position_in_items(board, position, items):
    return any([_position_is_item(board, position, item) for item in items])


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
            return constants.Action.Right
        else:
            return constants.Action.Left
    elif y == ny:
        if x < nx:
            return constants.Action.Down
        else:
            return constants.Action.Up
    raise constants.InvalidAction(
        "We did not receive a valid position transition.")


def get_next_position(position, direction):
    x, y = position
    if direction == constants.Action.Right:
        return (x, y + 1)
    elif direction == constants.Action.Left:
        return (x, y - 1)
    elif direction == constants.Action.Down:
        return (x + 1, y)
    elif direction == constants.Action.Up:
        return (x - 1, y)
    elif direction == constants.Action.Stop:
        return (x, y)
    raise constants.InvalidAction("We did not receive a valid direction: ",
                                   position, direction)


def get_next_position_tree(position, direction):
    if direction == constants.Action.Right:
        return 2*position + 2
    elif direction == constants.Action.Left:
        return 2*position + 1
    elif direction == constants.Action.Up:
        return (position - 1) // 2
    raise constants.InvalidAction("We did not receive a valid direction: ",
                                   position, direction)


def make_np_float(feature):
    return np.array(feature).astype(np.float32)


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL,seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator


class Timer:
    """With block timer.
    with Timer() as t:
      foo = blah()
    print('Request took %.03f sec.' % t.interval)
    """
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def is_int(s):
    try:
        s = int(s)
        return True
    except Exception as e:
        return False


def join_json_state(record_json_dir, agents, finished_at, config):
    jsonSchema = {
        "properties": {
            "state": {
                "mergeStrategy": "append"
            }
        }
    }

    jsonTemplate = {
        "agents": agents,
        "finished_at": finished_at,
        "config": config,
        "state": []
    }

    merger = Merger(jsonSchema)
    base = merger.merge({}, jsonTemplate)

    for root, dirs, files in os.walk(record_json_dir):
        for name in files:
            path = os.path.join(record_json_dir, name)
            if name.endswith('.json') and "game_state" not in name:
                with open(path) as data_file:
                    data = json.load(data_file)
                    head = {"state":[data]}
                    base = merger.merge(base, head)

    with open(os.path.join(record_json_dir, 'game_state.json'), 'w') as f:
        f.write(json.dumps(base, sort_keys=True, indent=4))

    for root, dirs, files in os.walk(record_json_dir):
        for name in files:
            if "game_state" not in name:
                os.remove(os.path.join(record_json_dir, name))
