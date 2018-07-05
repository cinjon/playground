"""The set of constants in the game.

This includes not just ints but also classes like Item, GameType, Action, etc.
"""
from enum import Enum

RENDER_FPS = 15
BOARD_SIZE = 11
NUM_RIGID = 36
NUM_WOOD = 36
NUM_ITEMS = 20
AGENT_VIEW_SIZE = 4
HUMAN_FACTOR = 32
DEFAULT_BLAST_STRENGTH = 2
DEFAULT_BOMB_LIFE = 10
# color for each of the 4 agents
AGENT_COLORS = [[231, 76, 60], [46, 139, 87], [65, 105, 225], [238, 130, 238]]
# color for each of the items.
ITEM_COLORS = [[240, 248, 255], [128, 128, 128], [210, 180, 140],
               [255, 153, 51], [241, 196, 15], [141, 137, 124]]
ITEM_COLORS += [(153, 153, 255), (153, 204, 204), (97, 169, 169), (48, 117,
                                                                   117)]
# If using collapsing boards, the step at which the board starts to collapse.
FIRST_COLLAPSE = 500
MAX_STEPS = 800
RADIO_VOCAB_SIZE = 8
RADIO_NUM_WORDS = 2

# Files for images and and fonts
RESOURCE_DIR = 'resources/'
file_names = [
    'Passage', 'Rigid', 'Wood', 'Bomb', 'Flames', 'Fog', 'ExtraBomb',
    'IncrRange', 'Kick', 'AgentDummy', 'Agent0', 'Agent1', 'Agent2', 'Agent3',
    'AgentDummy-No-Background', 'Agent0-No-Background', 'Agent1-No-Background',
    'Agent2-No-Background', 'Agent3-No-Background', 'X-No-Background'
]
IMAGES_DICT = {
    num: {
        'id': num,
        'file_name': '%s.png' % file_name,
        'name': file_name,
        'image': None
    } for num, file_name in enumerate(file_names)
}
FONTS_FILE_NAMES = ['Cousine-Regular.ttf']

# Human view board configurations
BORDER_SIZE = 20
MARGIN_SIZE = 10
TILE_SIZE = 50
BACKGROUND_COLOR = (41, 39, 51, 255)
TILE_COLOR = (248, 221, 82, 255)
TEXT_COLOR = (170, 170, 170, 255)


# Constants for easier setting of the "Easy" game.
BOARD_SIZE_EASY = 11
NUM_RIGID_EASY = 36
NUM_WOOD_EASY = 36
DEFAULT_BOMB_LIFE_EASY = 10
MAX_STEPS_EASY = 800
# NOTE: Should we get rid of can_kick? That's a hard one to use as well...
NUM_ITEMS_EASY = int(NUM_WOOD_EASY/2)
DEFAULT_BLAST_STRENGTH_EASY = 2


# Constants for easier setting of the "8x8" game.
BOARD_SIZE_8 = 8
NUM_RIGID_8 = 20
NUM_WOOD_8 = 12
DEFAULT_BOMB_LIFE_8 = 7
MAX_STEPS_8 = 500
NUM_ITEMS_8 = 12
DEFAULT_BLAST_STRENGTH_8 = 2

# Constants for the Grid with single agent and goal.
GRID_BOARD_SIZE = 24
GRID_MAX_STEPS = 200
GRID_NUM_RIGID = 0         # no walls
# num --> avg make_board_grid / avg num_inaccessible
# for min_path = 30:
# 220 --> 1.73 / 5.72, 180 --> 2.36 / 3.42, 160 --> 3.055 / 3.93,
# 120 --> 5.55 / 6.55, 150 --> 3.1 / 3.9
# for min_path = 25
# 150 --> 6.6 / 7.2, 180 --> 3.2 / 4.1
GRIDWALLS_NUM_RIGID = 180  # some rigid walls


class Item(Enum):
    """The Items in the game.

    When picked up:
      - ExtraBomb increments the agent's ammo by 1.
      - IncrRange increments the agent's blast strength by 1.
      - Kick grants the agent the ability to kick items.

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
    AgentDummy = 9
    Agent0 = 10
    Agent1 = 11
    Agent2 = 12
    Agent3 = 13
    Goal = 14

    
class GridItem(Enum):
    """The Items for the Grid env."""
    Passage = 0
    Wall = 1
    Goal = 2
    Agent = 3

    
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
    Grid = 4

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
