"""The set of constants in the game.

This includes not just ints but also classes like Item, GameType, Action, etc.
"""
from enum import Enum


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
