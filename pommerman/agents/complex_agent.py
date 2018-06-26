"""Complex Machine, h/t Yichen.

This is hardcoded to use a board size of 11.
"""
from collections import defaultdict
import copy
from enum import Enum
from functools import partial
import math
import queue
import random
import time

import numpy as np
import pommerman
from pommerman import constants
from pommerman import utility
from pommerman.agents import complex_agent_game_nodes as gn
from pommerman.agents import BaseAgent
from pommerman.characters import Flame


verbose = False

class State(Enum):
    Evader = 0
    Explorer = 1
    Attacker = 2


class ComplexAgent(BaseAgent):
    """This is a baseline agent. After you can beat it, submit your agent to compete."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_simple_agent = True

        # Keep track of recently visited uninteresting positions so that we don't keep visiting the same places.
        self._recently_visited_positions = []
        self._recently_visited_length = 6
        # Keep track of the previous direction to help with the enemy standoffs.
        self._no_safe_position_step = 0

        self._closest_safe_positions = ()
        self._prev_direction = []
        self._prev_position = []

        self._state = State.Explorer
        self._board_size = kwargs.get('board_size', 11)
        self._visit_map = np.zeros((self._board_size, self._board_size))
        self._target = None
        self.bombing_agents = {}
        self._evade_mcts = MCTSAgentExplore(board_size=self._board_size)
        self._mcts = MCTSAgentExploit(board_size=self._board_size)

    def act(self, obs, action_space):
        # print("##### obs #### \n", obs)
        # print("pos ", tuple(obs['position']))
        # print("\n\n\n")

        def convert_bombs(strength_map, life_map):
            ret = []
            locations = np.where(strength_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append({
                    'position': (r, c),
                    'blast_strength': int(strength_map[(r, c)]),
                    'bomb_life': int(life_map[(r, c)]),
                    'moving_direction': None
                })
            return ret

        def convert_flames(board):
            # Assuming that for each flame object, its life span is 2 ticks
            ret = []
            locations = np.where(board == 4)
            for r, c in zip(locations[0], locations[1]):
                ret.append(Flame((r, c)))
            return ret

        self.obs = obs
        self.my_position = tuple(obs['position'])
        self.board = np.array(obs['board'])
        self.bombs = convert_bombs(
            np.array(obs['bomb_blast_strength']), np.array(obs['bomb_life']))
        self.enemies = [constants.Item(e) for e in obs['enemies']]
        [self.blast_strength, self.ammo, self.flames] = [
            int(obs['blast_strength']),
            int(obs['ammo']),
            convert_flames(self.board)
        ]

        [self.items, self.dist, self.prev] = self._djikstra(
            self.board,
            self.my_position,
            self.bombs,
            self.enemies,
            depth=15,
            board_size=self._board_size)
        [self.dist, self.prev] = self.update_distance_to_items(
            self.items, self.dist, self.prev, self.board, self.enemies)
        self.find_bombing_agents(np.array(obs['bomb_life']), self.board)

        if self.my_position == self._closest_safe_positions:
            self._closest_safe_positions = ()
        elif self._no_safe_position_step >= 4:
            self._no_safe_position_step = 0
            self._closest_safe_positions = ()
        elif self._closest_safe_positions == (-1, -1):
            self._no_safe_position_step += 1

        if self._closest_safe_positions not in self.prev:
            self._closest_safe_positions = ()

# --------------------------------------
# ========Visit Map Initialization==============
# --------------------------------------
        if self.is_start():
            self._closest_safe_positions = ()
            self._prev_direction = []
            self._prev_position = []
            self._state = State.Explorer
            for i in range(len(self._visit_map)):
                for j in range(len(self._visit_map[i])):
                    if (self.board[i][j] == 1):
                        self._visit_map[i][j] = 99999
                    else:
                        self._visit_map[i][j] = 0

# --------------------------------------
# =========Yichen Safe Direction========
# --------------------------------------

        if len(self._closest_safe_positions
              ) != 0 and self._closest_safe_positions != (-1, -1):
            direction = get_next_direction_according_to_prev(
                self.my_position, self._closest_safe_positions, self.prev)
            agent_output([
                "my_position {}, {}".format(self.my_position[0],
                                            self.my_position[1]),
                "self._closest_safe_positions {}, {}".format(
                    self._closest_safe_positions[0],
                    self._closest_safe_positions[1]),
                "No. 100: {}".format(direction)
            ])

            if direction == self._prev_direction and self._prev_position == self.my_position:
                self._closest_safe_positions = ()
                # print("Safe Bomb")
                return constants.Action.Bomb.value
            elif direction is not None:

                # =======
                # If the next position to travel to is not safe, MCTS to survive
                # =======
                if check_if_in_bomb_range_threshold(
                        self.board, self.bombs,
                        utility.get_next_position(self.my_position, direction)):
                    # print("safe MCTS: ", self.obs['can_kick'])
                    # actions_space = range(5)
                    directions = [
                        constants.Action.Stop, constants.Action.Left,
                        constants.Action.Right, constants.Action.Up,
                        constants.Action.Down
                    ]
                    directions = self._filter_unsafe_directions(
                        self.board, self.my_position, directions, self.bombs,
                        self.items, self.dist, self.prev, self.enemies)
                    actions_space = [dir.value for dir in directions]
                    act = self._evade_mcts.find_next_move(
                        self.obs, actions_space,
                        partial(win_if_arrive, self._closest_safe_positions),
                        score_func_evade, self.bombing_agents)
                    agent_output(
                        ["139 check bomb", self._closest_safe_positions, act],
                        True)
                    act = act if type(act) == int else act.value
                    if act != -1:
                        return act

                self._prev_direction = direction
                self._prev_position = self.my_position
                # print(self.board)
                # print("Safe prev direction", direction)
                # if verbose:
                    # print("146", direction.value)
                return direction.value
            else:
                self._closest_safe_positions = ()

# --------------------------------------
# =============State Agent==============
# --------------------------------------
        [output, Att, Exp, Evd] = [constants.Action.Stop.value] + [-1] * 3
        if self.EvaderCondition():
            Evd = self.EvaderAction()
        elif self.AttackerCondition():
            Att = self.AttackerAction()
            # if verbose:
            #     print("ATTACK ACTION", Att)
            if Att == 5 and not self._maybe_bomb(
                    self.ammo, self.blast_strength, self.items, self.dist,
                    self.my_position, self.board, self.prev, self.enemies,
                    self.bombs):
                # print("Not safe, Evade")
                Att = 0  # self.EvaderAction()
            elif Att == -1:
                Att = self.ExplorerAction()
        elif self.ExplorerCondition():
            Exp = self.ExplorerAction()
        # print(self.board)
        # print(obs['bomb_life'])
        # print(self._closest_safe_positions)
        # print(Evd, Att, Exp)
        if Evd is not -1:
            output = Evd
            if verbose:
                print("170 Evader ", output)
        elif Att is not -1:
            output = Att
            if verbose:
                print("173 Attacker ", output)
        elif Exp is not -1:
            output = Exp
            if verbose:
                print("176 Explorer", output)

        # output = self.bomb_if_towards_negative(output)
        if type(output) != int:
            return output.value
        return output

    def AttackerCondition(self):
        # return self._state is State.Attacker
        return self._state is State.Attacker or (
            self.ammo >= 1 and
            self._near_enemy(self.my_position, self.items, self.dist, self.prev,
                             self.enemies, 5))

    def ExplorerCondition(self):
        return self._state is State.Explorer

    def EvaderCondition(self):
        self.unsafe_directions = self._directions_in_range_of_bomb(
            self.board, self.my_position, self.bombs, self.dist)
        return self.unsafe_directions

    def AttackerAction(self):
        # ============
        # PLACE BOMB IF NEAR ENEMY
        # ============
        # Lay pomme if we are adjacent to an enemy.
        # if self._is_adjacent_enemy(self.items, self.dist, self.enemies) and self._maybe_bomb(self.ammo, self.blast_strength, self.items, self.dist, self.my_position, self.board, self.prev, self.enemies):
        #    agent_output(["No. 300"])
        #    return constants.Action.Bomb.value

        # ===========
        # MCTS TREE SEARCH IF NEAR ENEMY
        # ===========
        # actions_space = range(6)
        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]  # , constants.Action.Bomb]
        directions = self._filter_unsafe_directions(
            self.board, self.my_position, directions, self.bombs, self.items,
            self.dist, self.prev, self.enemies)
        actions_space = [dir.value for dir in directions
                        ] + [constants.Action.Bomb.value]
        if self._target is not None:
            position = self.items.get(self._target, [])
            if not position or self.dist[position[0]] > 4:
                self._target = None
            else:
                # print("MCTS")
                # print(self.obs['board'], self.bombing_agents)

                return self._mcts.find_next_move(
                    self.obs, actions_space,
                    partial(win_cond_with_target, self._target),
                    partial(score_func_with_target, self._target),
                    self.bombing_agents)
        else:
            new_target = self._is_adjacent_enemy_target(self.items, self.dist,
                                                        self.enemies)
            if new_target is not None:
                self._target = new_target
                # print("MCTS")
                # print(self.obs['board'],self.bombing_agents)
                return self._mcts.find_next_move(
                    self.obs, actions_space,
                    partial(win_cond_with_target, self._target),
                    partial(score_func_with_target, self._target),
                    self.bombing_agents)

        # ============
        # MOVE TOWARDS ENEMY
        # ============
        enemy_detection_range = 6
        # Move towards an enemy if there is one in exactly ten reachable spaces
        direction = self._near_enemy(self.my_position, self.items, self.dist,
                                     self.prev, self.enemies,
                                     enemy_detection_range)
        if direction is not None:
            directions = self._filter_unsafe_directions(
                self.board, self.my_position, [direction], self.bombs,
                self.items, self.dist, self.prev, self.enemies)
            if directions:
                self._prev_direction = direction
                agent_output(["No. 400: {}".format(direction.value)])
                return direction.value

        # ===========
        # STOP CAUSE NOT SAFE
        # ===========
        return self.ExplorerAction()

    def ExplorerAction(self):
        # ============
        # BOMB NEGATIVE ITEM
        # ============
        # bomb_count = self._near_bomb_item(self.my_position, self.items, self.dist, self.prev, 1)
        # bomb_count = self.count_bomb_in_radius(my_position, bombs, items, radius=4)
        # agent_output(["NEGATIVE", bomb_count])
        # if bomb_count > 1:
        #     if self._maybe_bomb(self.ammo, self.blast_strength, self.items, self.dist, self.my_position, self.board, self.prev, self.enemies, self.bombs):
        #         agent_output(["No. 510"])
        #         return constants.Action.Bomb.value

        # ===========
        # MOVE TOWARDS GOOD ITEM
        # ===========
        # Move towards a good item if there is one within eight reachable spaces.
        # directions = list(filter(lambda x: x != self._prev_direction,
        # direction_to_items(self.my_position, self.items, self.dist, self.prev, 15)))
        directions = direction_to_items(self.my_position, self.items, self.dist,
                                        self.prev, 15)
        if directions is not None and len(directions) != 0:
            directions = self._filter_unsafe_directions(
                self.board, self.my_position, directions, self.bombs,
                self.items, self.dist, self.prev, self.enemies)
            if directions:
                agent_output(["No. 500"])
                self._prev_direction = directions[0]
                return directions[0]

        # ============
        # DESTROY WALL
        # ============
        # Maybe lay a bomb if we are within a space of a wooden wall.
        directions_bombwood = directions_wood = direction_to_woods(
            self.my_position, self.items, self.dist, self.prev, 1)
        if directions_bombwood:
            for d in directions_bombwood:
                new_pos = utility.get_next_position(self.my_position, d)
                if not check_if_in_bomb_range(self.board, self.bombs, new_pos) and\
                   self._maybe_bomb(self.ammo, self.blast_strength, self.items, self.dist, self.my_position, self.board, self.prev, self.enemies, self.bombs, "WOOD"):

                    agent_output(["No. 600"])
                    return constants.Action.Bomb.value
                else:
                    agent_output(["No. 610: 0"])
                    return constants.Action.Stop.value

        # ============
        # MOVE TOWARDS WOODS
        # ============
        # Move towards wooden wallS  within five reachable spaces
        directions_wood = direction_to_woods(self.my_position, self.items,
                                             self.dist, self.prev, 12)
        if directions_wood is not None or len(
                directions_wood
        ) is not 0:  # MOVE TOWARDS WOOD EVEN IF YOU DONT HAVE AMMO and self.ammo != 0:
            directions = self._filter_unsafe_directions(
                self.board, self.my_position, directions_wood, self.bombs,
                self.items, self.dist, self.prev, self.enemies)
            if directions:
                agent_output(["No. 700"])
                return directions[0]

        # ===========
        # MOVE TOWARDS UNFAMILIAR POSITION
        # ===========
        # Choose a random but valid direction.
        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]
        valid_directions = self._filter_invalid_directions(
            self.board, self.my_position, directions, self.enemies)
        directions = self._filter_unsafe_directions(
            self.board, self.my_position, valid_directions, self.bombs,
            self.items, self.dist, self.prev, self.enemies)
        if random.random() < 0.75:
            directions = self._filter_recently_visited(
                directions, self.my_position, self._recently_visited_positions)
        if len(directions) > 1:
            directions = [k for k in directions if k != constants.Action.Stop]
        if not len(directions):
            directions = [constants.Action.Stop]

        # Add this position to the recently visited uninteresting positions so we don't return immediately.
        self._recently_visited_positions.append(self.my_position)
        self._recently_visited_positions = self._recently_visited_positions[
            -self._recently_visited_length:]

        # visit map update
        self._visit_map[self.my_position[0]][self.my_position[1]] += 1

        # pick a dir with the smallest number
        values = []
        for d in directions:
            if d == constants.Action.Stop:
                values.append(
                    (self._visit_map[self.my_position[0]][self.my_position[1]],
                     d.value))
            if d == constants.Action.Up:
                values.append((self._visit_map[self.my_position[0] -
                                               1][self.my_position[1]],
                               d.value))
            if d == constants.Action.Down:
                values.append((self._visit_map[self.my_position[0] +
                                               1][self.my_position[1]],
                               d.value))
            if d == constants.Action.Left:
                values.append(
                    (self._visit_map[self.my_position[0]][self.my_position[1] -
                                                          1], d.value))
            if d == constants.Action.Right:
                values.append(
                    (self._visit_map[self.my_position[0]][self.my_position[1] +
                                                          1], d.value))
        rtn = min(values)
        agent_output(["randomly choose value {}".format(rtn[1])])

        # If visit_map is has one number greater than 10, then switch to attacker mode
        # if ((self._visit_map > 10) & (self._visit_map != 99999)).any():
        #    print(self._visit_map)
        #    self._state = State.Attacker

        return rtn[1]

    def EvaderAction(self):
        # ============
        # EVADING BOMB
        # ============
        # Move if we are in an unsafe place. 2. move to safe places if possible
        self._closest_safe_positions = self._update_safe_position(
            self.bombs, self.board, self.my_position, self.items, self.dist,
            self.prev, self.enemies)
        if self._closest_safe_positions == (-1, -1):
            agent_output([
                "Unsafe Directions", self.unsafe_directions, self.my_position,
                self._closest_safe_positions, "No. 201"
            ])
            directions = [
                constants.Action.Left, constants.Action.Right,
                constants.Action.Up, constants.Action.Down
            ]
            directions = self._filter_kicking_direction(
                self.board, self.my_position, directions, self.enemies)
            directions += [constants.Action.Stop]
            directions = self._filter_direction_toward_flames(
                self.board, self.my_position, directions, self.enemies)
            self._no_safe_position_step = 1
            rtn = random.choice(directions).value
            agent_output(
                ["308", self._closest_safe_positions,
                 constants.Action(rtn)], True)
            return rtn

            # MCTS to survive
            # return self._evade_mcts.find_next_move(self.obs, directions, \
            #                                     partial(win_if_arrive, self._closest_safe_positions), score_func_evade, self.bombing_agents);

        agent_output(["PRE 200", self._closest_safe_positions, self.prev])
        direction = get_next_direction_according_to_prev(
            self.my_position, self._closest_safe_positions, self.prev)
        agent_output([
            self.board, "Unsafe Directions", self.unsafe_directions,
            "next direction", direction, "cloest safe place",
            self._closest_safe_positions, "No. 200"
        ], True)

        # =======
        # If the next position to travel to is not safe, MCTS to survive
        # =======
        # if check_if_in_bomb_range_threshold(self.board, self.bombs,\
        #                                                 utility.get_next_position(self.my_position, direction)):
        #     #print(self.obs['bomb_life'])
        #     # actions_space = range(5)
        #     directions = [constants.Action.Stop, constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down]
        #     directions = self._filter_unsafe_directions(self.board, self.my_position, directions, self.bombs, self.items, self.dist, self.prev, self.enemies)
        #     agent_output(["333",directions], True)
        #     actions_space = [dir.value for dir in directions]
        #     return self._evade_mcts.find_next_move(self.obs, actions_space, \
        #                                          partial(win_if_arrive, self._closest_safe_positions), score_func_evade, self.bombing_agents)

        self._prev_direction = direction
        return direction

    # place down bomb if going towards negative item
    # def bomb_if_towards_negative(self, direction):
    #     if direction in [constants.Action.Up, constants.Action.Down, constants.Action.Left, constants.Action.Right]\
    #        and self.board[utility.get_next_position(self.my_position, direction)] == constants.Item.Skull.value:
    #         return constants.Action.Bomb.value
    #     else:
    #         return direction

    def find_bombing_agents(self, bomb_life_map, board):
        # only add initial bombs
        locations = np.where(bomb_life_map == constants.DEFAULT_BOMB_LIFE - 1)
        for r, c in zip(locations[0], locations[1]):
            b = board[r][c] - 11
            self.bombing_agents[(r, c)] = b

        # update kicked bombs
        # remove the older bombs
        keys_to_pop = []
        keys_to_add = []
        for key in self.bombing_agents.keys():
            if bomb_life_map[key[0]][key[
                    1]] == 0:  # or board[key[0]][key[1]] == 4:
                # check all directions
                # up
                r = key[0] - 1
                c = key[1]
                if (r >= 0):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # down
                r = key[0] + 1
                c = key[1]
                if (r < self._board_size):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # left
                r = key[0]
                c = key[1] - 1
                if (c >= 0):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # right
                r = key[0]
                c = key[1] + 1
                if (c < self._board_size):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                keys_to_pop.append((key[0], key[1]))
        for k in keys_to_pop:
            self.bombing_agents.pop(k, None)
        for k in keys_to_add:
            self.bombing_agents[k[0]] = k[1]
            # print(self.bombing_agents)
        # input("main mcts updating for kick")


# --------------------------------------
# ======================================
# --------------------------------------

    @staticmethod
    def _djikstra(board,
                  my_position,
                  bombs,
                  enemies,
                  depth=None,
                  exclude=None,
                  board_size=11):
        assert (depth is not None)

        if exclude is None:
            exclude = [constants.Item.Fog]  # , #constants.Item.Rigid,
            # constants.Item.Flames] # SKULL

        def out_of_range(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            return depth is not None and abs(y2 - y1) + abs(x2 - x1) > depth

        items = defaultdict(list)
        dist = {}
        prev = {}
        Q = queue.PriorityQueue()
        Q.put([0, my_position])

        mx, my = my_position
        for r in range(max(0, mx - depth), min(board_size, mx + depth)):
            for c in range(max(0, my - depth), min(board_size, my + depth)):
                position = (r, c)

                if any([
                        out_of_range(my_position, position),
                        utility.position_in_items(board, position, exclude),
                ]):
                    continue

                dist[position] = np.inf

                prev[position] = None
                # Q.put((dist[position], position))
                item = constants.Item(board[position])
                items[item].append(position)
        dist[my_position] = 0

        for bomb in bombs:
            if bomb['position'] == my_position:
                items[constants.Item.Bomb].append(my_position)
        while not Q.empty():
            _, position = Q.get()
            x, y = position
            val = dist[(x, y)] + 1
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (row + x, col + y)
                if utility.position_on_board(board, new_position):
                    if all([
                            new_position in dist,
                            utility.position_is_passable(
                                board, new_position, enemies)
                    ]):

                        new_val = val
                        # Manually increase the distance to the skull
                        # if board[new_position[0], new_position[1]] == constants.Item.Skull.value:
                        #     new_val += 4

                        if new_val < dist[new_position]:
                            dist[new_position] = new_val
                            prev[new_position] = position
                            Q.put((dist[new_position], new_position))
        return items, dist, prev

    @classmethod
    def _directions_in_range_of_bomb(self,
                                     board,
                                     my_position,
                                     bombs,
                                     dist,
                                     bomb_ticking_threshold=15,
                                     consider_bomb_life=True):
        ret = defaultdict(int)

        x, y = my_position

        # BOMB connection
        for i in range(len(bombs)):
            for j in range(len(bombs)):
                if i == j:
                    continue
                bombs[i], bombs[j] = self._connect_bomb(bombs[i], bombs[j])

        for bomb in bombs:
            position = bomb['position']
            bomb_life = bomb['bomb_life']
            distance = dist.get(position)

            path_bombable = path_is_bombable(board, my_position, position,
                                             bombs)
            if path_bombable:
                distance = get_manhattan_distance(my_position, position)

            if distance is None:
                continue

            if my_position == position:
                # We are on a bomb. All directions are in range of bomb.
                for direction in [
                        constants.Action.Right,
                        constants.Action.Left,
                        constants.Action.Up,
                        constants.Action.Down,
                ]:
                    ret[direction] = max(ret[direction], bomb['blast_strength'])

            bomb_range = bomb['blast_strength']

            if (distance > bomb_range and my_position != position and not path_bombable) \
                    or (bomb_life > distance + bomb_ticking_threshold and consider_bomb_life):
                continue

            if x == position[0]:
                if y < position[1]:
                    # Bomb is right.
                    ret[constants.Action.Right] = max(
                        ret[constants.Action.Right], bomb['blast_strength'])
                else:
                    # Bomb is left.
                    ret[constants.Action.Left] = max(ret[constants.Action.Left],
                                                     bomb['blast_strength'])
            elif y == position[1]:
                if x < position[0]:
                    # Bomb is down.
                    ret[constants.Action.Down] = max(ret[constants.Action.Down],
                                                     bomb['blast_strength'])
                else:
                    # Bomb is down.
                    ret[constants.Action.Up] = max(ret[constants.Action.Up],
                                                   bomb['blast_strength'])
        return ret

    def _update_safe_position(self, bombs, board, my_position, items, dist,
                              prev, enemies):

        sorted_dist = {
            k: v
            for k, v in dist.items()
            if v < 15 and not position_is_not_passible(board, k, enemies)
        }
        sorted_dist = sorted(
            sorted_dist, key=lambda position: dist[position]
        )  # + get_manhattan_distance(my_position, position))
        # bomb_count = self.count_bomb_in_radius(my_position, bombs, items, radius=4)
        safe_positions = queue.PriorityQueue()
        best_dist = 99999
        for position in sorted_dist:
            unsafe_directions = self._directions_in_range_of_bomb(
                board, position, bombs, dist,
                bomb_ticking_threshold=15)  # bomb_count * 2 + 3)
            # potential_unsafe_directions = self._directions_in_range_of_bomb(board, position, bombs, dist, bomb_ticking_threshold=15)#)bomb_count * 2 + 3, consider_bomb_life=False)
            position_is_bad_corner = self.is_bad_corner(
                board,
                my_position,
                position,
                items,
                dist,
                prev,
                enemies,
                distance_to_enemies=3,
                threshold_wall_count=2)

            if len(
                    unsafe_directions
            ) == 0 and not position_is_bad_corner:  # and len(potential_unsafe_directions) == 0:
                # agent_output(["SAFE POSITION BOARD",
                #                          position, my_position, board])
                if dist[position] <= best_dist:
                    best_dist = dist[position]
                    # calculate threat during escaping
                    num_threats = 0
                    curr_position = position
                    while prev[curr_position] != my_position:
                        unsafe_dir = self._directions_in_range_of_bomb(
                            board,
                            curr_position,
                            bombs,
                            dist,
                            bomb_ticking_threshold=15)
                        if len(unsafe_dir) != 0:
                            num_threats += 1
                        curr_position = prev[curr_position]
                    # append it to the queue
                    safe_positions.put((num_threats, position))
                elif best_dist != 99999:
                    break
                # return position
            # elif len(unsafe_directions) == 0 and not position_is_bad_corner:
            #     safe_positions.put((dist[position] + len(unsafe_directions) / 10.0, position))

        # append to safe position
        if not safe_positions.empty():
            position = safe_positions.get()[1]
            agent_output(["SAFE POSITION BOARD", position, my_position, board])
            return position
        else:
            # if there is no safe position, then
            return (-1, -1)

    def _find_safe_directions(self, board, my_position, unsafe_directions,
                              bombs, enemies):

        def is_stuck_direction(next_position, bomb_range, next_board, enemies):
            Q = queue.PriorityQueue()
            Q.put((0, next_position))
            seen = set()

            nx, ny = next_position
            is_stuck = True
            while not Q.empty():
                dist, position = Q.get()
                seen.add(position)

                px, py = position
                if nx != px and ny != py:
                    is_stuck = False
                    break

                if dist > bomb_range:
                    is_stuck = False
                    break

                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + px, col + py)
                    if new_position in seen:
                        continue

                    if not utility.position_on_board(next_board, new_position):
                        continue

                    if not utility.position_is_passable(next_board,
                                                        new_position, enemies):
                        continue

                    dist = abs(row + px - nx) + abs(col + py - ny)
                    Q.put((dist, new_position))
            return is_stuck

        # All directions are unsafe. Return a position that won't leave us locked.
        safe = []

        if len(unsafe_directions) == 4:
            next_board = board.copy()
            next_board[my_position] = constants.Item.Bomb.value

            for direction, bomb_range in unsafe_directions.items():
                next_position = utility.get_next_position(
                    my_position, direction)
                nx, ny = next_position
                if not utility.position_on_board(next_board, next_position) or \
                   not utility.position_is_passable(next_board, next_position, enemies):
                    continue

                if not is_stuck_direction(next_position, bomb_range, next_board,
                                          enemies):
                    # We found a direction that works. The .items provided
                    # a small bit of randomness. So let's go with this one.
                    return [direction]
            if not safe:
                safe = [constants.Action.Stop]
            return safe

        x, y = my_position
        disallowed = []  # The directions that will go off the board.

        for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            position = (x + row, y + col)
            direction = utility.get_direction(my_position, position)

            # Don't include any direction that will go off of the board.
            if not utility.position_on_board(board, position):
                disallowed.append(direction)
                continue

            # Don't include any direction that we know is unsafe.
            if direction in unsafe_directions:
                continue

            if utility.position_is_passable(board, position,
                                            enemies) or utility.position_is_fog(
                                                board, position):
                safe.append(direction)

        if not safe:
            # We don't have any safe directions, so return something that is allowed.
            safe = [k for k in unsafe_directions if k not in disallowed]

        if not safe:
            # We don't have ANY directions. So return the stop choice.
            return [constants.Action.Stop]

        return safe

    @staticmethod
    def _is_adjacent_enemy(items, dist, enemies):
        for enemy in enemies:
            for position in items.get(enemy, []):
                if dist[position] <= 2:
                    return True
        return False

    @staticmethod
    # Return the enemy ID on board
    def _is_adjacent_enemy_target(items, dist, enemies):
        for enemy in enemies:
            for position in items.get(enemy, []):
                if dist[position] <= 3:
                    return enemy
        return None

    @staticmethod
    def _has_bomb(obs):
        return obs['ammo'] >= 1

    # @staticmethod
    def _maybe_bomb(self,
                    ammo,
                    blast_strength,
                    items,
                    dist,
                    my_position,
                    board,
                    prev,
                    enemies,
                    bombs,
                    scope="NOTHING"):
        """Returns whether we can safely bomb right now.

        Decides this based on:
        1. Do we have ammo?
        2. If we laid a bomb right now, will we be stuck?
        """
        # Do we have ammo?
        if ammo < 1:
            return False

        # if self.count_bomb_in_radius(my_position, bombs, items, 4) >= 3:
        #     return False

        # if  self._directions_in_range_of_bomb(board, my_position, bombs, dist, consider_bomb_life=False): #current position connects other bombs
        #     return False

        copy_bombs = copy.deepcopy(self.bombs)
        copy_bombs.append({
            'position': my_position,
            'blast_strength': int(self.blast_strength),
            'bomb_life': 10,
            'moving_direction': None
        })

        # Will we be stuck?
        x, y = my_position
        for position in items.get(constants.Item.Passage):
            if dist[position] > 5 or utility.position_is_agent(board, position) \
               or self._directions_in_range_of_bomb(board, position, copy_bombs, dist, consider_bomb_life=False) \
               or self.is_bad_corner(board, my_position, position, items, dist, prev, enemies, distance_to_enemies=3, threshold_wall_count=3) \
               or self.susceptible_to_path_bombing(copy_bombs, my_position, position, dist, radius=4):
                continue

            # We can reach a passage that's outside of the bomb scope.
            px, py = position
            if px != x and py != y:
                return True

            path_bombable = path_is_bombable(board, my_position, position,
                                             bombs)
            if path_bombable:
                distance = get_manhattan_distance(my_position, position)
            else:
                distance = dist[position]
            # We can reach a passage that's outside of the bomb strength.
            if distance > blast_strength:
                return True

        return False

    @staticmethod
    def _nearest_position(dist, objs, items, radius):
        nearest = None
        # dist_to = max(dist.values())
        dist_to = 999999

        for obj in objs:
            for position in items.get(obj, []):
                d = dist[position]
                if d <= radius and d <= dist_to:
                    nearest = position
                    dist_to = d

        return nearest

    @staticmethod
    def _get_direction_towards_position(my_position, position, prev):
        if not position:
            return None

        next_position = position
        while prev[next_position] != my_position:
            next_position = prev[next_position]

        return utility.get_direction(my_position, next_position)

    @classmethod
    def _near_enemy(cls, my_position, items, dist, prev, enemies, radius):
        nearest_enemy_position = cls._nearest_position(dist, enemies, items,
                                                       radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_enemy_position, prev)

    @classmethod
    def _near_good_powerup(cls, my_position, items, dist, prev, radius):
        objs = [
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @classmethod
    def _near_item(cls, my_position, items, dist, prev, radius):
        objs = [
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @classmethod
    def _near_wood(cls, my_position, items, dist, prev, radius):
        objs = [constants.Item.Wood]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    def _near_bomb_item(self, my_position, items, dist, prev, radius):
        # objs = [constants.Item.Skull]
        # nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        counter = 0
        directions = [
            constants.Action.Up, constants.Action.Down, constants.Action.Left,
            constants.Action.Right
        ]
        for d in directions:
            new_pos = utility.get_next_position(my_position, d)
            if utility.position_on_board(self.board, new_pos) and\
               self.board[new_pos] == constants.Item.Bomb.value:
                counter += 1
        return counter

    @staticmethod
    def _filter_invalid_directions(board, my_position, directions, enemies):
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(
                    board, position) and utility.position_is_passable(
                        board, position,
                        enemies):  # and not position_is_skull(board, position):
                ret.append(direction)
        return ret

    @staticmethod
    def _count_adjacent_walls(board, position, items, enemies):
        walls_count = 0
        not_passible_items = items[constants.Item.
                                   Wood] + items[constants.Item.
                                                 Rigid] + items[constants.Item.
                                                                Bomb] + items[constants.
                                                                              Item.
                                                                              Flames]

        for enemy in enemies:
            not_passible_items += items.get(enemy, [])

        for direction in [
                constants.Action.Up, constants.Action.Down,
                constants.Action.Left, constants.Action.Right
        ]:
            new_pos = utility.get_next_position(position, direction)

            if not utility.position_on_board(board, new_pos) or \
               new_pos in not_passible_items:
                walls_count = walls_count + 1

        return walls_count

    @staticmethod
    def _check_enemy_near_hallway(board, my_position, new_position, enemies):

        def if_passable(pos):
            return not utility.position_on_board(board, pos)

        # check if it is a hallway
        pos_up = utility.get_next_position(new_position, constants.Action.Up)
        pos_down = utility.get_next_position(new_position,
                                             constants.Action.Down)
        pos_left = utility.get_next_position(new_position,
                                             constants.Action.Left)
        pos_right = utility.get_next_position(new_position,
                                              constants.Action.Right)
        # if if_passable(pos_up) and if_passable(pos_down):

        # elif if_passable(pos_left) and if_passable(pos_right):

        return False
        #    cls._get_direction_towards_position

    @classmethod
    def _filter_unsafe_directions(self, board, my_position, directions, bombs,
                                  items, dist, prev, enemies):
        ret = []
        bad_corner_surving_direction = []
        for direction in directions:
            if not utility.is_valid_direction(board, my_position, direction):
                continue
            x, y = utility.get_next_position(my_position, direction)

            is_bad = False
            unsafe_directions = self._directions_in_range_of_bomb(
                board, (x, y), bombs, dist)
            is_bad_corner = self.is_bad_corner(
                board,
                my_position, (x, y),
                items,
                dist,
                prev,
                enemies,
                distance_to_enemies=-1,
                threshold_wall_count=4)
            if len(unsafe_directions) != 0:
                is_bad = True

            if board[x, y] == constants.Item.Flames.value:
                is_bad = True

            if is_bad_corner and not is_bad:
                is_bad = True
                bad_corner_surving_direction.append(direction)

            if not is_bad:
                ret.append(direction)
        if not ret:
            return bad_corner_surving_direction
        else:
            return ret

    @staticmethod
    def _filter_recently_visited(directions, my_position,
                                 recently_visited_positions):
        ret = []
        for direction in directions:
            if not utility.get_next_position(
                    my_position, direction) in recently_visited_positions:
                ret.append(direction)

        if not ret:
            ret = directions
        return ret

    def update_distance_to_items(self, items, dist, prev, board, enemies):
        distance_to_items = {}
        path_to_items = {}
        for item, values in items.items():
            for position in values:
                if utility.position_is_passable(board, position, enemies):
                    # if passable, then the distance to the item is same as the dist
                    distance_to_items[position] = dist[position]
                    path_to_items[position] = prev[position]
                else:
                    x, y = position
                    min_dist = np.inf
                    for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_position = (row + x, col + y)

                        if utility.position_on_board(board, new_position) and \
                            new_position in dist:
                            if dist[new_position] + 1 < min_dist:
                                min_dist = dist[new_position] + 1
                                path_to_items[position] = new_position
                    distance_to_items[position] = min_dist
        return distance_to_items, path_to_items

    @classmethod
    def is_bad_corner(self,
                      board,
                      my_position,
                      target_position,
                      items,
                      dist,
                      prev,
                      enemies,
                      distance_to_enemies,
                      threshold_wall_count=3):
        wall_count = self._count_adjacent_walls(board, target_position, items,
                                                enemies)
        if distance_to_enemies == -1:
            if wall_count >= threshold_wall_count:
                return True
            else:
                return False
        else:
            if wall_count >= threshold_wall_count and self._near_enemy(
                    my_position, items, dist, prev, enemies,
                    distance_to_enemies):
                return True
            else:
                return False

    @classmethod
    def _connect_bomb(self, bomb1, bomb2):
        position1 = bomb1['position']
        x1, y1 = position1
        bomb_life1 = bomb1['bomb_life']
        bomb_range1 = bomb1['blast_strength']

        position2 = bomb2['position']
        x2, y2 = position2
        bomb_life2 = bomb2['bomb_life']
        bomb_range2 = bomb2['blast_strength']

        bomb_connected = False
        if x1 == x2:
            bomb_dist = abs(y1 - y2)
            if bomb_dist < bomb_range1 + 1 or bomb_dist < bomb_range2 + 1:
                bomb_connected = True
        elif y1 == y2:
            bomb_dist = abs(x1 - x2)
            if bomb_dist < bomb_range1 + 1 or bomb_dist < bomb_range2 + 1:
                bomb_connected = True
        if bomb_connected:
            min_bomb_life = min(bomb_life1, bomb_life2)
            bomb1['bomb_life'] = min_bomb_life
            bomb2['bomb_life'] = min_bomb_life

        return bomb1, bomb2

    def count_bomb_in_radius(self, my_position, bombs, items, radius):
        count = 0
        for position in items.get(constants.Item.Bomb, []):
            if get_manhattan_distance(position, my_position) <= radius:
                count += 1
        return count

    @staticmethod
    def _filter_kicking_direction(board, my_position, directions, enemies):
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(
                    board, position) and not utility.position_is_rigid(
                        board, position):
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_direction_toward_flames(board, my_position, directions,
                                        enemies):
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(
                    board, position) and not utility.position_is_flames(
                        board, position):
                ret.append(direction)
        return ret

    @classmethod
    def susceptible_to_path_bombing(self,
                                    bombs,
                                    my_position,
                                    position,
                                    dist,
                                    radius=4):
        copy_bombs = copy.deepcopy(bombs)
        for i in range(len(copy_bombs)):
            for j in range(len(copy_bombs)):
                if i == j:
                    continue
                copy_bombs[i], copy_bombs[j] = self._connect_bomb(
                    copy_bombs[i], copy_bombs[j])

        for bomb in copy_bombs:
            if bomb['position'] not in dist:
                continue
            if dist[bomb['position']] < radius and get_manhattan_distance(
                    my_position, position) + radius > bomb['bomb_life']:
                return True
        return False

    def is_start(self):
        # print(dir(self))
        # return self.my_position == self.start_position and \
        #     self.ammo == 1 and \
        #     self.is_alive == True and \
        #     self.blast_strength == constants.DEFAULT_BLAST_STRENGTH and \
        #     self.can_kick == False
        return False


def position_is_not_passible(board, position, enemies):
    # return not any([utility.position_is_agent(board, position), utility.position_is_powerup(board, position) or utility.position_is_passage(board, position)]) and not utility.position_is_enemy(board, position, enemies)
    return not utility.position_is_passable(board, position, enemies)


def get_next_direction_according_to_prev(my_position, target_position, prev):
    cached_position = target_position
    if not cached_position:

        return None
    while prev[cached_position] != my_position:

        cached_position = prev[cached_position]
        if cached_position is None:
            return None
    return utility.get_direction(my_position, cached_position)


# def position_is_skull(board, position):
#     return board[position] == constants.Item.Skull.value


# Change the default value for enabled to enable all output
def agent_output(output_array, enabled=0):
    if (enabled) and verbose:
        for s in output_array:
            print(s)


def position_is_flame(board, position):
    return utility._position_is_item(board, position, constants.Item.Flames)


def position_is_bombable(board, position, bombs):
    return any([
        utility.position_is_agent(board, position),
        utility.position_is_powerup(board, position),
        utility.position_is_passage(board, position),
        position_is_flame(board, position),
        position_is_bomb(bombs, position)
    ])


def path_is_bombable(board, position1, position2, bombs):
    x1, y1 = position1
    x2, y2 = position2

    positions_to_determine = []
    if x1 == x2:
        if y1 <= y2:
            positions_to_determine = [(x1, yk) for yk in range(y1, y2 + 1)]
        else:
            positions_to_determine = [(x1, yk) for yk in range(y2, y1 + 1)]
    elif y1 == y2:
        if x1 <= x2:
            positions_to_determine = [(xk, y1) for xk in range(x1, x2 + 1)]
        else:
            positions_to_determine = [(xk, y1) for xk in range(x2, x1 + 1)]
    else:
        return False
    return all(
        [position_is_bombable(board, p, bombs) for p in positions_to_determine])


def position_is_bomb(bombs, position):
    """Check if a given position is a bomb.

    We don't check the board because that is an unreliable source. An agent
    may be obscuring the bomb on the board.
    """
    for bomb in bombs:
        if position == bomb["position"]:
            return True
    return False


def get_manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# ==============================
# ------STATE AGENT HELPER------
# ==============================


# Get all positions of objects within a certain radius
def objects_within_radius(dist, objs, items, radius):
    obj_pos = []
    # dist_to = max(dist.values())
    dist_to = 999999

    for obj in objs:
        for position in items.get(obj, []):
            d = dist[position]
            if d <= radius:
                if d < dist_to:
                    obj_pos.insert(0, position)
                    dist_to = d
                else:
                    obj_pos.append(position)

    return obj_pos


# Get all wood positions and then returns directions towards them (array)
def direction_to_items(my_position, items, dist, prev, radius):
    objs = [
        constants.Item.ExtraBomb, constants.Item.IncrRange, constants.Item.Kick
    ]
    item_positions = objects_within_radius(dist, objs, items, radius)
    directions = []
    for pos in item_positions:
        d = get_next_direction_according_to_prev(my_position, pos, prev)
        if d not in directions:
            directions.append(d)
    return directions


# Get all wood positions and then returns directions towards them (array)
def direction_to_woods(my_position, items, dist, prev, radius):
    objs = [constants.Item.Wood]
    item_positions = objects_within_radius(dist, objs, items, radius)
    directions = []
    for pos in item_positions:
        d = get_next_direction_according_to_prev(my_position, pos, prev)
        if d not in directions:
            directions.append(d)
    return directions


# Check if the position is in the range of bomb
def check_if_in_bomb_range(board, bombs, position):
    for b in bombs:
        # Set the direction to trace
        direction = None
        if (b['position'][0] == position[0] and
                abs(b['position'][1] - position[1]) <= b['blast_strength']):
            if b['position'][1] < position[1]:
                direction = constants.Action.Right
            elif b['position'][1] > position[1]:
                direction = constants.Action.Left
        elif (b['position'][1] == position[1] and
              abs(b['position'][0] - position[0]) <= b['blast_strength']):
            if b['position'][0] < position[0]:
                direction = constants.Action.Down
            elif b['position'][0] > position[0]:
                direction = constants.Action.Up
        else:
            continue

        if direction is None:
            return True

        # Trace from bomb to see if there's block in the way
        new_pos = b['position']
        while new_pos != position:
            new_pos = utility.get_next_position(new_pos, direction)
            if board[new_pos] in [
                    constants.Item.Rigid.value, constants.Item.Wood.value
            ]:
                break
        if new_pos == position:
            return True
    return False


# Check if the position is in the range of bomb
def check_if_in_bomb_range_threshold(board, bombs, position, threshold=15):
    for b in bombs:
        if (b['bomb_life'] > threshold):
            continue

        # Set the direction to trace
        direction = None
        if (b['position'][0] == position[0] and
                abs(b['position'][1] - position[1]) <= b['blast_strength']):
            if b['position'][1] < position[1]:
                direction = constants.Action.Right
            elif b['position'][1] > position[1]:
                direction = constants.Action.Left
        elif (b['position'][1] == position[1] and
              abs(b['position'][0] - position[0]) <= b['blast_strength']):
            if b['position'][0] < position[0]:
                direction = constants.Action.Down
            elif b['position'][0] > position[0]:
                direction = constants.Action.Up
        else:
            continue

        if direction is None:
            return True

        # Trace from bomb to see if there's block in the way
        new_pos = b['position']
        while new_pos != position:
            new_pos = utility.get_next_position(new_pos, direction)
            if board[new_pos] in [
                    constants.Item.Rigid.value, constants.Item.Wood.value
            ]:
                break
        if new_pos == position:
            return True
    return False


def _filter_invalid_directions(board, my_position, directions, enemies):
    ret = []
    for direction in directions:
        position = utility.get_next_position(my_position, direction)
        if utility.position_on_board(
                board, position) and utility.position_is_passable(
                    board, position,
                    enemies) and not position_is_skull(board, position):
            ret.append(direction)
    return ret


# ================================
# FLOOD FILL WITH BOMB CHECKING
# ===============================
def score_func_with_target(target, obs):

    def convert_bombs(strength_map, life_map):
        ret = []
        locations = np.where(strength_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(strength_map[(r, c)]),
                'bomb_life': int(life_map[(r, c)]),
                'moving_direction': None
            })
        return ret

    # If we die, -100
    if obs['board'][obs['position']] not in\
       [constants.Item.Agent0.value, constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value]:
        return -100

    # Check four directions of the target
    score = 0
    target_pos = np.where(obs['board'] == target.value)
    if target_pos[0]:
        target_pos = (target_pos[0][0], target_pos[1][0])
        directions_to_check = [
            constants.Action.Left, constants.Action.Right, constants.Action.Up,
            constants.Action.Down
        ]
        bombs = convert_bombs(
            np.array(obs['bomb_blast_strength']), np.array(obs['bomb_life']))

        # Checking_range to limit search depth and calculate biggest possible area
        checking_range = 3
        frontier_positions = [target_pos]
        checked_positions = []

        total_area = 1
        passable_area = 0

        if not check_if_in_bomb_range(obs['board'], bombs, target_pos):
            passable_area += 1

        for i in range(checking_range):
            total_area += 4 * (i + 1)
            new_frontiers = []
            for front in frontier_positions:
                for direction in directions_to_check:
                    new_pos = utility.get_next_position(front, direction)
                    if new_pos not in checked_positions + frontier_positions and\
                       utility.position_on_board(obs['board'], new_pos) and\
                       obs['board'][new_pos] not in \
                       [constants.Item.Rigid.value, constants.Item.Wood.value, constants.Item.Bomb.value, constants.Item.Agent0.value, constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value]:

                        new_frontiers.append(new_pos)
                        if not check_if_in_bomb_range(obs['board'], bombs,
                                                      new_pos):
                            passable_area += 1

            checked_positions = checked_positions + frontier_positions
            frontier_positions = new_frontiers

        score += 100 - (100 * passable_area / total_area)
    else:
        # Target is dead
        return 100

    # if the agent is close to its enemy, then the score goes up
    tar, tac = obs['position']  # target agent row, target agent column
    sar, sac = target_pos
    distance = abs(tar - sar) + abs(
        tac - sac)  # (((tar - sar) ** 2 + (tac - sac) ** 2) ** 0.5
    if distance != 0:
        score += (int)(25 / distance)

    # print(passable_area, "/", total_area)
    # print("SCORE: \n",obs['board'], score)
    return score


#
# A SCORE FUNCTION THAT EVADES BOMBS
#
def score_func_evade(obs):

    def convert_bombs(strength_map, life_map):
        ret = []
        locations = np.where(strength_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(strength_map[(r, c)]),
                'bomb_life': int(life_map[(r, c)]),
                'moving_direction': None
            })
        return ret

    # If we die, -100
    if obs['board'][obs['position']] not in\
       [constants.Item.Agent0.value, constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value]:
        return -100

    position = obs['position']
    board = obs['board']
    bombs = convert_bombs(
        np.array(obs['bomb_blast_strength']), np.array(obs['bomb_life']))
    score = 100

    # Trace through bombs
    for b in bombs:
        # Set the direction to trace
        direction = None
        if (b['position'][0] == position[0] and
                abs(b['position'][1] - position[1]) <= b['blast_strength']):
            if b['position'][1] < position[1]:
                direction = constants.Action.Right
            elif b['position'][1] > position[1]:
                direction = constants.Action.Left
        elif (b['position'][1] == position[1] and
              abs(b['position'][0] - position[0]) <= b['blast_strength']):
            if b['position'][0] < position[0]:
                direction = constants.Action.Down
            elif b['position'][0] > position[0]:
                direction = constants.Action.Up
        # If no in bomb direction
        else:
            continue

        # If bomb is right on me
        if direction is None:
            score -= 100

        # Trace from bomb to see if there's block in the way
        new_pos = b['position']
        while new_pos != position:
            new_pos = utility.get_next_position(new_pos, direction)
            if board[new_pos] in [
                    constants.Item.Rigid.value, constants.Item.Wood.value,
                    constants.Item.Flames.value
            ]:
                break
        if new_pos == position and b['bomb_life'] < 10:
            score -= 25 * (11 - b['bomb_life']) / 10
    return score


# =====================================
# ----MCTS search winning condition----
# ----Target is the value of target----
# ----Cond 1:--------------------------
# -----Target surrounded---------------
# -------------------------------------
# ----Cond 2:--------------------------
# -----Target sur' by 2 bombs or walls-
# -----Cannot kick, but harder to win--
# =====================================
def win_cond_with_target(target, obs):
    # ============
    # --win-cond--
    # ============
    cond_num = 1
    if (cond_num == 1):
        board = np.array(obs['board'])
        coord = np.where(board == target.value)
        if not len(coord[0]):
            # Target Agent is dead, check if we're still alive
            return obs['board'][obs['position']] in [
                constants.Item.Agent0.value, constants.Item.Agent1.value,
                constants.Item.Agent2.value, constants.Item.Agent3.value
            ]

        pos = (coord[0][0], coord[1][0])
        for direction in\
            [constants.Action.Up, constants.Action.Down, constants.Action.Left, constants.Action.Right]:

            new_pos = utility.get_next_position(pos, direction)

            if utility.position_on_board(board, new_pos) and \
               board[new_pos] not in \
               [constants.Item.Bomb.value, constants.Item.Rigid.value, constants.Item.Wood.value, constants.Item.Agent0.value, constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value]:
                return False

        return True
    elif (cond_num == 2):
        if verbose:
            print("Unknown Cond_Num")
        return False
    if verbose:
        print("Unknown Cond_Num")
    return False


# Win if target is at location - used for traveling to safe position in evade
def win_if_arrive(target, obs):
    return obs['position'] == target


#
# A SCORE FUNCTION THAT CHECKS THE 4 DIRECTIONS OF THE AGENT
#
def score_func_with_target_FOUR(target, obs):
    # If we die, -100
    if obs['board'][obs['position']] not in\
       [constants.Item.Agent0.value, constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value]:
        return -100

    # Check four directions of the target
    score = 0
    target_pos = np.where(obs['board'] == target.value)
    if target_pos[0]:
        target_pos = (target_pos[0][0], target_pos[1][0])
        directions_to_check = [
            constants.Action.Left, constants.Action.Right, constants.Action.Up,
            constants.Action.Down
        ]

        checking_range = 4  # obs['blast_strength']

        for direction in directions_to_check:
            new_pos = target_pos
            for j in range(checking_range):
                new_pos = utility.get_next_position(new_pos, direction)
                if not utility.position_on_board(obs['board'], new_pos) or\
                   obs['board'][new_pos] in\
                   [constants.Item.Rigid.value, constants.Item.Wood.value]:  # , constants.Item.Agent0.value, constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value]:
                    score += (int)(25 * (checking_range - j) / checking_range)
                    break
                if obs['board'][new_pos] == constants.Item.Bomb.value:
                    score += 25
                    break
    else:
        # Target is dead
        return 100

    # if the agent is close to its enemy, then the score goes up
    tar, tac = obs['position']  # target agent row, target agent column
    sar, sac = target_pos
    distance = abs(tar - sar) + abs(
        tac - sac)  # (((tar - sar) ** 2 + (tac - sac) ** 2) ** 0.5
    if distance != 0:
        score += (int)(25 / distance)

    # print("SCORE: \n",obs['board'], score)
    return score

    # Check four directions of self
    self_pos = obs['position']
    directions_to_check = [
        constants.Action.Left, constants.Action.Right, constants.Action.Up,
        constants.Action.Down
    ]

    checking_range = 4  # obs['blast_strength']

    for direction in directions_to_check:
        new_pos = self_pos
        for j in range(checking_range):
            new_pos = utility.get_next_position(new_pos, direction)
            if not utility.position_on_board(obs['board'], new_pos) or\
               obs['board'][new_pos] in\
               [constants.Item.Rigid.value, constants.Item.Wood.value, constants.Item.Agent0.value, constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value]:
                score -= (int)(10 * (checking_range - j) / checking_range)
                break

            if obs['board'][new_pos] == constants.Item.Bomb.value:
                score -= 10
                break

    # print("SCORE: \n",obs['board'], score)

    return score


class MCTSAgent(BaseAgent):
    level = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bombing_agents = {}
        self._board_size = kwargs.get('board_size', 11)
        self.copy_board = np.zeros((self._board_size, self._board_size))
        self.copy_walls = True

    def find_next_move(self, obs, action_space, win_condition, score_func,
                       bombing_agents):
        # prep tuff
        self.win_condition = win_condition
        self.score_func = score_func
        self.bombing_agents = bombing_agents

        self.action_space = action_space
        my_pos = tuple(obs['position'])
        board = np.array(obs['board'])
        self.board = np.array(obs['board'])
        self._enemies = [constants.Item(e) for e in obs['enemies']]

        if (self.copy_walls):
            self.copy_board[board == 1] = 9999

        self.copy_board[my_pos[0]][my_pos[1]] += 1

        # check new bombs on field first
        bomb_life_map = np.array(obs['bomb_life'])

        # self.find_bombing_agents(bomb_life_map, board)

        # mcts stuff
        tree = gn.Tree(obs, True, self.bombing_agents, self._board_size)
        # get the root node
        self.rootNode = tree.get_root_node()

        # need way to find terminating condition
        self.end_time = 30
        start_time = time.time()
        elapsed = 0
        # while(elapsed < self.end_time):
        while (self.rootNode.visit_count < 250):
            promising_node = self.select_promising_node(self.rootNode)

            # expand that node
            # create the childs for that node
            if (promising_node == self.rootNode and
                    self.rootNode.visit_count == 0):
                self.expand_node(promising_node)

                # Check if any immediate children from the root node satisfies winning condition
                # If so, return that move directly
            if (promising_node == self.rootNode and
                    self.rootNode.visit_count == 0):
                winning_move = self.select_winning_move_from_root(
                    promising_node)
                if winning_move is not -1:
                    if winning_move is constants.Action.Stop or winning_move is 5:
                        return 5
                    return winning_move.value
            if (promising_node == self.rootNode):
                nodeToExplore = promising_node.get_random_child_node()
            else:
                nodeToExplore = promising_node  # promising_node.get_random_child_node()

            # simulate
            simulationResult = self.simulate_random_play(nodeToExplore)

            # propogate up
            self.back_propogation(nodeToExplore, simulationResult)

            elapsed = time.time() - start_time
            if (elapsed >= 0.095):
                # print("VISIT COUNT: ", self.rootNode.visit_count)
                # print(obs['board'])
                # for child in self.rootNode.childArray:
                #        print(child.my_move ,child.score, child.visit_count, self.UCB(child, child.get_win_score(), child.get_visit_count(), self.rootNode.get_visit_count(), True))
                break

        # winner is root node with child with big score
        # winner_node = rootNode.get_child_with_max_score()
        winner_node = None
        max_ucb = float('-inf')
        for child in self.rootNode.childArray:
            UCB1 = self.UCB(child, child.get_win_score(),
                            child.get_visit_count(),
                            self.rootNode.get_visit_count(), True)
            move = child.state.move
            if type(child.state.move) != int:
                move = child.state.move.value
            if UCB1 > max_ucb and move in self.action_space:
                max_ucb = UCB1
                winner_node = child
        if not (winner_node is None):
            self.bombing_agents = winner_node.state.bombing_agents
            return winner_node.state.move
        else:
            return -1

    def UCB(self,
            the_node,
            child_win_score,
            child_visit_count,
            current_visit_count,
            best=False):
        raise NotImplementedError()

    def select_promising_node(self, rootNode):
        parentVisit = rootNode.visit_count

        # check for children
        if (rootNode.childArray == []):
            return rootNode

        best = 0
        best_node = None

        for child in rootNode.childArray:
            UCB1 = self.UCB(child, child.get_win_score(),
                            child.get_visit_count(),
                            self.rootNode.get_visit_count())
            if UCB1 > best or best_node == None:
                best = UCB1
                best_node = child

        # currentNode = rootNode
        # while currentNode.childArray != []:
        #     best = 0
        #     best_node = None
        #     for child in currentNode.childArray:
        #         UCB1 = self.UCB(child, child.get_win_score(), child.get_visit_count(), self.rootNode.get_visit_count())
        #         if UCB1 > best or best_node == None:
        #             best = UCB1
        #             best_node = child

        #     currentNode = best_node

        return best_node

    def expand_node(self, promising_node):

        # get the node
        # get the state from that node
        # say these are all the possible states I can go to?
        possible_states = promising_node.state.get_all_possible_states()
        for state in possible_states:
            new_node = gn.Node(state._obs, True, state.bombing_agents,
                               self._board_size)
            new_node.set_state(state)
            new_node.bombing_agents = new_node.state.bombing_agents
            new_node.parent = promising_node
            new_node.increment_move_count(promising_node.get_move_count())
            new_node.my_move = state.move

            # New Changes TO MCM
            # new_node.score = state.score
            self.set_expand_node_state(new_node)

            promising_node.childArray.append(new_node)

    def set_expand_node_state(self, new_node):
        raise NotImplementedError()

    def simulate_random_play(self, nodeToExplore):

        temp_copy = nodeToExplore.clone()  #copy.deepcopy(nodeToExplore)
        temp_copy.state.score = 0
        state_won = False
        depth = random.randint(1, 3)
        while (temp_copy.get_move_count() < depth):
            temp_copy.state.look_randomPlay()
            temp_copy.increment_move_count(temp_copy.get_move_count())

            # temp_copy.state.score -= 10
            if (self.is_state_winner(temp_copy)):
                # print("WINNING SHIT:\n", temp_copy.state._board)
                state_won = True
                break

        # print(temp_copy.my_move)
        if (state_won):
            temp_copy.state.score = 100
        else:
            temp_copy.state.score = self.score_func(temp_copy.state._obs)
        temp_copy.score = temp_copy.state.score

        # CHANGED FOR OPTIMIZATION
        # nodeToExplore.state.score = temp_copy.state.score
        # nodeToExplore.score = temp_copy.state.score

        nodeToExplore.state.score += temp_copy.state.score
        nodeToExplore.score += temp_copy.state.score

        nodeToExplore.visit_count += 1

        return temp_copy.state.score

    def back_propogation(self, nodeToExplore, score):
        parent = nodeToExplore

        while parent.parent != None:
            parent = parent.parent

            parent.visit_count += 1
            # parent.win_score += win

            parent.score += score

    def find_bombing_agents(self, bomb_life_map, board):

        # only add initial bombs
        locations = np.where(bomb_life_map == constants.DEFAULT_BOMB_LIFE - 1)
        for r, c in zip(locations[0], locations[1]):
            b = board[r][c] - 10

            self.bombing_agents[(r, c)] = b

        # update kicked bombs
        # remove the older bombs
        keys_to_pop = []
        keys_to_add = []
        for key in self.bombing_agents.keys():
            if bomb_life_map[key[0]][key[
                    1]] == 0:  #or board[key[0]][key[1]] == 4:
                # check all directions
                # up
                r = key[0] - 1
                c = key[1]
                if (r >= 0):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # down
                r = key[0] + 1
                c = key[1]
                if (r < self._board_size):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # left
                r = key[0]
                c = key[1] - 1
                if (c >= 0):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # right
                r = key[0]
                c = key[1] + 1
                if (c < self._board_size):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                keys_to_pop.append((key[0], key[1]))
        for k in keys_to_pop:
            self.bombing_agents.pop(k, None)
        for k in keys_to_add:
            self.bombing_agents[k[0]] = k[1]
            # print(self.bombing_agents)
            # input("main mcts updating for kick")

    def is_state_winner(self, temp_node):
        return self.win_condition(temp_node.state._obs)

    def select_winning_move_from_root(self, rootNode):
        winning_move = -1
        for childNode in rootNode.childArray:
            if self.is_state_winner(childNode):
                winning_move = childNode.my_move
                break
        return winning_move


class MCTSAgentExploit(MCTSAgent):

    def UCB(self,
            the_node,
            child_win_score,
            child_visit_count,
            current_visit_count,
            best=False):

        if child_visit_count == 0:  #(the_node.childArray == []):
            return 100
        else:
            exploitation = the_node.score / child_visit_count
            # exploration = 50 * np.sqrt(2.0*np.log(current_visit_count) / child_visit_count)
            # print("ploit: ", exploitation, " explor: ", exploration)
            return exploitation  # + exploration

    def set_expand_node_state(self, new_node):
        new_node.score = 100
        new_node.visit_count = 1


class MCTSAgentExplore(MCTSAgent):

    def UCB(self,
            the_node,
            child_win_score,
            child_visit_count,
            current_visit_count,
            best=False):

        if child_visit_count == 0:  #(the_node.childArray == []):
            return 100
        else:

            if best:
                return the_node.score / child_visit_count  # + exploration
            else:
                return current_visit_count / child_visit_count

    def set_expand_node_state(self, new_node):
        new_node.score = 0
        new_node.visit_count = 0
