from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent
from .. import constants
from .. import utility


class SimpleAgent(BaseAgent):
    """This is a baseline agent. After you can beat it, submit your agent to
    compete.
    """

    def __init__(self, *args, **kwargs):
        super(SimpleAgent, self).__init__(*args, **kwargs)

        # Keep track of recently visited uninteresting positions so that we
        # don't keep visiting the same places.
        self._recently_visited_positions = []
        self._recently_visited_length = 6
        # Keep track of the previous direction to help with the enemy
        # standoffs.
        self._prev_direction = None
        self.reset_times()

    def reset_times(self):
        self._time_avg = defaultdict(float)
        self._time_max = defaultdict(float)
        self._time_cnt = defaultdict(int)

    def _update_times(self, t, key):
        avg = self._time_avg[key]
        cnt = self._time_cnt[key]
        new_avg = (float(avg)*float(cnt) + float(t))
        new_avg /= float(cnt + 1)
        self._time_cnt[key] = cnt + 1
        self._time_avg[key] = new_avg
        self._time_max[key] = max(self._time_max[key], float(t))

    def act(self, obs, action_space):
        def convert_bombs(bomb_map):
            ret = []
            locations = np.where(bomb_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append({'position': (r, c),
                            'blast_strength': int(bomb_map[(r, c)])})
            return ret

        my_position = tuple(obs['position'])
        board = np.array(obs['board'])
        with utility.Timer() as t:
            bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
        self._update_times(t.interval, 'convert_bombs')

        enemies = [constants.Item(e) for e in obs['enemies']]
        ammo = int(obs['ammo'])
        blast_strength = int(obs['blast_strength'])
        with utility.Timer() as t:
            items, dist, prev = self._djikstra(board, my_position, bombs,
                                               enemies, depth=10)
        self._update_times(t.interval, 'djikstra')

        # Move if we are in an unsafe place.
        with utility.Timer() as t:
            unsafe_directions = self._directions_in_range_of_bomb(
                board, my_position, bombs, dist)
        self._update_times(t.interval, 'directions_in_range_of_bomb')
        if unsafe_directions:
            with utility.Timer() as t:
                directions = self._find_safe_directions(
                    board, my_position, unsafe_directions, bombs, enemies)
            self._update_times(t.interval, 'find_safe_directions')
            return random.choice(directions).value

        # Lay pomme if we are adjacent to an enemy.
        if self._is_adjacent_enemy(items, dist, enemies) and \
           self._maybe_bomb(ammo, blast_strength, items, dist, my_position):
            return constants.Action.Bomb.value

        # Move to an enemy if there is one in exactly three reachable spaces.
        with utility.Timer() as t:
            direction = self._near_enemy(my_position, items, dist, prev,
                                         enemies, 3)
        self._update_times(t.interval, 'near_enemy')
        if direction is not None and \
           (self._prev_direction != direction or random.random() < .5):
            self._prev_direction = direction
            return direction.value

        # Move towards a good item if there is one within two reachable spaces.
        with utility.Timer() as t:
            direction = self._near_good_powerup(
                my_position, items, dist, prev, 2)

        self._update_times(t.interval, 'near_good_powerup')
        if direction is not None:
            return direction.value

        # Maybe lay a bomb if we are within a space of a wooden wall.
        with utility.Timer() as t:
            if self._near_wood(my_position, items, dist, prev, 1):
                if self._maybe_bomb(ammo, blast_strength, items, dist,
                                    my_position):
                    return constants.Action.Bomb.value
                else:
                    return constants.Action.Stop.value
        self._update_times(t.interval, 'near_wood_1')

        # Move towards a wooden wall if there is one within two reachable
        # spaces and you have a bomb.
        with utility.Timer() as t:
            direction = self._near_wood(my_position, items, dist, prev, 2)
        self._update_times(t.interval, 'near_wood_2')
        if direction is not None:
            with utility.Timer() as t:
                directions = self._filter_unsafe_directions(board, my_position,
                                                            [direction], bombs)
                if directions:
                    return directions[0].value
            self._update_times(t.interval, 'filter_unsafe_directions')

        # Choose a random but valid direction.
        directions = [constants.Action.Stop, constants.Action.Left,
                      constants.Action.Right, constants.Action.Up,
                      constants.Action.Down]
        with utility.Timer() as t:
            valid_directions = self._filter_invalid_directions(
                board, my_position, directions, enemies)
        self._update_times(t.interval, 'filter_invalid_directions')
        with utility.Timer() as t:
            directions = self._filter_unsafe_directions(
                board, my_position, valid_directions, bombs)
        self._update_times(t.interval, 'filter_unsafe_directions')
        with utility.Timer() as t:
            directions = self._filter_recently_visited(
                directions, my_position, self._recently_visited_positions)
        self._update_times(t.interval, 'filter_recently_visited')
        if len(directions) > 1:
            directions = [k for k in directions if k != constants.Action.Stop]
        if not len(directions):
            directions = [constants.Action.Stop]

        # Add this position to the recently visited uninteresting positions so
        # we don't return immediately.
        self._recently_visited_positions.append(my_position)
        self._recently_visited_positions = \
            self._recently_visited_positions[-self._recently_visited_length:]

        return random.choice(directions).value

    @staticmethod
    def _djikstra(board, my_position, bombs, enemies, depth=None,
                  exclude=None):
        assert(depth is not None)

        if exclude is None:
            exclude = [constants.Item.Fog, constants.Item.Rigid,
                       constants.Item.Skull, constants.Item.Flames]

        def out_of_range(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            return depth is not None and abs(y2 - y1) + abs(x2 - x1) > depth

        items = defaultdict(list)
        dist = {}
        prev = {}
        Q = queue.PriorityQueue()

        mx, my = my_position
        for r in range(max(0, mx - depth), min(13, mx + depth)):
            for c in range(max(0, my - depth), min(13, my + depth)):
                position = (r, c)
                if any([
                        out_of_range(my_position, position),
                        utility.position_in_items(board, position, exclude),
                ]):
                    continue

                if position == my_position:
                    dist[position] = 0
                else:
                    dist[position] = np.inf

                prev[position] = None
                Q.put((dist[position], position))

        for bomb in bombs:
            if bomb['position'] == my_position:
                items[constants.Item.Bomb].append(my_position)

        while not Q.empty():
            _, position = Q.get()

            if utility.position_is_passable(board, position, enemies):
                x, y = position
                val = dist[(x, y)] + 1
                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + x, col + y)
                    if new_position not in dist:
                        continue

                    if val < dist[new_position]:
                        dist[new_position] = val
                        prev[new_position] = position

            item = constants.Item(board[position])
            items[item].append(position)

        return items, dist, prev

    def _directions_in_range_of_bomb(self, board, my_position, bombs, dist):
        ret = defaultdict(int)

        x, y = my_position
        for bomb in bombs:
            position = bomb['position']
            distance = dist.get(position)
            if distance is None:
                continue

            bomb_range = bomb['blast_strength']
            if distance > bomb_range:
                continue

            if my_position == position:
                # We are on a bomb. All directions are in range of bomb.
                for direction in [
                    constants.Action.Right,
                    constants.Action.Left,
                    constants.Action.Up,
                    constants.Action.Down,
                ]:
                    ret[direction] = max(ret[direction],
                                         bomb['blast_strength'])
            elif x == position[0]:
                if y < position[1]:
                    # Bomb is right.
                    ret[constants.Action.Right] = max(
                        ret[constants.Action.Right], bomb['blast_strength'])
                else:
                    # Bomb is left.
                    ret[constants.Action.Left] = max(
                        ret[constants.Action.Left], bomb['blast_strength'])
            elif y == position[1]:
                if x < position[0]:
                    # Bomb is down.
                    ret[constants.Action.Down] = max(
                        ret[constants.Action.Down], bomb['blast_strength'])
                else:
                    # Bomb is down.
                    ret[constants.Action.Up] = max(ret[constants.Action.Up],
                                                   bomb['blast_strength'])
        return ret

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

        # All directions are unsafe. Return a position that isn't locked.
        safe = []

        if len(unsafe_directions) == 4:
            next_board = board.copy()
            next_board[my_position] = constants.Item.Bomb.value

            for direction, bomb_range in unsafe_directions.items():
                next_position = utility.get_next_position(my_position,
                                                          direction)
                nx, ny = next_position
                if not utility.position_on_board(next_board, next_position) \
                   or not utility.position_is_passable(next_board,
                                                       next_position, enemies):
                    continue

                if not is_stuck_direction(next_position, bomb_range,
                                          next_board, enemies):
                    # We found a direction that works. The .items provided
                    # a small bit of randomness. So let's go with this one.
                    return [direction]
            if not safe:
                safe = [constants.Action.Stop]
            return safe

        x, y = my_position
        disallowed = [] # The directions that will go off the board.

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

            if utility.position_is_passable(board, position, enemies) \
               or utility.position_is_fog(board, position):
                safe.append(direction)

        if not safe:
            # No safe directions, return something that is allowed.
            safe = [k for k in unsafe_directions if k not in disallowed]

        if not safe:
            # We don't have ANY directions. So return the stop choice.
            return [constants.Action.Stop]

        return safe

    @staticmethod
    def _is_adjacent_enemy(items, dist, enemies):
        for enemy in enemies:
            for position in items.get(enemy, []):
                if dist[position] == 1:
                    return True
        return False

    @staticmethod
    def _has_bomb(obs):
        return obs['ammo'] >= 1

    @staticmethod
    def _maybe_bomb(ammo, blast_strength, items, dist, my_position):
        """Returns whether we can safely bomb right now.
        Decides this based on:
        1. Do we have ammo?
        2. If we laid a bomb right now, will we be stuck?
        """
        # Do we have ammo?
        if ammo < 1:
            return False

        # Will we be stuck?
        x, y = my_position
        for position in items.get(constants.Item.Passage):
            if dist[position] == np.inf:
                continue

            # We can reach a passage that's outside of the bomb strength.
            if dist[position] > blast_strength:
                return True

            # We can reach a passage that's outside of the bomb scope.
            px, py = position
            if px != x and py != y:
                return True

        return False

    @staticmethod
    def _nearest_position(dist, objs, items, radius):
        nearest = None
        dist_to = max(dist.values())

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

    # @classmethod
    # def _near_enemy(cls, my_position, items, dist, prev, enemies, radius):
    def _near_enemy(self, my_position, items, dist, prev, enemies, radius):
        with utility.Timer() as t:
            nearest_enemy_position = self._nearest_position(dist, enemies,
                                                            items, radius)
        self._update_times(t.interval, 'nearest_position')
        with utility.Timer() as t:
            return self._get_direction_towards_position(
                my_position, nearest_enemy_position, prev)
        self._update_times(t.interval, 'get_direction_towards_position')

    # @classmethod
    # def _near_good_powerup(cls, my_position, items, dist, prev, radius):
    def _near_good_powerup(self, my_position, items, dist, prev, radius):
        objs = [
            constants.Item.ExtraBomb,
            constants.Item.IncrRange,
            constants.Item.Kick
        ]
        with utility.Timer() as t:
            nearest_item_position = self._nearest_position(dist, objs, items,
                                                           radius)
        self._update_times(t.interval, 'nearest_position')
        with utility.Timer() as t:
            return self._get_direction_towards_position(
                my_position, nearest_item_position, prev)
        self._update_times(t.interval, 'get_direction_towards_position')

    # @classmethod
    # def _near_wood(cls, my_position, items, dist, prev, radius):
    def _near_wood(self, my_position, items, dist, prev, radius):
        objs = [constants.Item.Wood]
        with utility.Timer() as t:
            nearest_item_position = self._nearest_position(dist, objs, items,
                                                           radius)
        self._update_times(t.interval, 'nearest_position')
        with utility.Timer() as t:
            return self._get_direction_towards_position(
                my_position, nearest_item_position, prev)
        self._update_times(t.interval, 'get_direction_towards_position')

    @staticmethod
    def _filter_invalid_directions(board, my_position, directions, enemies):
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(board, position) \
               and utility.position_is_passable(board, position, enemies):
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_unsafe_directions(board, my_position, directions, bombs):
        ret = []
        for direction in directions:
            x, y = utility.get_next_position(my_position, direction)
            is_bad = False
            for bomb in bombs:
                bx, by = bomb['position']
                blast_strength = bomb['blast_strength']
                if (x == bx and abs(by - y) <= blast_strength) or \
                   (y == by and abs(bx - x) <= blast_strength):
                    is_bad = True
                    break
            if not is_bad:
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_recently_visited(directions, my_position,
                                 recently_visited_positions):
        ret = []
        for direction in directions:
            positions = utility.get_next_position(my_position, direction)
            if not positions in recently_visited_positions:
                ret.append(direction)

        if not ret:
            ret = directions
        return ret
