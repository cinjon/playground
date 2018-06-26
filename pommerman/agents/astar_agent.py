from pommerman.agents import BaseAgent
from collections import defaultdict
import itertools
import heapq
import itertools
from scipy.spatial.distance import cityblock
from .. import characters

class AstarAgent(BaseAgent):
    # TODO: decide whether we want to keep all these functions understand
    # the AstarAgent class (since they are not used by any other class at the moment)
    # however they are quite general so maybe move them to a uitls file in agents?
    """This agent uses the A* algorithm to
    find the shortest path to an assigned goal
    and takes actions along that path.

    TODO: implement the dijkstra version as well
    shouldn't be any difference other than Dijkstra
    could be a bit slower."""

    def __init__(self, character=characters.Walker, *args, **kwargs):
        #TODO: do we need args kwargs below?
        super(AstarAgent, self).__init__(character, *args, **kwargs)

        self.is_simple_agent = True

    def act(self, obs, action_space):
        self.obs = obs  # single agent

        # TODO: check these work as expected
        self._agent_pos = tuple(self.obs['position'])
        self._goal_pos = tuple(self.obs['goal_position'])
        self._step = self.obs['step']
        self._board_size = len(self.obs['board'])

        # TODO: check this works as expected
        if self._step == 0:
            self._path = self._get_path()

        # Picking next action from precalculated path to goal
        next_loc = self._path[self._step]
        action = self._action_to_loc(next_loc)

        return action

    def _get_path(self):
        '''
        Returns shortest path to goal from
        the agent's initial position to its goal.
        '''
        # Find shortest path to goal using A* alg
        _, came_from = self._astar(self._agent_move_func,
                                   self._agent_pos, self._goal_pos)

        # Path to goal
        path = self._reconstruct_path(came_from, self._agent_pos,
                                      self._goal_pos)

        return path

    # TODO: this should be changed; check what each action does
    def _action_to_loc(self, next_loc):
        '''
        Returns the action the agent
        must take to get to next_loc.
        '''
        current_loc = self._agent_pos
        if not self._within_bounds(current_loc) or \
                not self._within_bounds(next_loc):

            raise Exception("One of the locations is \
            not valid for this maze.")

        x0, y0 = current_loc
        x, y = next_loc

        dx = next_loc[0] - current_loc[0]
        dy = next_loc[1] - current_loc[1]

        if dx == 1:
            return 4 # right
        elif dx == - 1:
            return 3 # left
        elif dy == 1:
            return 1 # up
        elif dy == - 1:
            return 2 # down
        else:
            return 0 # stop

    def _agent_move_func(self, loc):
        '''
        Returns all the locations the agent can move to
        from its current position: up, down, east, west.

        Note:
        The next position must be inside the maze.
        '''
        res = []
        x, y = loc
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not self._within_bounds((nx, ny)):
                continue
            res.append((nx, ny))
        return res

    def _dijkstra(self, move_func, start_loc, end_loc=None):
        '''
        Args:
            env -- the environment
            move_func -- f(loc) determines the locations you can move to from loc
            start_loc -- (x, y) tuple of start location
            end_loc -- (x, y) tuple of end location -- optional

        Returns:
            visited -- dictionary of {location: distance} pairs
            path -- dictionary of {location: previous_location} pairs

        Notes:
            - if end_loc is None, then returns visited and path for all the
        nodes in the graph (all locations in the maze)
            - if end_loc is not None, stop searching after finding path to end_loc
        and return the current visited and path (including for end_loc)
        '''

        # total distance from origin (max # in Python)
        visited = defaultdict(lambda: 1e309)
        visited[start_loc] = 0
        path = {}

        nodes = set(itertools.product(range(self._board_size), range(self._board_size)))

        while nodes:
            current = nodes.intersection(visited.keys())
            if not current:
                break
            min_node = min(current, key=visited.get)
            nodes.remove(min_node)
            current_weight = visited[min_node]
            x, y = min_node

            for edge in move_func(min_node):
                weight = current_weight + 1
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge] = min_node
                    # when end_loc is given, return after finding a path to it
                    if edge == end_loc:
                        return visited, path
        return visited, path


    def _astar(self, move_func, start, goal):
        '''
        Implements the Astar algorithm which finds the
        shortest path between start and goal, in gridworld env.

        Args:
            move_func -- f(loc) determines the locations you can move to from loc
            start -- (x, y) tuple of start location
            goal -- (x, y) tuple of end location -- optional

        Returns:
            cost_so_far -- dictionary of {location: path_cost} pairs
            came_from -- dictionary of {location: previous_location} pairs
        '''

        frontier = PriorityQueue()
        frontier.put(start, 0)

        came_from = {}
        cost_so_far = {}

        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            current = frontier.get()

            if current == goal:
                break

            for next in move_func(current):
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + cityblock(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current

        return cost_so_far, came_from


    def _reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()      # optional
        return path

    def _within_bounds(self, location):
        '''
        Checks whether a location is
        in the maze (i.e. within its bounds).
        '''
        x, y = location
        return 0 <= x < self._board_size and 0 <= y < self._board_size

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]
