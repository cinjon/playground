"""The Pommerman v1 Environment, which implements a collapsing board.

This environment is the same as v0.py, except that the board will collapse
according to a uniform schedule beginning at the kwarg first_collapse.

The collapsing works in the following manner:
1. Set the collapsing schedule. This is uniform starting at step first_collapse
   and ending at step max_steps.
2. Number the rings on the board from 0 to board_size-1 s.t. the outermost ring
   is 0 and the innermost ring is board_size-1. The cells in the former are
   [[0, i], [i, 0], [board_size-1, i], [i, board_size-1] for i in
   [0, board_size-1]] and the latter is, assuming an odd board_size,
   [[(board_size-1)/2, (board_size-1)/2]].
3. When we are at a step in the collapsing schedule, we take the matching ring
   and turn it into rigid walls. This has the effect of destroying any items,
   bombs (which don't go off), and agents in those squares.
"""
from .. import constants
from .. import utility
from . import v0


class Pomme(v0.Pomme):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'rgb_pixel'],
        'video.frames_per_second': constants.RENDER_FPS
    }

    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)
        first_collapse = kwargs.get('first_collapse')
        self.collapses = list(
            range(first_collapse, self._max_steps,
                  int((self._max_steps - first_collapse) / self._board_size)))

    def _collapse_board(self, ring):
        """Collapses the board at a certain ring radius.

        For example, if the board is 13x13 and ring is 0, then the the ring of the first row, last row,
        first column, and last column is all going to be turned into rigid walls. All agents in that ring
        die and all bombs are removed without detonating.

        For further rings, the values get closer to the center.

        Args:
          ring: Integer value of which cells to collapse.
        """
        board = self._board.copy()

        def collapse(r, c):
            if utility.position_is_agent(board, (r, c)):
                # Agent. Kill it.
                num_agent = board[r][c] - constants.Item.Agent0.value
                agent = self._agents[num_agent]
                agent.die()
            elif utility.position_is_bomb(self._bombs, (r, c)):
                # Bomb. Remove the bomb.
                self._bombs = [b for b in self._bombs if b.position != (r, c)]
            elif (r, c) in self._items:
                # Item. Remove the item.
                del self._items[(r, c)]
            board[r][c] = constants.Item.Rigid.value

        for cell in range(ring, self._board_size - ring):
            collapse(ring, cell)
            if ring != cell:
                collapse(cell, ring)

            end = self._board_size - ring - 1
            collapse(end, cell)
            if end != cell:
                collapse(cell, end)

        return board

    def get_json_info(self):
        ret = super().get_json_info()
        ret['collapses'] = json.dumps(self.collapses, cls=json_encoder)
        return ret

    def set_json_info(self):
        super().set_json_info()
        self.collapses = json.loads(self._init_game_state['collapses'])

    def step(self, actions):
        obs, reward, done, info = super().step(actions)

        for ring, collapse in enumerate(self.collapses):
            if self._step_count == collapse:
                self._board = self._collapse_board(ring)
                break

        return obs, reward, done, info
