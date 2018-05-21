"""
NOTE:

There are a few minor complications to fluid human control which make this
code a little more involved than trivial.

1. Key press-release cycles can be, and often are, faster than one tick of
   the game/simulation, but the player still wants that cycle to count, i.e.
   to lay a bomb!
2. When holding down a key, the player expects that action to be repeated,
   at least after a slight delay.
3. But when holding a key down (say, move left) and simultaneously doing a
   quick press-release cycle (put a bomb), we want the held-down key to keep
   being executed, but the cycle should have happened in-between.

The way we solve this problem is by separating key-state and actions-to-do.
We hold the actions that need be executed in a queue (`self._action_q`) and
a state for all considered keys.

1. When a key is pressed down, we note the time and mark it as down.
2. If it is released quickly thereafter, before a game tick could happen,
   we add its action into the queue. This often happens when putting bombs.
3. If it's still pressed down as we enter a game tick, we do some math to see
   if it's time for a "repeat" event and, if so, push an action to the queue.
4. Just work off one item from the queue each tick.

This way, the input is "natural" and things like dropping a bomb while doing
a diagonal walk from one end to the other "just work".
"""

from time import time

from . import BaseAgent
from .. import characters

REPEAT_DELAY = 0.2  # seconds
REPEAT_INTERVAL = 0.1


class Keystate:

    def __init__(self):
        self.keydown_time = time()
        self.last_repeat_time = None
        self.fired = False

    def should_fire(self):
        if self.last_repeat_time is None:
            # The first repetition:
            if time() - self.keydown_time > REPEAT_DELAY:
                return True
        else:
            # A repetition after the first:
            if time() - self.last_repeat_time > REPEAT_INTERVAL:
                return True

        # No repetition yet
        return False

    def mark_fired(self):
        self.last_repeat_time = time()
        self.fired = True


class PlayerAgent(BaseAgent):
    """The Player Agent that lets the user control a character."""

    def __init__(self, character=characters.Bomber, agent_control='arrows'):
        super(PlayerAgent, self).__init__(character)

        ##
        # @NOTE: DO NOT move this import outside the constructor. It will
        # not work in headless environments like a Docker container
        # and prevents Pommerman from running.
        #
        from pyglet.window import key
        CONTROLS = {
            'arrows': {
                key.UP: 1,
                key.DOWN: 2,
                key.LEFT: 3,
                key.RIGHT: 4,
                key.SPACE: 5,
                key.M: 6  # In Pommerman, this will freeze the game.
            },
            'wasd': {
                key.W: 1,
                key.S: 2,
                key.A: 3,
                key.D: 4,
                key.E: 5,
                key.Q: 6  # In Pommerman, this will freeze the game.
            }
        }

        assert agent_control in CONTROLS, "Unknown control: {}".format(
            agent_control)
        self._key2act = CONTROLS[agent_control]

        self._action_q = []
        self._keystate = {}

    def act(self, obs, action_space):
        # Go through the keys and fire for those that needs repetition (because they're held down)
        for k, state in self._keystate.items():
            if state.should_fire():
                self._action_q.append(k)
                state.mark_fired()

        act = 0
        if self._action_q:  # Work off the keys that are queued.
            act = self._key2act[self._action_q.pop(0)]
        return act

    @staticmethod
    def has_user_input():
        return True

    def on_key_press(self, k, mod):
        # Ignore if we're not handling the key. Avoids "shadowing" ticks in
        # multiplayer mode.
        if k in self._key2act:
            self._keystate[k] = Keystate()

    def on_key_release(self, k, mod):
        # We only need to act on keys for which we did something in the
        # `key_press` event, and ignore any other key releases.
        if k in self._keystate:
            # Only mark this as a "press" upon release if it was a quick one,
            # i.e. not held down and executed already
            if not self._keystate[k].fired:
                self._action_q.append(k)
            del self._keystate[k]
