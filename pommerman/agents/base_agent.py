from .. import characters


class BaseAgent:
    """Parent abstract Agent."""

    def __init__(self, character=characters.Bomber):
        self._character = character

    def __getattr__(self, attr):
        return getattr(self._character, attr)

    def act(self, obs, action_space):
        raise NotImplementedError()

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.

        Args:
          reward: The single reward scalar to this agent.
        """
        pass

    def init_agent(self, id, game_type):
        self._character = self._character(id, game_type)

    @staticmethod
    def has_user_input():
        return False

    def shutdown(self):
        pass

