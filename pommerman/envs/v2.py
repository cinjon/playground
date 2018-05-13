"""The Pommerman v2 Environment, which implements radio communication across the agents.

The communication works by allowing each agent to send a vector of radio_num_words (default = 2) from a vocabulary
of size radio_vocab_size (default = 8) to its teammate each turn. These vectors are passed into the observation
stream for each agent.
"""
from gym import spaces
import numpy as np

from .. import constants
from .. import utility
from . import v0


class Pomme(v0.Pomme):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'rgb_pixel'],
        'video.frames_per_second' : constants.RENDER_FPS
    }

    def __init__(self, *args, **kwargs):
        self._radio_vocab_size = kwargs.get('radio_vocab_size')
        self._radio_num_words = kwargs.get('radio_num_words')
        if (self._radio_vocab_size and not self._radio_num_words) or (not self._radio_vocab_size and self._radio_num_words):
            assert("Please provide both radio_vocab_size and radio_num_words to use the Radio environment.")

        self._radio_from_agent = {
            agent: (0, 0) for agent in [constants.Item.Agent0, constants.Item.Agent1, constants.Item.Agent2, constants.Item.Agent3]
        }
        super().__init__(*args, **kwargs)

    def _set_action_space(self):
        self.action_space = spaces.Tuple(tuple([spaces.Discrete(6)] + [spaces.Discrete(self._radio_vocab_size)]*self._radio_num_words))

    def _set_observation_space(self):
        """The Observation Space for each agent.

        There are a total of 3*board_size^2+12+radio_vocavb_size*radio_num_words observations:
        - all of the board (board_size^2)
        - bomb blast strength (board_size^2).
        - bomb life (board_size^2)
        - agent's position (2)
        - player ammo counts (1)
        - blast strength (1)
        - can_kick (1)
        - teammate (one of {AgentDummy.value, Agent3.value}).
        - enemies (three of {AgentDummy.value, Agent3.value}).
        - radio (radio_vocab_size * radio_num_words)
        """
        bss = self._board_size**2
        min_obs = [0]*3*bss + [0]*5 + [constants.Item.AgentDummy.value]*4
        max_obs = [len(constants.Item)]*bss + [self._board_size]*bss + [25]*bss
        max_obs += [self._board_size]*2 + [self._num_items]*2 + [1]
        max_obs += [constants.Item.Agent3.value]*4
        min_obs.extend([0]*self._radio_vocab_size*self._radio_num_words)
        max_obs.extend([1]*self._radio_vocab_size*self._radio_num_words)
        self.observation_space = spaces.Box(np.array(min_obs), np.array(max_obs))

    def get_observations(self):
        observations = super().get_observations()
        for obs in observations:
            obs['message'] = self._radio_from_agent[obs['teammate']]

        self.observations = observations
        return observations

    def step(self, actions):
        personal_actions = [action[0] for action in actions]
        radio_actions = [action[1:] for action in actions]

        for radio_action, agent in zip(radio_actions, self._agents):
            if not agent.is_alive:
                radio_action = (0, 0)
            else:
                radio_action = np.clip(radio_action, 1, 8).astype(np.uint8)
            self._radio_from_agent[getattr(constants.Item, 'Agent%d' % agent.agent_id)] = radio_action

        return super().step(personal_actions)

    @staticmethod
    def featurize(obs):
        ret = super().featurize(obs)
        message = obs['message']
        message = utility.make_np_float(message)
        return np.concatenate((ret, message))

    def get_json_info(self):
        ret = super().get_json_info()
        ret['radio_vocab_size'] = json.dumps(self._radio_vocab_size, cls=json_encoder)
        ret['radio_num_words'] = json.dumps(self._radio_num_words, cls=json_encoder)
        ret['_radio_from_agent'] = json.dumps(self._radio_from_agent, cls=json_encoder)
        return ret

    def set_json_info(self):
        super().set_json_info()
        self.radio_vocab_size = json.loads(self._init_game_state['radio_vocab_size'])
        self.radio_num_words = json.loads(self._init_game_state['radio_num_words'])
        self._radio_from_agent = json.loads(self._init_game_state['_radio_from_agent'])
