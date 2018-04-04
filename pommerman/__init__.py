import gym
import inspect
from . import agents
from . import configs
from . import constants
from . import forward_model
from . import utility

registry = None


def _register():
    global registry
    registry = []
    for name, f in inspect.getmembers(configs, inspect.isfunction):
        if not name.endswith('_env'):
            continue

        config = f()
        gym.envs.registration.register(
            id=config['env_id'],
            entry_point=config['env_entry_point'],
            kwargs=config['env_kwargs']
        )
        registry.append(config['env_id'])


# Register environments with gym
_register()


def make(config_id, agent_list, game_state_file=None):
    assert config_id in registry, "Unknown configuration '{}'. " \
        "Possible values: {}".format(config_id, registry)
    env = gym.make(config_id)

    for id, agent in enumerate(agent_list):
        assert isinstance(agent, agents.BaseAgent)
        # NOTE: This is IMPORTANT so that the agent character is initialized
        agent.init_agent(id, env.spec._kwargs['game_type'])

    env.set_agents(agent_list)
    env.set_init_game_state(game_state_file)
    return env
