import os

from .. import agents


use_game_servers = os.getenv("PLAYGROUND_USE_GAME_SERVERS")
game_servers = {id_: os.getenv("PLAYGROUND_GAME_INSTANCE_%d" % id_)
                for id_ in range(4)}


# NOTE: This routine is meant for internal usage.
def make_agent_from_string(agent_string, agent_id, docker_env_dict=None):
    agent_type, agent_control = agent_string.split("::")

    assert agent_type in ["player", "random", "docker", "test", "tensorforce"]

    agent_instance = None

    if agent_type == "player":
        agent_instance = agents.PlayerAgent(agent_control=agent_control)
    elif agent_type == "random":
        agent_instance = agents.RandomAgent()
    elif agent_type == "docker":
        port = agent_id + 1000
        if not use_game_servers:
            server = 'http://localhost'
        else:
            server = game_servers[agent_id]
        assert port is not None
        agent_instance = agents.DockerAgent(
            agent_control, port=port, server=server, env_vars=docker_env_dict)
    elif agent_type == "test":
        kwargs = {}
        if agent_control.endswith('ComplexAgent8'):
            agent_control = agent_control.strip('8')
            kwargs['board_size'] = 8
        agent_instance = eval(agent_control)(**kwargs)
    elif agent_type == "tensorforce":
        agent_instance = agents.TensorForceAgent(algorithm=agent_control)

    return agent_instance
