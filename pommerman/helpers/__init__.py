from .. import agents


# NOTE: This routine is meant for internal usage.
def make_agent_from_string(agent_string, port=None, docker_env_dict=None):
    agent_type, agent_control = agent_string.split("::")

    assert agent_type in ["player", "random", "docker", "test", "tensorforce"]

    agent_instance = None

    if agent_type == "player":
        agent_instance = agents.PlayerAgent(agent_control=agent_control)
    elif agent_type == "random":
        agent_instance = agents.RandomAgent()
    elif agent_type == "docker":
        assert port is not None
        agent_instance = agents.DockerAgent(agent_control, port=port, env_vars=docker_env_dict)
    elif agent_type == "test":
        agent_instance = eval(agent_control)()
    elif agent_type == "tensorforce":
        agent_instance = agents.TensorForceAgent(algorithm=agent_control)

    return agent_instance
