"""Train an agent with TensorForce.

Call this with a config, a game, and a list of agents, one of which should be a tensorforce agent. The script will start separate threads to operate the agents and then report back the result.

An example with all three simple agents running ffa:
python train_with_tensorforce.py --agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent --config=ffa_v0
"""
import atexit
import functools

import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym

from .. import configs, utility, agents


client = docker.from_env()


def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnv(OpenAIGym):
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def execute(self, actions):
        if self.visualize:
            self.gym.render()

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agents[0], actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent[0]])
        agent_reward = reward[self.gym.training_agent[0]]
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[self.gym.training_agents[0]])
        return agent_obs


def main():
    parser = argparse.ArgumentParser(description='Playground Flags.')
    parser.add_argument('--game',
                        default='pommerman',
                        help='Game to choose.')
    parser.add_argument('--config',
                        default='ffa_v0',
                        help='Configuration to execute.')
    parser.add_argument('--agents',
                        default='tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent',
                        help='Comma delineated list of agent types and docker locations to run the agents.')
    parser.add_argument('--record_dir',
                        help="Directory to record the PNGs of the game. Doesn't record if None.")
    args = parser.parse_args()

    config = utility.AttrDict(getattr(configs, args.config)())
    _agents = []
    for agent_id, agent_info in enumerate(args.agents.split(",")):
        agent = config.agent(agent_id, config.game_type)
        agent_type, agent_control = agent_info.split("::")
        assert agent_type in ["player", "random", "docker", "test", "tensorforce"]
        if agent_type == "player":
            agent = agents.PlayerAgent(agent, agent_control)
        elif agent_type == "random":
            agent = agents.RandomAgent(agent)
        elif agent_type == "docker":
            agent = agents.DockerAgent(
                agent,
                docker_image=agent_control,
                docker_client=client,
                port=agent_id+1000)
        elif agent_type == "test":
            agent = eval(agent_control)(agent)
        elif agent_type == "tensorforce":
            agent = agents.TensorForceAgent(agent, algorithm=agent_control)
            training_agent = agent
        _agents.append(agent)

    gym.envs.registration.register(
        id=config.env_id,
        entry_point=config.env_entry_point,
        kwargs=config.env_kwargs
    )
    env = config.env(**config.env_kwargs)
    env.set_agents(_agents)
    env.set_training_agents([training_agent.agent_id])
    env.seed(0)

    # Create a Proximal Policy Optimization agent
    agent = training_agent.initialize(env)

    atexit.register(functools.partial(clean_up_agents, _agents))
    wrapped_env = WrappedEnv(env, visualize=True)
    runner = Runner(agent=agent, environment=wrapped_env)
    runner.run(episodes=10, max_episode_timesteps=2000)
    print("Stats: ", runner.episode_rewards, runner.episode_timesteps, runner.episode_times)

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    main()
