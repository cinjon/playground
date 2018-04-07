"""Train an agent with TensorForce.

Call this with a config, a game, and a list of agents, one of which should be a
tensorforce agent. The script will start separate threads to operate the agents
and then report back the result.

An example with all three simple agents running ffa:
python train_with_tensorforce.py \
 --agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
 --config=ffa_v0
"""
import atexit
import functools
import os

import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym

from .. import helpers, make
from ..agents import TensorForceAgent


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
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game",
                        default="pommerman",
                        help="Game to choose.")
    parser.add_argument("--config",
                        default="PommeFFA-v0",
                        help="Configuration to execute. See env_ids in "
                        "configs.py for options.")
    parser.add_argument("--agents",
                        default="tensorforce::ppo,test::agents.SimpleAgent,"
                        "test::agents.SimpleAgent,test::agents.SimpleAgent",
                        help="Comma delineated list of agent types and docker "
                        "locations to run the agents.")
    parser.add_argument("--agent_env_vars",
                        help="Comma delineated list of agent environment vars "
                        "to pass to Docker. This is only for the Docker Agent."
                        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
                        "would send two arguments to Docker Agent 0 and one to"
                        " Docker Agent 3.",
                        default="")
    parser.add_argument("--record_pngs_dir",
                        default=None,
                        help="Directory to record the PNGs of the game. "
                        "Doesn't record if None.")
    parser.add_argument("--record_json_dir",
                        default=None,
                        help="Directory to record the JSON representations of "
                        "the game. Doesn't record if None.")
    parser.add_argument("--render",
                        default=True,
                        help="Whether to render or not. Defaults to True.")
    parser.add_argument("--game_state_file",
                        default=None,
                        help="File from which to load game state. Defaults to "
                        "None.")
    args = parser.parse_args()

    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        helpers.make_agent_from_string(agent_string, agent_id+1000)
        for agent_id, agent_string in enumerate(args.agents.split(","))
    ]

    env = make(config, agents, game_state_file)
    training_agent = None

    for agent in agents:
        if type(agent) == TensorForceAgent:
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            break

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    # Create a Proximal Policy Optimization agent
    agent = training_agent.initialize(env)

    atexit.register(functools.partial(clean_up_agents, agents))
    wrapped_env = WrappedEnv(env, visualize=args.render)
    runner = Runner(agent=agent, environment=wrapped_env)
    runner.run(episodes=10, max_episode_timesteps=2000)
    print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
          runner.episode_times)

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    main()
