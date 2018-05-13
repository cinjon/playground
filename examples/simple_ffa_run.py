import pommerman
from pommerman import agents


def main():
    # Print all possible environments in the Pommerman registry
    print(pommerman.registry)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFA-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
