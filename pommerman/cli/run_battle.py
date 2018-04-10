"""Run a battle among agents.

Call this with a config, a game, and a list of agents. The script will start separate threads to operate the agents
and then report back the result.

An example with all four test agents running ffa:
python run_battle.py --agents=test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent --config=ffa_v0

An example with one player, two random agents, and one test agent:
python run_battle.py --agents=player::arrows,test::agents.SimpleAgent,random::null,random::null --config=ffa_v0

An example with a docker agent:
python run_battle.py --agents=player::arrows,docker::pommerman/test-agent,random::null,random::null --config=ffa_v0
"""
import atexit
import os
import random
import time

import argparse
import numpy as np

from .. import helpers
from .. import make


def run(args, num_times=1, seed=None):
    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        helpers.make_agent_from_string(agent_string, agent_id+1000)
        for agent_id, agent_string in enumerate(args.agents.split(','))
    ]

    env = make(config, agents, game_state_file)

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    def _run(seed, record_pngs_dir=None, record_json_dir=None):
        env.seed(seed)
        print("Starting the Game.")
        obs = env.reset()
        steps = 0
        done = False
        while not done:
            steps += 1
            if args.render:
                env.render(record_pngs_dir=args.record_pngs_dir,
                           record_json_dir=args.record_json_dir)
            actions = env.act(obs)
            obs, reward, done, info = env.step(actions)

        for agent in agents:
            agent.episode_end(reward[agent.agent_id])
        
        print("Final Result: ", info)
        if args.render:
            time.sleep(5)
            env.render(record_pngs_dir=args.record_pngs_dir,
                       record_json_dir=args.record_json_dir, close=True)
        return info

    infos = []
    times = []
    for i in range(num_times):
        start = time.time()
        if seed is None:
            seed = random.randint(0, 1e6)
        np.random.seed(seed)
        random.seed(seed)

        record_pngs_dir_ = record_pngs_dir + '/%d' % (i+1) \
                           if record_pngs_dir else None
        record_json_dir_ = record_json_dir + '/%d' % (i+1) \
                           if record_json_dir else None
        infos.append(_run(seed, record_pngs_dir_, record_json_dir_))

        times.append(time.time() - start)
        print("Game Time: ", times[-1])

    atexit.register(env.close)
    return infos


def main():
    simple_agent = 'test::agents.SimpleAgent'
    player_agent = 'player::arrows'
    docker_agent = 'docker::pommerman/simple-agent'
    
    parser = argparse.ArgumentParser(description='Playground Flags.')
    parser.add_argument('--game',
                        default='pommerman',
                        help='Game to choose.')
    parser.add_argument('--config',
                        default='PommeFFA-v0',
                        help='Configuration to execute. See env_ids in '
                        'configs.py for options.')
    parser.add_argument('--agents',
                        default=','.join([simple_agent]*4),
                        # default=','.join([player_agent] + [simple_agent]*3]),
                        # default=','.join([docker_agent] + [simple_agent]*3]),
                        help='Comma delineated list of agent types and docker '
                        'locations to run the agents.')
    parser.add_argument('--agent_env_vars',
                        help='Comma delineated list of agent environment vars '
                        'to pass to Docker. This is only for the Docker Agent.'
                        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
                        'would send two arguments to Docker Agent 0 and one '
                        'to Docker Agent 3.',
                        default="")
    parser.add_argument('--record_pngs_dir',
                        default=None,
                        help='Directory to record the PNGs of the game. '
                        "Doesn't record if None.")
    parser.add_argument('--record_json_dir',
                        default=None,
                        help='Directory to record the JSON representations of '
                        "the game. Doesn't record if None.")
    parser.add_argument('--render',
                        default=True,
                        help="Whether to render or not. Defaults to True.")
    parser.add_argument('--game_state_file',
                        default=None,
                        help="File from which to load game state.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
