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
from collections import defaultdict
import os
import random
import time

import argparse
import numpy as np

from .. import helpers
from .. import make
from .. import utility


time_avg = defaultdict(float)
time_max = defaultdict(float)
time_cnt = defaultdict(int)

def run(args, num_times=1, seed=None, agents=None, training_agents=[],
        acting_agents=None):
    """Run the game a number of times.

    Args:
      args: The arguments we are passing through from CLI.
      num_times: The number of times to run the battle.
      seed: The random seed to use for the battles.
      agents: What agents to use. If not, we will make them from the args.
      training_agents: Which ids are the training_agents.
      acting_agents: Which, if any agents, use an act function.

    Returns:
      infos: The list of information dicts returned from these games.
    """
    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    game_state_file = args.game_state_file

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = agents or [
        helpers.make_agent_from_string(agent_string, agent_id+1000)
        for agent_id, agent_string in enumerate(args.agents.split(','))
    ]

    env = make(config, agents, game_state_file)
    env.set_training_agents(training_agents)
    if seed is None:
        seed = random.randint(0, 1e6)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    def _update_times(t, key):
        avg = time_avg[key]
        cnt = time_cnt[key]
        new_avg = (float(avg)*float(cnt) + float(t))
        new_avg /= float(cnt + 1)
        time_cnt[key] = cnt + 1
        time_avg[key] = new_avg
        time_max[key] = max(time_max[key], float(t))

    def _run(seed, acting_agents, record_pngs_dir=None, record_json_dir=None):
        global time_avg
        global time_max
        global time_cnt
        obs = env.reset()
        steps = 0
        done = False
        while not done:
            steps += 1
            if args.render:
                env.render(record_pngs_dir=args.record_pngs_dir,
                           record_json_dir=args.record_json_dir)
            actions = env.act(obs)
            for agent_id in acting_agents:
                with utility.Timer() as t:
                    agent_obs = obs[agent_id]
                    action = agents[agent_id].act(agent_obs, env.action_space)
                _update_times(t.interval,
                              '%s-%d' % (str(type(agents[agent_id])), agent_id))
                actions.insert(agent_id, action)

            obs, reward, done, info = env.step(actions)
            if type(done) == list:
                done = all(done)

            if done:
                print("Agent Run Times:")
                total = 0.0
                for k, v in env.model._time_avg.items():
                    time_avg[k] += v
                for k, v in env.model._time_cnt.items():
                    time_cnt[k] += v
                for k, v in env.model._time_max.items():
                    time_max[k] = max(time_max[k], v)

                for key in sorted(time_avg.keys()):
                    avg = time_avg[key]
                    cnt = time_cnt[key]
                    mx  = time_max[key]
                    print("\t%s: %.4f (%d) --> %.4f, %.4f" % (key, avg, cnt,
                                                              avg * cnt, mx))
                    total += avg * cnt
                print("\tTotal: %.4f" % total)
                env.model.reset_times()
                time_avg = defaultdict(float)
                time_max = defaultdict(float)
                time_cnt = defaultdict(int)

        for agent in agents:
            agent.episode_end(reward[agent.agent_id])

        if args.render:
            time.sleep(5)
            env.render(close=True)
        return info

    infos = []
    times = []
    for i in range(num_times):
        record_pngs_dir_ = record_pngs_dir + '/%d' % (i+1) \
                           if record_pngs_dir else None
        record_json_dir_ = record_json_dir + '/%d' % (i+1) \
                           if record_json_dir else None
        with utility.Timer() as t:
            info = _run(seed, acting_agents, record_pngs_dir_, record_json_dir_)
        infos.append(info)
        times.append(t.interval)
        print("Game %d final result (%.4f): " % (i, times[-1]), infos[-1])

    atexit.register(env.close)
    return infos


def main():
    simple_agent = 'test::agents.SimpleAgent'
    player_agent = 'player::arrows'
    docker_agent = 'docker::pommerman/simple-agent'

    parser = argparse.ArgumentParser(description='Playground Flags.')
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
