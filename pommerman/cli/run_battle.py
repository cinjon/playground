"""Run a battle among agents.
Call this with a config, a game, and a list of agents. The script will start separate threads to operate the agents
and then report back the result.

An example with all four test agents running ffa:
python run_battle.py --agents=test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent --config=PommeFFACompetition-v0

An example with one player, two random agents, and one test agent:
python run_battle.py --agents=player::arrows,test::agents.SimpleAgent,random::null,random::null --config=PommeFFACompetition-v0

An example with a docker agent:
python run_battle.py --agents=player::arrows,docker::pommerman/test-agent,random::null,random::null --config=PommeFFACompetition-v0
"""
import atexit
from collections import defaultdict
import os
import random
import time
from datetime import datetime

import argparse
import numpy as np

from .. import helpers
from .. import make

import pommerman
from pommerman import utility


def run(args, num_times=None, seed=None, agents=None, training_agent_ids=[],
        acting_agent_ids=None, training_agents=None, curriculum=False):
    """Run the game a number of times.

    Args:
      args: The arguments we are passing through from CLI.
      num_times: The number of times to run the battle.
      seed: The random seed to use for the battles.
      agents: What agents to use. If not, we will make them from the args.
      training_agent_ids: Which ids are the training_agents.
      acting_agent_ids: Which, if any agents, use an act function.

    Returns:
      infos: The list of information dicts returned from these games.
    """
    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    game_state_file = args.game_state_file
    num_times = num_times
    if not num_times:
        try:
            num_times = int(args.num_times)
        except Exception as e:
            num_times = 1

    render_mode = args.render_mode
    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.

    if not curriculum:
        agents = agents or [
            helpers.make_agent_from_string(agent_string, agent_id+1000)
            for agent_id, agent_string in enumerate(args.agents.split(','))
        ]
    else:
        training_agent_ids = [random.randint(0, 3)]
        agents = [pommerman.agents.SimpleAgent() for _ in range(3)]
        agents.insert(training_agent_ids[0], training_agents)

    env = make(config, agents, game_state_file, render_mode=render_mode)
    env.set_training_agents(training_agent_ids)
    env.enable_selfbombing()
    env.rank = 0

    if seed is None:
        seed = random.randint(0, 1e6)
    seed = random.randint(0, 1e6)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(args, 'state_directory'):
        state_directory = args.state_directory
        state_directory_distribution = args.state_directory_distribution
    else:
        state_directory = None
        state_directory_distribution = None
    env.set_state_directory(state_directory,
                            state_directory_distribution)

    if record_pngs_dir and not os.path.exists(record_pngs_dir):
        os.makedirs(record_pngs_dir)
    if record_json_dir and not os.path.exists(record_json_dir):
        os.makedirs(record_json_dir)

    def _run(seed, acting_agent_ids, record_pngs_dir=None, record_json_dir=None):
        obs = env.reset()
        steps = 0
        done = False
        acting_agent_ids = acting_agent_ids or []
        while not done:
            steps += 1
            if args.render:
                env.render(record_pngs_dir=record_pngs_dir,
                           record_json_dir=record_json_dir,
                           mode=render_mode)
            actions = env.act(obs, acting_agent_ids=acting_agent_ids)

            for agent_id in acting_agent_ids:
                agent_obs = obs[agent_id]
                action = agents[agent_id].act(agent_obs, env.action_space)
                actions.insert(agent_id, action)
                
            obs, reward, done, info = env.step(actions)
            if type(done) == list:
                done = all(done)

        for agent in agents:
            agent.episode_end(reward[agent.agent_id])

        print("Final Result: ", info)
        if args.render:
            env.render(record_pngs_dir=args.record_pngs_dir,
                       record_json_dir=args.record_json_dir,
                       mode=args.render_mode)
            time.sleep(5)
            env.render(close=True)

        if record_json_dir:
            finished_at = datetime.now().isoformat()
            _agents = args.agents.split(',')
            utility.join_json_state(record_json_dir, _agents, finished_at, config)

        return info

    infos = []
    times = []
    for i in range(num_times):
        start = time.time()
        record_pngs_dir_ = record_pngs_dir + '/%d' % (i+1) \
                           if record_pngs_dir else None
        record_json_dir_ = record_json_dir + '/%d' % (i+1) \
                           if record_json_dir else None
        with utility.Timer() as t:
            info = _run(seed, training_agent_ids, record_pngs_dir_, record_json_dir_)
        infos.append(info)
        times.append(t.interval)
        print("Game %d final result (%.4f): " % (i, times[-1]), infos[-1])

    atexit.register(env.close)
    return infos, training_agent_ids


def main():
    simple_agent = 'test::agents.ComplexAgent'
    player_agent = 'player::arrows'
    docker_agent = 'docker::pommerman/simple-agent'
    parser = argparse.ArgumentParser(description='Playground Flags.')
    parser.add_argument('--config',
                        default='PommeFFACompetition-v0',
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
    parser.add_argument("--render",
                        default=False,
                        action='store_true',
                        help="Whether to render or not. Defaults to False.")
    parser.add_argument('--render_mode',
                        default='human',
                        help="What mode to render. Options are human, "
                        "rgb_pixel, and rgb_array.")
    parser.add_argument('--game_state_file',
                        default=None,
                        help="File from which to load game state.")
    parser.add_argument('--num-times',
                        default=1,
                        help="The number of battles to run")
    parser.add_argument('--state-directory', type=str, default='',
                        help='a game state directory from which to load.')
    parser.add_argument('--state-directory-distribution', type=str,
                        default='', help='a distribution to load the '
                        'states in the directory. uniform will choose on'
                        'randomly. for the others, see envs.py.')
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
