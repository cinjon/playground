"""Eval script.

We have a target model ("target") that we are evaluating. The modes considered:
1. "ffa": Eval "target" against three agents in an FFA.
2. "homogenous-team": Eval "target" on a team with itself against two other
 agents that are also teamed together. Those two other agents are given by
 "opp1", "opp2". If "opp2" is None or empty str, then we duplicate "opp1".
3. "heterogenous-team": In this scenario, there is a second "target2". Aside
 from that, it is the same as #2 above.

In all circumstances, we run 100 battles and record the results afterward
in terms of Win/Loss/Tie as well as mean/std of numbers of steps in each kind.

Examples: 

python eval.py --eval-targets ppo::/path/to/model.pt --num-battles-eval 100
 --eval-opponents simple::null,simple::null,simple::null

TODO: Include an example using ssh.
"""
from collections import defaultdict
import os
import random

from pommerman import configs
from pommerman import agents
from pommerman.cli import run_battle
import numpy as np
import torch
from torch.autograd import Variable

from arguments import get_args
import dagger_agent
import envs as env_helpers
import networks
import ppo_agent
import utils


def build_agents(mode, targets, opponents, obs_shape, action_space, args):
    def _get_info(inp):
        model_type, model_path = inp.split('::')
        if all([model_path, model_path != 'null',
                not os.path.exists(model_path)]):
            print("Retrieving model %s from %s..." % \
                  (model_path, args.ssh_address))
            model_path = utils.scp_model_from_ssh(
                model_path,
                args.ssh_address,
                args.ssh_password,
                args.ssh_save_model_local)

        if model_type == 'ppo':
            return ppo_agent.PPOAgent, model_path
        elif model_type == 'dagger':
            return dagger_agent.DaggerAgent, model_path
        elif model_type == 'simple':
            return agents.SimpleAgent, None

    def _build(info):
        agent_type, path = info
        if path:
            model_str = args.model_str
            actor_critic = lambda state, board_size, num_channels: networks.get_actor_critic(model_str)(
                state, obs_shape[0], action_space, board_size, num_channels)

            print("Loading path %s as agent." % path)
            if args.cuda:
                loaded_model = torch.load(apath, map_location=lambda storage, loc: storage.cuda(args.cuda_device))
            else:
                loaded_model = torch.load(path, map_location=lambda storage, loc: storage)
            model_state_dict = loaded_model['state_dict']
            args_state_dict = loaded_model['args']
            model = actor_critic(model_state_dict, args_state_dict['board_size'],
                                 args_state_dict['num_channels'])
            agent = agent_type(model, num_stack=args_state_dict['num_stack'])
            if args.cuda:
                agent.cuda()
            return agent
        else:
            return agent_type()

    targets = targets.split(',')
    opponents = opponents.split(',')

    if mode == 'ffa':
        assert(len(targets) == 1), "Exactly 1 target for ffa."
        assert(len(opponents) == 3), "Exactly 3 opponents for ffa."
    elif mode == 'homogenous_team':
        assert(len(targets) == 1), "Exactly 1 target for homogenous-team."
        assert(len(opponents) in [1, 2]), \
            "Exactly 1 or 2 opponents for homogenous-team."
        targets += targets
        if len(opponents) == 1:
            opponents += opponents
    elif mode == 'heterogenous_team':
        assert(len(targets) == 1), "Exactly 2 targets for heterogenous-team."
        assert(len(opponents) in [1, 2]), \
            "Exactly 1 or 2 opponents for heterogenous-team."
        if len(opponents) == 1:
            opponents += opponents
    else:
        raise ValueError

    targets = [_build(_get_info(agent)) for agent in targets]
    opponents = [_build(_get_info(agent)) for agent in opponents]
    return targets, opponents


def eval():
    args = get_args()
    if args.cuda:
        os.environ['OMP_NUM_THREADS'] = '1'
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    mode = args.eval_mode
    obs_shape, action_space = env_helpers.get_env_shapes(args.config,
                                                         args.num_stack)
    targets = args.eval_targets
    opponents = args.eval_opponents
    targets, opponents = build_agents(mode, targets, opponents, obs_shape,
                                      action_space, args)

    # Run the model with run_battle.
    if mode == 'ffa':
        print('Starting FFA Battles.')
        wins = defaultdict(int)
        deads = defaultdict(list)
        ranks = defaultdict(list)
        for position in range(4):
            print("Running Battle Position %d..." % position)
            num_times = args.num_battles_eval // 4
            agents = [o for o in opponents]
            agents.insert(position, targets[0])
            infos = run_battle.run(args, num_times=num_times, seed=args.seed,
                                   agents=agents, training_agents=[position])
            for info in infos:
                if 'winners' in info and info['winners'] == [position]:
                    wins[position] += 1
                if 'step_info' in info and position in info['step_info']:
                    agent_step_info = info['step_info'][position]
                    for kv in agent_step_info:
                        k, v = kv.split(':')
                        if k == 'dead':
                            deads[position].append(int(v))
                        elif k == 'rank':
                            ranks[position].append(int(v))

            print("Position %d Result: " % position)
            print("Wins: ", wins)
            print("Dead: ", deads)
            print("Ranks: ", ranks)
            print("\n")
    elif mode == 'homogenous_team':
        print('Starting Homogenous Team Battles.')
        for position in range(2):
            training_agents = [position, position+2]
            print("Running Battle Position %d..." % position)
            num_times = args.num_battles_eval // 2
            agents = [o for o in opponents]
            agents.insert(position, targets[0])
            agents.insert(position+2, targets[1])
            infos = run_battle.run(
                args, num_times=num_times, seed=args.seed, agents=agents,
                training_agents=training_agents)
            print(infos)
    elif mode == 'heterogenous_team':
        print('Starting Heterogenous Team Battles.')
        for position in range(2):
            print("Running Battle Position %d..." % position)
            training_agents = [position, position+2]
            num_times = args.num_battles_eval // 2
            agents = [o for o in opponents]
            agents.insert(position, targets[0])
            agents.insert(position+2, targets[1])
            infos = run_battle.run(
                args, num_times=num_times, seed=args.seed, agents=agents,
                training_agents=training_agents)
            print(infos)


if __name__ == "__main__":
    eval()
