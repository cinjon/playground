"""Eval script.

We have a target model ("target") that we are evaluating. The modes considered:

1. "ffa": Eval "target" against three agents in an FFA.
2. "homogenous": Eval "target" on a team with itself against two other
 agents that are also teamed together. Those two other agents are given by
 "opp1", "opp2". If "opp2" is None or empty str, then we duplicate "opp1".
3. "heterogenous": In this scenario, there is a second "target2". Aside
 from that, it is the same as #2 above.

In all circumstances, we run 100 battles and record the results afterward
in terms of Win/Loss/Tie as well as mean/std of numbers of steps in each kind.

Examples:

On Cpu:
python eval.py --eval-targets ppo::/path/to/model.pt --num-battles-eval 100
 --eval-opponents simple::null,simple::null,simple::null --config PommeFFAFast-v3

On Gpu:
CUDA_VISIBLE_DEVICES=0 python eval.py --eval-targets ppo::/path/to/model.py \
 --num-battles-eval 200 --config PommeFFAFast-v3 --cuda-device 0

TODO: Include an example using ssh.
"""
import atexit
from collections import defaultdict
import os
import random
import time

import pommerman
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


def _get_info(inp, args):
    model_type, model_path = inp.split('::')
    if all([model_path, model_path != 'null',
            not os.path.exists(model_path)]):
        print("Retrieving model %s from %s..." % \
              (model_path, ssh_address))
        model_path = utils.scp_model_from_ssh(model_path,
                                              args.ssh_address,
                                              args.ssh_password,
                                              args.ssh_save_model_local)
    if model_type == 'ppo':
        return ppo_agent.PPOAgent, model_path
    elif model_type == 'dagger':
        return dagger_agent.DaggerAgent, model_path
    elif model_type == 'simple':
        return pommerman.agents.SimpleAgent, None


def _build(info, obs_shape, action_space, cuda, cuda_device):
    agent_type, path = info
    if path:
        model_str = args.model_str
        actor_critic = lambda state, board_size, num_channels: \
            networks.get_actor_critic(model_str)(state, obs_shape[0],
                                                 action_space, board_size,
                                                 num_channels)

        print("Loading path %s as agent." % path)
        if cuda:
            loaded_model = torch.load(path, map_location=lambda storage,
                                      loc: storage.cuda(cuda_device))
        else:
            loaded_model = torch.load(path, map_location=lambda storage,
                                      loc: storage)
        model_state_dict = loaded_model['state_dict']
        args_state_dict = loaded_model['args']
        model = actor_critic(model_state_dict,
                             args_state_dict['board_size'],
                             args_state_dict['num_channels'])
        agent = agent_type(model, num_stack=args_state_dict['num_stack'],
                           cuda=cuda)
        if cuda:
            agent.cuda()
        return agent
    else:
        return agent_type()


def build_agents(mode, targets, opponents, obs_shape, action_space, args):
    targets = targets.split(',')
    opponents = opponents.split(',')

    if mode == 'ffa':
        assert(len(targets) == 1), "Exactly 1 target for ffa."
        assert(len(opponents) == 3), "Exactly 3 opponents for ffa."
    elif mode == 'homogenous':
        assert(len(targets) == 1), "Exactly 1 target for homogenous."
        assert(len(opponents) in [1, 2]), \
            "Exactly 1 or 2 opponents for homogenous."
        targets += targets
        if len(opponents) == 1:
            opponents += opponents
    elif mode == 'heterogenous':
        assert(len(targets) == 1), "Exactly 2 targets for heterogenous."
        assert(len(opponents) in [1, 2]), \
            "Exactly 1 or 2 opponents for heterogenous."
        if len(opponents) == 1:
            opponents += opponents
    else:
        raise ValueError

    targets = [_build(_get_info(agent, args), obs_shape, action_space,
                      args.cuda, args.cuda_device) for agent in targets]
    opponents = [_build(_get_info(agent, args), obs_shape, action_space,
                        args.cuda, args.cuda_device) for agent in opponents]
    return targets, opponents


def eval(args=None, targets=None, opponents=None):
    args = args or get_args()
    if args.cuda:
        os.environ['OMP_NUM_THREADS'] = '1'
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    mode = args.eval_mode
    obs_shape, action_space = env_helpers.get_env_shapes(args.config,
                                                         args.num_stack)

    if not targets and not opponents:
        targets = args.eval_targets
        opponents = args.eval_opponents
        targets, opponents = build_agents(mode, targets, opponents, obs_shape,
                                          action_space, args)

    # Run the model with run_battle.
    if mode == 'ffa':
        print('Starting FFA Battles.')
        ties = defaultdict(int)
        wins = defaultdict(int)
        deads = defaultdict(list)
        ranks = defaultdict(list)
        for position in range(4):
            # TODO: Change this to use the parallel run_battle below.
            print("Running Battle Position %d..." % position)
            num_times = args.num_battles_eval // 4
            agents = [o for o in opponents]
            agents.insert(position, targets[0])
            training_agent_ids = []
            if not type(targets[0]) == pommerman.agents.SimpleAgent:
                training_agent_ids.append(position)
            infos = run_battle.run(
                args, num_times=num_times, seed=args.seed, agents=agents,
                training_agent_ids=training_agent_ids)
            for info in infos:
                if all(['result' in info,
                        info['result'] == pommerman.constants.Result.Tie,
                        not info.get('step_info')]):
                    ties[position] += 1
                if 'winners' in info and info['winners'] == [position]:
                    wins[position] += 1
                if 'step_info' in info and position in info['step_info']:
                    agent_step_info = info['step_info'][position]
                    for kv in agent_step_info:
                        if ':' not in kv:
                            continue
                        k, v = kv.split(':')
                        if k == 'dead':
                            deads[position].append(int(v))
                        elif k == 'rank':
                            ranks[position].append(int(v))

        print("Wins: ", wins)
        print("Dead: ", deads)
        print("Ranks: ", ranks)
        print("Ties: ", ties)
        print("\n")
        return wins, deads, ties, ranks
    elif mode == 'homogenous':
        print('Starting Homogenous Battles.') 
        ties = defaultdict(int)
        wins = defaultdict(int)
        one_dead = defaultdict(int)
        for position in range(2):
            training_agent_ids = [position, position+2]
            print("Running Battle Position %d..." % position)
            num_times = args.num_battles_eval // 2
            agents = [o for o in opponents]
            agents.insert(position, targets[0])
            agents.insert(position+2, targets[1])
            infos = run_battles(args, num_times, agents, action_space,
                                training_agent_ids)
            for info in infos:
                if info['result'] == pommerman.constants.Result.Tie:
                    ties[position] += 1
                else:
                    winners = info['winners']

                    is_win = False
                    if position in winners:
                        wins[position] += 1

                    # Count the number of times that one died and not other.
                    if is_win:
                        for id_ in [position, position+2]:
                            if id_ not in info['alive']:
                                one_dead[position] += 1

        print("Wins: ", wins)
        print("One Dead: ", one_dead)
        print("Ties: ", ties)
        print("\n")
        return wins, one_dead, ties
    elif mode == 'heterogenous':
        print('Starting Heterogenous Team Battles.')
        for position in range(2):
            print("Running Battle Position %d..." % position)
            training_agent_ids = [position, position+2]
            num_times = args.num_battles_eval // 2
            agents = [o for o in opponents]
            agents.insert(position, targets[0])
            agents.insert(position+2, targets[1])
            infos = run_battle.run(
                args, num_times=num_times, seed=args.seed, agents=agents,
                training_agent_ids=training_agent_ids)
            print(infos)


def run_battles(args, num_times, agents, action_space, training_agent_ids):
    """Run the game a number of times.

    We assume here that all of the agents are using their own act function.
    Another nice assumption is that the agents are not changing their position
    across the environments. We can make this assumption because of how this
    function is being called from above.

    Args:
      args: The arguments we are passing through from CLI.
      num_times: The number of times to run the battle.
      agents: What agents to use. If not, we will make them from the args.
      action_space: The action space for an environment. Likely Discrete(6).
      training_agent_ids: Which ids are the training_agents.

    Returns:
      infos: The list of information dicts returned from these games.
    """
    config = args.config
    seed = args.seed
    num_processes = args.num_processes

    if seed is None:
        seed = random.randint(0, 1e6)
    np.random.seed(seed)
    random.seed(seed)

    envs = env_helpers.make_eval_envs(
        config, args.how_train, seed, agents, training_agent_ids,
        args.num_stack, num_processes)
    atexit.register(envs.close)

    infos = []
    rewards = []
    times = []
    st = time.time()
    obs = envs.reset()
    while len(infos) != num_times:
        actions = [[None]*4 for _ in range(num_processes)]
        for num_agent, agent in enumerate(agents):
            agent_obs = [o[num_agent] for o in obs]
            agent_actions = agent.act(agent_obs, action_space)
            for num_process in range(num_processes):
                actions[num_process][num_agent] = agent_actions[num_process]

        obs, reward, done, info = envs.step(actions)
        for num, done_ in enumerate(done):
            if done_.all():
                print("FINISHE! ", info[num], len(infos))
                infos.append(info[num])
                rewards.append(reward[num])
    end = time.time()
    print("TIMEs: %.3f, %.3f" % (end - st, (end - st)/num_times))
    print(infos)
    print(times)

    envs.close()
    return infos


if __name__ == "__main__":
    eval()
