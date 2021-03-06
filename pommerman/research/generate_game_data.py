"""Generate data script.
For Saving Actions along with States:
Grid:
python generate_game_data.py --agents=astar::null --config=GridWalls-v4 \
--num-episodes 4 --num-processes 12 --num-stack 1 --how-train astar --seed 1 \
--record-json-dir /home/roberta/playground/trajectories/grid/4maps
Pomme:
python generate_game_data.py --agents=complex::null,complex::null,complex::null,complex::null \
 --config=PommeFFACompetition-v0 --num-episodes 4 --num-processes 12 \
 --how-train simple --seed 1 --record-json-dir /home/roberta/playground/trajectories/pomme/4maps
"""
import json
import os
import random
import shutil
import time

import numpy as np
import pommerman

from arguments import get_args
import envs as env_helpers


def _get_info(inp, args):
    model_type, model_path = inp.split('::')
    if all([model_path, model_path != 'null',
            not os.path.exists(model_path)]):
        print("Retrieving model %s from %s..." % \
              (model_path, args.ssh_address))
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
    elif model_type == 'complex':
        return pommerman.agents.ComplexAgent, None
    elif model_type == 'astar':
        return pommerman.agents.AstarAgent, None


def _build(info, obs_shape, action_space, cuda, cuda_device, model_str,
           board_size, character):
    agent_type, path = info
    if path:
        actor_critic = lambda state, num_channels: \
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
                             args_state_dict['num_channels'])
        agent = agent_type(model, num_stack=args_state_dict['num_stack'],
                           cuda=cuda, character=character)
        if cuda:
            agent.cuda()
        return agent
    else:
        return agent_type(character=character, prob_optim=args.prob_optim)


def build_agents(agents, obs_shape, board_size, args, action_space, character):
    agents = agents.split(',')

    agents = [_build(_get_info(agent, args), obs_shape, action_space,
                     args.cuda, args.cuda_device, args.model_str, board_size,
                     character)
               for agent in agents]
    acting_agent_ids = [num for num, agent in enumerate(agents) \
                        if not agent.is_simple_agent]

    return agents, acting_agent_ids


def generate(args, agents, action_space, acting_agent_ids):
    """Generate a number of episodes of training data.
    We want to save a number of episodes by their game states in order to then
    learn from these starting at the back end. After saving a game, we go back
    and record who won so that we can easily load agents to play as that agent.
    Args:
      args: The arguments we are passing through from CLI.
      num_episodes: The number of episodes to save.
      agents: What agents to use. If not, we will make them from the args.
      action_space: The action space for an environment. Likely Discrete(6).
      acting_agent_ids: Which ids are the acting agents.
    """
    config = args.config
    seed = args.seed
    num_processes = args.num_processes
    record_json_dir = args.record_json_dir
    actions_file_name = os.path.basename(os.path.normpath(record_json_dir)) + "_actions"
    record_actions_json_dir = os.path.join(record_json_dir, "../", actions_file_name)
    # record_actions_json_dir = os.path.join(record_json_dir, "actions")
    num_episodes = args.num_episodes
    init_num_episodes = args.num_episodes

    if record_json_dir and not os.path.exists(record_json_dir):
        os.makedirs(record_json_dir)
    if record_actions_json_dir and not os.path.exists(record_actions_json_dir):
        os.makedirs(record_actions_json_dir)

    if seed is None:
        seed = random.randint(0, 1e6)
    np.random.seed(seed)
    random.seed(seed)

    training_agent_ids = []
    envs = env_helpers.make_eval_envs(
        config, args.how_train, seed, agents, training_agent_ids,
        acting_agent_ids, args.num_stack, num_processes)

    steps = []
    process_dirs = list(range(num_processes))
    st = time.time()
    obs = envs.reset()
    milestones = [int(k*num_episodes/50) for k in range(50)]
    while num_episodes > 0:
        if milestones and num_episodes < milestones[-1]:
            mt = time.time()
            print("\nMilestone %d (%d). Total time %.3f / Avg time %.3f." % (
                50 - len(milestones), init_num_episodes - num_episodes,
                mt - st, 1.0*(mt-st)/(init_num_episodes - num_episodes)
            ))
            print("Mean / Median step count: %d / %d\n." % (
                np.mean(steps), np.median(steps)))
            milestones = milestones[:-1]
        actions = [[None]*len(acting_agent_ids) for _ in range(num_processes)]
        for num_action, acting_agent_id in enumerate(acting_agent_ids):
            agent_obs = [o[num_action] for o in obs]
            agent_actions = agents[acting_agent_id].act(agent_obs, action_space)
            for num_process in range(num_processes):
                actions[num_process][num_action] = agent_actions[num_process]

        for process_dir in process_dirs:
            directory = os.path.join(record_json_dir, '%d' % process_dir)
            if not os.path.exists(directory):
                os.makedirs(directory)

            actions_directory = os.path.join(record_actions_json_dir, '%d' % process_dir)
            if not os.path.exists(actions_directory):
                os.makedirs(actions_directory)

        envs.record_json([os.path.join(record_json_dir, '%d' % process_dir)
                          for process_dir in process_dirs])

        expert_actions = envs.get_actions()
        envs.record_actions_json([os.path.join(record_actions_json_dir, '%d' % process_dir)
                                  for process_dir in process_dirs], expert_actions)

        if args.render:
            envs.render()

        obs, reward, done, info = envs.step(actions)

        if args.render and done[0].all():
            time.sleep(2)

        for num, done_ in enumerate(done):
            if done_.all():
                directory = os.path.join(record_json_dir, '%d' % process_dirs[num])
                info_ = info[num]
                result = info_['result']
                winners = info_.get('winners', [])
                step_count = info_['step_count']
                targ_count = 35 if 'Grid' in args.config else 240
                if 'Grid' not in args.config:
                    is_targ_optimal = True
                else:
                    optimal_num_steps = info_['optimal_num_steps']
                    is_targ_optimal = step_count - optimal_num_steps == args.num_more_optimal
                    print("optimal {},  # steps {},  # optimal steps {},  desired # steps from optimal {} " \
                    .format(is_targ_optimal, step_count, optimal_num_steps, args.num_more_optimal))
                if any([result != pommerman.constants.Result.Win,
                        'Grid' not in args.config and not winners,
                        step_count < targ_count,
                        not is_targ_optimal]):
                    delete_data(directory)
                else:
                    save_endgame_info(directory, info_)
                    steps.append(info_['step_count'])
                    process_dirs[num] = max(process_dirs) + 1
                    num_episodes -= 1
            if num_episodes <= 0:
                break

    for process_dir in process_dirs:
        directory = os.path.join(record_json_dir, '%d' % process_dir)
        if os.path.exists(directory):
            shutil.rmtree(directory)

        actions_directory = os.path.join(record_actions_json_dir, '%d' % process_dir)
        if os.path.exists(actions_directory):
            shutil.rmtree(actions_directory)

    end = time.time()
    print("Generate Times (%d) --> Total: %.3f, Avg: %.3f" % (
        init_num_episodes, end - st, (end - st)/init_num_episodes))

    envs.close()
    print("Directories can be found at %s." % record_json_dir)


def delete_data(directory):
    # print("Removing directory %s..." % directory)
    shutil.rmtree(directory)


def save_endgame_info(directory, info):
    info['result'] = info['result'].value
    print("Completed directory %s with info %s." % (directory, info))
    with open(os.path.join(directory, 'endgame.json'), 'w') as f:
        f.write(json.dumps(info, sort_keys=True, indent=4))


if __name__ == "__main__":

    args = get_args()
    obs_shape, action_space, character, board_size = env_helpers.get_env_info(
        args.config, args.num_stack)
    agents, acting_agent_ids = build_agents(args.agents, obs_shape, board_size,
                                            args, action_space, character)

    print("Generating data for agents %s..." % args.agents)
    generate(args, agents, action_space, acting_agent_ids)
    print("Completed.")
