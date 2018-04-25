"""
Train script for qmix learning.

The number of samples used for an epoch is:
horizon * num_workers = num_steps * num_processes where num_steps is the number
of steps in a rollout (horizon) and num_processes is the number of parallel
processes/workers collecting data.

Example:

python train.py --how-train qmix --num-processes 10 --run-name test \
 --num-steps 50 --log-interval 5
"""

from collections import defaultdict
import os
import time

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.nn import MSELoss
import random

from storage import EpisodeBuffer
from arguments import get_args
import envs as env_helpers
import qmix_agent
import utils
import pommerman


def train():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'

    ## CLI params, hard-coded for now
    args.how_train = 'qmix'
    args.config = 'PommeTeam-v0'
    args.num_agents = 2
    args.run_name = 'qmix-train'
    args.model_str = 'QMIXNet'

    if args.cuda:
        torch.cuda.empty_cache()
        # torch.cuda.set_device(args.cuda_device)

    assert args.run_name
    print("\n###############")
    print("args ", args)
    print("##############\n")

    how_train, config, num_agents, num_stack, num_steps, num_processes, \
        num_epochs, reward_sharing = utils.get_train_vars(args)

    obs_shape, action_space = env_helpers.get_env_shapes(config, num_stack)
    global_obs_shape = (4, obs_shape[0] // 2, *obs_shape[1:]) # @TODO a better way?
    num_training_per_episode = utils.validate_how_train(how_train, num_agents)

    training_agents = utils.load_agents(
        obs_shape, action_space, num_training_per_episode, args,
        qmix_agent.QMIXMetaAgent, network_type='qmix')
    envs = env_helpers.make_envs(config, how_train, args.seed,
                                 args.game_state_file, training_agents,
                                 num_stack, num_processes, args.render)

    #####
    # Logging helpers.
    suffix = "{}.{}.{}.{}.nc{}.lr{}.mb{}.ns{}.seed{}".format(
        args.run_name, how_train, config, args.model_str, args.num_channels,
        args.lr, args.num_mini_batch, args.num_steps, args.seed)
    log_dir = os.path.join(args.log_dir, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)
    count_stats = defaultdict(int)
    array_stats = defaultdict(list)
    episode_rewards = torch.zeros([num_training_per_episode,
                                   num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode,
                                 num_processes, 1])

    episode_buffer = EpisodeBuffer(size=5000)

    # TODO: When we implement heterogenous, change this to be per agent.
    start_epoch = training_agents[0].num_epoch
    total_steps = training_agents[0].total_steps
    num_episodes = training_agents[0].num_episodes

    running_num_episodes = 0
    cumulative_reward = 0
    terminal_reward = 0
    success_rate = 0
    final_action_losses = [[] for agent in range(len(training_agents))]
    final_value_losses =  [[] for agent in range(len(training_agents))]


    def update_stats(info):
        # TODO: Change this stats computation when we use heterogenous.
        for i in info:
            for lst in i.get('step_info', {}).values():
                for l in lst:
                    if l.startswith('dead') or l.startswith('rank'):
                        key, count = l.split(':')
                        array_stats[key].append(int(count))
                    else:
                        count_stats[l] += 1
                        if 'bomb' in l:
                            count_stats['bomb'] += 1

    current_obs = envs.reset()
    current_global_obs = envs.get_global_obs()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()

    start = time.time()

    eps = args.eps_max
    eps_steps = 0
    gradient_steps = 0
    for num_epoch in range(start_epoch, num_epochs):
        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval):
            utils.save_agents("qmix-", num_epoch, training_agents, total_steps,
                              num_episodes, args, suffix)

        for agent in training_agents:
            agent.set_eval()

        ##
        # Each history is Python list is of length num_processes. This is a list because not all
        # episodes are of the same length and we don't want information of an episode
        # after it has ended. Each history item has the first dimension as time step, second
        # dimension as the number of training agents and then appropriate shape for the item
        #
        history = [
            [
                torch.zeros(0, *global_obs_shape), # 0: current global state
                torch.zeros(0, num_training_per_episode, *obs_shape), # 1: current agent(s) state
                torch.zeros(0, num_training_per_episode).long(), # 2: action
                torch.zeros(0, num_training_per_episode), # 3: reward
                torch.zeros(0, *global_obs_shape),  # 4: next global state
                torch.zeros(0, num_training_per_episode, *obs_shape),  # 5: next agent(s) state
                torch.zeros(0, num_training_per_episode).long(), # 6: done
            ]
            for _ in range(num_processes)
        ]

        # Run all episodes until the end

        # while True:
        for i in range(10):
            eps = args.eps_max + (args.eps_min - args.eps_max) / args.eps_max_steps * eps_steps

            # Ignore critic values during trajectory generation
            _, actions = training_agents[0].act(
                torch.FloatTensor(current_global_obs),
                torch.FloatTensor(current_obs), eps=eps)

            training_agent_actions = actions.data.numpy().tolist()
            obs, reward, done, info = envs.step(training_agent_actions)
            global_obs = envs.get_global_obs()
            reward = reward.astype(np.float)

            update_stats(info)

            if args.render:
                envs.render()

            global_state_tensor = torch.FloatTensor(current_global_obs).unsqueeze(1)
            state_tensor = torch.FloatTensor(current_obs).unsqueeze(1)
            action_tensor = torch.LongTensor(training_agent_actions).unsqueeze(1)
            reward_tensor = torch.from_numpy(reward).float().unsqueeze(1)
            next_global_state_tensor = torch.FloatTensor(global_obs).unsqueeze(1)
            next_state_tensor = torch.from_numpy(obs).float().unsqueeze(1)
            done_tensor = torch.from_numpy(done.astype(np.int)).long().unsqueeze(1)

            for i in range(num_processes):
                # Ignore history if done
                # if info[i]['result'] == pommerman.constants.Result.Incomplete:
                #     continue

                history[i][0] = torch.cat([history[i][0], global_state_tensor[i]], dim=0)
                history[i][1] = torch.cat([history[i][1], state_tensor[i]], dim=0)
                history[i][2] = torch.cat([history[i][2], action_tensor[i]], dim=0)
                history[i][3] = torch.cat([history[i][3], reward_tensor[i]], dim=0)
                history[i][4] = torch.cat([history[i][4], next_global_state_tensor[i]], dim=0)
                history[i][5] = torch.cat([history[i][5], next_state_tensor[i]], dim=0)
                history[i][6] = torch.cat([history[i][6], done_tensor[i]], dim=0)

                total_steps += 1

            # Update for next step
            eps_steps = min(eps_steps + 1, args.eps_max_steps)
            current_obs = obs
            current_global_obs = global_obs

            # Break loop if all parallel actors have finished their episodes
            all_episodes_done = True
            for episode_info in info:
                if episode_info['result'] == pommerman.constants.Result.Incomplete:
                    all_episodes_done = False
                    break

            if all_episodes_done:
                break

        episode_buffer.extend(history)

        def compute_q_loss(global_state, state, action, reward, next_global_state, next_state, done):
            current_q_values, _ = training_agents[0].act(global_state, state)
            max_next_q_values, _ = training_agents[0].target_act(next_global_state, next_state)
            max_next_q_values = max_next_q_values.max(1)[0].unsqueeze(1)
            # @TODO: Reward is per agent, how do we handle it?
            expected_q_values = reward + args.gamma * max_next_q_values

            loss = MSELoss()(current_q_values, expected_q_values.detach())
            loss.backward()

        if len(episode_buffer) >= args.episode_batch:
            for episode in episode_buffer.sample(args.episode_batch):
                compute_q_loss(*episode)
            training_agents[0].step()
            gradient_steps += 1
            if gradient_steps % args.target_update_steps == 0:
                training_agents[0].update_target()

        if running_num_episodes > args.log_interval:
            end = time.time()
            num_steps_sec = (end - start)
            num_episodes += running_num_episodes

            steps_per_sec = 1.0 * total_steps / (end - start)
            epochs_per_sec = 1.0 * args.log_interval / (end - start)
            episodes_per_sec =  1.0 * num_episodes / (end - start)

            mean_dist_entropy = np.mean([
                dist_entropy for dist_entropy in final_dist_entropies])
            std_dist_entropy = np.std([
                dist_entropy for dist_entropy in final_dist_entropies])

            mean_value_loss = np.mean([
                value_loss for value_loss in final_value_losses])
            std_value_loss = np.std([
                value_loss for value_loss in final_value_losses])

            mean_action_loss = np.mean([
                action_loss for action_loss in final_action_losses])
            std_action_loss = np.std([
                action_loss for action_loss in final_action_losses])

            utils.log_to_console(num_epoch, num_episodes, total_steps,
                                 steps_per_sec, epochs_per_sec, final_rewards,
                                 mean_dist_entropy, mean_value_loss, mean_action_loss,
                                 cumulative_reward, terminal_reward, success_rate,
                                 running_num_episodes)
            utils.log_to_tensorboard(writer, num_epoch, num_episodes,
                                     total_steps, steps_per_sec, episodes_per_sec,
                                     final_rewards,
                                     mean_dist_entropy, mean_value_loss,
                                     mean_action_loss, std_dist_entropy,
                                     std_value_loss, std_action_loss,
                                     count_stats, array_stats,
                                     cumulative_reward, terminal_reward, success_rate,
                                     running_num_episodes)

            # Reset stats so that plots are per the last log_interval.
            final_action_losses = [[] for agent in range(len(training_agents))]
            final_value_losses =  [[] for agent in range(len(training_agents))]
            final_dist_entropies = [[] for agent in \
                                    range(len(training_agents))]
            count_stats = defaultdict(int)
            array_stats = defaultdict(list)
            final_rewards = torch.zeros([num_training_per_episode,
                                         num_processes, 1])
            running_num_episodes = 0
            cumulative_reward = 0
            terminal_reward = 0
            success_rate = 0

    writer.close()


if __name__ == "__main__":
    train()
