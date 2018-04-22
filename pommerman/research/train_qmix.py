"""Train script for ppo learning.

Currently tested for how_train = "simple".
TODO: Get homogenous training working s.t. the reward is properly updated for
an agent in a team game that dies before the end of the game.
TODO: Test that homogenous training works.
TODO: Implement heterogenous training.

The number of samples used for an epoch is:
horizon * num_workers = num_steps * num_processes where num_steps is the number
of steps in a rollout (horizon) and num_processes is the number of parallel
processes/workers collecting data.

Example:

python train.py --how-train simple --num-processes 10 --run-name test \
 --num-steps 50 --log-interval 5
"""

from collections import defaultdict
import os
import time

import numpy as np
from tensorboardX import SummaryWriter
import torch
import random

from storage import EpisodeBuffer
from arguments import get_args
import envs as env_helpers
import ppo_agent
import utils


def train():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.cuda:
        torch.cuda.empty_cache()
        # torch.cuda.set_device(args.cuda_device)

    # assert(args.run_name)
    print("\n###############")
    print("args ", args)
    print("##############\n")

    how_train, config, num_agents, num_stack, num_steps, num_processes, \
        num_epochs = utils.get_train_vars(args)

    obs_shape, action_space = env_helpers.get_env_shapes(config, num_stack)
    num_training_per_episode = utils.validate_how_train(how_train, num_agents)

    training_agents = utils.load_agents(
        obs_shape, action_space, num_training_per_episode, args,
        ppo_agent.PPOAgent)
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
    def update_final_rewards(final_rewards, masks, episode_rewards):
        return final_rewards

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


    #####
    # NOTE: These four funcs have side effects where they set variable values.
    def update_current_obs(obs):
        return torch.from_numpy(obs).float().transpose(0,1)

    def update_actor_critic_results(result):
        value, action, action_log_prob, states = result
        value_agents.append(value)
        action_agents.append(action)
        action_log_prob_agents.append(action_log_prob)
        states_agents.append(states)
        return action.data.squeeze(1).cpu().numpy()

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

    current_obs = update_current_obs(envs.reset())
    # TODO: Update this when we implement heterogenous.
    training_agents[0].update_rollouts(obs=current_obs, timestep=0)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()

    episode_buffer = EpisodeBuffer(size=5000)

    start = time.time()
    for num_epoch in range(start_epoch, num_epochs):
        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval):
            utils.save_agents("qmix-", num_epoch, training_agents, total_steps,
                              num_episodes, args, suffix)

        for agent in training_agents:
            agent.set_eval()

        #############################################

        ##
        # Each history is Python list is of length num_processes. This is a list because not all
        # episodes are of the same length and we don't want information of an episode
        # after it has ended. Each history item has the first dimension as time step and then
        # appropriate shape
        #
        state_history = [torch.zeros(0, *obs_shape) for _ in range(num_processes)]
        action_history = [torch.zeros(0, 1) for _ in range(num_processes)]
        reward_history = [torch.zeros(0, 1) for _ in range(num_processes)]
        next_state_history = [torch.zeros(0, *obs_shape) for _ in range(num_processes)]
        done_history = [torch.zeros(0, 1).long() for _ in range(num_processes)]

        #############################################

        # @TODO: run for full episode length
        # while True:
        for step in range(10):

            current_obs.transpose_(0, 1)
            for i in range(num_processes):
                if len(done_history[i]) and done_history[i][-1][0] == 0:
                    state_history[i] = torch.cat([state_history[i], current_obs[i]], dim=0)
            current_obs.transpose_(0, 1)

            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []

            result = training_agents[0].actor_critic_act(step, 0)
            cpu_actions_agents = update_actor_critic_results(result)
            obs, reward, done, info = envs.step(cpu_actions_agents)
            reward = reward.astype(np.float)

            update_stats(info)

            if args.render:
                envs.render()

            running_num_episodes += sum([int(done_)
                                    for done_ in done])
            terminal_reward += reward[done.squeeze() == True].sum()

            success_rate += sum([int(s) for s in
                                [(done.squeeze() == True) & (reward.squeeze() > 0)][0] ])

            masks = torch.FloatTensor([
                [0.0]*num_training_per_episode if done_ \
                else [1.0]*num_training_per_episode
            for done_ in done])

            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            episode_rewards += reward
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            final_reward_arr = np.array(final_rewards.squeeze(0))
            cumulative_reward += final_reward_arr[done.squeeze() == True].sum()

            # TODO: figure out if the masking is done right
            current_obs = update_current_obs(obs)
            if args.cuda:
                masks = masks.cuda()
                current_obs = current_obs.cuda()

            # Mask out observations of completed agents from the prior round.
            # NOTE: we are ok with leaving in observations from agents that
            # died this round because their position mask is zeroed out, so
            # there is signal in the observations that they are dead.
            masks_all = masks.transpose(0,1).unsqueeze(2)
            current_obs *= masks_all.unsqueeze(2).unsqueeze(2)

            ############################################
            reward_all = reward.unsqueeze(2)
            states_all = utils.torch_numpy_stack(states_agents)
            action_all = utils.torch_numpy_stack(action_agents)
            action_log_prob_all = utils.torch_numpy_stack(
                action_log_prob_agents)
            value_all = utils.torch_numpy_stack(value_agents)

            done_all = torch.LongTensor(done).unsqueeze(1)

            action_all.transpose_(0, 1)
            reward_all.transpose_(0, 1)
            current_obs.transpose_(0, 1)

            for i in range(num_processes):
                if len(done_history[i]) and done_history[i][-1][0] == 0:
                    action_history[i] = torch.cat([action_history[i], current_obs[i]], dim=0)
                    reward_history[i] = torch.cat([reward_history[i], reward_all[i]], dim=0)
                    next_state_history[i] = torch.cat([next_state_history[i], current_obs[i]], dim=0)
                    done_history[i] = torch.cat([done_history[i], done_all[i]], dim=0)

            action_all.transpose_(0, 1)
            reward_all.transpose_(0, 1)
            current_obs.transpose_(0, 1)
            ###############################################

            training_agents[0].insert_rollouts(
                step, current_obs, states_all, action_all, action_log_prob_all,
                value_all, reward_all, masks_all)

            if done.all():
                break

        episode_buffer.extend(state_history, action_history, reward_history, next_state_history, done_history)

        # @TODO DQN here

        # Compute the advantage values.
        if how_train == 'simple' or how_train == 'homogenous':
            agent = training_agents[0]
            next_value_agents = [
                agent.actor_critic_call(step=-1, num_agent=num_agent)
                for num_agent in range(num_training_per_episode)
            ]
            advantages = [
                agent.compute_advantages(next_value_agents, args.use_gae,
                                         args.gamma, args.tau)
            ]

        # Run PPO Optimization.
        for num_agent, agent in enumerate(training_agents):
            agent.set_train()

            for _ in range(args.ppo_epoch):
                result = agent.ppo(advantages[num_agent], args.num_mini_batch,
                                   num_steps, args.clip_param,
                                   args.entropy_coef, args.max_grad_norm,) # anneal=True, lr=lr_anneal, eps=args.eps)
                action_losses, value_losses, dist_entropies = result
                final_action_losses[num_agent].extend(result[0])
                final_value_losses[num_agent].extend(result[1])
                final_dist_entropies[num_agent].extend(result[2])

            agent.after_epoch()

        total_steps += num_processes * num_steps

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
