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

from arguments import get_args
import envs as env_helpers
import ppo_agent
import utils


def train():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'

    torch.cuda.set_device(args.cuda_device)
    torch.cuda.empty_cache()

    assert(args.run_name)

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
    final_dist_entropies = [[] for agent in range(len(training_agents))]
    #####

    # NOTE: These four funcs have side effects where they set variable values.
    def update_current_obs(obs):
        current_obs = torch.from_numpy(obs).float()

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

    current_obs = torch.zeros(num_training_per_episode, num_processes,
                              *obs_shape)
    update_current_obs(envs.reset())

    # TODO: Update this when we implement heterogenous.
    training_agents[0].update_rollouts(obs=current_obs, timestep=0)
    if how_train == 'homogenous':
        # We use agent_died_now to keep track of which agents are dead in order
        # to feed them their teammate's reward.
        agent_died_now = [[False]*4 for _ in range(num_processes)]
        # We use agent_died_prior to keep track of which agents died in a prior
        # round of num_steps. If this is true, then we do NOT give them their
        # teammate's reward and instead mask their observations to be 0 so that
        # we can't get any training signal from it.
        agent_died_prior = [[False]*4 for _ in range(num_processes)]

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()

    start = time.time()
    for num_epoch in range(start_epoch, num_epochs + start_epoch):
        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval):
            utils.save_agents("ppo-", num_epoch, training_agents, total_steps,
                              num_episodes, args, suffix)

        for agent in training_agents:
            agent.set_eval()

        for step in range(num_steps):
            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            cpu_actions_agents = []

            if how_train == 'simple':
                result = training_agents[0].actor_critic_act(step, 0)
                cpu_actions_agents = update_actor_critic_results(result)
            elif how_train == 'homogenous':
                cpu_actions_agents = [[] for _ in range(num_processes)]
                for num_agent in range(4):
                    result = training_agents[0].actor_critic_act(
                        step=step, num_agent=num_agent)
                    cpu_actions = update_actor_critic_results(result)
                    for num_process in range(num_processes):
                        cpu_actions_agents[num_process].append(
                            cpu_actions[num_process])

            obs, reward, done, info = envs.step(cpu_actions_agents)
            reward = reward.astype(np.float)

            update_stats(info)

            if args.render:
                envs.render()


            if how_train == 'simple':
                terminal_reward += reward[done.squeeze() == True].sum()
                success_rate += sum([1 if x else 0 for x in
                                    [(done.squeeze() == True) & (reward.squeeze() > 0)][0] ])
                running_num_episodes += sum([1 if done_ else 0
                                             for done_ in done])
                masks = torch.FloatTensor([
                    [0.0]*num_training_per_episode if done_ \
                    else [1.0]*num_training_per_episode
                for done_ in done])
            elif how_train == 'homogenous':
                running_num_episodes += sum([int(done_.all())
                                             for done_ in done])

                # We want the masks to satisfy the following:
                # 1. If the agent is alive --> 1.0.
                # 2. If the agent died this round (agent_died_now) --> 1.0.
                # 3. If the agent died prior (agent_died_prior) --> 0.0.
                masks = [[None]*4 for _ in range(num_processes)]

                # We update so that if an agent dies, then it receives its
                # teammate's reward for the rest of this num_steps round. After
                # that, it gets masked out until the episode ends.
                for num_process in range(num_processes):
                    masks[num_process] = [1.0]*4
                    for id_ in range(4):
                        tid = (id_ + 2) % 4
                        if agent_died_now[num_process][id_]:
                            # Give the agent its teammate's reward because it
                            # died in this round.
                            reward[num_process][id_] = reward[num_process][tid]
                        elif done[num_process][id_]:
                            if agent_died_prior[num_process][id_]:
                                # The agent died in a prior round. No reward
                                # and masks this out.
                                masks[num_process][id_] = 0.0
                            else:
                                # The agent just died. Give it its teammate's
                                # reward AND update agent_died_now to be True.
                                reward[num_process][id_] = reward[num_process][tid]
                                agent_died_now[num_process][id_] = True

                masks = torch.FloatTensor(masks) \
                             .transpose(0, 1).unsqueeze(2).unsqueeze(2)

                # If the process completed, reset the agent_died_* vars.
                for num_process in range(num_processes):
                    if done[num_process].all():
                        agent_died_now[num_process] = [False]*4
                        agent_died_prior[num_process] = [False]*4

            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            episode_rewards += reward
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            final_reward_arr = np.array(final_rewards.squeeze(0))
            cumulative_reward += final_reward_arr[done.squeeze() == True].sum()

            if args.cuda:
                masks = masks.cuda()

            # Mask out observations of completed agents from the prior round.
            # NOTE: we are ok with leaving in observations from agents that
            # died this round because their position mask is zeroed out, so
            # there is signal in the observations that they are dead.
            if how_train == 'simple':
                masks_all = masks.transpose(0,1).unsqueeze(2)
                current_obs *= masks_all.unsqueeze(2).unsqueeze(2)
            elif how_train == 'homogenous':
                masks_all = masks
                current_obs *= masks_all.unsqueeze(2)
            update_current_obs(obs)

            reward_all = reward.unsqueeze(2)
            states_all = utils.torch_numpy_stack(states_agents)
            action_all = utils.torch_numpy_stack(action_agents)
            action_log_prob_all = utils.torch_numpy_stack(
                action_log_prob_agents)
            value_all = utils.torch_numpy_stack(value_agents)

            training_agents[0].insert_rollouts(
                step, current_obs, states_all, action_all, action_log_prob_all,
                value_all, reward_all, masks_all)

        if how_train == 'homogenous':
            # Update the agent trackers so that if one of the agents has died,
            # then that is passed through to the next block.
            for num_process in range(num_processes):
                for id_ in range(4):
                    agent_died_prior[num_process][id_] = any([
                        agent_died_prior[num_process][id_],
                        agent_died_now[num_process][id_]
                    ])
                    agent_died_now[num_process][id_] = False

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
        # for num_agent, agent in enumerate(training_agents):
        #     agent.set_train()
        #
        #     for _ in range(args.ppo_epoch):
        #         result = agent.ppo(advantages[num_agent], args.num_mini_batch,
        #                            num_steps, args.clip_param,
        #                            args.entropy_coef, args.max_grad_norm)
        #         action_losses, value_losses, dist_entropies = result
        #         final_action_losses[num_agent].extend(result[0])
        #         final_value_losses[num_agent].extend(result[1])
        #         final_dist_entropies[num_agent].extend(result[2])
        #
        #     agent.after_epoch()

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
