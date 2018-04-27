"""Train script for ppo learning.

TODO: Implement heterogenous training.

The number of samples used for an epoch is:
horizon * num_workers = num_steps * num_processes where num_steps is the number
of steps in a rollout (horizon) and num_processes is the number of parallel
processes/workers collecting data.

Simple Example:
python train_ppo.py --how-train simple --num-processes 10 --run-name test \
 --num-steps 50 --log-interval 5

Distillation Example:
python train_ppo.py --how-train simple --num-processes 10 --run-name distill \
 --num-steps 100 --log-interval 5 \
 --distill-epochs 100 --distill-target dagger::/path/to/model.pt

Homogenous Example:
python train_ppo.py --how-train homogenous --num-processes 10 \
 --run-name distill --num-steps 100 --log-interval 5 --distill-epochs 100 \
 --distill-target dagger::/path/to/model.pt --config PommeTeam-v0 \
 --eval-mode homogenous --num-battles-eval 100 --seed 100
"""
from collections import defaultdict
import os
import time

import numpy as np
from pommerman.agents import SimpleAgent
from pommerman import utility
from tensorboardX import SummaryWriter
import torch
import random

from arguments import get_args
import envs as env_helpers
from eval import eval as run_eval
import ppo_agent
import utils


def train():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.cuda:
        torch.cuda.empty_cache()

    assert(args.run_name)
    print("\n###############")
    print("args ", args)
    print("##############\n")

    how_train, config, num_agents, num_stack, num_steps, num_processes, \
        num_epochs, reward_sharing = utils.get_train_vars(args)

    obs_shape, action_space = env_helpers.get_env_shapes(config, num_stack)
    num_training_per_episode = utils.validate_how_train(how_train, num_agents)

    training_agents = utils.load_agents(
        obs_shape, action_space, num_training_per_episode, args,
        ppo_agent.PPOAgent)
    if how_train == 'homogenous':
        bad_guys = [SimpleAgent(), SimpleAgent()]
        model = training_agents[0].model
        good_guys = []
        for _ in range(2):
            guy = ppo_agent.PPOAgent(model, num_stack=args.num_stack, cuda=args.cuda)
            if args.cuda:
                guy.cuda()
            good_guys.append(guy)
        eval_round = 0
    envs = env_helpers.make_train_envs(config, how_train, args.seed,
                                       args.game_state_file, training_agents,
                                       num_stack, num_processes)

    suffix = "{}.{}.{}.{}.nc{}.lr{}.mb{}.ns{}.clip{}.valc{}.seed{}".format(
        args.run_name, how_train, config, args.model_str, args.num_channels,
        args.lr, args.num_mini_batch, args.num_steps, args.clip_param,
        args.value_loss_coef, args.seed)

    distill_target = args.distill_target
    distill_epochs = args.distill_epochs
    do_distill = distill_target is not '' and \
                 distill_epochs > training_agents[0].num_epoch
    if do_distill:
        distill_agent = utils.load_distill_agent(obs_shape, action_space, args)
        distill_agent.set_eval()
        # NOTE: We have to call init_agent, but the agent_id won't matter
        # because we will use the observations from the ppo_agent.
        distill_agent.init_agent(0, envs.get_game_type())
        distill_type = distill_target.split('::')[0]
        suffix += ".dstl{}.dstlepi{}".format(distill_type, distill_epochs)
        # TODO: Should we not run this against the distill_agent as the first
        # opponent? The problem is that the distill_agent will just stall.
        if how_train == 'homogenous':
            distill_agent2 = utils.load_distill_agent(obs_shape, action_space, args)
            distill_agent2.set_eval()
            bad_guys = [distill_agent, distill_agent2]

    log_dir = os.path.join(args.log_dir, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #####
    # Logging helpers.
    writer = SummaryWriter(log_dir)
    count_stats = defaultdict(int)
    array_stats = defaultdict(list)
    episode_rewards = torch.zeros([num_training_per_episode,
                                   num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode,
                                 num_processes, 1])

    # TODO: When we implement heterogenous, change this to be per agent.
    start_epoch = training_agents[0].num_epoch
    total_steps = training_agents[0].total_steps
    num_episodes = training_agents[0].num_episodes

    running_num_episodes = 0
    cumulative_reward = 0
    terminal_reward = 0
    success_rate = 0
    prev_epoch = start_epoch
    final_action_losses = [[] for agent in range(len(training_agents))]
    final_value_losses =  [[] for agent in range(len(training_agents))]
    final_dist_entropies = [[] for agent in range(len(training_agents))]
    if do_distill:
        final_kl_losses = [[] for agent in range(len(training_agents))]
    final_total_losses =  [[] for agent in range(len(training_agents))]

    def update_current_obs(obs):
        return torch.from_numpy(obs).float().transpose(0,1)

    def update_actor_critic_results(result):
        value, action, action_log_prob, states, _, log_probs = result
        value_agents.append(value)
        action_agents.append(action)
        action_log_prob_agents.append(action_log_prob)
        states_agents.append(states)
        action_log_prob_distr.append(log_probs)
        return action.data.squeeze(1).cpu().numpy()

    def update_stats(info):
        # NOTE: This func has a side effect where it sets variable values.
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

    # Start the environment and set the current_obs appropriately.
    current_obs = update_current_obs(envs.reset())

    if how_train == 'simple' or how_train == 'homogenous':
        # NOTE: Here, we put the first observation into the rollouts.
        training_agents[0].update_rollouts(obs=current_obs, timestep=0)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()
        if do_distill:
            distill_agent.cuda()
            if how_train == 'homogenous':
                distill_agent2.cuda()

    if how_train == 'homogenous':
        win_rate, tie_rate, die_rate = evaluate_homogenous(
            args, good_guys, bad_guys, 0, writer, 0)
        print("Homog test beforehand: (%d)--> Win %.3f, Tie %.3f, Die %.3f" % (
            args.num_battles_eval, win_rate, tie_rate, die_rate))

    start = time.time()
    for num_epoch in range(start_epoch, num_epochs):
        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval) \
           and how_train == 'simple':
            # Only save at regular epochs if using "simple". The others save
            # upon successful evaluation.
            utils.save_agents("ppo-", num_epoch, training_agents, total_steps,
                              num_episodes, args, suffix)

        for agent in training_agents:
            agent.set_eval()

        if do_distill:
            if args.set_distill_kl >= 0:
                distill_factor = args.set_distill_kl
            else:
                distill_factor = distill_epochs - num_epoch
                distill_factor = 1.0 * distill_factor / distill_epochs
                distill_factor = max(distill_factor, 0.0)
            print("Epoch %d - distill factor %.3f." % (
                num_epoch, distill_factor))
        else:
            distill_factor = 0

        for step in range(num_steps):
            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            action_log_prob_distr = []
            dagger_prob_distr = []

            if how_train == 'simple':
                training_agent = training_agents[0]

                if do_distill:
                    data = training_agent.get_rollout_data(step, 0)
                    _, _, _, _, probs, _ = distill_agent.act_on_data(
                        *data, deterministic=True)
                    dagger_prob_distr.append(probs)

                result = training_agent.actor_critic_act(step, 0)
                cpu_actions_agents = update_actor_critic_results(result)
            elif how_train == 'homogenous':
                # Reshape to do computation once rather than four times.
                with utility.Timer() as t:
                    cpu_actions_agents = [[] for _ in range(num_processes)]
                    data = training_agents[0].get_rollout_data(
                        step=step, num_agent=0, num_agent_end=4)
                    observations, states, masks = data
                    observations = observations.view([num_processes * 4,
                                                      *observations.shape[2:]])
                    states = states.view([num_processes * 4, *states.shape[2:]])
                    masks = masks.view([num_processes * 4, *masks.shape[2:]])
                    if do_distill:
                        _, _, _, _, probs, _ = distill_agent.act_on_data(
                            observations, states, masks, deterministic=True)
                        probs = probs.view([num_processes, 4, *probs.shape[1:]])
                        for num_agent in range(4):
                            dagger_prob_distr.append(probs[:, num_agent])

                    result = training_agents[0].act_on_data(
                        observations, states, masks, deterministic=False)
                    result = [datum.view([num_processes, 4, *datum.shape[1:]])
                              for datum in result]
                    for num_agent in range(4):
                        agent_result = [datum[:, num_agent] for datum in result]
                        cpu_actions = update_actor_critic_results(agent_result)
                        for num_process in range(num_processes):
                            cpu_actions_agents[num_process].append(
                                cpu_actions[num_process])

            with utility.Timer() as t:
                obs, reward, done, info = envs.step(cpu_actions_agents)
            reward = reward.astype(np.float)

            update_stats(info)

            if args.render:
                envs.render()

            if how_train == 'simple':
                running_num_episodes += sum([int(done_) for done_ in done])
                terminal_reward += reward[done.squeeze() == True].sum()
                success_rate += sum([int(s) for s in \
                                     [(done.squeeze() == True) &
                                      (reward.squeeze() > 0)][0]])

                # NOTE: The masking for simple should be such that:
                # - final_rewards is masked out on every step except for the
                # last step of a process. at that point, it becomes the
                # episode_rewards.
                # - episode_rewards accumulates the rewards at every step and
                # is masked out only at the last step of a process.
                # - current_obs consists of the num_stack observations. when
                # the game resets, the observations do as well, so we won't
                # have an issue with the frames overlapping. this means that we
                # shouldn't be using the masking on the current_obs.
                masks = torch.FloatTensor([
                    [0.0]*num_training_per_episode if done_ \
                    else [1.0]*num_training_per_episode
                for done_ in done])
            elif how_train == 'homogenous':
                running_num_episodes += sum([int(done_.all())
                                             for done_ in done])

                # NOTE: The masking for homogenous should be such that:
                # 1. If the agent is alive, then it follows the same process as
                # in `simple`. This means that it's 1.0.
                # 2. If the agent died, then it's still going to get rewards
                # according to the team_reward_sharing attribute. This means
                # that masking should be 1.0 as well. However, note that this
                # could be problematic for the rollout if the agent's
                # observations don't specify that it's dead. That's why we
                # amended the featurize3D function in networks to be a zero map
                # for the agent's position if it's not alive.
                # TODO: Consider additionally changing the agent's action and
                # associated log probs to be the Stop action.
                masks = torch.FloatTensor([[0.0]*4 if done_.all() else [1.0]*4
                                           for done_ in done]) \
                             .transpose(0, 1).unsqueeze(2).unsqueeze(2)
                for num_process in range(num_processes):
                    for id_ in range(4):
                        tid = (id_ + 2) % 4
                        self_reward = reward[num_process][id_]
                        teammate_reward = reward[num_process][tid]
                        my_reward = (1 - reward_sharing) * self_reward + \
                                    reward_sharing * teammate_reward
                        reward[num_process][id_] = my_reward

            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            # NOTE: These don't mean anything for homogenous training
            episode_rewards += reward
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            final_reward_arr = np.array(final_rewards.squeeze(0))
            if how_train == 'simple':
                final_sum = final_reward_arr[done.squeeze() == True].sum()
                cumulative_reward += final_sum
            elif how_train == 'homogenous':
                where_done = np.array([done_.all() for done_ in done]) == True
                final_sum = final_reward_arr.squeeze().transpose()
                final_sum = final_sum[where_done].sum()
                cumulative_reward += final_sum

            current_obs = update_current_obs(obs)
            if args.cuda:
                masks = masks.cuda()
                current_obs = current_obs.cuda()

            if how_train == 'simple':
                masks_all = masks.transpose(0,1).unsqueeze(2)
            elif how_train == 'homogenous':
                masks_all = masks

            reward_all = reward.unsqueeze(2)
            states_all = utils.torch_numpy_stack(states_agents)
            action_all = utils.torch_numpy_stack(action_agents)
            action_log_prob_all = utils.torch_numpy_stack(
                action_log_prob_agents)
            if do_distill:
                dagger_prob_distr = utils.torch_numpy_stack(dagger_prob_distr)
                action_log_prob_distr = utils.torch_numpy_stack(
                    action_log_prob_distr)
            else:
                dagger_prob_distr = None
                action_log_prob_distr = None

            value_all = utils.torch_numpy_stack(value_agents)

            if how_train == 'simple' or how_train == 'homogenous':
                training_agents[0].insert_rollouts(
                    step, current_obs, states_all, action_all,
                    action_log_prob_all, value_all, reward_all, masks_all,
                    action_log_prob_distr, dagger_prob_distr)

        # Compute the advantage values.
        if how_train == 'simple' or how_train == 'homogenous':
            training_agent = training_agents[0]
            next_value_agents = [
                training_agent.actor_critic_call(step=-1, num_agent=num_agent)
                for num_agent in range(num_training_per_episode)
            ]
            advantages = [
                training_agent.compute_advantages(
                    next_value_agents, args.use_gae, args.gamma, args.tau)
            ]

        # Run PPO Optimization.
        for num_agent, agent in enumerate(training_agents):
            agent.set_train()

            for _ in range(args.ppo_epoch):
                with utility.Timer() as t:
                    result = agent.ppo(advantages[num_agent],
                                       args.num_mini_batch,
                                       num_steps, args.clip_param,
                                       args.entropy_coef, args.value_loss_coef,
                                       args.max_grad_norm,
                                       kl_factor=distill_factor)
                action_losses, value_losses, dist_entropies, \
                    kl_losses, total_losses = result

                final_action_losses[num_agent].extend(result[0])
                final_value_losses[num_agent].extend(result[1])
                final_dist_entropies[num_agent].extend(result[2])
                if do_distill:
                    final_kl_losses[num_agent].extend(result[3])
                final_total_losses[num_agent].extend(result[4])

            agent.after_epoch()

        total_steps += num_processes * num_steps

        if running_num_episodes > args.log_interval:
            end = time.time()
            num_steps_sec = (end - start)
            num_episodes += running_num_episodes

            steps_per_sec = 1.0 * total_steps / (end - start)
            epochs_per_sec = 1.0 * (num_epoch - prev_epoch) / (end - start)
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

            mean_total_loss = np.mean([
                total_loss for total_loss in final_total_losses])
            std_total_loss = np.std([
                total_loss for total_loss in final_total_losses])

            if how_train == 'homogenous':
                win_rate, tie_rate, die_rate = evaluate_homogenous(
                    args, good_guys, bad_guys, eval_round, writer, num_epoch)
                print("Epoch %d (%d)--> Win %.3f, Tie %.3f, Die %.3f" % (
                    num_epoch, args.num_battles_eval, win_rate, tie_rate,
                    die_rate))
                if win_rate >= .60:
                    suffix = suffix + ".wr%.3f.evlrnd%d" % (win_rate, eval_round)
                    saved_paths = utils.save_agents(
                        "ppo-", num_epoch, training_agents, total_steps,
                        num_episodes, args, suffix)
                    eval_round += 1
                    bad_guys = []
                    for _ in range(2):
                        guy = utils.torch_load(saved_paths[0], args.cuda,
                                               args.cuda_device)
                        if args.cuda:
                            guy.cuda()
                        bad_guys.append(guy)

            if do_distill:
                mean_kl_loss = np.mean([
                    kl_loss for kl_loss in final_kl_losses])
                std_kl_loss = np.std([
                    kl_loss for kl_loss in final_kl_losses])
            else:
                mean_kl_loss = None

            utils.log_to_console(num_epoch, num_episodes, total_steps,
                                 steps_per_sec, epochs_per_sec, final_rewards,
                                 mean_dist_entropy, mean_value_loss,
                                 mean_action_loss, cumulative_reward,
                                 terminal_reward, success_rate,
                                 running_num_episodes, mean_total_loss,
                                 mean_kl_loss)

            utils.log_to_tensorboard(writer, num_epoch, num_episodes,
                                     total_steps, steps_per_sec,
                                     episodes_per_sec, final_rewards,
                                     mean_dist_entropy, mean_value_loss,
                                     mean_action_loss, std_dist_entropy,
                                     std_value_loss, std_action_loss,
                                     count_stats, array_stats,
                                     cumulative_reward, terminal_reward,
                                     success_rate, running_num_episodes,
                                     mean_total_loss, mean_kl_loss)

            # Reset stats so that plots are per the last log_interval.
            final_action_losses = [[] for agent in range(len(training_agents))]
            final_value_losses =  [[] for agent in range(len(training_agents))]
            final_dist_entropies = [[] for agent in \
                                    range(len(training_agents))]
            if do_distill:
                final_kl_losses = [[] for agent in range(len(training_agents))]
            final_total_losses =  [[] for agent in range(len(training_agents))]

            count_stats = defaultdict(int)
            array_stats = defaultdict(list)
            final_rewards = torch.zeros([num_training_per_episode,
                                         num_processes, 1])
            running_num_episodes = 0
            cumulative_reward = 0
            terminal_reward = 0
            success_rate = 0
            prev_epoch = num_epoch

    writer.close()


def evaluate_homogenous(args, good_guys, bad_guys, eval_round, writer, epoch):
    print("Starting homogenous eval at epoch %d..." % epoch)
    with utility.Timer() as t:
        wins, one_dead, ties = run_eval(
            args=args, targets=good_guys, opponents=bad_guys)
    print("Eval took %.4fs." % t.interval)

    descriptor = 'homogenous_eval_round%d/' % eval_round
    num_battles = args.num_battles_eval
    win_count = sum(wins.values())
    tie_count = sum(ties.values())
    one_dead_count  = sum(one_dead.values())

    win_rate = 1.0*win_count/num_battles
    tie_rate = 1.0*tie_count/num_battles
    die_rate = 1.0*(num_battles - win_count - tie_count)/num_battles
    one_dead_per_battle = 1.0*one_dead_count/num_battles
    one_dead_per_win = 1.0*one_dead_count/win_count if win_count else 0
    writer.add_scalar('%s/win_rate' % descriptor, win_rate)
    writer.add_scalar('%s/tie_rate' % descriptor, tie_rate)
    writer.add_scalar('%s/die_rate' % descriptor, die_rate)
    writer.add_scalar('%s/one_dead_per_battle' % descriptor, one_dead_per_battle)
    writer.add_scalar('%s/one_dead_per_win' % descriptor, one_dead_per_win)
    return win_rate, tie_rate, die_rate


if __name__ == "__main__":
    train()
