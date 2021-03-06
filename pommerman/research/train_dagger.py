"""Train script for dagger learning.
Currently uses SimpleAgent as the expert.
The training is performed on one processor,
but evaluation is run on multiple processors
TODO:
make code less redundant
if not using the value loss it will store and do many unnecessary operations
Example args:
python train_dagger.py --num-processes 16 --run-name a --how-train dagger \
 --minibatch-size 5000 --num-steps 5000 --log-interval 10 --save-interval 10 \
 --lr 0.005 --expert-prob 0.5 --num-steps-eval 500 --use-value-loss
The --use-value-loss setting makes it so that the value loss is considered.
The --stop-grads-value setting stops the gradients from the value loss in going
through the rest of the shared params of the network. Both default to false.
"""

from collections import defaultdict
import os
import random
import time

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable

from arguments import get_args
import dagger_agent
import envs as env_helpers
import networks
import utils


def train():
    os.environ['OMP_NUM_THREADS'] = '1'

    args = get_args()
    assert(args.run_name)
    print("\n###############")
    print("args ", args)
    print("##############\n")

    if args.cuda:
        torch.cuda.empty_cache()

    num_training_per_episode = utils.validate_how_train(args)
    how_train, config, num_agents, num_stack, num_steps, num_processes, \
        num_epochs, reward_sharing, batch_size, num_mini_batch = \
        utils.get_train_vars(args, num_training_per_episode)
    assert(num_processes % 4 == 0), "Num Processes should be a multiple of " \
        "four so that the distribution of positions is even."

    obs_shape, action_space, character, board_size = env_helpers.get_env_info(config, num_stack)

    training_agents = utils.load_agents(
        obs_shape, action_space, num_training_per_episode, num_steps, args,
        agent_type=dagger_agent.DaggerAgent, character=character, board_size=board_size)
    agent = training_agents[0]

    #####
    # Logging helpers.
    suffix = "{}.{}.{}.{}.nc{}.lr{}.mb{}.ned{}.prob{}.nopt{}.seed{}.maxaggr{}.pt" \
             .format(args.run_name, args.how_train, config, args.model_str,
                     args.num_channels, args.lr, args.minibatch_size,
                     args.num_episodes_dagger, args.expert_prob,
                     args.dagger_epoch, args.seed,
                     args.max_aggregate_agent_states)
    if args.state_directory_distribution:
        suffix += ".%s" % args.state_directory_distribution

    log_dir = os.path.join(args.log_dir, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    start_epoch = agent.num_epoch
    total_steps = agent.total_steps
    num_episodes = agent.num_episodes

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        agent.cuda()

    aggregate_agent_states = []
    aggregate_expert_actions = []
    aggregate_returns = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    dummy_states = torch.zeros(1,1)
    dummy_masks = torch.zeros(1,1)
    if args.cuda:
        dummy_states = dummy_states.cuda()
        dummy_masks = dummy_masks.cuda()

    envs = env_helpers.make_train_envs(
        config, how_train, args.seed, args.game_state_file, training_agents,
        num_stack, num_processes, state_directory=args.state_directory,
        state_directory_distribution=args.state_directory_distribution,
        step_loss=args.step_loss, bomb_reward=args.bomb_reward,
        item_reward=args.item_reward)

    # [num_proc, num_frame*19, board_size, board_size]
    agent_obs = torch.from_numpy(envs.reset()).float().squeeze(1)
    if args.cuda:
        agent_obs = agent_obs.cuda()

    dummy_states_eval = torch.zeros(num_processes, 1)
    dummy_masks_eval = torch.zeros(num_processes, 1)
    if args.cuda:
        dummy_states_eval = dummy_states_eval.cuda()
        dummy_masks_eval = dummy_masks_eval.cuda()

    episode_rewards = torch.zeros([num_training_per_episode,
                                   num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode,
                                 num_processes, 1])

    running_num_episodes = 0
    cumulative_reward = 0
    terminal_reward = 0
    success_rate = 0

    done = np.array([[False]])

    agent_act_arr = []
    expert_act_arr = []

    start = time.time()
    for num_epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        if num_epoch > 0:
            print("Avg Epoch Time: %.3f (%d)" % ((epoch_start_time - start)*1.0/num_epoch, num_epoch))

        if args.anneal_expert_prob:
            expert_prob = args.expert_prob - num_epoch * args.anneal_factor
        else:
            expert_prob = args.expert_prob

        agent.set_eval()
        agent_states_list = [[] for _ in range(num_processes)]
        expert_actions_list = [[] for _ in range(num_processes)]
        returns_list = [[] for _ in range(num_processes)]

        ########
        # Collect data using DAGGER
        ########
        count_episodes = 0
        current_ep_len = [0 for _ in range(num_processes)]
        while count_episodes < args.num_episodes_dagger:

            expert_obs = envs.get_expert_obs()
            expert_actions = envs.get_expert_actions(expert_obs, 'ComplexAgent')

            for num_process in range(num_processes):
                agent_states_list[num_process].append(agent_obs[num_process])
                expert_actions_list[num_process].append(expert_actions[num_process])
                # expert_actions_list.append(expert_action_tensor)

            if random.random() <= expert_prob:
                env_actions = expert_actions
                expert_act_arr.append(expert_actions)
            else:
                result = agent.act_on_data(
                    Variable(agent_obs, volatile=True),
                    Variable(dummy_states, volatile=True),
                    Variable(dummy_masks, volatile=True))
                _, action, _, _, _, _ = result
                env_actions = action.data.squeeze(1).cpu().numpy()
                agent_act_arr.append(env_actions)
                del result  # for reducing memory usage

            obs, reward, done, info = envs.step(env_actions)

            agent_obs = torch.from_numpy(obs).float().squeeze(1)
            if args.cuda:
                agent_obs = agent_obs.cuda()

            for num_process, done_ in enumerate(done):
                returns_list[num_process].append(float(reward[num_process][0]))

                if not done_[0]:
                    current_ep_len[num_process] += 1
                    continue

                # NOTE: In a FFA game, at this point it's over for the agent so
                # we call it. However, in a team game, it may not be over yet.
                # That depends on if the returned Result is Incomplete. We
                # could do something awkward and try to manage the rewards. We
                # could also change it so that the agent keeps going a la the
                # other setups. However, that's not really the point here and
                # so we keep it simple and give it zero reward.
                count_episodes += 1

                total_data_len = len(returns_list[num_process])
                start_current_ep = total_data_len - current_ep_len[num_process] - 1
                for step in range(total_data_len - 2, start_current_ep, -1):
                    next_return = returns_list[num_process][step+1]
                    future_value = float(next_return * args.gamma)
                    returns_list[num_process][step] += future_value

                current_ep_len[num_process] = 0

        if num_epoch % args.log_interval == 0:
            agent_act_arr = utils.flatten(agent_act_arr)
            expert_act_arr = utils.flatten(expert_act_arr)
            if len(agent_act_arr) > 0:
                agent_mean_act_prob = [
                    len([i for i in agent_act_arr if i == k]) * \
                    1.0/len(agent_act_arr) for k in range(6)
                ]
                for k in range(6):
                    print("mean act {} probs {}".format(k, agent_mean_act_prob[k]))
                print("")

            if len(expert_act_arr) > 0:
                expert_mean_act_prob = [
                    len([i for i in expert_act_arr if i == k]) * \
                    1.0/len(expert_act_arr) for k in range(6)
                ]
                for k in range(6):
                    print("expert mean act {} probs {}".format(
                        k, expert_mean_act_prob[k]))
                print("")

        expert_act_arr = []
        agent_act_arr = []

        total_steps += num_processes * num_steps
        #########
        # Train using DAGGER (with supervision from the expert)
        #########
        agent.set_train()

        agent_states_list = utils.flatten(agent_states_list)
        expert_actions_list = utils.flatten(expert_actions_list)
        returns_list = utils.flatten(returns_list)

        if len(aggregate_agent_states) >= args.max_aggregate_agent_states:
            indices_replace = np.arange(0, len(aggregate_agent_states)) \
                                .tolist()
            random.shuffle(indices_replace)
            for k in range(len(agent_states_list)):
                indice = indices_replace[k]
                aggregate_agent_states[indice] = agent_states_list[k]
                aggregate_expert_actions[indice] = expert_actions_list[k]
                aggregate_returns[indice] = returns_list[k]
        else:
            aggregate_agent_states += agent_states_list
            aggregate_expert_actions += expert_actions_list
            aggregate_returns += returns_list

        del agent_states_list, expert_actions_list, returns_list

        indices = np.arange(0, len(aggregate_agent_states)).tolist()
        random.shuffle(indices)

        for j in range(args.dagger_epoch):
            if j == args.dagger_epoch - 1:
                action_losses = []
                value_losses = []
            # TODO: make this loop more efficient - maybe you can move part of
            # minibatching outside and only select in the tensors?
            for i in range(0, len(aggregate_agent_states),
                           args.minibatch_size):
                indices_minibatch = indices[i: i + args.minibatch_size]
                agent_states_minibatch = [aggregate_agent_states[k]
                                          for k in indices_minibatch]
                expert_actions_minibatch = [aggregate_expert_actions[k]
                                            for k in indices_minibatch]
                returns_minibatch = [aggregate_returns[k]
                                    for k in indices_minibatch]

                agent_states_minibatch = torch.stack(agent_states_minibatch, 0)
                expert_actions_minibatch = torch.from_numpy(np.array(expert_actions_minibatch).squeeze(1))
                returns_minibatch = torch.FloatTensor(returns_minibatch) \
                                    .unsqueeze(1)

                if args.cuda:
                    agent_states_minibatch = agent_states_minibatch.cuda()
                    expert_actions_minibatch = expert_actions_minibatch.cuda()
                    returns_minibatch = returns_minibatch.cuda()

                values, action_scores = agent.get_values_action_scores(
                    Variable(agent_states_minibatch),
                    Variable(dummy_states).detach(),
                    Variable(dummy_masks).detach())
                action_loss = cross_entropy_loss(
                    action_scores, Variable(expert_actions_minibatch))
                value_loss = (Variable(returns_minibatch) - values) \
                                .pow(2).mean()

                agent.optimize(action_loss, value_loss, args.max_grad_norm, \
                               use_value_loss=args.use_value_loss,
                               stop_grads_value=args.stop_grads_value,
                               add_nonlin=args.add_nonlin_valhead)

                if j == args.dagger_epoch - 1:
                    action_losses.append(action_loss.data[0])
                    value_losses.append(value_loss.data[0])

                del action_scores, action_loss, value_loss, values

        del indices_minibatch, returns_minibatch, agent_states_minibatch,\
            expert_actions_minibatch

        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval):
            utils.save_agents("dagger-", num_epoch, training_agents,
                              total_steps, num_episodes, args)


        ######
        # Eval the current policy
        ######
        # TODO: make eval deterministic
        if num_epoch % args.log_interval == 0:
            agent.set_eval()
            eval_time = time.time()
            eval_envs = env_helpers.make_train_envs(
                config, 'simple', args.seed, args.game_state_file,
                training_agents, num_stack, num_processes,
                state_directory=args.state_directory,
                state_directory_distribution=args.state_directory_distribution,
                do_filter_team=False, step_loss=args.step_loss,
                bomb_reward=args.bomb_reward, item_reward=args.item_reward)

            dagger_obs = torch.from_numpy(eval_envs.reset()) \
                              .float().squeeze(0).squeeze(1)
            if args.cuda:
                dagger_obs = dagger_obs.cuda()
            while running_num_episodes < args.num_steps_eval:
                result_eval = agent.act_on_data(
                    Variable(dagger_obs, volatile=True),
                    Variable(dummy_states_eval, volatile=True),
                    Variable(dummy_masks_eval, volatile=True),
                    deterministic=True)
                _, actions_eval, _, _, _, _ = result_eval
                cpu_actions_eval = actions_eval.data.squeeze(1).cpu().numpy()
                cpu_actions_agents_eval = cpu_actions_eval
                obs_eval, reward_eval, done_eval, info_eval = eval_envs.step(
                    cpu_actions_agents_eval)
                del result_eval

                dagger_obs = torch.from_numpy(
                    obs_eval.reshape(num_processes, *obs_shape)).float()
                if args.cuda:
                    dagger_obs = dagger_obs.cuda()

                running_num_episodes += sum([1 if done_ else 0
                                             for done_ in done_eval])
                terminal_reward += reward_eval[done_eval.squeeze() == True] \
                                   .sum()
                success_rate += sum([1 if x else 0 for x in
                                    [(done_eval.squeeze() == True) & \
                                     (reward_eval.squeeze() > 0)][0] ])

                masks = torch.FloatTensor([
                    [0.0]*num_training_per_episode if done_ \
                    else [1.0]*num_training_per_episode
                for done_ in done_eval])

                reward_eval = utils.torch_numpy_stack(reward_eval, False) \
                                   .transpose(0, 1)
                episode_rewards += reward_eval[:, :, None]
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                final_reward_arr = np.array(final_rewards.squeeze(0))
                cumulative_reward += final_reward_arr[
                    done_eval.squeeze() == True].sum()

                if args.render:
                    eval_envs.render()

            print("Eval Time: ", time.time() - eval_time)

            cumulative_reward = 1.0 * cumulative_reward / running_num_episodes
            terminal_reward = 1.0 * terminal_reward / running_num_episodes
            success_rate = 1.0 * success_rate / running_num_episodes

            end = time.time()
            steps_per_sec = 1.0 * total_steps / (end - start)
            epochs_per_sec = 1.0 * num_epoch / (end - start)

            print("###########")
            print("Epoch {}, # steps: {} SPS {} EPS {} \n success rate {} " \
                  "mean final reward {} mean total reward {} " \
                  .format(num_epoch, len(aggregate_agent_states),
                          steps_per_sec, epochs_per_sec, success_rate,
                          terminal_reward, cumulative_reward))
            print("###########\n")

            utils.log_to_tensorboard_dagger(
                writer, num_epoch, total_steps, np.mean(action_losses),
                cumulative_reward, success_rate, terminal_reward,
                np.mean(value_losses), epochs_per_sec, steps_per_sec,
                agent_mean_act_prob, expert_mean_act_prob)

            running_num_episodes = 0
            cumulative_reward = 0
            terminal_reward = 0
            success_rate = 0

            eval_envs.close()

    writer.close()

if __name__ == "__main__":
    train()
