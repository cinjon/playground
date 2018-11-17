'''
Train script for behavioral cloning
(using saved trajectories)

The --use-value-loss setting makes it so that a value predicting network
is also trained with supervision (to avoid policy overwritting by
value loss gradients when fine-tuning using PPO).

The --stop-grads-value setting stops the gradients from the value loss in going
through the rest of the shared params of the network (to avoid changing the
policy since we want the policy to imitate the trajectory as well as possible).

The --add-nonlin-valhead addsa a nonlinearity in the valuehead so that it has
more expressive power.

All the above args default to false.

Example Run:

Grid:
python train_bc.py --traj-directory-bc /home/roberta/playground/trajectories/grid/4maps/ \
--run-name a --how-train bc --minibatch-size 160 --num-stack 1 \
--config GridWalls-v4 --num-processes 1 --log-interval 100 \
--num-channels 5 --model-str GridCNNPolicy --save-interval 1000 --log-dir ./logs/bc

Pomme:
python train_bc.py --traj-directory-bc /home/roberta/playground/trajectories/pomme/4maps \
--run-name a --how-train bc --minibatch-size 1753 \
--config PommeFFAEasy-v0 --num-processes 4 \
--num-stack 1 --num-channels 19 --log-interval 100 --lr 0.001 \
--model-str PommeCNNPolicySmall --save-interval 1000 --log-dir ./logs/bc
'''

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
    num_epochs = 1000000

    obs_shape, action_space, character, board_size = env_helpers.get_env_info(config, num_stack)
    training_agents = utils.load_agents(
        obs_shape, action_space, num_training_per_episode, num_steps, args,
        agent_type=dagger_agent.DaggerAgent, character=character, board_size=board_size)
    agent = training_agents[args.expert_id]

    #####
    # Logging helpers.
    suffix = "{}.{}.{}.{}.nc{}.lr{}.mb{}.nopt{}.traj-{}.seed{}.pt" \
             .format(args.run_name, args.how_train, config, args.model_str,
                     args.num_channels, args.lr, args.minibatch_size,
                     args.dagger_epoch, args.traj_directory_bc, args.seed)

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

    envs = env_helpers.make_train_envs(
        config, how_train, args.seed, args.game_state_file, training_agents,
        num_stack, num_processes, state_directory=args.state_directory,
        state_directory_distribution=args.state_directory_distribution,
        step_loss=args.step_loss, bomb_reward=args.bomb_reward,
        item_reward=args.item_reward)

    if config == 'GridWalls-v4':
        how_train_eval = 'grid'
    else:
        how_train_eval = 'simple'
    eval_envs = env_helpers.make_train_envs(
        config, how_train_eval, args.seed, args.game_state_file, training_agents,
        num_stack, num_processes, state_directory=args.state_directory,
        state_directory_distribution=args.state_directory_distribution,
        step_loss=args.step_loss, bomb_reward=args.bomb_reward,
        item_reward=args.item_reward)

    #################################################
    # Load Trajectories (State, Action)-Pairs from File
    #################################################
    states, actions, files = envs.get_states_actions_json(args.traj_directory_bc)
    expert_states = states[0]
    expert_actions = actions[0]
    expert_files = files[0]

    agent_obs_lst = []
    if args.num_stack == 1:
        for s in expert_states:
            agent_obs_lst.append(torch.from_numpy(envs.observation(s)[0]).squeeze(0).float())
    elif args.num_stack == 2:
        curr_state = torch.from_numpy(envs.observation(expert_states[0])[0]).squeeze(0).float()
        agent_obs_lst.append(torch.cat((curr_state, curr_state), 0))
        for i in range(1, len(expert_states)):
            prev_state = torch.from_numpy(envs.observation(expert_states[i - 1])[0]).squeeze(0).float()
            curr_state = torch.from_numpy(envs.observation(expert_states[i])[0]).squeeze(0).float()
            agent_obs_lst.append(torch.cat((prev_state, curr_state), 0))
    else:
        raise ValueError("Only works with num_stack 1 or 2.")

    init_states = envs.get_init_states_json(args.traj_directory_bc)[0]

    init_states_lst = []
    for s in init_states:
        init_states_lst.append(torch.from_numpy(envs.observation(s)[0]).float())
    if args.cuda:
        init_states_lst = [s.cuda() for s in init_states_lst]

    print("\n# states: {}\n".format(len(agent_obs_lst)))

    expert_actions_lst = []
    for a in expert_actions:
        x = torch.LongTensor(1,1)
        x[0][0] = int(a[args.expert_id])
        expert_actions_lst.append(x)

    indices = np.arange(0, len(agent_obs_lst)).tolist()

    dummy_states = torch.zeros(1,1)
    dummy_masks = torch.zeros(1,1)
    if args.cuda:
        dummy_states = dummy_states.cuda()
        dummy_masks = dummy_masks.cuda()

    dummy_states_eval = torch.zeros(1,1)
    dummy_masks_eval = torch.zeros(1,1)
    if args.cuda:
        dummy_states_eval = dummy_states_eval.cuda()
        dummy_masks_eval = dummy_masks_eval.cuda()

    episode_rewards = torch.zeros([num_training_per_episode,
                                   num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode,
                                 num_processes, 1])
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    returns = [0 for k in agent_obs_lst]


    #################################################
    # Train Policy using Behavioral Cloning
    #################################################
    for num_epoch in range(start_epoch, num_epochs):
        action_losses = []
        value_losses = []
        num_correct_actions = 0
        random.shuffle(indices)
        agent.set_train()

        for i in range(0, len(agent_obs_lst), args.minibatch_size):
            indices_minibatch = indices[i:i + args.minibatch_size]
            agent_obs_mb = [agent_obs_lst[k] for k in indices_minibatch]
            expert_actions_mb = [expert_actions_lst[k] for k in indices_minibatch]
            returns_mb = [returns[k] for k in indices_minibatch]

            agent_obs_mb = torch.stack(agent_obs_mb, 0)
            expert_actions_mb = torch.from_numpy(np.array(expert_actions_mb))
            returns_mb = torch.from_numpy(np.array(returns_mb)).float()

            if args.cuda:
                agent_obs_mb = agent_obs_mb.cuda()
                expert_actions_mb = expert_actions_mb.cuda()
                returns_mb = returns_mb.cuda()

            values, action_scores = agent.get_values_action_scores(
                Variable(agent_obs_mb),
                Variable(dummy_states).detach(),
                Variable(dummy_masks).detach())
            action_loss = cross_entropy_loss(
                action_scores, Variable(expert_actions_mb))
            # value_loss = (values - values) \
            #                 .pow(2).mean()
            value_loss = (Variable(returns_mb) - values) \
                            .pow(2).mean()

            agent.optimize(action_loss, value_loss, args.max_grad_norm, \
                           use_value_loss=args.use_value_loss,
                           stop_grads_value=args.stop_grads_value,
                           add_nonlin=args.add_nonlin_valhead)

            action_losses.append(action_loss.data[0])
            value_losses.append(value_loss.data[0])

            ###############
            # Measure percentage of correct actions (identical to x)
            ###############
            result_train = agent.act_on_data(
                Variable(agent_obs_mb, volatile=True),
                Variable(dummy_states, volatile=True),
                Variable(dummy_masks, volatile=True),
                deterministic=True)
            _, actions_train, _, _, _, _ = result_train
            cpu_actions_train = actions_train.data.squeeze(1).cpu().numpy()
            expert_actions_train = expert_actions_mb.cpu().numpy()

            num_correct_actions += sum(sum([cpu_actions_train == expert_actions_train]))

        percent_correct = num_correct_actions / len(agent_obs_lst)
        mean_action_loss = np.sum(action_losses) / len(agent_obs_lst)
        mean_value_loss = np.sum(value_losses) / len(agent_obs_lst)
        if num_epoch % args.log_interval == 0:
            print("\n*********************************")
            print("EPOCH {}:".format(num_epoch))
            print("% correct action ", percent_correct)
            print("action loss ", mean_action_loss)
            print("value loss ", mean_value_loss)
            print("**********************************\n")

            utils.log_to_tensorboard_bc(writer, num_epoch, percent_correct,
                                     mean_action_loss, mean_value_loss)

        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval):
            utils.save_agents("bc-", num_epoch, training_agents,
                              total_steps, num_episodes, args)

        #################################################
        # Estimate Value Function Using Current Policy
        #################################################
        nmaps = 0
        returns = []
        for state_file in expert_files:
            nmaps += 1
            running_num_episodes = 0
            cumulative_reward = 0
            success_rate = 0
            state_returns = []
            for j in range(args.num_eps_eval):
                state_eval = eval_envs.reset_state_file(state_file)
                obs_eval = torch.from_numpy(eval_envs.observation(state_eval[0])[0]).float()
                if args.cuda:
                    obs_eval = obs_eval.cuda()
                done_eval = False
                current_ep_len = 0
                rewards = []
                while not done_eval:
                    result_eval = agent.act_on_data(
                        Variable(obs_eval, volatile=True),
                        Variable(dummy_states_eval, volatile=True),
                        Variable(dummy_masks_eval, volatile=True),
                        deterministic=True)
                    _, actions_eval, _, _, _, _ = result_eval
                    cpu_actions_eval = actions_eval.data.squeeze(1).cpu().numpy()
                    cpu_actions_agents_eval = cpu_actions_eval

                    obs_eval, reward_eval, done_eval, info_eval = eval_envs.step(cpu_actions_agents_eval)
                    obs_eval = torch.from_numpy(obs_eval.reshape(*obs_shape)).float().unsqueeze(0)
                    if args.cuda:
                        obs_eval = obs_eval.cuda()
                    rewards.append(reward_eval)

                    running_num_episodes += sum([1 if done_ else 0
                                                 for done_ in done_eval])
                    success_rate += sum([1 if x else 0 for x in
                                        [(done_eval.squeeze() == True) & \
                                        (reward_eval.squeeze() > 0)] ])
                    masks = torch.FloatTensor([[0.0]*num_training_per_episode if done_ \
                                                else [1.0]*num_training_per_episode
                                                for done_ in done_eval])
                    reward_eval = utils.torch_numpy_stack(reward_eval, False) \
                                                        .transpose(0, 1)
                    episode_rewards += reward_eval[:, :, None]
                    final_rewards *= masks
                    final_rewards += (1 - masks) * episode_rewards
                    episode_rewards *= masks
                    final_reward_arr = np.array(final_rewards.squeeze(0))
                    cumulative_reward += final_reward_arr[done_eval.squeeze() == True].sum()

                # compute return for this episode
                for step in range(len(rewards) - 2, 0, -1):
                    next_return = rewards[step+1]
                    future_value = float(next_return * args.gamma)
                    rewards[step] += future_value

                # save the return for this episode (corresp to state_file)
                state_returns.append(rewards[0])

            # take the mean over all returns from state_file
            returns.append(np.mean(state_returns))
            cumulative_reward = 1.0 * cumulative_reward / args.num_eps_eval
            success_rate = 1.0 * success_rate / args.num_eps_eval

            if num_epoch % args.save_interval == 0:
                print("###########")
                print("Epoch {}, map {}: \n success rate {} " \
                      "mean total reward {} " \
                      .format(num_epoch, nmaps, success_rate, cumulative_reward))
                print("###########\n")
                # utils.log_to_tensorboard_bc(
                #     writer, num_epoch, total_steps, np.mean(action_losses),
                #     cumulative_reward, success_rate, terminal_reward,
                #     np.mean(value_losses), epochs_per_sec, steps_per_sec,
                #     agent_mean_act_prob, expert_mean_act_prob)

    writer.close()

if __name__ == "__main__":
    train()
