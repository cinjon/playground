"""Train script for dagger learning.

Currently only works for single processor and uses SimpleAgent as the expert.

NOTE: Run this with how-train dagger so that the agent's position is random
among the four possibilities.

Example args:

python train_dagger.py --num-processes 1 --run-name a --how-train dagger \
 --minibatch-size 5000 --num-steps 5000 --log-interval 10 --lr 0.1 \
 --expert-prob 0.5 --save-interval 10 --num-steps-eval 100
"""

from collections import defaultdict
import os
import random
import time

import numpy as np
from pommerman.agents import SimpleAgent
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
    torch.cuda.empty_cache()

    args = get_args()
    assert(args.run_name)
    print("\n###############")
    print("args ", args)
    print("##############\n")

    how_train, config, num_agents, num_stack, num_steps, num_processes, \
        num_epochs = utils.get_train_vars(args)
    dagger_num_processes = 1
    # assert(num_processes == 1), "Doesn't work on more than one process."

    obs_shape, action_space = env_helpers.get_env_shapes(config, num_stack)
    num_training_per_episode = utils.validate_how_train(how_train, num_agents)

    training_agents = utils.load_agents(
        obs_shape, action_space, num_training_per_episode, args,
        dagger_agent.DaggerAgent)
    agent = training_agents[0]


    #####
    # Logging helpers.
    suffix = "{}.ht-{}.cfg-{}.m-{}.nc-{}.lr-{}-.mb-{}.prob-{}.anneal-{}.seed-{}.pt" \
             .format(args.run_name, args.how_train, config, args.model_str,
                     args.num_channels, args.lr, args.minibatch_size, args.expert_prob,
                     args.anneal_expert_prob, args.seed)
    log_dir = os.path.join(args.log_dir, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # TODO: Remove the num_processes here.
    writer = SummaryWriter(log_dir)

    start_epoch = agent.num_epoch
    total_steps = agent.total_steps
    num_episodes = agent.num_episodes

    expert_act_arr = []
    agent_act_arr = []
    #####

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        agent.cuda()

    aggregate_agent_states = []
    aggregate_expert_actions = []
    aggregate_agent_masks = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    expert = SimpleAgent()
    action_loss = 0

    dummy_states = torch.zeros(1,1)
    dummy_masks = torch.zeros(1,1)
    if args.cuda:
        dummy_states = dummy_states.cuda()
        dummy_masks = dummy_masks.cuda()


    eval_how_train = 'simple'
    eval_envs = env_helpers.make_envs(config, eval_how_train, args.seed,
                                 args.game_state_file, training_agents,
                                 num_stack, num_processes, args.render)

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

    start = time.time()
    for num_epoch in range(start_epoch, num_epochs):
        if num_epoch > start_epoch:
            envs.close()
        envs = env_helpers.make_envs(config, how_train, args.seed,
                                     args.game_state_file, training_agents,
                                     num_stack, dagger_num_processes, args.render)
        agent_obs = torch.from_numpy(envs.reset()).float().squeeze(0)
        if args.cuda:
            agent_obs = agent_obs.cuda()

        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval):
            utils.save_agents("dagger-", num_epoch, training_agents,
                              total_steps, num_episodes, args)

        # expert_prob 0.5 --> 0 after 100 epochs with 0.005 annealing factor.
        if args.anneal_expert_prob:
            expert_prob = args.expert_prob - num_epoch * args.anneal_factor
        else:
            expert_prob = args.expert_prob

        agent.set_eval()
        agent_states_list = []
        expert_actions_list = []

        ########
        # Collect data using DAGGER
        ########

        for step in range(args.num_steps):
            # NOTE: moved envs inside the loop so that you get dif init position for the dagger agent each epoch
            if step > 0 and done[0][0]:
                envs.close()
                envs = env_helpers.make_envs(config, how_train, args.seed,
                                             args.game_state_file, training_agents,
                                             num_stack, dagger_num_processes, args.render)
                agent_obs = torch.from_numpy(envs.reset()).float().squeeze(0)
                if args.cuda:
                    agent_obs = agent_obs.cuda()

            expert_obs = envs.get_expert_obs()
            expert_obs = expert_obs[0][0]
            expert_action = expert.act(expert_obs, action_space=action_space)
            expert_action_tensor = torch.LongTensor(1)
            expert_action_tensor[0] = expert_action

            agent_states_list.append(agent_obs.squeeze(0))
            expert_actions_list.append(expert_action_tensor)

            # TODO: debug - figure out if expert_obs (for expert) is the same with current_obs (for agent)
            if random.random() <= expert_prob:
                # take action provided by expert
                list_expert_action = []
                list_expert_action.append(expert_action)
                arr_expert_action = np.array(list_expert_action)
                obs, reward, done, info = envs.step(arr_expert_action)
                expert_act_arr.append(expert_action)
            else:
                # take action provided by learning policy
                result = agent.dagger_act(
                    Variable(agent_obs, volatile=True),
                    Variable(dummy_states, volatile=True),
                    Variable(dummy_masks, volatile=True))
                value, action, action_log_prob, states = result
                cpu_actions = action.data.squeeze(1).cpu().numpy()
                cpu_actions_agents = cpu_actions
                obs, reward, done, info = envs.step(cpu_actions_agents)

                agent_act_arr.append(cpu_actions_agents)

            agent_obs = torch.from_numpy(obs).float().squeeze(0)
            if args.cuda:
                agent_obs = agent_obs.cuda()

        # NOTE: figure which num_processes want to use here
        total_steps += num_processes * num_steps

        #########
        # Train using DAGGER (with supervision from the expert)
        #########
        agent.set_train()

        if len(aggregate_agent_states) >= 50000:
            indices_replace = np.arange(0, len(aggregate_agent_states)) \
                                .tolist()
            random.shuffle(indices_replace)
            for k in range(args.num_steps):
                indice = indices_replace[k]
                aggregate_agent_states[indice] = agent_states_list[k]
                aggregate_expert_actions[indice] = expert_actions_list[k]
        else:
            aggregate_agent_states += agent_states_list
            aggregate_expert_actions += expert_actions_list

        indices = np.arange(0, len(aggregate_agent_states)).tolist()
        random.shuffle(indices)

        agent_states_minibatch = torch.FloatTensor(
            args.minibatch_size, obs_shape[0], obs_shape[1], obs_shape[2])
        expert_actions_minibatch = torch.FloatTensor(
            args.minibatch_size,  obs_shape[0], obs_shape[1], obs_shape[2])
        action_losses = []
        num_opts = 0
        for i in range(0, len(aggregate_agent_states), args.minibatch_size):
            indices_minibatch = indices[i: i + args.minibatch_size]
            agent_states_minibatch = [aggregate_agent_states[k]
                                      for k in indices_minibatch]
            expert_actions_minibatch = [aggregate_expert_actions[k]
                                        for k in indices_minibatch]

            agent_states_minibatch = torch.stack(agent_states_minibatch, 0)
            expert_actions_minibatch = torch.stack(expert_actions_minibatch, 0)

            if args.cuda:
                agent_states_minibatch = agent_states_minibatch.cuda()
                expert_actions_minibatch = expert_actions_minibatch.cuda()

            action_scores = agent.get_action_scores(
                Variable(agent_states_minibatch),
                Variable(dummy_states),
                Variable(dummy_masks))
            action_loss = cross_entropy_loss(
                action_scores, Variable(expert_actions_minibatch.squeeze(1)))

            agent.optimize(action_loss, args.max_grad_norm)
            action_losses.append(action_loss.data[0])

            if num_opts < 3:
                print("Opt Step %d: action_loss %.5f." % (num_opts,
                                                          action_loss))
            num_opts += 1

        # TODO: What are the right hyperparams to use?
        # TODO: Should we optimize multiple times each epoch? On same data?

        num_epoch_steps = len(aggregate_agent_states) // args.minibatch_size
        num_optim_steps = (num_epoch + 1) * num_epoch_steps
        print("###########")
        print("epoch {} steps {} action loss mean {:.3f} / std {:.3f}" \
              .format(num_epoch, num_epoch_steps, np.mean(action_losses),
                      np.std(action_losses)))
        print("###########\n")

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

        ######
        # Eval the current policy
        ######
        if num_epoch % args.log_interval == 0:
            st = time.time()
            dagger_obs = torch.from_numpy(eval_envs.reset()).float().squeeze(0).squeeze(1)
            if args.cuda:
                dagger_obs = dagger_obs.cuda()
            while running_num_episodes < args.num_steps_eval:
                result = agent.dagger_act(Variable(dagger_obs, volatile=True), \
                                            Variable(dummy_states_eval, volatile=True), \
                                            Variable(dummy_masks_eval, volatile=True))
                _, actions, _, _ = result
                cpu_actions = actions.data.squeeze(1).cpu().numpy()
                cpu_actions_agents = cpu_actions
                obs, reward, done, info = eval_envs.step(cpu_actions_agents)

                dagger_obs = torch.from_numpy(obs.reshape(num_processes, *obs_shape)).float()
                if args.cuda:
                    dagger_obs = dagger_obs.cuda()

                running_num_episodes += sum([1 if done_ else 0 for done_ in done])
                terminal_reward += reward[done.squeeze() == True].sum()
                success_rate += sum([1 if x else 0 for x in
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

                if args.render:
                    eval_envs.render()

            cumulative_reward = 1.0 * cumulative_reward / running_num_episodes
            terminal_reward = 1.0 * terminal_reward / running_num_episodes
            success_rate = 1.0 * success_rate / running_num_episodes

            end = time.time()
            steps_per_sec = 1.0 * total_steps / (end - start)
            epochs_per_sec = 1.0 * num_epoch / (end - start)

            et = time.time()
            print("TIME {} for {} num_evals & {} num_episodes".format(et - st, args.num_steps_eval, running_num_episodes))
            print("###########")
            print("Epoch {}, # steps: {} SPS {} EPS {} \n success rate {} mean final reward {} mean total reward {} ".format(num_epoch, \
                    len(aggregate_agent_states), steps_per_sec, epochs_per_sec, success_rate, terminal_reward, cumulative_reward))
            print("###########\n")

            utils.log_to_tensorboard_dagger(writer, num_epoch, total_steps, np.mean(action_losses), \
                                            cumulative_reward, success_rate, terminal_reward)

            running_num_episodes = 0
            cumulative_reward = 0
            terminal_reward = 0
            success_rate = 0

        if success_rate >= 0.24:   # early stopping when performance is same with SimpleAgent
            break


    writer.close()

if __name__ == "__main__":
    train()
