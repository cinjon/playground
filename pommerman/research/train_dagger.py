"""Train script for dagger learning.

Currently only works for single processor and uses SimpleAgent as the expert.

NOTE: Run this with how-train dagger so that the agent's position is random
among the four possibilities.
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
    assert(num_processes == 1), "Doesn't work on more than one process."

    obs_shape, action_space = env_helpers.get_env_shapes(config, num_stack)
    num_training_per_episode = utils.validate_how_train(how_train, num_agents)

    training_agents = utils.load_agents(
        obs_shape, action_space, num_training_per_episode, args,
        dagger_agent.DaggerAgent)
    agent = training_agents[0]


    #####
    # Logging helpers.
    suffix = "{}.ht-{}.cfg-{}.m-{}.lr-{}-.mb-{}.prob-{}.anneal-{}.seed-{}.pt" \
             .format(args.run_name, args.how_train, config, args.model_str,
                     args.lr, args.minibatch_size, args.expert_prob,
                     args.anneal_expert_prob, args.seed)
    log_dir = os.path.join(args.log_dir, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # TODO: Remove the num_processes here.
    writer = SummaryWriter(log_dir)
    episode_rewards = torch.zeros([num_training_per_episode,
                                   num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode,
                                 num_processes, 1])

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

    start = time.time()

    action_loss_mean = 0  # NOTE: no need to initialize - only needed when using eval-only part to log to tensorboard
    for num_epoch in range(start_epoch, num_epochs):
        # NOTE: moved envs inside the loop so that you get dif init position for the dagger agent each epoch
        envs = env_helpers.make_envs(config, how_train, args.seed,
                                     args.game_state_file, training_agents,
                                     num_stack, num_processes, args.render)
        agent_obs = torch.from_numpy(envs.reset()).float().squeeze(0)
        if args.cuda:
            agent_obs = agent_obs.cuda()

        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval):
            utils.save_agents("dagger", num_epoch, training_agents,
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
                obs, reward, done, info = envs.step(arr_expert_action) # obs: 1x1x36x13x13
                expert_act_arr.append(expert_action)
            else:
                # take action provided by learning policy
                result = agent.dagger_act(Variable(agent_obs, volatile=True), \
                                            Variable(dummy_states, volatile=True), \
                                            Variable(dummy_masks, volatile=True))
                value, action, action_log_prob, states = result
                cpu_actions = action.data.squeeze(1).cpu().numpy()
                cpu_actions_agents = cpu_actions
                obs, reward, done, info = envs.step(cpu_actions_agents)   # obs: 1x1x36x13x13

                agent_act_arr.append(cpu_actions_agents)

            agent_obs = torch.from_numpy(obs).float().squeeze(0)
            if args.cuda:
                agent_obs = agent_obs.cuda()

        total_steps += num_processes * num_steps

        #########
        # Train using DAGGER (with supervision from the expert)
        #########
        agent.set_train()

        if len(aggregate_agent_states) >= 50000:
            indices_replace = np.arange(0, len(aggregate_agent_states)).tolist()
            random.shuffle(indices_replace)
            for k in range(args.num_steps):
                aggregate_agent_states[indices_replace[k]] = agent_states_list[k]
                aggregate_expert_actions[indices_replace[k]] = expert_actions_list[k]
        else:
            aggregate_agent_states += agent_states_list
            aggregate_expert_actions += expert_actions_list

        indices = np.arange(0, len(aggregate_agent_states)).tolist()
        random.shuffle(indices)

        agent_states_minibatch = torch.FloatTensor(args.minibatch_size, obs_shape[0], obs_shape[1], obs_shape[2])
        expert_actions_minibatch = torch.FloatTensor(args.minibatch_size,  obs_shape[0], obs_shape[1], obs_shape[2])
        total_action_loss = 0
        for i in range(0, len(aggregate_agent_states), args.minibatch_size):
            indices_minibatch = indices[i: i + args.minibatch_size]
            agent_states_minibatch = [aggregate_agent_states[k] for k in indices_minibatch]
            expert_actions_minibatch = [aggregate_expert_actions[k] for k in indices_minibatch]

            agent_states_minibatch = torch.stack(agent_states_minibatch, 0)
            expert_actions_minibatch = torch.stack(expert_actions_minibatch, 0)

            if args.cuda:
                agent_states_minibatch = agent_states_minibatch.cuda()
                expert_actions_minibatch = expert_actions_minibatch.cuda()

            action_scores = agent.get_action_scores(Variable(agent_states_minibatch), Variable(dummy_states), Variable(dummy_masks))
            action_loss = cross_entropy_loss(action_scores, Variable(expert_actions_minibatch.squeeze(1)))

            agent.optimize(action_loss, args.max_grad_norm)
            total_action_loss += action_loss


        # TODO: figure out what are the right hyperparams to use: num-steps, minibatch-size etc.
        # TODO: figure out whether you need to optimize multilpe times each epoch? on same data?

        action_loss_mean = total_action_loss.data[0]/len(aggregate_agent_states)
        print("###########")
        print("epoch {}, # steps: {} action loss {} ".format(num_epoch, len(aggregate_agent_states), action_loss_mean))
        print("###########\n")

        if len(agent_act_arr) > 0:
            agent_mean_act_prob = [len([i for i in agent_act_arr if i == k])*1.0/len(agent_act_arr) for k in range(6)]
            for k in range(6):
                print("mean act {} probs {}".format(k, agent_mean_act_prob[k]))
            print("")

        if len(expert_act_arr) > 0:
            expert_mean_act_prob = [len([i for i in expert_act_arr if i == k])*1.0/len(expert_act_arr) for k in range(6)]
            for k in range(6):
                print("expert mean act {} probs {}".format(k, expert_mean_act_prob[k]))
            print("")

        expert_act_arr = []
        agent_act_arr = []

        ######
        # Eval the current policy
        ######
        if num_epoch % args.log_interval == 0:
            success_rate = 0
            final_reward_mean = 0
            total_reward_mean = 0
            for k in range(args.num_steps_eval):
                dagger_obs = torch.from_numpy(envs.reset().reshape(1, *obs_shape)).float()
                if args.cuda:
                    dagger_obs = dagger_obs.cuda()

                total_rewards = torch.zeros([num_training_per_episode, num_processes, 1])
                success = 0
                done = [[False]]
                while done[0][0] == False:
                    # take action provided by learning policy
                    # TODO: change the below to make it deterministic - need change in the dagger_act fct
                    result = agent.dagger_act(Variable(dagger_obs, volatile=True), \
                                                Variable(dummy_states, volatile=True), \
                                                Variable(dummy_masks, volatile=True))
                    _, action, _, _ = result
                    cpu_actions = action.data.squeeze(1).cpu().numpy()
                    cpu_actions_agents = cpu_actions
                    obs, reward, done, info = envs.step(cpu_actions_agents)   # obs: 1x1x36x13x13

                    dagger_obs = torch.from_numpy(obs.reshape(1, *obs_shape)).float()
                    if args.cuda:
                        dagger_obs = dagger_obs.cuda()

                    reward = utils.torch_numpy_stack(reward).transpose(0, 1)
                    episode_rewards += reward
                    masks = torch.FloatTensor([
                        [0.0]*num_training_per_episode if done_ \
                        else [1.0]*num_training_per_episode
                    for done_ in done])
                    total_rewards *= masks
                    total_rewards += (1 - masks) * episode_rewards
                    episode_rewards *= masks

                    if args.render:
                        envs.render()

                # NOTE: can add this when eval-only / rendering
                # print("**** episode {} ***** ".format(k))
                # print("win          {} ".format(success))
                # print("final reward {} ".format(reward[0][0]))
                # print("total reward {} \n".format(total_rewards[0][0][0]))

                if reward[0][0] > 0:
                    success_rate += success
                final_reward_mean += reward[0][0]
                total_reward_mean += total_rewards[0][0][0]

            final_reward_mean = final_reward_mean / args.num_steps_eval
            total_reward_mean = total_reward_mean / args.num_steps_eval
            success_rate = success_rate / args.num_steps_eval

            end = time.time()
            steps_per_sec = 1.0 * total_steps / (end - start)
            epochs_per_sec = 1.0 * num_epoch / (end - start)

            print("###########")
            print("Epoch {}, # steps: {} SPS {} EPS {} \n success rate {} mean final reward {} mean total reward {} ".format(num_epoch, \
                    len(aggregate_agent_states), steps_per_sec, epochs_per_sec, success_rate, final_reward_mean,  total_reward_mean))
            print("###########\n")

            utils.log_to_tensorboard_dagger(writer, num_epoch, total_steps, action_loss_mean, \
                                            total_reward_mean, success_rate, final_reward_mean)

    writer.close()

if __name__ == "__main__":
    train()
