"""Train script for dagger learning.

Currently only works for single processor and uses SimpleAgent as the expert.
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
from pommerman.agents import SimpleAgent
import utils


def train():
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.cuda.empty_cache()

    args = get_args()
    assert(args.run_name)

    how_train, config, num_agents, num_stack, num_steps, num_processes, \
        num_epochs = utils.get_train_vars(args)
    assert(num_processes == 1), "Doesn't work on more than one process."

    obs_shape, action_space = env_helpers.get_env_shapes(config, num_stack)
    num_training_per_episode = utils.validate_how_train(how_train, num_agents)

    training_agents = utils.load_agents(
        obs_shape, action_space, num_training_per_episode, args,
        dagger_agent.DaggerAgent)
    agent = training_agents[0]

    envs = env_helpers.make_envs(config, how_train, args.seed,
                                 args.game_state_file, training_agents,
                                 num_stack, num_processes, args.render)

    # NOTE: These two funcs have side effects where they set variable values.
    def update_current_obs(obs):
        current_obs = torch.from_numpy(obs).float()

    def update_stats(info):
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

    current_obs = torch.zeros(num_training_per_episode, *obs_shape)
    update_current_obs(envs.reset())

    #####
    # Logging helpers.
    suffix = "{}.train.ht-{}.cfg-{}.m-{}.seed-{}".format(
        args.run_name, how_train, config, args.model_str, args.seed)
    log_dir = os.path.join(args.log_dir, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)
    count_stats = defaultdict(int)
    array_stats = defaultdict(list)
    # TODO: Remove the num_processes here.
    episode_rewards = torch.zeros([num_training_per_episode,
                                   num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode,
                                 num_processes, 1])

    start_epoch = agent.num_epoch
    total_steps = agent.total_steps
    num_episodes = agent.num_episodes

    running_num_episodes = 0
    final_action_losses = []
    final_value_losses =  []
    final_dist_entropies = []
    #####

    # TODO: Do we need this?
    agent.update_rollouts(obs=current_obs, timestep=0)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        current_obs = current_obs.cuda()
        agent.cuda()

    aggregate_agent_states = []
    aggregate_expert_actions = []
    aggregate_agent_masks = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    expert = SimpleAgent()
    action_loss = 0

    start = time.time()
    for num_epoch in range(start_epoch, num_epochs):
        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval):
            utils.save_agents("dagger", num_epoch, training_agents,
                              total_steps, num_episodes, args)

        agent.set_eval()

        agent_states_list = []
        expert_actions_list = []

        ########
        # Collect data using DAGGER
        ########
        for step in range(args.num_steps):
            expert_obs = envs.get_expert_obs()
            # Cinjon: I think that this is going to get the first agent's obs
            # in the first environment. The latter is true because we are
            # assuming one process. However we don't want the former to be true
            # because then we're always agent 0 and start in the same place.
            expert_obs = expert_obs[0][0]
            expert_action = expert.act(expert_obs, action_space=action_space)
            expert_action_tensor = torch.LongTensor(1)
            expert_action_tensor[0] = expert_action

            agent_states_list.append(current_obs.squeeze(0))
            expert_actions_list.append(expert_action_tensor)

            # TODO: debug - figure out if expert_obs (for expert) is the same with current_obs (for agent)
            if random.random() <= args.expert_prob:
                # take action provided by expert
                list_expert_action = []
                list_expert_action.append(expert_action)
                arr_expert_action = np.array(list_expert_action)
                obs, reward, done, info = envs.step(arr_expert_action) # obs: 1x1x36x13x13
            else:
                # take action provided by learning policy
                result = agent.run(step, 0, use_act=True)
                value, action, action_log_prob, states = result
                cpu_actions = action.data.squeeze(1).cpu().numpy()
                cpu_actions_agents = cpu_actions
                obs, reward, done, info = envs.step(cpu_actions_agents)   # obs: 1x1x36x13x13

            # update current observation with the one observed by taking a step with above action
            update_current_obs(obs.reshape(1, *obs_shape))

            if args.render:
                envs.render()

        total_steps += num_processes * num_steps

        #########
        # Train using DAGGER (with supervision from the expert)
        #########
        agent.set_train()

        aggregate_agent_states += agent_states_list
        aggregate_expert_actions += expert_actions_list

        num_steps_dagger = len(aggregate_agent_states)
        indices = np.arange(0, num_steps_dagger).tolist()
        random.shuffle(indices)

        dummy_hidden_state = Variable(torch.zeros(1,1))
        dummy_mask = Variable(torch.zeros(1,1))
        agent_states_minibatch = torch.FloatTensor(args.minibatch_size, obs_shape[0], obs_shape[1], obs_shape[2])
        expert_actions_minibatch = torch.FloatTensor(args.minibatch_size,  obs_shape[0], obs_shape[1], obs_shape[2])
        indices_minibatch = torch.LongTensor(args.minibatch_size,  obs_shape[0], obs_shape[1], obs_shape[2])
        total_action_loss = 0
        for i in range(0, num_steps_dagger, args.minibatch_size):
            indices_minibatch = indices[i: i + args.minibatch_size]
            agent_states_minibatch = [aggregate_agent_states[k] for k in indices_minibatch]
            expert_actions_minibatch = [aggregate_expert_actions[k] for k in indices_minibatch]

            agent_states_minibatch = torch.stack(agent_states_minibatch, 0)
            expert_actions_minibatch = torch.stack(expert_actions_minibatch, 0)

            if args.cuda:
                agent_states_minibatch = agent_states_minibatch.cuda()
                expert_actions_minibatch = expert_actions_minibatch.cuda()
                dummy_hidden_state = dummy_hidden_state.cuda()
                dummy_mask = dummy_mask.cuda()

            action_scores = agent.get_action_scores(Variable(agent_states_minibatch), dummy_hidden_state, dummy_mask)
            action_loss = cross_entropy_loss(action_scores, Variable(expert_actions_minibatch.squeeze(1)))

            agent.optimize(action_loss, args.max_grad_norm)
            total_action_loss += action_loss

        # TODO: figure out what are the right hyperparams to use: num-steps, minibatch-size etc.
        # TODO: figure out whether you need to optimize multilpe times each epoch? on same data?

        print("###########")
        print("epoch {}, # steps: {} action loss {} ".format(num_epoch, num_steps_dagger, total_action_loss.data[0]/num_steps_dagger))
        print("###########\n")

        ########
        # Evaluate Current Policy
        ########
        agent.set_eval()

        for step in range(args.num_steps_eval):
            step_time = time.time()

            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            cpu_actions_agents = []

            result = agent.run(step, 0, use_act=True)

            value, action, action_log_prob, states = result
            value_agents.append(value)
            action_agents.append(action)
            action_log_prob_agents.append(action_log_prob)
            states_agents.append(states)
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            cpu_actions_agents = cpu_actions

            obs, reward, done, info = envs.step(cpu_actions_agents)

            update_stats(info)

            if args.render:
                envs.render()

            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            episode_rewards += reward

            running_num_episodes += sum([1 if done_ else 0 for done_ in done])
            masks = torch.FloatTensor([
                [0.0]*num_training_per_episode if done_ \
                else [1.0]*num_training_per_episode for done_ in done])

            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            if args.cuda:
                masks = masks.cuda()

            reward_all = reward.unsqueeze(2)
            masks_all = masks.transpose(0,1).unsqueeze(2)
            current_obs *= masks_all.unsqueeze(2)

            states_all = utils.torch_numpy_stack(states_agents)
            action_all = utils.torch_numpy_stack(action_agents)
            action_log_prob_all = utils.torch_numpy_stack(action_log_prob_agents)
            value_all = utils.torch_numpy_stack(value_agents)

            agent.insert_rollouts(
                step, current_obs, states_all, action_all, action_log_prob_all,
                value_all, reward_all, masks_all)

        #####
        # Log to console and to Tensorboard.
        #####
        if running_num_episodes > args.log_interval:
            end = time.time()
            num_steps_sec = (end - start)
            num_episodes += running_num_episodes

            steps_per_sec = 1.0 * total_steps / (end - start)
            epochs_per_sec = 1.0 * args.log_interval / (end - start)
            episodes_per_sec =  1.0 * num_episodes / (end - start)

            mean_dist_entropy = np.mean(final_dist_entropies)
            std_dist_entropy = np.std(final_dist_entropies)

            mean_value_loss = np.mean(final_value_losses)
            std_value_loss = np.std(final_value_losses)

            mean_action_loss = np.mean(final_action_losses)
            std_action_loss = np.std(final_action_losses)

            utils.log_to_console(num_epoch, num_episodes, total_steps,
                                 steps_per_sec, final_rewards,
                                 mean_dist_entropy, mean_value_loss)
            utils.log_to_tensorboard(writer, num_epoch, num_episodes,
                                     total_steps, steps_per_sec, final_rewards,
                                     mean_dist_entropy, mean_value_loss,
                                     mean_action_loss, std_dist_entropy,
                                     std_value_loss, std_action_loss,
                                     count_stats, array_stats,
                                     running_num_episodes)

            # Reset stats so that plots are per the last log_interval.
            final_action_losses = []
            final_value_losses =  []
            final_dist_entropies = []
            count_stats = defaultdict(int)
            array_stats = defaultdict(list)
            final_rewards = torch.zeros([num_training_per_episode,
                                         num_processes, 1])
            running_num_episodes = 0

    writer.close()

if __name__ == "__main__":
    train()
