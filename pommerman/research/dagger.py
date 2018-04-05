from collections import defaultdict
import copy
import glob
import os
import time
import sys
import random

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
import numpy as np
from pommerman import configs
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils

from arguments import get_args
from envs import make_env
from model import PommeResnetPolicy, PommeCNNPolicySmall

import ppo_agent
from subproc_vec_env import SubprocVecEnvRender

from pommerman.agents import SimpleAgent

args = get_args()

print("uses cuda: ", args.cuda)

# num_updates = number of samples collected in one round of updates.
# num_steps = number of steps in a rollout (horizon)
# num_processes = number of parallel processes/workers collecting data.
# number of samples used for a round of updates = horizon * num_workers = num_steps_rollout * num_parallel_processes
num_updates = int(args.num_frames) // args.num_steps // args.num_processes
print("NUM UPDATES {} num frames {} num steps {} num processes {}".format(
    num_updates, args.num_frames, args.num_steps, args.num_processes), "\n")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

def main():
    torch.cuda.empty_cache()
    os.environ['OMP_NUM_THREADS'] = '1'
    assert(args.run_name)
    assert(args.num_processes == 1), "Doesn't work with multi-processing yet."

    dummy_env = make_env(config=args.config, how_train='dummy', seed=None,
                         rank=-1, game_state_file=args.game_state_file,
                         training_agents=[], num_stack=args.num_stack)()
    envs_shape = dummy_env.observation_space.shape[1:]
    obs_shape = (envs_shape[0], *envs_shape[1:])
    action_space = dummy_env.action_space
    if args.model == 'convnet':
        actor_critic = lambda saved_model: PommeCNNPolicySmall(
            obs_shape[0], action_space, args)
    elif args.model == 'resnet':
        actor_critic = lambda saved_model: PommeResnetPolicy(
            obs_shape[0], action_space, args)

    # We need to get the agent = config.agent(agent_id, config.game_type) and
    # then pass that agent into the agent.PPOAgent
    training_agents = []
    saved_models = args.saved_models
    saved_models = saved_models.split(',') if saved_models \
                   else [None]*args.nagents
    assert(len(saved_models)) == args.nagents
    for saved_model in saved_models:
        # TODO: implement the model loading.
        model = actor_critic(saved_model)
        agent = ppo_agent.PPOAgent(model)
        training_agents.append(agent)

    # Simple trains a single agent against three SimpleAgents.
    assert(args.nagents == 1), "Simple training should have one agent."
    num_training_per_episode = 1


    suffix = "{}.train.ht-{}.cfg-{}.m-{}.seed-{}".format(
        args.run_name, args.how_train, args.config, args.model, args.seed)
    writer = SummaryWriter(os.path.join(args.log_dir, suffix))

    # NOTE: I didn't think that this should work because I thought that the
    # agent object would overwrite the agent_id on each env. Somehow it is
    # working though. This might be due to the separate threads, but that
    # doesn't make sense to me. TODO: Look into why.
    envs = [
        make_env(config=args.config, how_train=args.how_train, seed=args.seed,
                 rank=rank, game_state_file=args.game_state_file,
                 training_agents=training_agents, num_stack=args.num_stack)
        for rank in range(args.num_processes)
    ]
    envs = SubprocVecEnvRender(envs) if args.render else SubprocVecEnv(envs)

    for agent in training_agents:
        agent.initialize(args, obs_shape, action_space,
                         num_training_per_episode)

    # NOTE: only use one process so removed the second dim = num_processes
    current_obs = torch.zeros(num_training_per_episode, *obs_shape)
    def update_current_obs(obs):
        current_obs = torch.from_numpy(obs).float()

    def torch_numpy_stack(value):
        return torch.from_numpy(np.stack([x.data for x in value])).float()

    obs = update_current_obs(envs.reset())
    training_agents[0].update_rollouts(obs=current_obs, timestep=0)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_training_per_episode,
                                   args.num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode,
                                 args.num_processes, 1])

    final_dist_entropies = torch.zeros([num_training_per_episode,
                                 args.num_processes, 1])

    count_stats = defaultdict(int)
    array_stats = defaultdict(list)

    if args.cuda:
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()

    # TODO: Set the total_steps count when you load the model.
    start = time.time()
    time_counts = defaultdict(list)

    num_episodes = 0
    running_num_episodes = 0
    final_action_losses = [[] for agent in range(len(training_agents))]
    final_value_losses =  [[] for agent in range(len(training_agents))]
    final_dist_entropies = [[] for agent in range(len(training_agents))]

    aggr_agent_states = []
    aggr_expert_actions = []
    aggr_agent_masks = []

    cross_entropy_loss = nn.CrossEntropyLoss()
    expert = SimpleAgent()
    expert_action_tensor = torch.LongTensor(1)
    action_loss = 0

    for j in range(num_updates):
        # NOTE: do we still need the set_eval? since we got rid of the batchnorm?
        for agent in training_agents:
            agent.set_eval()

        agent_states = []
        expert_actions = []

        ########
        # Collect data using DAGGER
        ########
        for step in range(args.num_steps):
            expert_obs = envs.get_expert_obs()[0][0]
            expert_action = expert.act(expert_obs, action_space=action_space)
            expert_action_tensor[0] = expert_action

            agent_states.append(current_obs.squeeze(0))
            expert_actions.append(expert_action_tensor)

            # take action provided by expert
            if random.random() <= args.expert_prob:
                list_expert_action = []
                list_expert_action.append(expert_action)
                arr_expert_action = np.array(list_expert_action)
                obs, reward, done, info = envs.step(arr_expert_action)

            # take action provided by learning policy
            else:
                result = training_agents[0].run(step, 0, use_act=True)
                value, action, action_log_prob, states = result
                cpu_actions = action.data.squeeze(1).cpu().numpy()
                cpu_actions_agents = cpu_actions
                obs, reward, done, info = envs.step(cpu_actions_agents)

            update_current_obs(obs)

            if args.render:
                envs.render()

        #########
        # Train using DAGGER (with supervision from the expert)
        #########
        for agent in training_agents:
            agent.set_train()

        aggr_agent_states += agent_states
        aggr_expert_actions += expert_actions

        num_steps_dagger = len(aggr_agent_states)
        indices = np.arange(0, num_steps_dagger).tolist()
        random.shuffle(indices)

        dummy_hidden_state = Variable(torch.zeros(1,1))
        dummy_mask = Variable(torch.zeros(1,1))
        agent_states_minibatch = torch.FloatTensor(args.minibatch_size, obs_shape[0], obs_shape[1], obs_shape[2])
        expert_actions_minibatch = torch.FloatTensor(args.minibatch_size, obs_shape[0], obs_shape[1], obs_shape[2])
        indices_minibatch = torch.LongTensor(args.minibatch_size, obs_shape[0], obs_shape[1], obs_shape[2])
        total_action_loss = 0
        for i in range(0, num_steps_dagger, args.minibatch_size):
            indices_minibatch = indices[i: i + args.minibatch_size]
            agent_states_minibatch = [aggr_agent_states[k] for k in indices_minibatch]
            expert_actions_minibatch = [aggr_expert_actions[k] for k in indices_minibatch]

            agent_states_minibatch = torch.stack(agent_states_minibatch, 0)
            expert_actions_minibatch = torch.stack(expert_actions_minibatch, 0)

            if args.cuda:
                agent_states_minibatch = agent_states_minibatch.cuda()
                expert_actions_minibatch = expert_actions_minibatch.cuda()

            action_scores = agent.get_action_scores(Variable(agent_states_minibatch), dummy_hidden_state, dummy_mask)
            action_loss = cross_entropy_loss(action_scores, Variable(expert_actions_minibatch.squeeze(1)))
            agent.optimize_dagger(action_loss, args.max_grad_norm)
            total_action_loss += action_loss


        # TODO: figure out what are the right hyperparams to use: num-steps, minibatch-size etc.
        # TODO: figure out whether you need to optimize multilpe times each update? on same data?


        print("###########")
        print("update {}, # steps: {} action loss {} ".format(j, num_steps_dagger, total_action_loss.data[0]/num_steps_dagger))
        print("###########\n")

        # TODO: figure out a way to run it on multiple processes

        ########
        # Evaluate Current Policy
        ########
        for agent in training_agents:
            agent.set_eval()

        for step in range(args.num_steps_eval):
            step_time = time.time()

            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            cpu_actions_agents = []

            result = training_agents[0].run(step, 0, use_act=True)

            value, action, action_log_prob, states = result
            value_agents.append(value)
            action_agents.append(action)
            action_log_prob_agents.append(action_log_prob)
            states_agents.append(states)
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            cpu_actions_agents = cpu_actions

            obs, reward, done, info = envs.step(cpu_actions_agents)

            # TODO: Change this when we use heterogenous.
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

            if args.render:
                envs.render()
            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            episode_rewards += reward

            # NOTE: if how-train simple always has num_training_per_episode = 1
            # then we don't need the conditions below
            running_num_episodes += sum([1 if done_ else 0
                                         for done_ in done])
            masks = torch.FloatTensor([
                [0.0]*num_training_per_episode if done_ \
                else [1.0]*num_training_per_episode
            for done_ in done])

            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            if args.cuda:
                masks = masks.cuda()

            reward_all = reward.unsqueeze(2)
            masks_all = masks.transpose(0,1).unsqueeze(2)

            current_obs *= masks_all.unsqueeze(2)

            states_all = torch_numpy_stack(states_agents)
            action_all = torch_numpy_stack(action_agents)
            action_log_prob_all = torch_numpy_stack(action_log_prob_agents)
            value_all = torch_numpy_stack(value_agents)

            training_agents[0].insert_rollouts(
                step, current_obs, states_all, action_all, action_log_prob_all,
                value_all, reward_all, masks_all)


        # TODO: This is relative to the loaded model if exists.
        total_steps = (j + 1) * args.num_processes * args.num_steps

        #####
        # Save model.
        #####
        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            for num_agent, agent in enumerate(training_agents):
                save_model = agent.get_model()
                save_optimizer = agent.get_optimizer()
                save_dict = {
                    'epoch': j,
                    'arch': args.model,
                    'state_dict': save_model.state_dict(),
                    'optimizer' : save_optimizer.state_dict(),
                    'total_steps': num_steps_dagger,
                }
                save_dict['args'] = vars(args)
                suffix = "{}.train.ht-{}.cfg-{}.m-{}.nc-{}.lr-{}.epoch-{}.steps-{}.seed-{}.pt" \
                         .format(args.run_name, args.how_train, args.config,
                                 args.model, args.num_channels, args.lr, j, num_steps_dagger, args.seed)
                torch.save(save_dict, os.path.join(save_path, suffix))

        #####
        # Log to console and to Tensorboard.
        #####
        if running_num_episodes > args.log_interval:
            end = time.time()
            num_steps_sec = (end - start)
            total_steps = (j + 1) * args.num_processes * args.num_steps
            num_episodes += running_num_episodes

            steps_per_sec = 1.0 * total_steps / (end - start)
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

            print("Updates {}, total num steps {}, action classif loss {}, num timesteps {}, FPS {}, mean/median reward "
                  "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, avg entropy "
                  "{:.5f}, avg value loss {:.5f}, avg policy loss {:.5f}".
                format(j, num_steps_dagger,
                       total_action_loss,
                       total_steps,
                       steps_per_sec,
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(),
                       mean_dist_entropy,
                       mean_value_loss,
                       mean_action_loss))

            writer.add_scalar('action_classif_loss_per_steps', total_action_loss, num_steps_dagger)
            writer.add_scalar('action_classif_loss_per_updates', total_action_loss, j)

            writer.add_scalars('entropy', {
                'mean': mean_dist_entropy,
                'var': std_dist_entropy,
            }, num_steps_dagger)

            writer.add_scalars('action_loss', {
                'mean': mean_action_loss,
                'var': std_action_loss,
            }, num_steps_dagger)

            writer.add_scalars('value_loss', {
                'mean': mean_value_loss,
                'var': std_value_loss,
            }, num_steps_dagger)

            writer.add_scalars('rewards', {
                'mean': final_rewards.mean(),
                'std': final_rewards.std(),
                'median': final_rewards.median(),
            }, num_steps_dagger)

            writer.add_scalar('updates', j, num_steps_dagger)
            writer.add_scalar('steps_per_sec', steps_per_sec, num_steps_dagger)
            writer.add_scalar('episodes_per_sec', episodes_per_sec, num_steps_dagger)

            for title, count in count_stats.items():
                if title.startswith('bomb:'):
                    continue
                writer.add_scalar(title, 1.0 * count / running_num_episodes,
                                  num_steps_dagger)

            writer.add_scalars('bomb_distances', {
                key.split(':')[1]: 1.0 * count / running_num_episodes
                for key, count in count_stats.items() \
                if key.startswith('bomb:')
            }, num_steps_dagger)

            if array_stats['rank']:
                writer.add_scalars('rank', {
                    'mean': np.mean(array_stats['rank']),
                    'std': np.std(array_stats['rank']),
                }, num_steps_dagger)

            if array_stats['dead']:
                writer.add_scalars('dying_step', {
                    'mean': np.mean(array_stats['dead']),
                    'std': np.std(array_stats['dead']),
                }, num_steps_dagger)
                writer.add_scalar(
                    'percent_dying_per_ep',
                    1.0 * len(array_stats['dead']) / running_num_episodes,
                    num_steps_dagger)

            # always reset these stats so that means in the plots are per the last log_interval
            final_action_losses = [[] for agent in range(len(training_agents))]
            final_value_losses =  [[] for agent in range(len(training_agents))]
            final_dist_entropies = [[] for agent in range(len(training_agents))]
            count_stats = defaultdict(int)
            array_stats = defaultdict(list)
            final_rewards = torch.zeros([num_training_per_episode, args.num_processes, 1])
            running_num_episodes = 0

    writer.close()

if __name__ == "__main__":
    main()
