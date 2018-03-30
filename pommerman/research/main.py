from collections import defaultdict
import copy
import glob
import os
import time
import sys

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
import numpy as np
from pommerman import configs
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import utils

from arguments import get_args
from envs import make_env
from model import PommeResnetPolicy, PommeCNNPolicySmall

import ppo_agent
from subproc_vec_env import SubprocVecEnvRender


args = get_args()

# num_updates = number of samples collected in one round of updates.
# num_steps = number of steps in a rollout (horizon)
# num_processes = number of parallel processes/workers collecting data.
# number of samples used for a round of updates = horizon * num_workers = num_steps_rollout * num_parallel_processes
num_updates = int(args.num_frames) // args.num_steps // args.num_processes
print("NUM UPDATES {} num frames {} num steps {} num processes {}".format(num_updates, args.num_frames, args.num_steps, args.num_processes), "\n")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized.")
    print("Use a monitor (see envs.py) or visdom plot to get true rewards.")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'
    assert(args.run_name)

    # Instantiate the environment
    config = getattr(configs, args.config)()

    # We make this in order to get the shapes.
    dummy_agent = config['agent'](game_type=config['game_type'])
    dummy_agent = ppo_agent.PPOAgent(dummy_agent, None)
    dummy_env = make_env(args, config, -1, [dummy_agent])()
    envs_shape = dummy_env.observation_space.shape[1:]
    obs_shape = (envs_shape[0], *envs_shape[1:])
    action_space = dummy_env.action_space
    if args.model == 'convnet':
        actor_critic = lambda saved_model: PommeCNNPolicySmall(
            obs_shape[0], action_space, args)
    elif args.model == 'resnet':
        actor_critic = lambda saved_model: PommeResnetPolicy(
            obs_shape[0], action_space, args)

    # We need to get the agent = config.agent(agent_id, config.game_type) and then
    # pass that agent into the agent.PPOAgent
    training_agents = []
    saved_models = args.saved_models
    saved_models = saved_models.split(',') if saved_models else [None]*args.nagents
    assert(len(saved_models)) == args.nagents
    for saved_model in saved_models:
        # TODO: implement the model loading.
        model = actor_critic(saved_model)
        agent = config['agent'](game_type=config['game_type'])
        agent = ppo_agent.PPOAgent(agent, model)
        training_agents.append(agent)

    if args.how_train == 'simple':
        # Simple trains a single agent against three SimpleAgents.
        assert(args.nagents == 1), "Simple training should have a single agent."
        num_training_per_episode = 1
    elif args.how_train == 'homogenous':
        # Homogenous trains a single agent against itself (self-play).
        assert(args.nagents == 1), "Homogenous toraining should have a single agent."
        num_training_per_episode = 4
    elif args.how_train == 'heterogenous':
        assert(args.nagents > 1), "Heterogenous training should have more than one agent."
        print("Heterogenous training is not implemented yet.")
        return

    suffix = "{}.train.ht-{}.cfg-{}.m-{}.event".format(
        args.run_name, args.how_train, args.config, args.model)
    writer = SummaryWriter(os.path.join(args.log_dir, suffix))

    # NOTE: Does this work correctly? Will the threads operate independently?
    envs = [make_env(args, config, i, training_agents)
            for i in range(args.num_processes)]
    if args.render:
        envs = SubprocVecEnvRender(envs)
    else:
        envs = SubprocVecEnv(envs)

    for agent in training_agents:
        agent.initialize(args, obs_shape, action_space, num_training_per_episode)

    current_obs = torch.zeros(num_training_per_episode, args.num_processes, *obs_shape)
    def update_current_obs(obs):
        current_obs = torch.from_numpy(obs).float()

    def torch_numpy_stack(value):
        return torch.from_numpy(np.stack([x.data for x in value])).float()

    obs = update_current_obs(envs.reset())

    if args.how_train == 'simple':
        training_agents[0].update_rollouts(obs=current_obs, timestep=0)
    elif args.how_train == 'homogenous':
        training_agents[0].update_rollouts(obs=current_obs, timestep=0)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_training_per_episode, args.num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode, args.num_processes, 1])
    accumulated_stats = defaultdict(int)

    if args.cuda:
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()

    # TODO: Set the total_steps count when you load the model.
    start = time.time()

    for j in range(num_updates):
        for agent in training_agents:
            agent.set_eval()

        for step in range(args.num_steps):
            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            cpu_actions_agents = []

            if args.how_train == 'simple':
                value, action, action_log_prob, states = training_agents[0].run(step, 0, use_act=True)
                value_agents.append(value)
                action_agents.append(action)
                action_log_prob_agents.append(action_log_prob)
                states_agents.append(states)
                cpu_actions = action.data.squeeze(1).cpu().numpy()
                cpu_actions_agents = cpu_actions
            elif args.how_train == 'homogenous':
                cpu_actions_agents = [[] for _ in range(args.num_processes)]
                for i in range(4):
                    value, action, action_log_prob, states = training_agents[0].run(step=step, num_agent=i, use_act=True)
                    value_agents.append(value)
                    action_agents.append(action)
                    action_log_prob_agents.append(action_log_prob)
                    states_agents.append(states)
                    cpu_actions = action.data.squeeze(1).cpu().numpy()
                    for num_process in range(args.num_processes):
                        cpu_actions_agents[num_process].append(cpu_actions[num_process])

            obs, reward, done, info = envs.step(cpu_actions_agents)

            # TODO: Change this when we use heterogenous.
            for i in info:
                for lst in i.get('step_info', {}).values():
                    for l in lst:
                        if l == 'died':
                            accumulated_stats['dead'].append(l.split(':')[1])
                        accumulated_stats[l] += 1
                        if 'bomb' in l:
                            accumulated_stats['bomb'] += 1

            if args.render:
                envs.render(q)
            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            episode_rewards += reward

            # NOTE: if how-train simple always has num_training_per_episode = 1 then we don't need the conditions below
            if args.how_train == 'simple':
                if num_training_per_episode == 1:
                    masks = torch.FloatTensor([
                        [0.0]*num_training_per_episode if done_ else [1.0]*num_training_per_episode
                    for done_ in done])
                else:
                    masks = torch.FloatTensor(
                        [[[0.0] if done_[i] else [1.0] for i in range(len(done_))] for done_ in done])

            elif args.how_train == 'homogenous':
                masks = torch.FloatTensor(
                    [[[0.0] if done_[i] else [1.0] for i in range(len(done_))] for done_ in done]).transpose(0,1).unsqueeze(2)

            # print("REWARD / DONE / MASKS: ", reward, done, masks.squeeze())
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            if args.cuda:
                masks = masks.cuda()

            reward_all = reward.unsqueeze(2)
            if args.how_train == 'simple':
                masks_all = masks.transpose(0,1).unsqueeze(2)
            elif args.how_train == 'homogenous':
                masks_all = masks

            if args.how_train == 'simple':
                current_obs *= masks_all.unsqueeze(2).unsqueeze(2)
            elif args.how_train == 'homogenous':
                current_obs *= masks_all.unsqueeze(2)
            update_current_obs(obs)

            states_all = torch_numpy_stack(states_agents)
            action_all = torch_numpy_stack(action_agents)
            action_log_prob_all = torch_numpy_stack(action_log_prob_agents)
            value_all = torch_numpy_stack(value_agents)

            training_agents[0].insert_rollouts(
                step, current_obs, states_all, action_all, action_log_prob_all,
                value_all, reward_all, masks_all)

        next_value_agents = []
        if args.how_train == 'simple':
            agent = training_agents[0]
            next_value_agents.append(agent.run(step=-1, num_agent=0))
            advantages = [agent.compute_advantages(next_value_agents, args.use_gae, args.gamma, args.tau)]
        elif args.how_train == 'homogenous':
            agent = training_agents[0]
            next_value_agents = [agent.run(step=-1, num_agent=num_agent)
                                 for num_agent in range(4)]
            advantages = [agent.compute_advantages(next_value_agents, args.use_gae, args.gamma, args.tau)]

        final_action_losses = []
        final_value_losses = []
        final_dist_entropies = []

        for agent in training_agents:
            agent.set_train()

        for num_agent, agent in enumerate(training_agents):
            for _ in range(args.ppo_epoch):
                data_generator = agent.feed_forward_generator(advantages[num_agent], args)

                for sample in data_generator:
                    observations_batch, states_batch, actions_batch, \
                        return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, states = agent.evaluate_actions(
                        Variable(observations_batch),
                        Variable(states_batch),
                        Variable(masks_batch),
                        Variable(actions_batch))

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()
                    value_loss = (Variable(return_batch) - values).pow(2).mean()
                    agent.optimize(value_loss, action_loss, dist_entropy, args.entropy_coef, args.max_grad_norm)

            final_action_losses.append(action_loss)
            final_value_losses.append(value_loss)
            final_dist_entropies.append(dist_entropy)

            agent.after_update()

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

            # XXX: new way for saving model
            # XXX: we should also add the optimizer along with the state_dict
            for num_agent, agent in enumerate(training_agents):
                save_model = agent.get_model()
                save_optimizer = agent.get_optimizer()
                save_dict = {
                    'epoch': j,
                    'arch': args.model,
                    'state_dict': save_model.state_dict(),
                    'optimizer' : save_optimizer.state_dict(),
                    'total_steps': total_steps,
                }
                save_dict['args'] = vars(args)
                suffix = "{}.train.ht-{}.cfg-{}.m-{}.num-{}.epoch-{}.steps-{}.pt" \
                         .format(args.run_name, args.how_train, args.config,
                                 args.model, num_agent, j, total_steps)
                torch.save(save_dict, os.path.join(save_path, suffix))

        #####
        # Log to console and to Tensorboard.
        #####
        if j % args.log_interval == 0:
            end = time.time()
            num_steps_sec = (end - start)
            total_steps = (j + 1) * args.num_processes * args.num_steps
            steps_per_sec = int(total_steps / (end - start))

            mean_dist_entropy = np.mean([
                dist_entropy.data[0] for dist_entropy in final_dist_entropies])
            std_dist_entropy = np.std([
                dist_entropy.data[0] for dist_entropy in final_dist_entropies])

            mean_value_loss = np.mean([
                value_loss.data[0] for value_loss in final_value_losses])
            mean_action_loss = np.mean([
                action_loss.data[0] for action_loss in final_action_losses])
            std_value_loss = np.std([
                value_loss.data[0] for value_loss in final_value_losses])
            std_action_loss = np.std([
                action_loss.data[0] for action_loss in final_action_losses])

            print("Updates {}, num timesteps {}, FPS {}, mean/median reward "
                  "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, avg entropy "
                  "{:.5f}, avg value loss {:.5f}, avg policy loss {:.5f}".
                format(j, total_steps,
                       steps_per_sec,
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(),
                       mean_dist_entropy,
                       mean_value_loss,
                       mean_action_loss))

            # TODO: Update all of this when we get model loading working.

            # TODO: make these work so that you can show mean +/- variance on the same plot
            # writer.add_scalar('entropy', {
            # 'mean_dist_entropy' : mean_dist_entropy,
            # 'var_max_dist_entropy': mean_dist_entropy + std_dist_entropy,
            # 'var_min_dist_entropy': mean_dist_entropy - std_dist_entropy,
            # }, total_steps)

            # writer.add_scalar('reward', {
            # 'mean_reward': final_rewards.mean(),
            # 'var_max_reward': final_rewards.mean() + final_rewards.std(),
            # 'var_min_reward': final_rewards.mean() - final_rewards.std(),
            # }, total_steps)

            writer.add_scalars('entropy', {
                'mean': mean_dist_entropy,
                'var': std_dist_entropy,
            }, total_steps)

            writer.add_scalars('action_loss', {
                'mean': mean_action_loss,
                'var': std_action_loss,
            }, total_steps)

            writer.add_scalars('value_loss', {
                'mean': mean_value_loss,
                'var': std_value_loss,
            }, total_steps)

            writer.add_scalars('rewards', {
                'mean': final_rewards.mean(),
                'std': final_rewards.std(),
                'median': final_rewards.median(),
            }, total_steps)

            writer.add_scalar('updates', j, total_steps)
            writer.add_scalar('steps_per_sec', steps_per_sec, total_steps)
                              
            for title, count in accumulated_stats.items():
                if title == 'dead':
                    continue
                writer.add_scalar(title, 1.0 * count / total_steps,
                                  total_steps)

            writer.add_scalars('dying_step', {
                'mean': np.mean(accumulated_stats['dead']),
                'std': np.std(accumulated_stats['dead']),
            }, total_steps)
            writer.add_scalar(
                'percent_dying',
                1.0 * len(accumulated_stats['dead']) / total_steps,
                total_steps)

    writer.close()

if __name__ == "__main__":
    main()
