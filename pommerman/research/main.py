import copy
import glob
import os
import time
import sys

import gym
from pommerman import configs
import numpy as np
import torch
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from model import CNNPolicy, MLPPolicy, PommeCNNPolicy, PommeResnetPolicy, \
    PommeCNNPolicySmall
import ppo_agent
from subproc_vec_env import SubprocVecEnvRender
from visualize import visdom_plot

import utils

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

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    # Instantiate the environment
    config = getattr(configs, args.config)()

    # We make this in order to get the shapes.
    dummy_env = make_env(args, config, -1, [config['agent'](game_type=config['game_type'])])()
    envs_shape = dummy_env.observation_space.shape[1:]
    obs_shape = (envs_shape[0], *envs_shape[1:])
    action_space = dummy_env.action_space
    if len(envs_shape) == 3:
        if args.model == 'convnet':
            actor_critic = lambda saved_model: PommeCNNPolicySmall(obs_shape[0], action_space, args)
        elif args.model == 'resnet':
            actor_critic = lambda saved_model: PommeResnetPolicy(obs_shape[0], action_space, args)
    else:
        actor_critic = lambda saved_model: MLPPolicy(obs_shape[0], action_space)

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

    # NOTE: Does this work correctly? Will the threads operate independently?
    envs = [make_env(args, config, i, training_agents) for i in range(args.num_processes)]
    if args.render:
        envs = SubprocVecEnvRender(envs)
    else: 
        envs = SubprocVecEnv(envs)

    # TODO: Figure out how to render this for testing purposes. The following link may help:
    # https://github.com/MG2033/A2C/blob/master/envs/subproc_vec_env.py

    for agent in training_agents:
        agent.initialize(args, obs_shape, action_space, num_training_per_episode)

    current_obs = torch.zeros(num_training_per_episode, args.num_processes, *obs_shape)
    def update_current_obs(obs):
        current_obs = torch.from_numpy(obs).float()

    obs = envs.reset()
    update_current_obs(obs)
    if args.how_train == 'simple':
        training_agents[0].update_rollouts(obs=current_obs, timestep=0)
    elif args.how_train == 'homogenous':
        training_agents[0].update_rollouts(obs=current_obs, timestep=0)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_training_per_episode, args.num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode, args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()

    stats = utils.init_stats(args)
    start = time.time()
    for j in range(num_updates):
        for agent in training_agents:
            agent.eval()

        for step in range(args.num_steps):
            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            cpu_actions_agents = []

            if args.how_train == 'simple':
                value, action, action_log_prob, states = training_agents[0].act_pytorch(step, 0)
                value_agents.append(value)
                action_agents.append(action)
                action_log_prob_agents.append(action_log_prob)
                states_agents.append(states)
                cpu_actions = action.data.squeeze(1).cpu().numpy()
                cpu_actions_agents = cpu_actions
            elif args.how_train == 'homogenous':
                cpu_actions_agents = [[] for _ in range(args.num_processes)]
                for i in range(4):
                    value, action, action_log_prob, states = training_agents[0].act_pytorch(step, i)
                    value_agents.append(value)
                    action_agents.append(action)
                    action_log_prob_agents.append(action_log_prob)
                    states_agents.append(states)
                    cpu_actions = action.data.squeeze(1).cpu().numpy()
                    for num_process in range(args.num_processes):
                        cpu_actions_agents[num_process].append(cpu_actions[num_process])

            obs, reward, done, info = envs.step(cpu_actions_agents)
            if args.render:
                envs.render()
            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            episode_rewards += reward


            # import pdb; pdb.set_trace()
            if args.how_train == 'simple':
                masks = torch.FloatTensor([
                    [0.0]*num_training_per_episode if done_ else [1.0]*num_training_per_episode
                    for done_ in done])
            elif args.how_train == 'homogenous':
                masks = torch.FloatTensor([
                    [0.0]*num_training_per_episode if done_ else [1.0]*num_training_per_episode
                    for done_ in done]).transpose(0,1).unsqueeze(2)

            print("REWARD / DONE / MASKS: ", reward, done, masks.squeeze())
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

            current_obs *= masks_all.unsqueeze(2).unsqueeze(2)
            update_current_obs(obs)

            states_all = torch.from_numpy(np.stack([x.data for x in states_agents])).float()
            action_all = torch.from_numpy(np.stack([x.data for x in action_agents])).float()
            action_log_prob_all = torch.from_numpy(np.stack([x.data for x in action_log_prob_agents])).float()
            value_all = torch.from_numpy(np.stack([x.data for x in value_agents])).float()

            if args.how_train in ['simple', 'homogenous']:
                training_agents[0].insert_rollouts(
                    step, current_obs, states_all, action_all, action_log_prob_all,
                    value_all, reward_all, masks_all)

        next_value_agents = []
        if args.how_train == 'simple':
            agent = training_agents[0]
            next_value_agents.append(agent.run_actor_critic(-1, 0))
            advantages = [agent.compute_advantages(next_value_agents, args.use_gae, args.gamma, args.tau)]
        elif args.how_train == 'homogenous':
            agent = training_agents[0]
            next_value_agents = [agent.run_actor_critic(-1, num_agent) for num_agent in range(4)]
            advantages = [agent.compute_advantages(next_value_agents, args.use_gae, args.gamma, args.tau)]

        final_action_losses = []
        final_value_losses = []
        final_dist_entropies = []

        for agent in training_agents:
            agent.train()

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
                torch.save(
                {
                    'epoch': j,
                    'arch': args.model,
                    'state_dict': save_model.state_dict(),
                    'optimizer' : save_optimizer.state_dict(),

                },
                os.path.join(save_path, "train={}-config={}-model={}-agent={}.pt".format(args.how_train, args.config, args.model, num_agent)))


        #####
        # Log to console.
        #####
        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, avg entropy {:.5f}, avg value loss {:.5f}, avg policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(),
                       np.mean([dist_entropy.data[0] for dist_entropy in final_dist_entropies]),
                       np.mean([value_loss.data[0] for value_loss in final_value_losses]),
                       np.mean([action_loss.data[0] for action_loss in final_action_losses])))

            # save stats to h5 file
            # TODO: need to fix this error
            # stats = utils.log_stats(args, stats, j, int(total_num_steps / (end - start)), \
            #     final_rewards.mean(), final_rewards.median(), final_rewards.min(), final_rewards.max(), \
            #     np.mean([action_loss.data[0] for action_loss in final_action_losses]), \
            #     np.mean([value_loss.data[0] for value_loss in final_value_losses]), \
            #     np.mean([dist_entropy.data[0] for dist_entropy in final_dist_entropies]))
            #
            # log_path = os.path.join(args.log_dir)
            # filename_stats = '%s/stats.h5' % log_path
            # utils.save_dict(filename_stats, stats)



        #####
        # Log to Visdom.
        #####
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, 'ppo')
            except IOError:
                pass

if __name__ == "__main__":
    main()
