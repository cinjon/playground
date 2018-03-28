'''
Evaluation pipeline PPO v. 3 SimpleAgents in FFA and
for self-play PPO in FFA and for two different versions
(newer v. 3 older versions of the policy).
'''

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
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model import CNNPolicy, MLPPolicy, PommeCNNPolicy, PommeResnetPolicy, PommeCNNPolicySmall
import ppo_agent
from visualize import visdom_plot


args = get_args()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)



# XXX: decide whether I want to make two functions or  for testing against prior version and against simple agent
def eval_pomme(saved_models='train=simple-config=ffa_v0-model=convnet-agent=0.pt'):
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom(server=args.server, port=8097)
        # viz = Visdom(port=args.port)
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


    # TODO: this only works for simple - need a list of checkpoints for self-play
    # We need to get the agent = config.agent(agent_id, config.game_type) and then
    # pass that agent into the agent.PPOAgent
    training_agents = []

    # TODO: this is a bit hacky and doesn't work for more than 1 model
    # saved_models = args.saved_models

    save_path = os.path.join(args.save_dir)
    saved_models = [os.path.join(save_path, saved_models)]
    # saved_models = saved_models.split(',') if saved_models else [None]*args.nagents

    assert(len(saved_models)) == args.nagents
    if len(envs_shape) == 3:
        if args.model == 'convnet':
            actor_critic_model = PommeCNNPolicySmall(obs_shape[0], action_space, args)
        elif args.model == 'resnet':
            actor_critic_model = PommeResnetPolicy(obs_shape[0], action_space, args)
    else:
        assert not args.recurrent_policy, \
            "Recurrent policy is not implemented for the MLP controller"
        actor_critic_model = MLPPolicy(obs_shape[0], action_space)

    print("****")
    for saved_model in saved_models:
        # TODO: implement the model loading.
        loaded_model = torch.load(saved_model)
        print("epoch of model {} is: {}".format(saved_model, loaded_model['epoch']))
        loaded_actor_critic_model = actor_critic_model.load_state_dict(loaded_model['state_dict'])
        model = actor_critic(loaded_actor_critic_model)
        model.eval()
        agent = config['agent'](game_type=config['game_type'])
        agent = ppo_agent.PPOAgent(agent, model)
        training_agents.append(agent)
    print("****")

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
    envs = SubprocVecEnv(envs) if args.num_processes > 1 else DummyVecEnv(envs)

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

    start = time.time()
    for j in range(args.num_steps_eval):
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
            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            episode_rewards += reward


            if args.how_train == 'simple':
                masks = torch.FloatTensor([
                    [0.0]*num_training_per_episode if done_ else [1.0]*num_training_per_episode
                    for done_ in done])
            elif args.how_train == 'homogenous':
                masks = torch.FloatTensor([
                    [0.0]*num_training_per_episode if done_ else [1.0]*num_training_per_episode
                    for done_ in done]).transpose(0,1)

            masks = torch.FloatTensor([
                [0.0]*num_training_per_episode if done_ else [1.0]*num_training_per_episode
                for done_ in done]).transpose(0,1)

            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            if args.cuda:
                masks = masks.cuda()

            reward_all = reward.unsqueeze(2)

            masks_all = masks.unsqueeze(2)

            if args.how_train == 'simple':
                masks_all = masks.transpose(0,1).unsqueeze(2)
            elif args.how_train == 'homogenous':
                masks_all = masks.unsqueeze(2)

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


        if step % args.log_interval == 0:
            print("step ", step)
            end = time.time()
            total_num_steps = (step + 1) * args.num_processes * args.num_steps_eval
            final_rewards_tr = torch.zeros([args.num_processes, args.nagents, 1])
            final_rewards_tr.copy_(final_rewards)
            final_rewards_tr = final_rewards_tr.view(args.num_processes, args.nagents).transpose(0, 1)
            for i in range(args.nagents):
                print("agent # ", i)
                print("Updates {}, Agent {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
                    format(step, i, total_num_steps,
                           int(total_num_steps / (end - start)),
                           final_rewards_tr[i].mean(),
                           final_rewards_tr[i].median(),
                           final_rewards_tr[i].min(),
                           final_rewards_tr[i].max()),
                           "\n")
            print("\n")

        if args.vis and step % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name)
            except IOError:
                pass


# test against prior version
if __name__ == "__main__":
    eval_pomme()
