"""Eval script. Merge this in with main.py's training.

Evaluation pipeline PPO v. 3 SimpleAgents in FFA and
for self-play PPO in FFA and for two different versions
(newer v. 3 older versions of the policy).

Examples:

python eval.py --cims-save-model-local ~/Code/selfplayground/models \
 --cims-password $CIMSP --cims-address $CIMSU \
 --saved-models /path/to/model.pt --num-channels 128
 
python eval.py --num-channels 128 --saved-models /path/to/model.pt
"""

import copy
import glob
import os
import subprocess
import sys
import time

import gym
from pommerman import configs
import numpy as np
import torch
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from model import PommeResnetPolicy, PommeCNNPolicySmall
import ppo_agent
from subproc_vec_env import SubprocVecEnvRender

args = get_args()

torch.manual_seed(args.seed)


def eval():
    os.environ['OMP_NUM_THREADS'] = '1'
    saved_models = args.saved_models
    assert(saved_models), "Please include saved_models path."
    assert(len(saved_models.split(",")) == 1), "Only one model please."

    # Instantiate the environment
    config = getattr(configs, args.config)()

    if args.cims_address:
        assert(args.cims_password)
        cims_model_address = ":".join([args.cims_address, saved_models])
        local_model_address = os.path.join(
            args.cims_save_model_local, saved_models.split('/')[-1])
        subprocess.call(['sshpass', '-p', '%s' % args.cims_password,
                         'scp', cims_model_address, local_model_address])
    else:
        local_model_address = saved_models

    # We make this in order to get the shapes.
    dummy_agent = config['agent'](game_type=config['game_type'])
    dummy_agent = ppo_agent.PPOAgent(dummy_agent, None)
    dummy_env = make_env(args, config, -1, [dummy_agent])()
    envs_shape = dummy_env.observation_space.shape[1:]
    obs_shape = (envs_shape[0], *envs_shape[1:])
    action_space = dummy_env.action_space
    if args.model == 'convnet':
        actor_critic_model = PommeCNNPolicySmall(
            obs_shape[0], action_space, args)
        actor_critic_lambda = lambda saved_model: actor_critic_model
    elif args.model == 'resnet':
        actor_critic_model = PommeResnetPolicy(
            obs_shape[0], action_space, args)
        actor_critic_lambda = lambda saved_model: actor_critic_model

    # TODO: this only works for simple - need a list of checkpoints for self-play
    # We need to get the agent = config.agent(agent_id, config.game_type) and then
    # pass that agent into the agent.PPOAgent
    training_agents = []

    print("****")
    model_name = local_model_address.split('/')[-1]
    loaded_model = torch.load(local_model_address)
    state_dict = loaded_model['state_dict']
    print("epoch for {} is: {}".format(model_name, loaded_model['epoch']))
    # TODO: Make this cleaner. It's silly rn with the lambda above.
    actor_critic_model.load_state_dict(state_dict)
    agent = config['agent'](game_type=config['game_type'])
    agent = ppo_agent.PPOAgent(agent, actor_critic_model)
    training_agents = [agent]
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
    envs = [make_env(args, config, i, training_agents)
            for i in range(args.num_processes)]
    envs = SubprocVecEnvRender(envs)

    for agent in training_agents:
        agent.initialize(args, obs_shape, action_space,
                         num_training_per_episode)

    current_obs = torch.zeros(num_training_per_episode, args.num_processes,
                              *obs_shape)
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
    episode_rewards = torch.zeros([num_training_per_episode,
                                   args.num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode,
                                 args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()

    for agent in training_agents:
        agent.set_eval()

    start = time.time()
    eval_steps = args.num_steps_eval

    for j in range(eval_steps):
        for step in range(args.num_steps):
            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            cpu_actions_agents = []

            if args.how_train == 'simple':
                result = training_agents[0].run(step, 0, use_act=True)
                value, action, action_log_prob, states = result
                value_agents.append(value)
                action_agents.append(action)
                action_log_prob_agents.append(action_log_prob)
                states_agents.append(states)
                cpu_actions = action.data.squeeze(1).cpu().numpy()
                cpu_actions_agents = cpu_actions
            elif args.how_train == 'homogenous':
                cpu_actions_agents = [[] for _ in range(args.num_processes)]
                for i in range(4):
                    result = training_agents[0].run(step=step, num_agent=i,
                                                    use_act=True)
                    value, action, action_log_prob, states = result
                    value_agents.append(value)
                    action_agents.append(action)
                    action_log_prob_agents.append(action_log_prob)
                    states_agents.append(states)
                    cpu_actions = action.data.squeeze(1).cpu().numpy()
                    for num_process in range(args.num_processes):
                        cpu_actions_agents[num_process].append(cpu_actions[num_process])

            obs, reward, done, info = envs.step(cpu_actions_agents)

            envs.render()
            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            episode_rewards += reward


            # NOTE: if how-train simple always has num_training_per_episode = 1
            # then we don't need the conditions below
            if args.how_train == 'simple':
                if num_training_per_episode == 1:
                    masks = torch.FloatTensor([
                        [0.0]*num_training_per_episode if done_ \
                        else [1.0]*num_training_per_episode
                    for done_ in done])
                else:
                    masks = torch.FloatTensor(
                        [[[0.0] if done_[i] else [1.0]
                          for i in range(len(done_))] for done_ in done])
            elif args.how_train == 'homogenous':
                masks = torch.FloatTensor(
                    [[[0.0] if done_[i] else [1.0] for i in range(len(done_))]
                     for done_ in done]).transpose(0,1).unsqueeze(2)

            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            if args.cuda:
                masks = masks.cuda()

            reward_all = reward.unsqueeze(2)
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


# test against prior version
if __name__ == "__main__":
    eval()
