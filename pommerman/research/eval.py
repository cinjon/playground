"""Eval script.

Evaluation pipeline:
1. Receives a set of paths to saved models.
2. Told which one is being tested.
3. If there are less than four paths given, then it will fill the rest of the
   agents with SimpleAgent.
4. Runs 100 episodes. 
5. Records and logs the number of wins in that episode for the target agent.

Examples:

python eval.py --cims-save-model-local ~/Code/selfplayground/models \
 --cims-password $CIMSP --cims-address $CIMSU \
 --saved-models /path/to/model.pt --num-channels 128
 
python eval.py --num-channels 128 --saved-models /path/to/model.pt
"""

import copy
import glob
import os
import sys
import time

import gym
from pommerman import configs
import numpy as np
import torch
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import envs as env_helpers
from model import PommeCNNPolicySmall
import ppo_agent
from subproc_vec_env import SubprocVecEnvRender


torch.manual_seed(args.seed)


def eval():
    os.environ['OMP_NUM_THREADS'] = '1'
    args = get_args()

    target_paths = args.target_eval_paths.split(',')
    assert(target_paths), "Please include target_paths."
    saved_paths = args.saved_paths
    assert(saved_paths), "Please include saved_paths."

    for path in target_paths:
        assert(path in saved_paths), "Path %s not in saved_paths." % path

    new_saved_paths = []
    for path in saved_paths.split(','):
        if os.path.exists(path):
            new_saved_paths.append(path)
        else:
            new_saved_paths.append(
                utils.scp_model_from_cims(
                    saved_paths, args.cims_address,
                    args.cims_password, args.cims_save_model_local)
            )

    obs_shape, action_space = env_helpers.get_env_shapes(config, num_stack)
    # TODO: This doesn't quite work the same as in train ... Needs adjustment.
    num_training_per_episode = utils.validate_how_train(args.how_train,
                                                        args.num_agents)

    training_agents = utils.load_agents(
        obs_shape, action_space, args.board_size, args.num_channels, config,
        num_stack, num_training_per_episode, args.model_str, new_saved_paths,
        args.lr, args.eps, num_steps, num_processes)

    training_agents = utils.load_agents(obs_shape, action_space,
                                        args.board_size, args.num_channels,
                                        args.config, args.num_stack,
                                        args.model, local_model_address,
                                        args.lr, args.eps, num_steps,
                                        num_processes)
    model_name = local_model_address.split('/')[-1]
    print("num episodes for {} is: {}".format(model_name,
                                              training_agents[0].num_episodes)

    envs = utils.make_envs(args.config, args.how_train, args.seed,
                           args.game_state_file, training_agents,
                           args.num_stack, num_processes, args.render)

    def update_current_obs(obs):
        current_obs = torch.from_numpy(obs).float()
    current_obs = torch.zeros(num_training_per_episode, num_processes,
                              *obs_shape)
    update_current_obs(envs.reset())

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

            states_all = utils.torch_numpy_stack(states_agents)
            action_all = utils.torch_numpy_stack(action_agents)
            action_log_prob_all = utils.torch_numpy_stack(action_log_prob_agents)
            value_all = utils.torch_numpy_stack(value_agents)

            training_agents[0].insert_rollouts(
                step, current_obs, states_all, action_all, action_log_prob_all,
                value_all, reward_all, masks_all)


# test against prior version
if __name__ == "__main__":
    eval()
