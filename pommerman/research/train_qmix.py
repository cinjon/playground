"""
Train script for qmix learning.

The number of samples used for an epoch is:
horizon * num_workers = num_steps * num_processes where num_steps is the number
of steps in a rollout (horizon) and num_processes is the number of parallel
processes/workers collecting data.

Example:

python train_qmix.py --num-processes 8 --how-train qmix \
    --save-dir $TRAINED_MODELS_DIR --log-dir $LOG_DIR --log-interval 5 --save-interval 1000 \
    --run-name qmix --seed 1 --config PommeTeam-v0 --num-agents 2 --model-str QMIXNet \
    --num-steps 1000
"""

from collections import defaultdict
import os
import time

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.nn import MSELoss
from torch.autograd import Variable

from storage import EpisodeBuffer
from arguments import get_args
import envs as env_helpers
import qmix_agent
import utils
import pommerman


def train():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.cuda:
        torch.cuda.empty_cache()
        torch.cuda.set_device(args.cuda_device)

    assert args.run_name
    print("\n###############")
    print("args ", args)
    print("##############\n")

    how_train, config, num_agents, num_stack, num_steps, num_processes, \
        num_epochs, reward_sharing = utils.get_train_vars(args)

    obs_shape, action_space = env_helpers.get_env_shapes(config, num_stack)
    global_obs_shape = (4, obs_shape[0] // 2, *obs_shape[1:]) # @TODO a better way?
    num_training_per_episode = utils.validate_how_train(how_train, num_agents)

    training_agents = utils.load_agents(
        obs_shape, action_space, num_training_per_episode, args,
        qmix_agent.QMIXMetaAgent, network_type='qmix')
    envs = env_helpers.make_train_envs(config, how_train, args.seed,
                                       args.game_state_file, training_agents,
                                       num_stack, num_processes)

    #####
    # Logging helpers.
    suffix = args.run_name
    log_dir = os.path.join(args.log_dir, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    episode_buffer = EpisodeBuffer(size=5000)

    # TODO: When we implement heterogenous, change this to be per agent.
    start_epoch = training_agents[0].num_epoch
    total_steps = training_agents[0].total_steps
    num_episodes = training_agents[0].num_episodes

    # Initialize observations
    current_obs = envs.reset()
    current_global_obs = envs.get_global_obs()
    training_ids = envs.get_training_ids()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        for agent in training_agents:
            agent.cuda()

    def init_history_instance():
        history_init = [
            torch.zeros(0, *global_obs_shape),  # current global state
            torch.zeros(0, num_training_per_episode, *obs_shape),  # current agent(s) state
            # torch.zeros(0, num_training_per_episode).long(),  # action
            torch.zeros(0, num_training_per_episode),  # reward
            torch.zeros(0, *global_obs_shape),  # next global state
            torch.zeros(0, num_training_per_episode, *obs_shape),  # next agent(s) state
            # torch.zeros(0, num_training_per_episode).long(),  # done
        ]
        return history_init

    ##
    # Each history is Python list is of length num_processes. This is a list because not all
    # episodes are of the same length and we don't want information of an episode
    # after it has ended. Each history item has the first dimension as time step, second
    # dimension as the number of training agents and then appropriate shape for the item
    #
    history = [init_history_instance() for _ in range(num_processes)]
    eps = args.eps_max
    eps_steps = 0
    gradient_steps = 0

    # Collected Statistics
    start = time.time()
    running_team_wins = 0
    running_team_ties = 0
    running_num_episodes = 0
    running_mean_episode_length = 0
    value_losses = []

    def compute_q_loss(global_state, state, reward, next_global_state, next_state):
        if args.cuda:
            global_state = global_state.cuda()
            state = state.cuda()
            reward = reward.cuda()
            next_global_state = next_global_state.cuda()
            next_state = next_state.cuda()

        current_q_values, _ = training_agents[0].act(
            Variable(global_state, requires_grad=True),
            Variable(state, requires_grad=True))
        max_next_q_values, _ = training_agents[0].target_act(
            Variable(next_global_state, volatile=True),
            Variable(next_state, volatile=True))
        max_next_q_values = max_next_q_values.max(1)[0]
        # sum the rewards for individual agents
        expected_q_values = reward.sum(dim=1) + args.gamma * max_next_q_values.data

        loss = MSELoss()(current_q_values, Variable(expected_q_values))
        loss.backward()

        value_losses.append(loss.cpu().data[0])

    def run_dqn():
        if len(episode_buffer) >= args.episode_batch:
            for episode in episode_buffer.sample(args.episode_batch):
                compute_q_loss(*episode)
            training_agents[0].optimizer_step()
            return 1
        return 0

    for num_epoch in range(start_epoch, num_epochs):
        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval):
            utils.save_agents("qmix-", num_epoch, training_agents, total_steps,
                              num_episodes, args, suffix)

        for agent in training_agents:
            agent.set_eval()

        # Keep collecting episodes
        for _ in range(num_steps):
            eps = args.eps_max + (args.eps_min - args.eps_max) / args.eps_max_steps * eps_steps

            current_global_obs_tensor = torch.FloatTensor(current_global_obs)
            current_obs_tensor = torch.FloatTensor(current_obs)
            if args.cuda:
                current_global_obs_tensor = current_global_obs_tensor.cuda()
                current_obs_tensor = current_obs_tensor.cuda()

            # Ignore critic values during trajectory generation (decentralized actors)
            _, actions = training_agents[0].act(
                Variable(current_global_obs_tensor, volatile=True),
                Variable(current_obs_tensor, volatile=True),
                eps=eps)

            training_agent_actions = actions.cpu().data.numpy().tolist()
            obs, reward, done, info = envs.step(training_agent_actions)
            global_obs = envs.get_global_obs()
            reward = reward.astype(np.float)

            if args.render:
                envs.render()

            global_state_tensor = current_global_obs_tensor.cpu().unsqueeze(1)
            state_tensor = current_obs_tensor.cpu().unsqueeze(1)
            reward_tensor = torch.from_numpy(reward).float().unsqueeze(1)
            next_global_state_tensor = torch.FloatTensor(global_obs).unsqueeze(1)
            next_state_tensor = torch.from_numpy(obs).float().unsqueeze(1)

            for i in range(num_processes):
                history[i][0] = torch.cat([history[i][0], global_state_tensor[i]], dim=0)
                history[i][1] = torch.cat([history[i][1], state_tensor[i]], dim=0)
                history[i][2] = torch.cat([history[i][2], reward_tensor[i]], dim=0)
                history[i][3] = torch.cat([history[i][3], next_global_state_tensor[i]], dim=0)
                history[i][4] = torch.cat([history[i][4], next_state_tensor[i]], dim=0)

                total_steps += 1

                if info[i]['result'] != pommerman.constants.Result.Incomplete:
                    # Update stats
                    running_num_episodes += 1
                    running_mean_episode_length = running_mean_episode_length + \
                                                  (history[i][0].size(0) - running_mean_episode_length) / running_num_episodes
                    if 'winners' in info[i] and info[i]['winners'] == training_ids[i]:
                        running_team_wins += 1
                    elif info[i]['result'] == pommerman.constants.Result.Tie:
                        running_team_ties += 1

                    # Flush completed episode into buffer and clear current episode's history for next episode
                    episode_buffer.append(history[i])
                    history[i] = init_history_instance()

                    gradient_steps += run_dqn()
                    if gradient_steps % args.target_update_steps == 0:
                        training_agents[0].update_target()

            # Update for next step
            eps_steps = min(eps_steps + 1, args.eps_max_steps)
            current_obs = obs
            current_global_obs = global_obs

        history = [init_history_instance() for _ in range(num_processes)]

        if running_num_episodes:
            end = time.time()

            num_episodes += running_num_episodes

            steps_per_sec = 1.0 * total_steps / (end - start)

            mean_value_loss = np.mean(value_losses).item()
            std_value_loss = np.std(value_losses).item()

            writer.add_scalar('num_episodes', num_episodes, num_epoch)
            writer.add_scalar('running_num_episodes', running_num_episodes , num_epoch)
            writer.add_scalar('running_mean_episode_length', running_mean_episode_length, num_epoch)
            writer.add_scalar('win_rate', (running_team_wins * 100.0 / running_num_episodes), num_epoch)
            writer.add_scalar('tie_rate', (running_team_ties * 100.0 / running_num_episodes), num_epoch)
            writer.add_scalar('mean_value_loss', mean_value_loss, num_epoch)
            writer.add_scalar('std_value_loss', std_value_loss, num_epoch)
            writer.add_scalar('steps_per_sec', steps_per_sec, num_epoch)

            # Partial Stats
            print('[{} steps/s] Num Episodes: {}, Running Team Win Rate: {}%, Mean Running Loss: {}'.format(
                  steps_per_sec, num_episodes, (running_team_wins * 100.0 / running_num_episodes), mean_value_loss))

            running_num_episodes = 0
            running_mean_episode_length = 0
            running_team_wins = 0
            running_team_ties = 0
            value_losses = []

    writer.close()


if __name__ == "__main__":
    train()
