'''
Train script for behavioral cloning
(using saved trajectories)

The --use-value-loss setting makes it so that a value predicting network
is also trained with supervision (to avoid policy overwritting by
value loss gradients when fine-tuning using PPO).

The --stop-grads-value setting stops the gradients from the value loss in going
through the rest of the shared params of the network (to avoid changing the
policy since we want the policy to imitate the trajectory as well as possible).

The --add-nonlin-valhead addsa a nonlinearity in the valuehead so that it has
more expressive power.

All the above args default to false.

Example Run:

python train_bc.py --traj-directory-bc /home/roberta/playground/trajectories/pomme/4maps \
--num-processes 16 --run-name a --how-train bc --minibatch-size 5000 --num-steps 5000 \
--num-steps-eval 500 --config PommeFFAEasy-v0 --num-processes 4 --how-train dagger \
--num-processes 1

'''

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
import networks
import utils

def train():
    os.environ['OMP_NUM_THREADS'] = '1'

    args = get_args()
    assert(args.run_name)
    print("\n###############")
    print("args ", args)
    print("##############\n")

    if args.cuda:
        torch.cuda.empty_cache()

    num_training_per_episode = utils.validate_how_train(args)
    how_train, config, num_agents, num_stack, num_steps, num_processes, \
        num_epochs, reward_sharing, batch_size, num_mini_batch = \
        utils.get_train_vars(args, num_training_per_episode)

    obs_shape, action_space, character, board_size = env_helpers.get_env_info(config, num_stack)

    training_agents = utils.load_agents(
        obs_shape, action_space, num_training_per_episode, num_steps, args,
        agent_type=dagger_agent.DaggerAgent, character=character, board_size=board_size)
    agent = training_agents[0]

    #####
    # Logging helpers.
    suffix = "{}.{}.{}.{}.nc{}.lr{}.mb{}.nopt{}.traj-{}.seed{}.pt" \
             .format(args.run_name, args.how_train, config, args.model_str,
                     args.num_channels, args.lr, args.minibatch_size,
                     args.dagger_epoch, args.traj_directory_bc, args.seed)

    log_dir = os.path.join(args.log_dir, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    start_epoch = agent.num_epoch
    total_steps = agent.total_steps
    num_episodes = agent.num_episodes

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        agent.cuda()

    states = []
    expert_actions = []
    actions = []
    returns = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    envs = env_helpers.make_train_envs(
        config, how_train, args.seed, args.game_state_file, training_agents,
        num_stack, num_processes, state_directory=args.state_directory,
        state_directory_distribution=args.state_directory_distribution,
        step_loss=args.step_loss, bomb_reward=args.bomb_reward,
        item_reward=args.item_reward)

    dummy_states = torch.zeros(1,1)
    dummy_masks = torch.zeros(1,1)
    dummy_states_eval = torch.zeros(num_processes, 1)
    dummy_masks_eval = torch.zeros(num_processes, 1)
    agent_obs = torch.from_numpy(envs.reset()).float().squeeze(1)
    if args.cuda:
        dummy_states = dummy_states.cuda()
        dummy_masks = dummy_masks.cuda()
        dummy_states_eval = dummy_states_eval.cuda()
        dummy_masks_eval = dummy_masks_eval.cuda()
        agent_obs = agent_obs.cuda()

    episode_rewards = torch.zeros([num_training_per_episode,
                                   num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode,
                                 num_processes, 1])

    running_num_episodes = 0
    cumulative_reward = 0
    terminal_reward = 0
    success_rate = 0

    done = np.array([[False]])

    agent_act_arr = []
    expert_act_arr = []

    start = time.time()
    for num_epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        if num_epoch > 0:
            print("Avg Epoch Time: %.3f (%d)" % ((epoch_start_time - start)*1.0/num_epoch, num_epoch))

        agent.set_eval()
        agent_states_list = [[] for _ in range(num_processes)]
        expert_actions_list = [[] for _ in range(num_processes)]
        returns_list = [[] for _ in range(num_processes)]

        #################################################
        # Load Data from File
        #################################################
        states, actions = envs.get_states_actions_json(args.traj_directory_bc)
        expert_states = states[0]
        expert_actions = actions[0]

        #################################################
        # Train Policy using Behavioral Cloning
        #################################################


if __name__ == "__main__":
    train()
