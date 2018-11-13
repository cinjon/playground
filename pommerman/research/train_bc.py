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

Grid:
python train_bc.py --traj-directory-bc /home/roberta/playground/trajectories/grid/4maps/ \
--run-name a --how-train bc --minibatch-size 5000 --num-steps 5000 --num-stack 1 \
--num-steps-eval 500 --config GridWalls-v4 --how-train dagger --num-processes 1 \
--num-channels 5

Pomme:
python train_bc.py --traj-directory-bc /home/roberta/playground/trajectories/pomme/4maps \
--run-name a --how-train bc --minibatch-size 5000 --num-steps 5000 \
--num-steps-eval 500 --config PommeFFAEasy-v0 --how-train dagger --num-processes 4
--num-stack 1 --num-channels 19 \

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
    agent = training_agents[args.expert_id]

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

    envs = env_helpers.make_train_envs(
        config, how_train, args.seed, args.game_state_file, training_agents,
        num_stack, num_processes, state_directory=args.state_directory,
        state_directory_distribution=args.state_directory_distribution,
        step_loss=args.step_loss, bomb_reward=args.bomb_reward,
        item_reward=args.item_reward)



    #################################################
    # Load Trajectories (State, Action)-Pairs from File
    #################################################
    states, actions = envs.get_states_actions_json(args.traj_directory_bc)
    expert_states = states[0]
    expert_actions = actions[0]

    agent_obs_lst = []
    for s in expert_states:
        agent_obs_lst.append(torch.from_numpy(envs.observation(s)[0]).squeeze(0).float())

    expert_obs_lst = []
    for a in expert_actions:
        expert_obs_lst.append(int(a[args.expert_id]))

    import pdb; pdb.set_trace()


    # agent_obs_lst = agent_obs.tolist()
    # agent_actions_lst = agent_actions.tolist()
    # expert_actions_lst = expert_actions.tolist()

    indices = np.arange(0, len(agent_obs_lst)).tolist()

    dummy_states = torch.zeros(agent_obs.shape[0],1)
    dummy_masks = torch.zeros(agent_obs.shape[0],1)
    if args.cuda:
        dummy_states = dummy_states.cuda()
        dummy_masks = dummy_masks.cuda()

    cross_entropy_loss = torch.nn.CrossEntropyLoss()


    #################################################
    # Train Policy using Behavioral Cloning
    #################################################

    start = time.time()
    agent.set_train()
    for num_epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        if num_epoch > 0:
            print("Avg Epoch Time: %.3f (%d)" % ((epoch_start_time - start)*1.0/num_epoch, num_epoch))

        # result = agent.act_on_data(
        #             Variable(agent_obs, volatile=True),
        #             Variable(dummy_states, volatile=True),
        #             Variable(dummy_masks, volatile=True))
        # _, actions, _, _, _, _ = result
        # agent_actions = actions.data.squeeze(1).cpu().numpy()

        random.shuffle(indices)
        action_losses = []
        value_losses = []
        for i in range(0, len(agent_obs_lst), args.minibatch_size):
            import pdb; pdb.set_trace()
            indices_minibatch = indices[i:i + args.minibatch_size]
            agent_obs_mb = [agent_obs_lst[k] for k in indices_minibatch]
            # agent_actions_mb = [agent_actions_lst[k] for k in indices_minibatch]
            expert_actions_mb = [expert_actions_lst[k] for k in indices_minibatch]

            agent_obs_mb = torch.stack(agent_obs_mb, 0)
            # agent_actions_mb = torch.from_numpy(np.array(agent_actions_mb).squeeze(1))
            expert_actions_mb = torch.from_numpy(np.array(expert_actions_mb).squeeze(1))

            if args.cuda:
                agent_obs_mb = agent_obs_mb.cuda()
                # agent_actions_mb = agent_actions_mb.cuda()
                expert_actions_mb = expert_actions_mb.cuda()

            values, action_scores = agent.get_values_action_scores(
                Variable(agent_obs_mb),
                Variable(dummy_states).detach(),
                Variable(dummy_masks).detach())
            action_loss = cross_entropy_loss(
                action_scores, Variable(expert_actions_mb))
            value_loss = (values - values) \
                            .pow(2).mean()
            # value_loss = (Variable(returns_minibatch) - values) \
            #                 .pow(2).mean()

            agent.optimize(action_loss, value_loss, args.max_grad_norm, \
                           use_value_loss=args.use_value_loss,
                           stop_grads_value=args.stop_grads_value,
                           add_nonlin=args.add_nonlin_valhead)

            action_losses.append(action_loss.data[0])
            value_losses.append(value_loss.data[0])


    if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval):
        utils.save_agents("dagger-", num_epoch, training_agents,
                          total_steps, num_episodes, args)




if __name__ == "__main__":
    train()
