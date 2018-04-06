import os
import subprocess

import numpy as np
import torch
import torch.nn as nn

import networks
import ppo_agent


def load_agents(obs_shape, action_space, board_size, num_channels, config,
                num_stack, num_agents, num_training_per_episode, model_str,
                paths, lr, eps, num_steps, num_processes):
    actor_critic = lambda state_dict: networks.get_actor_critic(model_str)(
        state_dict, obs_shape[0], action_space, board_size, num_channels)

    if not type(paths) == list:
        paths = paths.split(',') if paths else [None]*num_agents
            
    assert(len(paths)) == num_agents

    training_agents = []
    for path in paths:
        if path:
            loaded_model = torch.load(path)
            model_state_dict = loaded_model['state_dict']
            optimizer_state_dict = loaded_model['optimizer']
            num_episodes = loaded_model['num_episodes']
            total_steps = loaded_model['total_steps']
            num_epoch = loaded_model['num_epoch']
            model = actor_critic(state_dict)
        else:
            num_episodes = 0
            total_steps = 0
            num_epoch = 0
            optimizer_state_dict = None
            model = actor_critic(None)

        agent = ppo_agent.PPOAgent(model)
        agent.initialize(lr, eps, num_steps, num_processes, num_epoch, obs_shape,
                         action_space, num_training_per_episode, num_episodes,
                         total_steps, optimizer_state_dict)
        training_agents.append(agent)
    return training_agents


def save_agents(num_epoch, training_agents, total_steps, num_episodes, args):
    """Save the model.

    Args:
      num_epoch: The int epoch.
      training_agents: The agent classes being trained.
      total_steps: The current number of steps.
      num_episodes: The number of episodes thus far.
      run_name: The name to save this under.
      model_str: The name of the model we are using.
      seed: The int seed we are using.
    """
    run_name = args.run_name
    config = args.config
    how_train = args.how_train
    model_str = args.model_str
    seed = args.seed
    save_dir = args.save_dir
    if not save_dir:
        return

    try:
        os.makedirs(save_dir)
    except OSError:
        pass

    for num_agent, agent in enumerate(training_agents):
        model = agent.model
        optimizer = agent.optimizer
        save_dict = {
            'num_epoch': num_epoch,
            'model_str': model_str,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'total_steps': total_steps,
            'num_episodes': num_episodes,
        }
        save_dict['args'] = vars(args)
        suffix = "{}.ht-{}.cfg-{}.m-{}.num-{}.epoch-{}.steps-{}.seed-{}.pt" \
                 .format(run_name, how_train, config,
                         model_str, num_agent, num_epoch, total_steps, seed)
        torch.save(save_dict, os.path.join(save_dir, suffix))


def scp_model_from_cims(saved_paths, cims_address, cims_password,
                        cims_save_model_local):
    try:
        assert(cims_password)
        cims_model_address = ":".join([cims_address, saved_paths])
        local_model_address = os.path.join(
            cims_save_model_local, saved_paths.split('/')[-1])
        subprocess.call(['sshpass', '-p', '%s' % cims_password, 'scp',
                         cims_model_address, local_model_address])
        return local_model_address
    except Exception as e:
        return None


def validate_how_train(how_train, nagents):
    if how_train == 'simple':
        # Simple trains a single agent against three SimpleAgents.
        assert(nagents == 1), "Simple training should have one agent."
        return 1
    elif how_train == 'homogenous':
        # Homogenous trains a single agent against itself (self-play).
        assert(nagents == 1), "Homogenous training should have one agent."
        return 4
    elif how_train == 'heterogenous':
        s = "Heterogenous training should have multiple agents."
        assert(nagents > 1), s
        print("Heterogenous training is not implemented yet.")
        return None


def torch_numpy_stack(value):
    return torch.from_numpy(np.stack([x.data for x in value])).float()
