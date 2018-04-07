import os
import subprocess

import numpy as np
import torch
import torch.nn as nn

import networks


def load_agents(obs_shape, action_space, num_training_per_episode, args,
                agent_type):
    actor_critic = lambda state: networks.get_actor_critic(args.model_str)(
        state, obs_shape[0], action_space, args.board_size, args.num_channels)

    paths = args.saved_paths
    if not type(paths) == list:
        paths = paths.split(',') if paths else [None] * args.num_agents
            
    assert(len(paths)) == args.num_agents

    training_agents = []
    for path in paths:
        if path:
            print("Loading path %s as agent." % path)
            loaded_model = torch.load(path)
            model_state_dict = loaded_model['state_dict']
            optimizer_state_dict = loaded_model['optimizer']
            num_episodes = loaded_model['num_episodes']
            total_steps = loaded_model['total_steps']
            num_epoch = loaded_model['num_epoch']
            model = actor_critic(model_state_dict)
        else:
            num_episodes = 0
            total_steps = 0
            num_epoch = 0
            optimizer_state_dict = None
            model = actor_critic(None)

        agent = agent_type(model)
        agent.initialize(args, obs_shape, action_space,
                         num_training_per_episode, num_episodes, total_steps,
                         num_epoch, optimizer_state_dict)
        training_agents.append(agent)
    return training_agents


def is_save_epoch(num_epoch, start_epoch, save_interval):
    if num_epoch % save_interval != 0:
        return False
    return num_epoch == 0 or num_epoch != start_epoch


def save_agents(prefix, num_epoch, training_agents, total_steps, num_episodes,
                args):
    """Save the model.

    Args:
      prefix: A prefix string to prepend to the run_name.
      num_epoch: The int epoch.
      training_agents: The agent classes being trained.
      total_steps: The current number of steps.
      num_episodes: The number of episodes thus far.
      args: The args from arguments.py
    """
    name = prefix + args.run_name
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
                 .format(name, how_train, config, model_str, num_agent,
                         num_epoch, total_steps, seed)
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


def get_train_vars(args):
    how_train = args.how_train
    config = args.config
    num_agents = args.num_agents
    num_stack = args.num_stack
    num_steps = args.num_steps
    num_processes = args.num_processes
    num_epochs = int(args.num_frames // num_steps // num_processes)
    print("NumEpochs {} NumFrames {} NumSteps {} NumProcesses {} Cuda {}\n"
          .format(num_epochs, args.num_frames, num_steps, num_processes,
                  args.cuda))
    return how_train, config, num_agents, num_stack, num_steps, \
        num_processes, num_epochs


def log_to_console(num_epoch, num_episodes, total_steps, steps_per_sec,
                   final_rewards, mean_dist_entropy, mean_value_loss,
                   mean_action_loss):
    print("Epochs {}, num episodes {}, num timesteps {}, FPS {}, epochs "
          "per sec {}, mean/median reward {:.1f}/{:.1f}, min/max reward "
          "{:.1f}/{:.1f}, avg entropy {:.5f}, avg value loss {:.5f}, avg "
          "policy loss {:.5f}"
          .format(num_epoch, num_episodes, total_steps, steps_per_sec,
                  epochs_per_sec, final_rewards.mean(),
                  final_rewards.median(), final_rewards.min(),
                  final_rewards.max(), mean_dist_entropy, mean_value_loss,
                  mean_action_loss))


def log_to_tensorboard(writer, num_epoch, num_episodes, total_steps,
                       steps_per_sec, final_rewards, mean_dist_entropy,
                       mean_value_loss, mean_action_loss, std_dist_entropy,
                       std_value_loss, std_action_loss, count_stats,
                       array_stats, running_num_episodes):
    writer.add_scalar('entropy', {
        'mean' : mean_dist_entropy,
        'std_max': mean_dist_entropy + std_dist_entropy,
        'std_min': mean_dist_entropy - std_dist_entropy,
    }, num_episodes)

    writer.add_scalar('reward', {
        'mean': final_rewards.mean(),
        'std_max': final_rewards.mean() + final_rewards.std(),
        'std_min': final_rewards.mean() - final_rewards.std(),
    }, num_episodes)

    writer.add_scalars('action_loss', {
        'mean': mean_action_loss,
        'std_max': mean_action_loss + std_action_loss,
        'std_min': mean_action_loss - std_action_loss,
    }, num_episodes)

    writer.add_scalars('value_loss', {
        'mean': mean_value_loss,
        'std_max': mean_value_loss + std_value_loss,
        'std_min': mean_value_loss - std_value_loss,
    }, num_episodes)

    writer.add_scalar('epochs', num_epoch, num_episodes)
    writer.add_scalar('steps_per_sec', steps_per_sec, num_episodes)
    writer.add_scalar('episodes_per_sec', episodes_per_sec, num_episodes)

    for title, count in count_stats.items():
        if title.startswith('bomb:'):
            continue
        writer.add_scalar(title, 1.0 * count / running_num_episodes,
                          num_episodes)

    writer.add_scalars('bomb_distances', {
        key.split(':')[1]: 1.0 * count / running_num_episodes
        for key, count in count_stats.items() \
        if key.startswith('bomb:')
    }, num_episodes)

    if array_stats.get('rank'):
        writer.add_scalar('mean_rank', np.mean(array_stats['rank']),
                          num_episodes)

    if array_stats.get('dead'):
        writer.add_scalar('mean_dying_step', np.mean(array_stats['dead']),
                          num_episodes)
        writer.add_scalar(
            'percent_dying_per_episode',
            1.0 * len(array_stats['dead']) / running_num_episodes,
            num_episodes)


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
