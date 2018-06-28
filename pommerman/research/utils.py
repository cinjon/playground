import logging
import os
import subprocess

import numpy as np
import torch

import dagger_agent
import networks


def load_agents(obs_shape, action_space, num_training_per_episode,num_steps,
                args, agent_type, network_type='ac'):
    if network_type == 'qmix':
        net = lambda state: networks.get_q_network(args.model_str)(
            state, obs_shape[0], action_space, obs_shape[1],
            args.num_channels, args.num_agents)
    else:
        net = lambda state: networks.get_actor_critic(args.model_str)(
            state, obs_shape[0], action_space, args.board_size,
            args.num_channels, args.recurrent_policy)

    paths = args.saved_paths
    if not type(paths) == list:
        paths = paths.split(',') if paths else [None] * args.num_agents

    assert(len(paths)) == args.num_agents

    training_agents = []
    for path in paths:
        if path:
            print("Loading path %s as agent." % path)
            loaded_model = torch_load(path, args.cuda, args.cuda_device)
            model_state_dict = loaded_model['state_dict']
            model = net(model_state_dict)
            optimizer_state_dict = loaded_model['optimizer']
            if args.restart_counts:
                num_episodes = 0
                total_steps = 0
                num_epoch = 0
                uniform_v = None
                uniform_v_prior = None
            else:
                num_episodes = loaded_model['num_episodes']
                total_steps = loaded_model['total_steps']
                num_epoch = loaded_model['num_epoch']
                uniform_v = loaded_model.get('uniform_v')
                uniform_v_prior = loaded_model.get('uniform_v_prior')
        else:
            num_episodes = 0
            total_steps = 0
            num_epoch = 0
            uniform_v = None
            uniform_v_prior = None
            optimizer_state_dict = None
            model = net(None)

        agent = agent_type(model, num_stack=args.num_stack, cuda=args.cuda,
                           num_processes=args.num_processes,
                           recurrent_policy = args.recurrent_policy)
        agent.initialize(args, obs_shape, action_space,
                         num_training_per_episode, num_episodes, total_steps,
                         num_epoch, optimizer_state_dict, num_steps, uniform_v,
                         uniform_v_prior)
        training_agents.append(agent)

    return training_agents


def load_inference_agent(path, agent_type, network_type, action_space,
                         obs_shape, num_processes, args):
    print("Loading inference agent %s from path %s." % (agent_type, path))
    if network_type == 'qmix':
        net = lambda state: networks.get_q_network(args.model_str)(
            state, obs_shape[0], action_space, obs_shape[1],
            args.num_channels, args.num_agents)
    else:
        net = lambda state: networks.get_actor_critic(args.model_str)(
            state, obs_shape[0], action_space, args.board_size,
            args.num_channels, args.recurrent_policy)

    loaded_model = torch_load(path, args.cuda, args.cuda_device)
    model_state_dict = loaded_model['state_dict']
    model = net(model_state_dict)
    agent = agent_type(model, num_stack=args.num_stack, cuda=args.cuda,
                       num_processes=num_processes,
                       recurrent_policy=args.recurrent_policy)
    if args.cuda:
        agent.cuda()
    return agent

def load_distill_agent(obs_shape, action_space, args):
    model_type, path = args.distill_target.split('::')
    if model_type == 'dagger':
        print("Loading %s as distill agent." % path)
        loaded_model = torch_load(path, args.cuda, args.cuda_device)
        model_state_dict = loaded_model['state_dict']
        # TODO: Remove this hardcoded obs_shape after retraining dagger agent.
        # obs_shape = [36, 13, 13]
        model = networks.get_actor_critic(args.model_str)(
            model_state_dict, obs_shape[0], action_space, args.board_size,
            args.num_channels, args.recurrent_policy)
        return dagger_agent.DaggerAgent(model, cuda=args.cuda,
                                        num_stack=args.num_stack)
    else:
        raise ValueError("We do not support distilling from %s." % model_type)


def is_save_epoch(num_epoch, start_epoch, save_interval):
    if num_epoch % save_interval != 0:
        return False
    return num_epoch == 0 or num_epoch != start_epoch


def save_agents(prefix, num_epoch, training_agents, total_steps, num_episodes,
                args, suffix=None, uniform_v=None, uniform_v_prior=None, clear_saved=False):
    """Save the model.
    Args:
      prefix: A prefix string to prepend to the run_name.
      num_epoch: The int epoch.
      training_agents: The agent classes being trained.
      total_steps: The current number of steps.
      num_episodes: The number of episodes thus far.
      args: The args from arguments.py.
      suffix: A given suffix to call each agent's saved model.
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

    ret = []
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
            'uniform_v': uniform_v,
            'uniform_v_prior': uniform_v_prior,
        }
        save_dict['args'] = vars(args)
        if not suffix:
            if how_train == 'dagger':
                suffix = "{}.{}.{}.{}.nc{}.lr{}.bs{}.ne{}.prob{}.nopt{}.epoch{}.steps{}.seed{}.pt" \
                         .format(name, how_train, config, model_str,
                                 args.num_channels, args.lr, args.batch_size,
                                 args.num_episodes_dagger, args.expert_prob,
                                 args.dagger_epoch, num_epoch, total_steps, seed)
            else:
                suffix = "{}.{}.{}.{}.nc{}.lr{}.bs{}.ns{}.gam{}.gae{}.epoch{}.steps{}.seed{}.pt" \
                         .format(name, how_train, config, model_str,
                                 args.num_channels, args.lr, args.batch_size,
                                 args.num_steps, args.gamma, args.use_gae,
                                 num_epoch, total_steps, seed)
        else:
            suffix += ".epoch{}.steps{}".format(num_epoch, total_steps)

        if not suffix.endswith('.pt'):
            suffix += '.pt'
        save_path = os.path.join(save_dir, "agent%d-%s" % (num_agent, suffix))
        torch.save(save_dict, save_path)
    ret.append(save_path)
    return ret


def scp_model_from_ssh(saved_paths, ssh_address, ssh_password,
                       ssh_save_model_local):
    try:
        assert(ssh_password)
        ssh_model_address = ":".join([ssh_address, saved_paths])
        local_model_address = os.path.join(
            ssh_save_model_local, saved_paths.split('/')[-1])
        subprocess.call(['sshpass', '-p', '%s' % ssh_password, 'scp',
                         ssh_model_address, local_model_address])
        return local_model_address
    except Exception as e:
        return None


def get_train_vars(args, num_training_per_episode):
    how_train = args.how_train
    config = args.config
    num_agents = args.num_agents
    if args.recurrent_policy:
        assert(args.num_stack == 1), \
        "If you are using a recurrent model, than num_stack must be 1."
    num_stack = args.num_stack
    num_mini_batch = args.num_mini_batch
    batch_size = args.batch_size
    num_processes = args.num_processes
    num_steps = int(batch_size // num_training_per_episode // num_processes) + 1
    num_epochs = int(args.num_frames // num_steps // num_processes)
    reward_sharing = args.reward_sharing
    s = "NumEpochs {} NumFrames {} NumSteps {} NumProcesses {} Cuda {} " \
        "RewardSharing {} Batch Size {} Num Mini Batch {}\n" \
        .format(num_epochs, args.num_frames, num_steps, num_processes,
                args.cuda, reward_sharing, batch_size, num_mini_batch)
    print(s)
    return how_train, config, num_agents, num_stack, num_steps, \
        num_processes, num_epochs, reward_sharing, batch_size, num_mini_batch


def log_to_console(num_epoch, num_episodes, total_steps, steps_per_sec,
                   epochs_per_sec, final_rewards, mean_dist_entropy,
                   mean_value_loss, mean_action_loss,
                   cumulative_reward, terminal_reward, success_rate,
                   success_rate_alive, running_num_episodes, mean_total_loss,
                   mean_kl_loss=None, mean_pg_loss = None, distill_factor=0,
                   reinforce_only=False, start_step_ratios=None, start_step_beg_ratios=None):
    print("Epochs {}, num episodes {}, num timesteps {}, FPS {}, "
          "epochs per sec {} mean cumulative reward {:.3f} "
          "mean terminal reward {:.3f}, mean success rate {:.3f} "
          "mean success rate learning agent alive at the end {:.3f} "
          "mean final reward {:.3f}, min/max finals reward {:.3f}/{:.3f} "
          "mean total loss {:.3f}"
          .format(num_epoch, num_episodes, total_steps, steps_per_sec,
                  epochs_per_sec, 1.0*cumulative_reward/running_num_episodes,
                  1.0*terminal_reward/running_num_episodes,
                  1.0*success_rate/running_num_episodes,
                  1.0*success_rate_alive/running_num_episodes,
                  final_rewards.mean(),
                  final_rewards.min(), final_rewards.max(), mean_total_loss))

    if reinforce_only:
        print(" mean pg loss {:.3f} ".format(mean_pg_loss))
    else:
        print(" avg entropy {:.3f} avg value loss {:.3f} "
            "avg policy loss {:.3f} "
            .format(mean_dist_entropy, mean_value_loss, mean_action_loss))
    if distill_factor > 0:
        print(" mean kl loss {} ".format(mean_kl_loss))

    if start_step_ratios:
        print("Start Step Win Ratios: %s" % ", ".join(
            ["%d: %.3f" % (k, v) for k, v in sorted(start_step_ratios.items())])
        )
    if start_step_beg_ratios:
        print("Start Step Beg Win Ratios: %s" % ", ".join(
            ["%d: %.3f" % (k, v) for k, v in sorted(start_step_beg_ratios.items())])
        )


def log_to_tensorboard_dagger(writer, num_epoch, total_steps, action_loss,
                              total_reward, success_rate, final_reward,
                              value_loss):

    writer.add_scalar('final_reward_epoch', final_reward, num_epoch)
    writer.add_scalar('final_reward_steps', final_reward, total_steps)

    writer.add_scalar('total_reward_epoch', total_reward, num_epoch)
    writer.add_scalar('total_reward_steps', total_reward, total_steps)

    writer.add_scalar('action_loss_epoch', action_loss, num_epoch)
    writer.add_scalar('action_loss_steps', action_loss, total_steps)

    writer.add_scalar('value_loss_epoch', value_loss, num_epoch)
    writer.add_scalar('value_loss_steps', value_loss, total_steps)

    writer.add_scalar('success_rate_epoch', success_rate, num_epoch)
    writer.add_scalar('success_rate_steps', success_rate, total_steps)


def log_to_tensorboard(writer, num_epoch, num_episodes, total_steps,
                       steps_per_sec, episodes_per_sec, final_rewards,
                       mean_dist_entropy, mean_value_loss, mean_action_loss,
                       std_dist_entropy, std_value_loss, stmd_action_loss,
                       count_stats, array_stats, cumulative_reward,
                       terminal_reward, success_rate, success_rate_alive,
                       running_num_episodes, mean_total_loss,
                       mean_kl_loss=None, mean_pg_loss=None, lr=None,
                       distill_factor=0, reinforce_only=False,
                       start_step_ratios=None, start_step_all=None,
                       start_step_beg_ratios=None, start_step_all_beg=None,                       
                       bomb_penalty_lambda=None, action_choices=None,
                       action_probs=None, uniform_v=None,
                       mean_running_success_rate=None,
                       running_total_game_step_counts=[]):
    # writer.add_scalar('entropy', {
    #     'mean' : mean_dist_entropy,
    #     'std_max': mean_dist_entropy + std_dist_entropy,
    #     'std_min': mean_dist_entropy - std_dist_entropy,
    # }, num_episodes)
    #
    # writer.add_scalar('reward', {
    #     'mean': final_rewards.mean(),
    #     'std_max': final_rewards.mean() + final_rewards.std(),
    #     'std_min': final_rewards.mean() - final_rewards.std(),
    # }, num_episodes)
    #
    # writer.add_scalars('action_loss', {
    #     'mean': mean_action_loss,
    #     'std_max': mean_action_loss + std_action_loss,
    #     'std_min': mean_action_loss - std_action_loss,
    # }, num_episodes)
    #
    # writer.add_scalars('value_loss', {
    #     'mean': mean_value_loss,
    #     'std_max': mean_value_loss + std_value_loss,
    #     'std_min': mean_value_loss - std_value_loss,
    # }, num_episodes)

    # x-axis: # steps
    if mean_kl_loss and not np.isnan(mean_kl_loss):
        writer.add_scalar('kl_loss_step', mean_kl_loss, total_steps)
    writer.add_scalar('total_loss_step', mean_total_loss, total_steps)

    writer.add_scalar('success_rate_step',
                    1.0 * success_rate / running_num_episodes, total_steps)

    writer.add_scalar('steps_per_sec', steps_per_sec, total_steps)
    # writer.add_scalar('bomb_penalty_lambda', bomb_penalty_lambda, num_epoch)

    # if array_stats.get('rank'):
    #     writer.add_scalar('mean_rank_step', np.mean(array_stats['rank']),
    #                       total_steps)

    # if array_stats.get('dead'):
    #     writer.add_scalar('mean_dying_step_step', np.mean(array_stats['dead']),
    #                       total_steps)
    #     writer.add_scalar(
    #         'percent_dying_per_episode_step',
    #         1.0 * len(array_stats['dead']) / running_num_episodes,
    #         total_steps)

    writer.add_histogram('action_choices_epoch', action_choices, num_epoch, bins='doane')
    # writer.add_histogram('action_choices_steps', action_choices, total_steps, bins='doane')
    # for num in range(len(action_probs)):
    #     writer.add_histogram('action_probs_epoch/%d' % num, action_probs[num],
    #                          num_epoch, bins='doane')
    #     writer.add_histogram('action_probs_steps/%d' % num, action_probs[num],
    #                          total_steps, bins='doane')

    if uniform_v is not None:
        writer.add_scalar('uniform_v_epoch', uniform_v, num_epoch)
    if mean_running_success_rate is not None and not np.isnan(mean_running_success_rate):
        writer.add_scalar('mean_running_success_rate', mean_running_success_rate,
                          num_epoch)
    # if running_total_game_step_counts:
    #     avg = np.mean(running_total_game_step_counts)
    #     std = np.std(running_total_game_step_counts)
    #     writer.add_scalar('game_step_counts/mean', avg, num_epoch)
    #     writer.add_scalar('game_step_counts/std', std, num_epoch)
    
    # x-axis: # episodes
    # if reinforce_only:
    #     writer.add_scalar('pg_loss_epi', mean_pg_loss, num_episodes)
    # else:
    #     writer.add_scalar('entropy_epi', mean_dist_entropy, num_episodes)
    #     writer.add_scalar('action_loss_epi', mean_action_loss, num_episodes)
    #     writer.add_scalar('value_loss_epi', mean_value_loss, num_episodes)
    # if mean_kl_loss and not np.isnan(mean_kl_loss):
    #     writer.add_scalar('kl_loss_epi', mean_kl_loss, num_episodes)
    # writer.add_scalar('total_loss_epi', mean_total_loss, num_episodes)

    # writer.add_scalar('final_reward_epi', final_rewards.mean(), num_episodes)
    # writer.add_scalar('cumulative_reward_epi',
    #                 1.0 * cumulative_reward / running_num_episodes, num_episodes)
    # writer.add_scalar('terminal_reward_epi',
    #                 1.0 * terminal_reward / running_num_episodes, num_episodes)
    # writer.add_scalar('success_rate_epi',
    #                 1.0 * success_rate / running_num_episodes, num_episodes)
    # writer.add_scalar('success_rate_alive_epi',
    #                 1.0 * success_rate_alive / running_num_episodes,
    #                 num_episodes)

    for start_step, ratio in start_step_ratios.items():
        writer.add_scalar("win_startstep_epoch/%d" % start_step, ratio,
                          num_epoch)
    for start_step, ratio in start_step_beg_ratios.items():
        writer.add_scalar("win_startstep_beg_epoch/%d" % start_step, ratio,
                          num_epoch)

    for start_step, count in start_step_all.items():
        writer.add_scalar("all_startstep_epoch/%d" % start_step, count, num_epoch)
    for start_step, count in start_step_all_beg.items():
        writer.add_scalar("all_startstep_beg_epoch/%d" % start_step, count, num_epoch)

    # x-axis: # epochs / updates
    if lr is not None:
        writer.add_scalar('learning_rate', lr, num_epoch)

    if reinforce_only:
        writer.add_scalar('pg_loss_epoch', mean_pg_loss, num_epoch)
    else:
        writer.add_scalar('entropy_epoch', mean_dist_entropy, num_epoch)
        writer.add_scalar('action_loss_epoch', mean_action_loss, num_epoch)
        writer.add_scalar('value_loss_epoch', mean_value_loss, num_epoch)
    if mean_kl_loss and not np.isnan(mean_kl_loss):
        writer.add_scalar('kl_loss_epoch', mean_kl_loss, num_epoch)
    if distill_factor > 0:
        writer.add_scalar('kl_factor_epoch', distill_factor, num_epoch)
    writer.add_scalar('total_loss_epoch', mean_total_loss, num_epoch)

    writer.add_scalar('final_reward_epoch', final_rewards.mean(), num_epoch)
    writer.add_scalar('cumulative_reward_epoch',
                    1.0 * cumulative_reward / running_num_episodes, num_epoch)
    # writer.add_scalar('terminal_reward_epoch',
    #                 1.0 * terminal_reward / running_num_episodes, num_epoch)
    writer.add_scalar('success_rate_epoch',
                    1.0 * success_rate / running_num_episodes, num_epoch)
    # writer.add_scalar('success_rate_alive_epoch',
    #                 1.0 * success_rate_alive / running_num_episodes,
    #                 num_epoch)


    # for title, count in count_stats.items():
    #     if title.startswith('bomb:'):
    #         continue
    #     writer.add_scalar(title, 1.0 * count / running_num_episodes,
    #                       num_epoch)

    # writer.add_scalars('bomb_distances_epoch', {
    #     key.split(':')[1]: 1.0 * count / running_num_episodes
    #     for key, count in count_stats.items() \
    #     if key.startswith('bomb:')
    # }, num_epoch)

    # if array_stats.get('rank'):
    #     writer.add_scalar('mean_rank_epoch', np.mean(array_stats['rank']),
    #                       num_epoch)

    # if array_stats.get('dead'):
    #     writer.add_scalar('mean_dying_step_epoch', np.mean(array_stats['dead']),
    #                       num_epoch)
    #     writer.add_scalar(
    #         'percent_dying_per_episode_epoch',
    #         1.0 * len(array_stats['dead']) / running_num_episodes,
    #         num_epoch)


def validate_how_train(args):
    how_train = args.how_train
    nagents = args.num_agents
    if how_train == 'simple':
        # Simple trains a single agent with three SimpleAgents.
        assert(nagents == 1), "Simple training should have one agent."
        return 1
    elif how_train == 'dagger':
        # Dagger trains a single agent with three SimpleAgents.
        assert(nagents == 1), "Dagger training should have one agent."
        return 1
    elif how_train == 'homogenous':
        # Homogenous trains a single agent alongside itself (self-play), with
        # the other two agents being prior versions. The initial opponents are
        # the initialized versions.
        assert(nagents == 1), "Homogenous training should have one agent."
        return 2
    elif how_train == 'qmix':
        assert nagents == 2
        return 2
    elif how_train == 'heterogenous':
        s = "Heterogenous training should have multiple agents."
        assert(nagents > 1), s
        print("Heterogenous training is not implemented yet.")
        return None
    elif how_train == 'astar':
        return 1


def torch_numpy_stack(value, data=True):
    return torch.from_numpy(np.stack([x.data if data else x for x in value])) \
                .float()


def torch_load(path, cuda, cuda_device):
    if cuda:
        # NOTE: we specify the cuda_device to avoid memory/segfault issues.
        return torch.load(path, map_location=lambda storage,
                          loc: storage.cuda(cuda_device))
    else:
        return torch.load(path, map_location=lambda storage, loc: storage)


def flatten(lst):
    return [item for sublist in lst for item in sublist]
