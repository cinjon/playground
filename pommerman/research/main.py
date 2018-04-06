from collections import defaultdict
import os
import time

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
import numpy as np
from pommerman import configs
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import utils

from arguments import get_args
from envs import make_env
from model import PommeResnetPolicy, PommeCNNPolicySmall

import ppo_agent
from subproc_vec_env import SubprocVecEnvRender


args = get_args()

print("uses cuda: ", args.cuda)

# num_updates = number of samples collected in one round of updates.
# num_steps = number of steps in a rollout (horizon)
# num_processes = number of parallel processes/workers collecting data.
# number of samples used for a round of updates = horizon * num_workers = num_steps_rollout * num_parallel_processes
num_updates = int(args.num_frames) // args.num_steps // args.num_processes
print("NUM UPDATES {} num frames {} num steps {} num processes {}".format(
    num_updates, args.num_frames, args.num_steps, args.num_processes), "\n")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized.")
    print("Use a monitor (see envs.py) or visdom plot to get true rewards.")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'
    assert(args.run_name)

    dummy_env = make_env(config=args.config, how_train='dummy', seed=None,
                         rank=-1, game_state_file=args.game_state_file,
                         training_agents=[], num_stack=args.num_stack)()
    envs_shape = dummy_env.observation_space.shape[1:]
    obs_shape = (envs_shape[0], *envs_shape[1:])
    action_space = dummy_env.action_space
    if args.model == 'convnet':
        actor_critic = lambda saved_model: PommeCNNPolicySmall(
            obs_shape[0], action_space, args)
    elif args.model == 'resnet':
        actor_critic = lambda saved_model: PommeResnetPolicy(
            obs_shape[0], action_space, args)

    # We need to get the agent = config.agent(agent_id, config.game_type) and
    # then pass that agent into the agent.PPOAgent
    training_agents = []
    saved_models = args.saved_models
    saved_models = saved_models.split(',') if saved_models \
                   else [None]*args.nagents
    assert(len(saved_models)) == args.nagents
    for saved_model in saved_models:
        # TODO: implement the model loading.
        model = actor_critic(saved_model)
        agent = ppo_agent.PPOAgent(model)
        training_agents.append(agent)

    if args.how_train == 'simple':
        # Simple trains a single agent against three SimpleAgents.
        assert(args.nagents == 1), "Simple training should have one agent."
        num_training_per_episode = 1
    elif args.how_train == 'homogenous':
        # Homogenous trains a single agent against itself (self-play).
        assert(args.nagents == 1), "Homogenous training should have one agent."
        num_training_per_episode = 4
    elif args.how_train == 'heterogenous':
        s = "Heterogenous training should have multiple agents."
        assert(args.nagents > 1), s
        print("Heterogenous training is not implemented yet.")
        return

    suffix = "{}.train.ht-{}.cfg-{}.m-{}.seed-{}".format(
        args.run_name, args.how_train, args.config, args.model, args.seed)
    writer = SummaryWriter(os.path.join(args.log_dir, suffix))

    # NOTE: I didn't think that this should work because I thought that the
    # agent object would overwrite the agent_id on each env. Somehow it is
    # working though. This might be due to the separate threads, but that
    # doesn't make sense to me. TODO: Look into why.
    envs = [
        make_env(config=args.config, how_train=args.how_train, seed=args.seed,
                 rank=rank, game_state_file=args.game_state_file,
                 training_agents=training_agents, num_stack=args.num_stack)
        for rank in range(args.num_processes)
    ]
    envs = SubprocVecEnvRender(envs) if args.render else SubprocVecEnv(envs)

    for agent in training_agents:
        agent.initialize(args, obs_shape, action_space,
                         num_training_per_episode)

    current_obs = torch.zeros(num_training_per_episode, args.num_processes,
                              *obs_shape)
    def update_current_obs(obs):
        current_obs = torch.from_numpy(obs).float()

    def torch_numpy_stack(value):
        return torch.from_numpy(np.stack([x.data for x in value])).float()

    import pdb; pdb.set_trace()
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

    final_dist_entropies = torch.zeros([num_training_per_episode,
                                 args.num_processes, 1])

    count_stats = defaultdict(int)
    array_stats = defaultdict(list)

    if args.cuda:
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()

    # TODO: Set the total_steps count when you load the model.
    start = time.time()
    time_counts = defaultdict(list)

    num_episodes = 0
    running_num_episodes = 0
    final_action_losses = [[] for agent in range(len(training_agents))]
    final_value_losses =  [[] for agent in range(len(training_agents))]
    final_dist_entropies = [[] for agent in range(len(training_agents))]

    for j in range(num_updates):
        for agent in training_agents:
            agent.set_eval()

        for step in range(args.num_steps):
            step_time = time.time()

            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            cpu_actions_agents = []

            if args.how_train == 'simple':
                _s = time.time()
                result = training_agents[0].run(step, 0, use_act=True)
                time_counts['run_time'].append(time.time() - _s)

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
                        cpu_actions_agents[num_process].append(
                            cpu_actions[num_process])

            _s = time.time()
            obs, reward, done, info = envs.step(cpu_actions_agents)
            time_counts['env_step_time'].append(time.time() - _s)

            # TODO: Change this when we use heterogenous.
            for i in info:
                for lst in i.get('step_info', {}).values():
                    for l in lst:
                        if l.startswith('dead') or l.startswith('rank'):
                            key, count = l.split(':')
                            array_stats[key].append(int(count))
                        else:
                            count_stats[l] += 1
                            if 'bomb' in l:
                                count_stats['bomb'] += 1

            if args.render:
                envs.render()
            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            episode_rewards += reward

            # NOTE: if how-train simple always has num_training_per_episode = 1
            # then we don't need the conditions below
            if args.how_train == 'simple':
                running_num_episodes += sum([1 if done_ else 0
                                             for done_ in done])
                masks = torch.FloatTensor([
                    [0.0]*num_training_per_episode if done_ \
                    else [1.0]*num_training_per_episode
                for done_ in done])
            elif args.how_train == 'homogenous':
                running_num_episodes += sum([1 if done_.all() else 0 for done_ in done])

                masks = torch.FloatTensor(
                    [[[0.0] if done_[i] else [1.0] for i in range(len(done_))]
                     for done_ in done]).transpose(0,1).unsqueeze(2)

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

            import pdb; pdb.set_trace()
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

            time_counts['step_time'].append(time.time() - step_time)

        next_value_agents = []
        if args.how_train == 'simple':
            agent = training_agents[0]

            _s = time.time()
            next_value_agents.append(agent.run(step=-1, num_agent=0))
            advantages = [
                agent.compute_advantages(next_value_agents, args.use_gae,
                                         args.gamma, args.tau)
            ]
            time_counts['adv_time'].append(time.time() - _s)
        elif args.how_train == 'homogenous':
            agent = training_agents[0]
            next_value_agents = [agent.run(step=-1, num_agent=num_agent)
                                 for num_agent in range(4)]
            advantages = [
                agent.compute_advantages(next_value_agents, args.use_gae,
                                         args.gamma, args.tau)
            ]

        for agent in training_agents:
            agent.set_train()

        for num_agent, agent in enumerate(training_agents):
            for _ in range(args.ppo_epoch):
                ppo_time = time.time()

                data_generator = agent.feed_forward_generator(
                    advantages[num_agent], args)

                for sample in data_generator:
                    observations_batch, states_batch, actions_batch, \
                        return_batch, masks_batch, \
                        old_action_log_probs_batch, adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    _s = time.time()
                    import pdb; pdb.set_trace()
                    result = agent.evaluate_actions(
                        Variable(observations_batch),
                        Variable(states_batch),
                        Variable(masks_batch),
                        Variable(actions_batch))
                    time_counts['evaluate_time'].append(time.time() - _s)
                    values, action_log_probs, dist_entropy, states = result

                    adv_targ = Variable(adv_targ)
                    ratio = action_log_probs
                    ratio -= Variable(old_action_log_probs_batch)
                    ratio = torch.exp(ratio)

                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(
                        ratio, 1.0 - args.clip_param, 1.0 + args.clip_param)
                    surr2 *= adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()
                    value_loss = (Variable(return_batch) - values) \
                                 .pow(2).mean()
                    _s = time.time()
                    agent.optimize(value_loss, action_loss, dist_entropy,
                                   args.entropy_coef, args.max_grad_norm)
                    time_counts['optimize_time'].append(time.time() - _s)

                    final_action_losses[num_agent].append(action_loss.data[0])
                    final_value_losses[num_agent].append(value_loss.data[0])
                    final_dist_entropies[num_agent].append(dist_entropy.data[0])

                time_counts['ppo_time'].append(time.time() - ppo_time)
            agent.after_update()

        # for title, counts in time_counts.items():
        #     print("%s (%d): mean = %.3f, std = %.3f, total = %.3f." % (
        #         title, len(counts), np.mean(counts), np.std(counts),
        #         sum(counts)
        #     ))

        # TODO: This is relative to the loaded model if exists.
        total_steps = (j + 1) * args.num_processes * args.num_steps

        #####
        # Save model.
        #####
        if False and j % args.save_interval == 0 and args.save_dir != "":
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
                save_dict = {
                    'epoch': j,
                    'arch': args.model,
                    'state_dict': save_model.state_dict(),
                    'optimizer' : save_optimizer.state_dict(),
                    'total_steps': total_steps,
                }
                save_dict['args'] = vars(args)
                suffix = "{}.train.ht-{}.cfg-{}.m-{}.num-{}.epoch-{}.steps-{}.seed-{}.pt" \
                         .format(args.run_name, args.how_train, args.config,
                                 args.model, num_agent, j, total_steps, args.seed)
                torch.save(save_dict, os.path.join(save_path, suffix))

        #####
        # Log to console and to Tensorboard.
        #####
        if False and running_num_episodes > args.log_interval:
            end = time.time()
            num_steps_sec = (end - start)
            total_steps = (j + 1) * args.num_processes * args.num_steps
            num_episodes += running_num_episodes

            steps_per_sec = 1.0 * total_steps / (end - start)
            if j == 0:
                updates_per_sec = int(args.log_interval / (end - start))
            episodes_per_sec =  1.0 * num_episodes / (end - start)

            mean_dist_entropy = np.mean([
                dist_entropy for dist_entropy in final_dist_entropies])
            std_dist_entropy = np.std([
                dist_entropy for dist_entropy in final_dist_entropies])

            mean_value_loss = np.mean([
                value_loss for value_loss in final_value_losses])
            std_value_loss = np.std([
                value_loss for value_loss in final_value_losses])

            mean_action_loss = np.mean([
                action_loss for action_loss in final_action_losses])
            std_action_loss = np.std([
                action_loss for action_loss in final_action_losses])

            print("Updates {}, num episodes {}, num timesteps {}, FPS {}, updates per sec {}, mean/median reward "
                  "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, avg entropy "
                  "{:.5f}, avg value loss {:.5f}, avg policy loss {:.5f}".
                format(j, num_episodes, total_steps,
                       steps_per_sec,
                       updates_per_sec,
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(),
                       mean_dist_entropy,
                       mean_value_loss,
                       mean_action_loss))

            writer.add_scalars('entropy', {
                'mean': mean_dist_entropy,
                'var': std_dist_entropy,
            }, num_episodes)

            writer.add_scalars('action_loss', {
                'mean': mean_action_loss,
                'var': std_action_loss,
            }, num_episodes)

            writer.add_scalars('value_loss', {
                'mean': mean_value_loss,
                'var': std_value_loss,
            }, num_episodes)

            writer.add_scalars('rewards', {
                'mean': final_rewards.mean(),
                'std': final_rewards.std(),
                'median': final_rewards.median(),
            }, num_episodes)

            writer.add_scalar('updates', j, num_episodes)
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

            if array_stats['rank']:
                writer.add_scalars('rank', {
                    'mean': np.mean(array_stats['rank']),
                    'std': np.std(array_stats['rank']),
                }, num_episodes)

            if array_stats['dead']:
                writer.add_scalars('dying_step', {
                    'mean': np.mean(array_stats['dead']),
                    'std': np.std(array_stats['dead']),
                }, num_episodes)
                writer.add_scalar(
                    'percent_dying_per_ep',
                    1.0 * len(array_stats['dead']) / running_num_episodes,
                    num_episodes)

            # always reset these stats so that means in the plots are per the last log_interval
            final_action_losses = [[] for agent in range(len(training_agents))]
            final_value_losses =  [[] for agent in range(len(training_agents))]
            final_dist_entropies = [[] for agent in range(len(training_agents))]
            count_stats = defaultdict(int)
            array_stats = defaultdict(list)
            final_rewards = torch.zeros([num_training_per_episode, args.num_processes, 1])
            running_num_episodes = 0

    writer.close()

if __name__ == "__main__":
    main()
