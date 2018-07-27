"""Train script for ppo learning.
TODO: Implement heterogenous training.

The number of samples used for an epoch is:
horizon * num_workers = num_steps * num_processes where num_steps is the number
of steps in a rollout (horizon) and num_processes is the number of parallel
processes/workers collecting data.

Simple Example:
python train_ppo.py --how-train simple --num-processes 10 --run-name test \
 --num-steps 50 --log-interval 5

Distillation Example:
python train_ppo.py --how-train simple --num-processes 10 --run-name distill \
 --num-steps 100 --log-interval 5 \
 --distill-epochs 100 --distill-target dagger::/path/to/model.pt

Homogenous Example:
python train_ppo.py --how-train homogenous --num-processes 10 \
 --run-name distill --num-steps 100 --log-interval 5 --distill-epochs 100 \
 --distill-target dagger::/path/to/model.pt --config PommeTeam-v0 \
 --eval-mode homogenous --num-battles-eval 100 --seed 100

Lower Complexity example:
python train_ppo.py --how-train simple --num-processes 10 --run-name test \
 --num-steps 50 --log-interval 5 --config PommeFFAEasy-v0 --board-size 11

Reverse Curriculum with Eval:
python train_ppo.py --run-name test --num-processes 12 --config PommeFFAEasy-v0 \
--how-train simple --lr 1e-4 --save-interval 100 --log-interval 1 --gamma 0.95 \
 --model-str PommeCNNPolicySmall --board_size 11 --num-battles-eval 100 \
 --eval-mode ffa-curriculum --state-directory-distribution uniform21 \
 --state-directory /home/roberta/pommerman_spring18/pomme_games/ffaeasyv0-seed1 \
"""
from collections import defaultdict
from collections import deque
import os
import time

import numpy as np
from pommerman.agents import SimpleAgent
from pommerman.agents import ComplexAgent
from pommerman import utility
from tensorboardX import SummaryWriter
import torch
import random

from arguments import get_args
import envs as env_helpers
from eval import eval as run_eval
import ppo_agent, reinforce_agent
import utils
from torch.autograd import Variable

import pommerman.constants as constants
from statistics import mean as mean

def train():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.cuda:
        torch.cuda.empty_cache()

    assert(args.run_name)
    print("\n###############")
    print("args ", args)
    print("##############\n")

    num_training_per_episode = utils.validate_how_train(args)
    how_train, config, num_agents, num_stack, num_steps, num_processes, \
        num_epochs, reward_sharing, batch_size, num_mini_batch = \
        utils.get_train_vars(args, num_training_per_episode)

    obs_shape, action_space, character, board_size = env_helpers.get_env_info(
        config, num_stack)

    if args.reinforce_only:
        training_agents = utils.load_agents(
            obs_shape, action_space, num_training_per_episode, num_steps, args,
            reinforce_agent.ReinforceAgent, character=character, board_size=board_size)
    else:
        training_agents = utils.load_agents(
            obs_shape, action_space, num_training_per_episode, num_steps,
            args, ppo_agent.PPOAgent, character=character, board_size=board_size)

    model_str = args.model_str.replace('PommeCNNPolicy', '')
    config_str = config.strip('Pomme').replace('Short', 'Sh').replace('FFACompetition-v0', 'FFACmp')
    suffix = "{}.{}.{}.{}.nc{}.lr{}.bs{}.ns{}.gam{}.seed{}".format(
        args.run_name, how_train, config_str, model_str, args.num_channels,
        args.lr, args.batch_size, num_steps, args.gamma, args.seed)
    if args.use_gae:
        suffix += ".gae"
    if args.half_lr_epochs:
        suffix += ".halflr%d" % args.half_lr_epochs
    if args.use_lr_scheduler:
        suffix += ".ulrs"
    if args.state_directory_distribution:
        suffix += ".%s" % args.state_directory_distribution
    if args.anneal_bomb_penalty_epochs:
        suffix += ".abpe%d" % args.anneal_bomb_penalty_epochs
    if args.item_reward:
        suffix += ".itemrew%.3f" % args.item_reward
    if args.step_loss:
        suffix += ".stl%.3f" % args.step_loss

    if args.use_second_place:
        suffix += ".usp"
    elif args.use_both_places:
        suffix += ".ubp"

    envs = env_helpers.make_train_envs(
        config, how_train, args.seed, args.game_state_file, training_agents,
        num_stack, num_processes, state_directory=args.state_directory,
        state_directory_distribution=args.state_directory_distribution,
        step_loss=args.step_loss, bomb_reward=args.bomb_reward,
        item_reward=args.item_reward, use_second_place=args.use_second_place,
        use_both_places=args.use_both_places
    )
    game_type = envs.get_game_type()

    uniform_v = None
    running_success_rate = []
    if args.state_directory_distribution == 'uniformAdapt':
        uniform_v = 33
        uniform_v_factor = args.uniform_v_factor
        running_success_rate_maxlen = 10 # corresponds to roughly 200 epochs of FFA.
        running_success_rate = deque([], maxlen=running_success_rate_maxlen)
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformScheduleA':
        uniform_v = 33
        uniform_v_factor = 2
        uniform_v_incr = 3000
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformScheduleB':
        uniform_v = 33
        uniform_v_factor = 2
        uniform_v_incr = 1000
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformScheduleC':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 500
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformScheduleD':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 250
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformScheduleE':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 1500
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformScheduleF':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 2000
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformScheduleG':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 2500
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsA':
        # (0, 32), (24, 64), (56, 128), (120, 256), (248, 512), ...
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 2500
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsB':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 1000
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsC':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 750
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsD':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 1500
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsE':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 2000
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsF':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 500
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsG':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 50
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsH':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 100
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsI':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 150
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsJ':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 75
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsK':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 40
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsL':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 85
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsGrA':
        uniform_v = 4
        uniform_v_factor = 2
        uniform_v_incr = 50
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsGrB':
        uniform_v = 4
        uniform_v_factor = 2
        uniform_v_incr = 100
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsGrC':
        uniform_v = 4
        uniform_v_factor = 2
        uniform_v_incr = 250
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformBoundsGrD':
        uniform_v = 4
        uniform_v_factor = 2
        uniform_v_incr = 500
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'grUniformBoundsA':
        uniform_v = 4
        uniform_v_factor = 2
        uniform_v_incr = 200
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'grUniformBoundsB':
        uniform_v = 4
        uniform_v_factor = 2
        uniform_v_incr = 350
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'grUniformBoundsC':
        uniform_v = 4
        uniform_v_factor = 2
        uniform_v_incr = 500
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformForwardA':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 250
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformForwardB':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 500
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)
    elif args.state_directory_distribution == 'uniformForwardC':
        uniform_v = 32
        uniform_v_factor = 2
        uniform_v_incr = 1000
        uniform_v_prior = 0
        envs.set_uniform_v(uniform_v)

    set_distill_kl = args.set_distill_kl
    distill_target = args.distill_target
    distill_epochs = args.distill_epochs
    init_kl_factor = args.init_kl_factor
    distill_expert = args.distill_expert

    if distill_expert is not None:
        assert(distill_epochs > training_agents[0].num_epoch), \
        "If you are distilling, distill_epochs > trianing_agents[0].num_epoch."
    elif distill_expert == 'DaggerAgent':
        assert(distill_target is not ''), \
        "If you are distilling from Dagger you need to specify distill_target."

    # NOTE: you need to set the distill expert in order to do distillation
    do_distill = distill_expert is not None
    if do_distill:
        print("Distilling: {} from {} \n".format(do_distill, distill_expert))
        if distill_expert == 'DaggerAgent':
            distill_agent = utils.load_distill_agent(obs_shape, action_space, args)
            distill_agent.set_eval()
            # NOTE: We have to call init_agent, but the agent_id won't matter
            # because we will use the observations from the ppo_agent.
            distill_agent.init_agent(0, game_type)
            distill_type = distill_target.split('::')[0]
            if set_distill_kl >= 0:
                suffix += ".dstlDagKL{}".format(set_distill_kl)
            else:
                suffix += ".dstlDagEp{}".format(distill_epochs)
            suffix += ".ikl{}".format(init_kl_factor)
        elif distill_expert == 'SimpleAgent':
            if set_distill_kl >= 0:
                suffix += ".dstlSimKL{}".format(set_distill_kl)
            else:
                suffix += ".dstlSimEp{}".format(distill_epochs)
            suffix += ".ikl{}".format(init_kl_factor)
        elif distill_expert == 'ComplexAgent':
            if set_distill_kl >= 0:
                suffix += ".dstlComKL{}".format(set_distill_kl)
            else:
                suffix += ".dstlComEp{}".format(distill_epochs)
            suffix += ".ikl{}".format(init_kl_factor)
        else:
            raise ValueError("Only distill from Dagger, Simple, or Complex.")

    log_dir = os.path.join(args.log_dir, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #####
    # Logging helpers.
    writer = SummaryWriter(log_dir)
    count_stats = defaultdict(int)
    array_stats = defaultdict(list)
    episode_rewards = torch.zeros([num_training_per_episode,
                                   num_processes, 1])
    final_rewards = torch.zeros([num_training_per_episode,
                                 num_processes, 1])

    # TODO: When we implement heterogenous, change this to be per agent.
    start_epoch = training_agents[0].num_epoch
    total_steps = training_agents[0].total_steps
    num_episodes = training_agents[0].num_episodes
    if training_agents[0].uniform_v is not None:
        print("UNFIROM V IS NOT NONE")
        uniform_v = training_agents[0].uniform_v
        uniform_v_prior = training_agents[0].uniform_v_prior
        envs.set_uniform_v(uniform_v)
        if args.state_directory_distribution.startswith('setBounds'):
            while uniform_v_vals and uniform_v_vals[0] < uniform_v:
                uniform_v_vals.pop(0)
                uniform_v_incrs.pop(0)

    start_step_wins = defaultdict(int)
    start_step_all = defaultdict(int)
    start_step_position_wins = defaultdict(int)
    start_step_wins_beg = defaultdict(int)
    start_step_all_beg = defaultdict(int)
    start_step_position_wins_beg = defaultdict(int)

    running_num_episodes = 0
    cumulative_reward = 0
    terminal_reward = 0
    all_agent_success_rate = 0
    per_agent_success_rate = [0]*4
    success_rate = 0
    success_rate_alive = 0
    prev_epoch = start_epoch
    game_step_counts = [0 for _ in range(num_processes)]
    running_total_game_step_counts = []
    running_optimal_info = []
    optimal_by_file = defaultdict(list)

    if args.reinforce_only:
        final_pg_losses = [[] for agent in range(len(training_agents))]
    else:
        final_action_losses = [[] for agent in range(len(training_agents))]
        final_value_losses =  [[] for agent in range(len(training_agents))]
        final_dist_entropies = [[] for agent in range(len(training_agents))]
    if do_distill:
        final_kl_losses = [[] for agent in range(len(training_agents))]
    final_total_losses =  [[] for agent in range(len(training_agents))]

    def update_current_obs(obs):
        return torch.from_numpy(obs).float().transpose(0,1)

    def update_actor_critic_results(result):
        value, action, action_log_prob, states, probs, log_probs = result
        value_agents.append(value)
        action_agents.append(action)
        action_log_prob_agents.append(action_log_prob)
        states_agents.append(states)
        action_log_prob_distr.append(log_probs)
        return action.data.squeeze(1).cpu().numpy(), probs.data.squeeze().cpu().numpy()

    def update_stats(info):
        # NOTE: This func has a side effect where it sets variable values.
        # TODO: Change this stats computation when we use heterogenous.
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

    if how_train == 'simple' or how_train == 'grid':
        expert_actions_onehot = torch.FloatTensor(
            num_processes, action_space.n)
    elif how_train in ['homogenous', 'backselfplay']:
        expert_actions_onehot = torch.FloatTensor(
            num_processes, num_training_per_episode, action_space.n)
    else:
        raise ValueError("Only Simple and Homogenous training regimes have been \
            implemented.")
    onehot_dim = len(expert_actions_onehot.shape) - 1

    if how_train == 'homogenous':
        good_guys = [
            ppo_agent.PPOAgent(training_agents[0].model,
                               num_stack=args.num_stack, cuda=args.cuda,
                               num_processes=args.num_processes // 2,
                               recurrent_policy=args.recurrent_policy)
            for _ in range(2)
        ]
        if args.cuda:
            for guy in good_guys:
                guy.cuda()
        saved_paths = utils.save_agents(
            "ppo", 0, training_agents, total_steps,
            num_episodes, args, suffix, uniform_v, uniform_v_prior)
        if args.homogenous_init == 'self':
            bad_guys_eval = [
                utils.load_inference_agent(saved_paths[0], ppo_agent.PPOAgent,
                                           "ppo", action_space, obs_shape,
                                           args.num_processes // 2, args)
                for _ in range(2)
            ]
            bad_guys_train = [
                    utils.load_inference_agent(saved_paths[0], ppo_agent.PPOAgent,
                                               "ppo", action_space, obs_shape,
                                               args.num_processes, args)
                for _ in range(2)
            ]
        eval_round = 0


    # NOTE: only works for how_train simple because we assume training_ids
    # has a single element
    def get_win_alive(info, envs):
        '''win_list: the training agent's team won
        alive_list: the training agent's team won and the training agent
        is alive at the end of the game'''

        training_ids = [x[0] for x in envs.get_training_ids()]
        win_list = []
        alive_list = []
        for inf, id_ in zip(info, training_ids):
            result = inf.get('result', {})
            if result == constants.Result.Win:
                alives = inf.get('alive', {})
                winners = inf.get('winners', {})
                win = False
                alive = False
                for w in winners:
                    if w == id_:
                        win_list.append(True)
                        win = True
                        break
                if not win:
                    win_list.append(False)
                    alive_list.append(False)
                else:
                    for a in alives:
                        if a == id_:
                            alive_list.append(True)
                            alive = True
                            break
                    if not alive:
                        alive_list.append(False)
            else:
                win_list.append(False)    # True iff training agent's team won
                alive_list.append(False)  # True iff training won and alive
        return np.array(win_list), np.array(alive_list)

    def get_wins(info, envs):
        training_ids = envs.get_training_ids()
        position_wins = defaultdict(int)
        position_games = defaultdict(int)
        game_results = []

        for info_, ids in zip(info, training_ids):
            result = info_.get('result', {})
            game_result = None
            if result == constants.Result.Win:
                winners = info_.get('winners', {})
                for id_ in ids:
                    position_games[id_] += 1
                    if id_ in winners:
                        position_wins[id_] += 1
                        game_result = id_
            game_results.append(game_result)

        return position_wins, game_results


    # TODO: make the function below less hacky
    def make_onehot(actions):
        actions_tensor = torch.from_numpy(actions)
        if how_train in ['homogenous', 'backselfplay']:
            actions_tensor = torch.from_numpy(actions).unsqueeze(onehot_dim)
        expert_actions_onehot.zero_()
        expert_actions_onehot.scatter_(onehot_dim, actions_tensor, 1)

    # Start the environment and set the current_obs appropriately.
    current_obs = update_current_obs(envs.reset())
    if how_train in ['simple', 'homogenous', 'grid', 'backselfplay']:
        # NOTE: Here, we put the first observation into the rollouts.
        training_agents[0].update_rollouts(obs=current_obs, timestep=0)

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()
        if do_distill and distill_expert == 'DaggerAgent':
            distill_agent.cuda()

    if how_train == 'homogenous':
        win_rate, tie_rate, loss_rate = evaluate_homogenous(
                args, good_guys, bad_guys_eval, 0, writer, 0)
        print("Homog test before: (%d) --> Win %.3f, Tie %.3f, Loss %.3f" % (
                args.num_battles_eval, win_rate, tie_rate, loss_rate))
        for agent in good_guys + bad_guys_eval:
            agent.clear_obs_stack()

    start = time.time()
    # NOTE: assumes just one agent.
    action_choices = []
    action_probs = [[] for _ in range(action_space.n)]

    anneal_bomb_penalty_epochs = args.anneal_bomb_penalty_epochs
    bomb_penalty_lambda = 1.0

    for num_epoch in range(start_epoch, num_epochs):
        if num_epoch >= args.begin_selfbombing_epoch:
            envs.enable_selfbombing()

        if utils.is_save_epoch(num_epoch, start_epoch, args.save_interval) \
           and how_train in ['simple', 'grid', 'backselfplay']:
            # Only save at regular epochs if using "simple" or "grid". The others save
            # upon successful evaluation.
            utils.save_agents("ppo-", num_epoch, training_agents, total_steps,
                              num_episodes, args, suffix, clear_saved=True)

        for agent in training_agents:
            agent.set_eval()

        if do_distill:
            if args.set_distill_kl >= 0:
                distill_factor = args.set_distill_kl
            else:
                distill_factor = (distill_epochs - num_epoch) * init_kl_factor
                distill_factor = 1.0 * distill_factor / distill_epochs
                distill_factor = max(distill_factor, 0.0)
            if num_epoch % 50 == 0:
                print("Epoch %d - distill factor %.3f." % (
                    num_epoch, distill_factor))
        else:
            distill_factor = 0

        for step in range(num_steps):
            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            action_log_prob_distr = []
            dagger_prob_distr = []

            if how_train == 'simple':
                training_agent = training_agents[0]

                if do_distill:
                    if distill_expert == 'DaggerAgent':
                        data = training_agent.get_rollout_data(step, 0)
                        _, _, _, _, probs, _ = distill_agent.act_on_data(
                            *data, deterministic=True)
                        dagger_prob_distr.append(probs)
                    elif distill_expert in ['SimpleAgent', 'ComplexAgent']:
                        expert_obs = envs.get_expert_obs()
                        expert_actions = envs.get_expert_actions(expert_obs,
                                                                 distill_expert)
                        make_onehot(expert_actions)
                        dagger_prob_distr.append(expert_actions_onehot)
                    else:
                        raise ValueError("We only support distilling from \
                            DaggerAgent, SimpleAgent, or ComplexAgent")

                result = training_agent.actor_critic_act(
                    step, 0, deterministic=args.eval_only)
                # [num_processor,] ..., [num_processor, 6]
                cpu_actions_agents, cpu_probs = update_actor_critic_results(result)
                action_choices.extend(cpu_actions_agents)
                for num in range(action_space.n):
                    action_probs[num].extend([p[num] for p in cpu_probs])
            elif how_train == 'backselfplay':
                # Reshape to do computation once rather than four times.
                cpu_actions_agents = [[] for _ in range(num_processes)]
                data = training_agents[0].get_rollout_data(
                    step=step,
                    num_agent=0,
                    num_agent_end=num_training_per_episode)
                observations, states, masks = data
                observations = observations.view([
                    num_processes * num_training_per_episode,
                    *observations.shape[2:]
                ])
                states = states.view([
                    num_processes * num_training_per_episode,
                    *states.shape[2:]
                ])
                masks = masks.view([
                    num_processes * num_training_per_episode,
                    *masks.shape[2:]
                ])

                training_acts = training_agents[0].act_on_data(
                    observations, states, masks, deterministic=False)
                training_acts = [
                    datum.view([
                        num_processes, num_training_per_episode,
                        *datum.shape[1:]
                    ])
                    for datum in training_acts
                ]

                dead_agent_indices = envs.get_dead_agents()
                for num_agent in range(2):
                    # This is: value, action, action_log_prob, states, probs, log_probs
                    # Each of them are num_processes x <specific shape>
                    agent_results = [datum[:, num_agent]
                                     for datum in training_acts]

                    # NOTE: If the agent is not alive, change the action to Pass.
                    for num_process in range(num_processes):
                        if num_agent in dead_agent_indices[num_process]:
                            agent_results[1][num_process] = constants.Action.Stop.value
                    actions, probs = update_actor_critic_results(agent_results)
                    for num_process in range(num_processes):
                        cpu_actions_agents[num_process].append(
                            actions[num_process])

                    action_choices.extend(actions)
                    for num in range(action_space.n):
                        action_probs[num].extend([p[num] for p in probs])
            elif how_train == 'homogenous':
                # Reshape to do computation once rather than four times.
                cpu_actions_agents = [[] for _ in range(num_processes)]
                data = training_agents[0].get_rollout_data(
                    step=step,
                    num_agent=0,
                    num_agent_end=num_training_per_episode)
                observations, states, masks = data
                observations = observations.view([
                    num_processes * num_training_per_episode,
                    *observations.shape[2:]
                ])
                states = states.view([
                    num_processes * num_training_per_episode,
                    *states.shape[2:]
                ])
                masks = masks.view([
                    num_processes * num_training_per_episode,
                    *masks.shape[2:]
                ])
                if do_distill:
                    if distill_expert == 'DaggerAgent':
                        _, _, _, _, probs, _ = distill_agent.act_on_data(
                            observations, states, masks,
                            deterministic=True)
                        probs = probs.view([
                            num_processes, num_training_per_episode,
                            *probs.shape[1:]
                        ])
                        for num_agent in range(num_training_per_episode):
                            dagger_prob_distr.append(probs[:, num_agent])
                    elif distill_expert in ['SimpleAgent', 'ComplexAgent']:
                        # TODO: change this so that you get actions for all the agents
                        expert_obs = envs.get_expert_obs()
                        expert_actions = envs.get_expert_actions(
                            expert_obs, distill_expert)
                        make_onehot(expert_actions)
                        for num_agent in range(num_training_per_episode):
                            dagger_prob_distr.append(
                                expert_actions_onehot[:, num_agent])
                    else:
                        raise ValueError("We only support distilling from \
                        DaggerAgent, SimpleAgent, or ComplexAgent")

                training_acts = training_agents[0].act_on_data(
                    observations, states, masks, deterministic=False)
                training_acts = [
                    datum.view([
                        num_processes, num_training_per_episode,
                        *datum.shape[1:]
                    ])
                    for datum in training_acts
                ]
                # cpu_training_actions: num_process x 2 list of actions
                training_actions = [[] for _ in range(num_processes)]
                training_probs = [[] for _ in range(num_processes)]
                for num_agent in range(2):
                    agent_results = [datum[:, num_agent]
                                     for datum in training_acts]
                    actions, probs = update_actor_critic_results(agent_results)
                    for num_process in range(num_processes):
                        training_actions[num_process].append(
                            actions[num_process])
                        training_probs[num_process].append(
                            probs[num_process])

                    action_choices.extend(actions)
                    for num in range(action_space.n):
                        action_probs[num].extend([p[num] for p in probs])

                non_training_obs = envs.get_non_training_obs()
                if hasattr(bad_guys_train[0], 'is_simple_agent'):
                    non_training_actions = envs.get_expert_actions(
                        non_training_obs)
                else:
                    non_training_actions = bad_guys_train[0].act(
                        non_training_obs, action_space)
                    non_training_actions = non_training_actions.reshape(
                        (num_processes, 2))

                for num_agent in range(4):
                    for num_process in range(num_processes):
                        is_training_agent = any([
                            num_process % 2 == 0 and num_agent in [0, 2],
                            num_process % 2 == 1 and num_agent in [1, 3]
                        ])
                        if is_training_agent:
                            actions = training_actions[num_process]
                            action = actions[num_agent // 2]
                            cpu_actions_agents[num_process].append(action)
                        else:
                            actions = non_training_actions[num_process]
                            action = actions[num_agent // 2]
                            cpu_actions_agents[num_process].append(action)
            elif how_train == 'grid':
                training_agent = training_agents[0]
                result = training_agent.actor_critic_act(
                    step, 0, deterministic=args.eval_only)
                cpu_actions_agents, cpu_probs = update_actor_critic_results(result)
                action_choices.extend(cpu_actions_agents)
                for num in range(action_space.n):
                    action_probs[num].extend([p[num] for p in cpu_probs])
            obs, reward, done, info = envs.step(cpu_actions_agents)

            reward = reward.astype(np.float)
            update_stats(info)
            game_ended = np.array([done_.all() for done_ in done])
            for num_process, ended_ in enumerate(game_ended):
                game_step_counts[num_process] += 1
                if ended_:
                    step_count = game_step_counts[num_process]
                    running_total_game_step_counts.append(step_count)
                    info_ = info[num_process]
                    optimal = info_.get('optimal_num_steps')
                    game_state_file = info_.get('game_state_file')
                    is_win = info_.get('result', -1) == constants.Result.Win
                    if optimal:
                        running_optimal_info.append((
                            optimal, step_count, step_count - optimal))
                        if game_state_file:
                            optimal_by_file[game_state_file].append((
                                step_count - optimal, is_win))
                    elif game_state_file:
                        optimal_by_file[game_state_file].append((0, is_win))
                    game_step_counts[num_process] = 0

            if how_train == 'simple':
                win, alive_win = get_win_alive(info, envs)
            elif how_train == 'backselfplay':
                position_wins, game_results = get_wins(info, envs)

            game_state_start_steps = np.array([
                info_.get('game_state_step_start') for info_ in info])
            game_state_start_steps_beg = np.array([
                info_.get('game_state_step_start_beg') for info_ in info])

            if args.render:
                envs.render(args.record_pngs_dir, game_step_counts, num_env=0,
                            game_type=game_type)

            if how_train in ['simple', 'grid']:
                # NOTE: The masking for simple should be such that:
                # - final_rewards is masked out on every step except for the
                # last step of a process. at that point, it becomes the
                # episode_rewards.
                # - episode_rewards accumulates the rewards at every step and
                # is masked out only at the last step of a process.
                # - current_obs consists of the num_stack observations. when
                # the game resets, the observations do as well, so we won't
                # have an issue with the frames overlapping. this means that we
                # shouldn't be using the masking on the current_obs.

                # NOTE: this mask is for the game: 0 if finished -
                # i.e. if the training agent is dead and the game ended
                # (both agents in any of the teams died or there is
                # at most one agent alive) --
                # used to get discounted rewards for the training agent
                # if it dies first but wins as a team
                masks = torch.FloatTensor([
                    [0.0] if ended_ else [1.0]
                for ended_ in game_ended])

                if game_type == constants.GameType.Team:
                    for num_process in range(num_processes):
                        self_reward = reward[num_process][0]
                        teammate_reward = reward[num_process][1]
                        my_reward = (1 - reward_sharing) * self_reward + \
                                    reward_sharing * teammate_reward
                        reward[num_process][0] = my_reward
                    # from here on we only need the training agent's reward & done
                    reward = np.array([[reward_[0]] for reward_ in reward])
                    done = np.array([[done_[0]] for done_ in done])

                    # NOTE: success only for the training agent
                    # this counts as success only the times when the
                    # training agent wins the game and is alive at the end
                    success_rate_alive += sum([int(s) for s in \
                                               ((alive_win == True) &
                                                (game_ended == True))])

                # NOTE: masks_kl is 0 if the agent die (game may still be going)
                # this must be executed after adjusting done to contain only
                # the learning agent's done
                masks_kl = torch.FloatTensor([
                    [0.0]*num_training_per_episode if done_ \
                    else [1.0]*num_training_per_episode
                for done_ in done])

                # NOTE: terminal_reward and success for the entire team this
                # counts as success all the times training agent's team wins
                # if config is team, it gets the shared reward
                running_num_episodes += sum([int(ended_)
                                            for ended_ in game_ended])
                terminal_reward += reward[game_ended == True].sum()
                success_rate += sum([int(s) for s in \
                                     ((game_ended == True) &
                                      (win == True))])
                if args.eval_only and any([done_ for done_ in done]):
                    print("Num completed %d --> %d success." % (
                        running_num_episodes, success_rate))
                    using_opt = False
                    if 'optimal_num_steps' in info[0]:
                        using_opt = True
                        num_optimal = len([k for k in running_optimal_info \
                                           if k[2] == 0])
                        avg_over = np.mean([k[2] for k in running_optimal_info])
                        std_over = np.std([k[2] for k in running_optimal_info])
                    if running_num_episodes >= 5000:
                        print("Num completed %d --> %d success." % (
                            running_num_episodes, success_rate))
                        if using_opt:
                            print('Num optimal %d / Avg optimal %.3f / Std optimal %.3f.' % (
                                num_optimal, avg_over, std_over))
                        if optimal_by_file:
                            print("\n")
                            means = {f: np.mean([k[0] for k in lst])
                                     for f, lst in sorted(optimal_by_file.items())}
                            is_wins = {f: 1.0*sum([k[1] for k in lst])/len(lst)
                                       for f, lst in sorted(optimal_by_file.items())}
                            counts = {f: len(lst) for f, lst in sorted(optimal_by_file.items())}
                            buckets = defaultdict(int)
                            for f in sorted(optimal_by_file):
                                if not using_opt:
                                    print(f,
                                          ", avg over optimal: %.3f, " % means[f],
                                          "percent wins %.3f, " % is_wins[f],
                                          "num ran %d." % counts[f])
                                elif means[f] == 0:
                                    buckets[0] += 1
                                else:
                                    next_five = ((means[f] // 5) + 1) * 5
                                    buckets[next_five] += 1
                            for bucket, count in sorted(buckets.items()):
                                print("Bucket %d: %d" % (bucket, count))
                            print("\n")
                        raise

                ### NOTE: Use the below if you want to make a game video of just
                ### the first successful game.
                # if game_ended[0] and win[0]:
                #     print("WE DONE YO")
                #     raise
                # elif game_ended[0]:
                #     print("DEL THAT SHIT")
                #     os.rmdir(args.record_pngs_dir)

                for e, w, ss, sb in zip(
                        game_ended, win, game_state_start_steps,
                        game_state_start_steps_beg):
                    if not e or ss is None or sb is None:
                        continue
                    if w:
                        start_step_wins[ss] += 1
                        start_step_wins_beg[sb] += 1
                    start_step_all[ss] += 1
                    start_step_all_beg[sb] += 1
            elif how_train == 'backselfplay':
                running_num_episodes += sum([int(done_.all())
                                             for done_ in done])
                per_agent_success_rate = [
                    per_agent_success_rate[id_] + position_wins[id_]
                    for id_ in range(4)
                ]
                all_agent_success_rate += sum([position_wins[id_] for id_ in range(4)])
                success_rate = all_agent_success_rate

                # NOTE: The masking for backselfplay should be such that:
                # 1. If the agent is alive, then it follows the same process as
                # in `simple`. This means that it's 1.0.
                # 2. If the agent died, then it's not going to get rewards.
                # This means the masking should be 0.0. Note that this
                # could be problematic for the rollout if the agent's
                # observations don't specify that it's dead. That's why we
                # amended the featurize3D function in networks to be a zero map
                # for the agent's position if it's not alive.
                # TODO: Consider additionally changing the agent's action and
                # associated log probs to be the Stop action.
                masks = torch.FloatTensor([[int(d) for d in done_]
                                           for done_ in done]) \
                             .transpose(0, 1).unsqueeze(2)

                for e, pos, ss, sb in zip(game_ended,
                                          game_results,
                                          game_state_start_steps,
                                          game_state_start_steps_beg):
                    if not e or ss is None or sb is None:
                        continue
                    if pos:
                        start_step_wins[ss] += 1
                        start_step_wins_beg[sb] += 1
                        start_step_position_wins[(pos, ss)] += 1
                        start_step_position_wins_beg[(pos, sb)] += 1
                    start_step_all[ss] += 1
                    start_step_all_beg[sb] += 1
            elif how_train == 'homogenous':
                # We have to clear any observations from done so that the stacks are pure.
                for num, done_ in enumerate(done):
                    if done_.all():
                        bad_guys_train[0].clear_obs_stack(num)

                running_num_episodes += sum([int(done_.all())
                                             for done_ in done])
                # NOTE: The masking for homogenous should be such that:
                # 1. If the agent is alive, then it follows the same process as
                # in `simple`. This means that it's 1.0.
                # 2. If the agent died, then it's still going to get rewards
                # according to the team_reward_sharing attribute. This means
                # that masking should be 1.0 as well. However, note that this
                # could be problematic for the rollout if the agent's
                # observations don't specify that it's dead. That's why we
                # amended the featurize3D function in networks to be a zero map
                # for the agent's position if it's not alive.
                # TODO: Consider additionally changing the agent's action and
                # associated log probs to be the Stop action.
                masks = torch.FloatTensor([[0.0]*2 if done_.all() else [1.0]*2
                                           for done_ in done]) \
                             .transpose(0, 1).unsqueeze(2).unsqueeze(2)
                masks_kl = [[0.0]*2 for _ in range(num_processes)]
                for num_process in range(num_processes):
                    temp_rewards = []
                    for id_ in range(2):
                        tid = (id_ + 1) % 2
                        self_reward = reward[num_process][id_]
                        teammate_reward = reward[num_process][tid]
                        my_reward = (1 - reward_sharing) * self_reward + \
                                        reward_sharing * teammate_reward
                        temp_rewards.append(my_reward)
                        if not done[num_process][id_]:
                            masks_kl[num_process][id_] = 1.0
                    reward[num_process] = temp_rewards

                # NOTE: masks_kl is 0 if the agent died.
                masks_kl = torch.FloatTensor(masks_kl).transpose(0, 1) \
                                                      .unsqueeze(2)

            reward = torch.from_numpy(np.stack(reward)).float().transpose(0, 1)
            episode_rewards += reward[:, :, None]
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            final_reward_arr = np.array(final_rewards.squeeze(0))
            if how_train in ['simple', 'grid']:
                final_sum = final_reward_arr[done.squeeze() == True].sum()
                cumulative_reward += final_sum
            elif how_train in ['homogenous', 'backselfplay']:
                where_done = np.array([done_.all() for done_ in done]) == True
                final_sum = final_reward_arr.squeeze().transpose()
                final_sum = final_sum[where_done].sum()
                cumulative_reward += final_sum

            current_obs = update_current_obs(obs)
            if args.cuda:
                masks = masks.cuda()
                current_obs = current_obs.cuda()

            if how_train in ['simple', 'grid']:
                masks_all = masks.transpose(0,1).unsqueeze(2)
            elif how_train in ['homogenous', 'backselfplay']:
                masks_all = masks

            reward_all = reward.unsqueeze(2)
            states_all = utils.torch_numpy_stack(states_agents)
            action_all = utils.torch_numpy_stack(action_agents)
            action_log_prob_all = utils.torch_numpy_stack(
                action_log_prob_agents)
            if do_distill:
                if distill_expert == 'DaggerAgent':
                    dagger_prob_distr = utils.torch_numpy_stack(dagger_prob_distr)
                elif distill_expert in ['SimpleAgent', 'ComplexAgent']:
                    dagger_prob_distr = utils.torch_numpy_stack(dagger_prob_distr,\
                        data=False)
                else:
                    raise ValueError("We only support distilling from \
                        DaggerAgent or SimpleAgent \n")
                action_log_prob_distr = utils.torch_numpy_stack(
                    action_log_prob_distr)
                if args.cuda:
                    dagger_prob_distr.cuda()

                action_log_prob_distr *= masks_kl
                dagger_prob_distr *= masks_kl
            else:
                dagger_prob_distr = None
                action_log_prob_distr = None

            value_all = utils.torch_numpy_stack(value_agents)

            if how_train in ['simple', 'homogenous', 'grid', 'backselfplay']:
                training_agents[0].insert_rollouts(
                    step, current_obs, states_all, action_all,
                    action_log_prob_all, value_all, reward_all, masks_all,
                    action_log_prob_distr, dagger_prob_distr)

        # Compute the advantage values.
        if not args.eval_only and how_train in ['simple', 'homogenous', 'grid', 'backselfplay']:
            training_agent = training_agents[0]
            next_value_agents = [
                training_agent.actor_critic_call(step=-1, num_agent=num_agent)
                for num_agent in range(num_training_per_episode)
            ]
            advantages = [
                training_agent.compute_advantages(
                    next_value_agents, args.use_gae, args.gamma, args.tau)
            ]

        # Run PPO Optimization.
        if args.eval_only:
            pass
        elif args.reinforce_only:
            for num_agent, agent in enumerate(training_agents):
                agent.set_train()
                with utility.Timer() as t:
                    result = agent.reinforce(advantages[num_agent],
                                       args.num_mini_batch, num_steps,
                                       args.max_grad_norm,
                                       kl_factor=distill_factor)
                pg_losses, kl_losses, total_losses, lr = result

                final_pg_losses[num_agent].extend(pg_losses)
                if do_distill:
                    final_kl_losses[num_agent].extend(kl_losses)
                final_total_losses[num_agent].extend(total_losses)

                agent.after_epoch()
                if args.half_lr_epochs > 0 and num_epoch > 0 and num_epoch % args.half_lr_epochs == 0:
                    agent.halve_lr()
        else:
            for num_agent, agent in enumerate(training_agents):
                agent.set_train()
                for _ in range(args.ppo_epoch):
                    with utility.Timer() as t:
                        result = agent.ppo(advantages[num_agent],
                                           num_mini_batch,
                                           batch_size,
                                           num_steps, args.clip_param,
                                           args.entropy_coef, args.value_loss_coef,
                                           args.max_grad_norm,
                                           kl_factor=distill_factor)
                    action_losses, value_losses, dist_entropies, \
                        kl_losses, total_losses, lr = result

                    final_action_losses[num_agent].extend(action_losses)
                    final_value_losses[num_agent].extend(value_losses)
                    final_dist_entropies[num_agent].extend(dist_entropies)
                    if do_distill:
                        final_kl_losses[num_agent].extend(kl_losses)
                    final_total_losses[num_agent].extend(total_losses)

                agent.after_epoch()
                if args.half_lr_epochs > 0 and num_epoch > 0 and num_epoch % args.half_lr_epochs == 0:
                    agent.halve_lr()

        total_steps += num_processes * num_steps

        if args.eval_only:
            pass
        elif running_num_episodes > args.log_interval:
            end = time.time()
            num_steps_sec = (end - start)
            num_episodes += running_num_episodes

            steps_per_sec = 1.0 * total_steps / (end - start)
            epochs_per_sec = 1.0 * (num_epoch - prev_epoch) / (end - start)
            episodes_per_sec =  1.0 * num_episodes / (end - start)

            if args.reinforce_only:
                mean_pg_loss = np.mean([
                    pg_loss for pg_loss in final_pg_losses])
                std_pg_loss = np.std([
                    pg_loss for pg_loss in final_pg_losses])

                mean_value_loss = None
                mean_action_loss = None
                mean_dist_entropy = None
                std_value_loss = None
                std_action_loss = None
                std_dist_entropy = None
            else:
                mean_value_loss = np.mean([
                    value_loss for value_loss in final_value_losses])
                std_value_loss = np.std([
                    value_loss for value_loss in final_value_losses])

                mean_action_loss = np.mean([
                    action_loss for action_loss in final_action_losses])
                std_action_loss = np.std([
                    action_loss for action_loss in final_action_losses])

                mean_dist_entropy = np.mean([
                    dist_entropy for dist_entropy in final_dist_entropies])
                std_dist_entropy = np.std([
                    dist_entropy for dist_entropy in final_dist_entropies])

                mean_pg_loss = None
                std_pg_loss = None

            mean_total_loss = np.mean([
                total_loss for total_loss in final_total_losses])
            std_total_loss = np.std([
                total_loss for total_loss in final_total_losses])

            # TODO: Compute stats for backselfplay wrt first or second player winning.
            if how_train == 'homogenous':
                win_rate, tie_rate, loss_rate = evaluate_homogenous(
                    args, good_guys, bad_guys_eval, eval_round, writer, num_epoch)
                for agent in good_guys + bad_guys_eval:
                    agent.clear_obs_stack()
                print("Epoch %d (%d)-> Win %.3f, Tie %.3f, Loss %.3f" % (
                    num_epoch, args.num_battles_eval, win_rate, tie_rate,
                    loss_rate))
                if win_rate >= .60:
                    suffix = suffix + ".wr%.3f.evlrnd%d" % (win_rate, eval_round)
                    saved_paths = utils.save_agents(
                            "ppo-", num_epoch, training_agents, total_steps,
                            num_episodes, args, suffix)
                    eval_round += 1
                    bad_guys_eval = [
                            utils.load_inference_agent(
                                    saved_paths[0], ppo_agent.PPOAgent, "ppo",
                                    action_space, obs_shape, args.num_processes // 2, args)
                            for _ in range(2)
                    ]
                    bad_guys_train = [
                            utils.load_inference_agent(
                                    saved_paths[0], ppo_agent.PPOAgent, "ppo",
                                    action_space, obs_shape, args.num_processes, args)
                            for _ in range(2)
                    ]

            if do_distill and len(final_kl_losses):
                mean_kl_loss = np.mean([
                    kl_loss for kl_loss in final_kl_losses])
                std_kl_loss = np.std([
                    kl_loss for kl_loss in final_kl_losses])
            else:
                mean_kl_loss = None

            start_step_ratios = {k:1.0 * start_step_wins.get(k, 0) / v
                                 for k, v in start_step_all.items()}
            start_step_beg_ratios = {k:1.0 * start_step_wins_beg.get(k, 0) / v
                                     for k, v in start_step_all_beg.items()}
            start_step_position_ratios = {
                (pos, ss):1.0 * v / start_step_all.get(ss, 1)
                for (pos, ss), v in start_step_position_wins.items()}
            start_step_position_beg_ratios = {
                (pos, sb):1.0 * v / start_step_all_beg.get(sb, 1)
                for (pos, sb), v in start_step_position_wins_beg.items()}

            utils.log_to_console(num_epoch, num_episodes, total_steps,
                                 steps_per_sec, epochs_per_sec, final_rewards,
                                 mean_dist_entropy, mean_value_loss,
                                 mean_action_loss, cumulative_reward,
                                 terminal_reward, success_rate,
                                 success_rate_alive, running_num_episodes,
                                 mean_total_loss, mean_kl_loss, mean_pg_loss,
                                 distill_factor, args.reinforce_only,
                                 start_step_ratios, start_step_beg_ratios,
                                 running_optimal_info,
                                 start_step_position_ratios,
                                 start_step_position_beg_ratios)

            utils.log_to_tensorboard(writer, num_epoch, num_episodes,
                                     total_steps, steps_per_sec,
                                     episodes_per_sec, final_rewards,
                                     mean_dist_entropy, mean_value_loss,
                                     mean_action_loss, std_dist_entropy,
                                     std_value_loss, std_action_loss,
                                     count_stats, array_stats,
                                     cumulative_reward, terminal_reward,
                                     success_rate, success_rate_alive,
                                     running_num_episodes, mean_total_loss,
                                     mean_kl_loss, mean_pg_loss, lr,
                                     distill_factor, args.reinforce_only,
                                     start_step_ratios, start_step_all,
                                     start_step_beg_ratios, start_step_all_beg,
                                     bomb_penalty_lambda,
                                     np.array(action_choices),
                                     np.array(action_probs), uniform_v,
                                     np.mean(running_success_rate),
                                     running_total_game_step_counts,
                                     running_optimal_info,
                                     start_step_position_ratios,
                                     start_step_position_beg_ratios
                                     per_agent_success_rate)


            start_step_all = defaultdict(int)
            start_step_wins = defaultdict(int)
            start_step_all_beg = defaultdict(int)
            start_step_wins_beg = defaultdict(int)
            start_step_positions_wins = defaultdict(int)
            start_step_position_wins_beg = defaultdict(int)

            if args.state_directory_distribution == 'uniformAdapt':
                rate_ = 1.0 * success_rate / running_num_episodes
                running_success_rate.append(rate_)
                if len(running_success_rate) == running_success_rate_maxlen \
                   and np.mean(running_success_rate) > .8:
                    print("Updating Mean Success Rate: ", uniform_v, running_success_rate)
                    uniform_v = int(uniform_v * uniform_v_factor)
                    envs.set_uniform_v(uniform_v)
                    running_success_rate = deque(
                        [], maxlen=running_success_rate_maxlen)

            # Reset stats so that plots are per the last log_interval.
            if args.reinforce_only:
                final_pg_losses = [[] for agent in range(len(training_agents))]
            else:
                final_action_losses = [[] for agent in range(len(training_agents))]
                final_value_losses =  [[] for agent in range(len(training_agents))]
                final_dist_entropies = [[] for agent in \
                                        range(len(training_agents))]
            if do_distill:
                final_kl_losses = [[] for agent in range(len(training_agents))]
            final_total_losses =  [[] for agent in range(len(training_agents))]

            count_stats = defaultdict(int)
            array_stats = defaultdict(list)
            final_rewards = torch.zeros([num_training_per_episode,
                                         num_processes, 1])
            running_total_game_step_counts = []
            running_optimal_info = []
            running_num_episodes = 0
            cumulative_reward = 0
            terminal_reward = 0
            all_agent_success_rate = 0
            per_agent_success_rate = [0]*4
            success_rate = 0
            success_rate_alive = 0
            prev_epoch = num_epoch
            action_choices = []
            action_probs = [[] for _ in range(6)]

        if args.eval_only:
            pass
        elif any([
                args.state_directory_distribution.startswith('uniformSchedul'),
                args.state_directory_distribution.startswith('uniformBounds'),
                args.state_directory_distribution.startswith('uniformForward'),
        ]) and num_epoch - uniform_v_incr >= uniform_v_prior:
            uniform_v_prior = num_epoch
            uniform_v = int(uniform_v * uniform_v_factor)
            envs.set_uniform_v(uniform_v)
        elif args.state_directory_distribution.startswith('grUniformBounds') \
             and num_epoch - uniform_v_incr >= uniform_v_prior:
            uniform_v_prior = num_epoch
            uniform_v = min(int(uniform_v * uniform_v_factor), 128)
            envs.set_uniform_v(uniform_v)

    writer.close()


def evaluate_homogenous(args, good_guys, bad_guys, eval_round, writer, epoch):
    print("Starting homogenous eval at epoch %d..." % epoch)
    with utility.Timer() as t:
        wins, one_dead, ties, losses = run_eval(
                args=args, targets=good_guys, opponents=bad_guys)
    print("Eval took %.4fs." % t.interval)

    descriptor = 'homogenous_eval_round%d' % eval_round
    num_battles = args.num_battles_eval

    win_count = len(wins)
    tie_count = len(ties)
    loss_count = len(losses)
    one_dead_count  = len(one_dead)

    mean_win_time = np.mean(wins)
    mean_tie_time = np.mean(ties)
    mean_loss_time = np.mean(losses)
    mean_all_time = np.mean(wins + ties + losses)

    win_rate = 1.0*win_count/num_battles
    tie_rate = 1.0*tie_count/num_battles
    loss_rate = 1.0*loss_count/num_battles
    one_dead_per_battle = 1.0*one_dead_count/num_battles
    one_dead_per_win = 1.0*one_dead_count/win_count if win_count else 0
    writer.add_scalar('eval_round', eval_round, epoch)
    writer.add_scalar('%s/win_rate' % descriptor, win_rate, epoch)
    writer.add_scalar('%s/tie_rate' % descriptor, tie_rate, epoch)
    writer.add_scalar('%s/loss_rate' % descriptor, loss_rate, epoch)
    if not np.isnan(mean_win_time):
        writer.add_scalar('%s/mean_win_time' % descriptor, mean_win_time, epoch)
    if not np.isnan(mean_tie_time):
        writer.add_scalar('%s/mean_tie_time' % descriptor, mean_tie_time, epoch)
    if not np.isnan(mean_loss_time):
        writer.add_scalar('%s/mean_loss_time' % descriptor, mean_loss_time, epoch)

    writer.add_scalar('%s/mean_all_time' % descriptor, mean_all_time, epoch)
    writer.add_scalar('%s/one_dead_per_battle' % descriptor,
                                             one_dead_per_battle, epoch)
    writer.add_scalar('%s/one_dead_per_win' % descriptor, one_dead_per_win,
                                             epoch)
    return win_rate, tie_rate, loss_rate


def evaluate_simple(args, good_guys, bad_guys, eval_round, writer, epoch):
    print("Starting simple eval at epoch %d..." % epoch)
    descriptor = 'simple_eval_round%d' % eval_round
    num_battles = args.num_battles_eval

    win_count = 0; tie_count = 0; loss_count = 0; rank_count = 0;
    mean_win_time = 0; mean_tie_time = 0; mean_loss_time = 0; mean_all_time = 0;
    for i in range(num_battles):
        with utility.Timer() as t:
            wins, ties, losses, ranks = run_eval(
                args=args, targets=good_guys, opponents=bad_guys, nbattle=i)
        print("Eval took %.4fs." % t.interval)

        win_count += len(wins)
        tie_count += len(ties)
        loss_count += len(losses)
        rank_count  += len(ranks)

        mean_win_time += np.mean(wins)
        mean_tie_time += np.mean(ties)
        mean_loss_time += np.mean(losses)
        mean_all_time += np.mean(wins + ties + losses)

    mean_win_time = 1.0 * mean_win_time / num_battles
    mean_tie_time = 1.0 * mean_tie_time / num_battles
    mean_loss_time = 1.0 * mean_loss_time / num_battles
    mean_all_time = 1.0 * mean_all_time / num_battles

    win_rate = 1.0*win_count/num_battles
    tie_rate = 1.0*tie_count/num_battles
    loss_rate = 1.0*loss_count/num_battles
    rank_per_battle = 1.0*rank_count/num_battles
    rank_per_win = 1.0*rank_count/win_count if win_count else 0
    writer.add_scalar('%s/win_rate' % descriptor, win_rate, epoch)
    writer.add_scalar('%s/tie_rate' % descriptor, tie_rate, epoch)
    writer.add_scalar('%s/loss_rate' % descriptor, loss_rate, epoch)
    writer.add_scalar('%s/mean_win_time' % descriptor, mean_win_time, epoch)
    writer.add_scalar('%s/mean_tie_time' % descriptor, mean_tie_time, epoch)
    writer.add_scalar('%s/mean_loss_time' % descriptor, mean_loss_time, epoch)
    writer.add_scalar('%s/mean_all_time' % descriptor, mean_all_time, epoch)
    writer.add_scalar('%s/rank_per_battle' % descriptor,
                      rank_per_battle, epoch)
    writer.add_scalar('%s/rank_per_win' % descriptor, rank_per_win,
                      epoch)
    return win_rate, tie_rate, loss_rate


if __name__ == "__main__":
    train()
