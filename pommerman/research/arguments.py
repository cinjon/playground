import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # general for PPO
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (1e-5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (1)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (16).')
    parser.add_argument('--num-steps', type=int, default=100,
                        help='number of forward steps')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (4)')
    parser.add_argument('--num-mini-batch', type=int, default=2,
                        help='number of batches for ppo (2)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (0.2)')
    parser.add_argument('--num-stack', type=int, default=2,
                        help='number of frames to stack (2)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='log interval, one log per n updates (10)')
    parser.add_argument('--save-interval', type=int, default=250,
                        help='save interval, one save per n updates (10)')
    parser.add_argument('--num-frames', type=int, default=10e7,
                        help='number of frames to train (10e6)')
    parser.add_argument('--log-dir', default='./logs',
                        help='directory to save agent logs (./logs)')
    parser.add_argument('--save-dir', default='./trained_models',
                        help='directory to save models (./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (8097)')
    parser.add_argument('--run-name', default='',
                        help='save this run with this name. must be set')
    parser.add_argument('--distill-target', type=str, default='',
                        help='local path to target model and model type to '
                        'which we will distill the PPO agent, e.g. '
                        'dagger::/path/to/model.pt.')
    parser.add_argument('--distill-epochs', type=int, default=200,
                        help='the number of training epochs over which we '
                        'distill the distill-target into the model. at epoch '
                        '0, the probability is 1.0 (i.e. only use target), at '
                        'epoch distill-step and onwards, it is 0.0.')
    parser.add_argument('--init-kl-factor', type=float, default=1.0,
                        help='the initial kl factor.')
    parser.add_argument('--set-distill-kl', type=float, default=-1.0,
                        help='if we want a constant factor for the kl distill '
                        'loss, then set this to be that nonnegative value.')
    parser.add_argument('--restart-counts', default=False, action='store_true',
                        help='if True, then restart the saved counts for a '
                        'loaded model')
    parser.add_argument('--distill-expert', type=str, default=None,
                        help='expert to use for distillation: \
                        SimpleAgent/DaggerAgent')
    parser.add_argument('--use-lr-scheduler', default=False, action='store_true',
                        help='whether to use the pytorch ReduceLROnPlateau')
    parser.add_argument('--half-lr-epochs', default=0, type=int,
                        help='after how many epochs to halve the learning rate. '
                        'if 0, this never halves the learning rate.')
    parser.add_argument('--suffix', type=str, default='',
                        help='added suffix for the model to use.')
    parser.add_argument('--reinforce-only', action='store_true', default=False,
                        help='train only with reinforce')
    parser.add_argument('--eval-only', action='store_true', default=False,
                        help='run train script but only do eval.')
    parser.add_argument('--use-second-place', action='store_true', default=False,
                        help='whether we run from 2nd place instead of 1st.')
    parser.add_argument('--use-both-places', action='store_true', default=False,
                        help='whether we use both positions (1st and 2nd).')
    parser.add_argument('--add-nonlin-valhead', action='store_true', default=False,
                        help='add nonlinearity to value head')
    parser.add_argument('--batch-size', type=int, default=5120,
                        help='batch size used for training')
    parser.add_argument('--anneal-bomb-penalty-epochs', type=int, default=0,
                        help='number of epochs to anneal from 0 --> 1 the neg '
                        'reward that the training agents receive from dying. '
                        'disabled if set to 0.')
    parser.add_argument('--recurrent-policy', default=False, action='store_true',
                        help='whether to use recurrency in the policy networks')

    # specific to Pommerman
    parser.add_argument('--agents',
                        default=','.join(['simple::null']*4),
                        help='Comma delineated list of agent types to run in '
                        'run_battle or generate_game_data.')
    parser.add_argument('--board-size', type=int, default=11,
                        help='size of the board')
    parser.add_argument('--config', type=str, default='PommeFFA-v3',
                        help='Configuration to execute. See env_ids in '
                        'configs.py for options.')
    parser.add_argument('--num-agents', type=int, default=1,
                        help='number of training agents. independent of the '
                        'number of agents battling.')
    parser.add_argument('--model-str', type=str, default='PommeCNNPolicySmall',
                        help='str name of model (models.py) we are using.')
    parser.add_argument('--saved-paths', type=str, default='',
                        help='comma separated paths to the saved models.')
    parser.add_argument('--game-state-file', type=str, default='',
                        help='a game state file from which to load.')
    parser.add_argument('--uniform-v-factor', type=float, default=1.5,
                        help='a factor for which to multiple the uniform_v')
    parser.add_argument('--state-directory', type=str, default='',
                        help='a game state directory from which to load.')
    parser.add_argument('--state-directory-distribution', type=str,
                        default='', help='a distribution to load the '
                        'states in the directory. uniform will choose on'
                        'randomly. for the others, see envs.py.')
    parser.add_argument('--how-train', type=str, default='simple',
                        help='how to train: simple, homogenous, heterogenous, '
                        'dagger, backselfplay.')
    parser.add_argument('--homogenous-init', type=str, default='self',
                        help='whether the initial opponent for homomgeous is '
                        'self or simple agent.')
    parser.add_argument('--step-loss', type=float, default=0.0,
                        help='the loss to apply per-step. this should be '
                        'positive if used. 0.0 otherwise.')
    parser.add_argument('--bomb-reward', type=float, default=0.0,
                        help='the reward to apply per-step for using the bomb '
                        'action. this should be positive if used. default=0.')
    parser.add_argument('--item-reward', type=float, default=0.0,
                        help='the per-step reward for picking up an item.')
    parser.add_argument('--num-channels', type=int, default=256,
                        help='number of channels in the convolutional layers')
    parser.add_argument('--render', default=False, action='store_true',
                        help='whether to render the first process.')
    parser.add_argument('--render-mode',
                        default='rgb_pixel',
                        help="What mode to render. Options are human, "
                        "rgb_pixel, and rgb_array.")
    parser.add_argument('--eval-render', default=False, action='store_true',
                        help='whether to render the first process in eval.')
    parser.add_argument('--cuda-device', type=int, default=0,
                        help='gpu id to be used')
    parser.add_argument('--do-filter-team', default=True, action='store_true',
                        help='whether we should filter the full team: \
                        if True, for simple and team, \
                        env.step returns the rewards and done for both \
                        the training agent and its teammate')
    parser.add_argument('--begin-selfbombing-epoch', default=0, type=int,
                        help='the epoch at which to begin dying from your own '
                        'bombs. this applies to all agents, even simple ones. ')

    ### Eval Specific
    parser.add_argument('--eval-mode', type=str, default='ffa',
                        help='mode for evaluation. see eval.py for options.')
    parser.add_argument('--eval-targets', type=str, default='',
                        help='local path to target comma-delineated model(s). '
                        'each model is consists of a path to the model and '
                        'the model type, e.g. "ppo::/path/to/model". '
                        'not all modes expect multiple models. will retrieve '
                        'from ssh location if provided.')
    parser.add_argument('--eval-opponents', type=str, default='',
                        help='similar to eval-targets but for opponents. if '
                        'looking to test against SimpleAgents, then use '
                        'simple::null.')
    parser.add_argument('--ssh-address', type=str, default='',
                        help='ssh address, e.g. resnick@access.cims.nyu.edu. '
                        'if empty, then assumed that we are using local.')
    parser.add_argument('--ssh-password', type=str, default='',
                        help='ssh password to copy over the model.')
    parser.add_argument('--ssh-save-model-local', type=str, default='',
                        help='directory where to save a ssh model locally.')
    parser.add_argument('--num-battles-eval', type=int, default=100,
                        help='number of battles to run for evaluation.')
    parser.add_argument('--record-pngs-dir',
                        default=None,
                        help='Directory to record the PNGs of the game. '
                        "Doesn't record if None.")
    parser.add_argument('--record-json-dir',
                        default=None,
                        help='Directory to record the JSON representations of '
                        "the game. Doesn't record if None.")

    ### Team Specific
    parser.add_argument('--reward-sharing', type=float, default=0.5,
                        help="what percent p of the reward is blended between "
                        "the team. agent a's reward is p*r_b + (1-p)*r_a. the "
                        "default is 0.5, which means that the agents share "
                        "all the rewards. 0.0 would be selfish, 1.0 selfless.")

    # for Dagger
    parser.add_argument('--expert-prob', type=float, default=0.5,
                        help='probability that the agent will act using \
                        the experts action')
    parser.add_argument('--anneal-factor', type=float, default=0.005,
                        help='probability that the agent will act using \
                        the experts action')
    parser.add_argument('--anneal-expert-prob', action='store_true', default=False,
                        help='anneal the probability of using the expert')
    parser.add_argument('--minibatch-size', type=int, default=5120,
                        help='size of the minibatch for training on \
                        the aggregated dataset')
    parser.add_argument('--max-aggregate-agent-states', type=int, default=50000,
                        help='Max size of the aggregate agent states.')
    parser.add_argument('--num-steps-eval', type=int, default=500,
                        help='number of steps in the evaluation of the Dagger')
    parser.add_argument('--dagger-epoch', type=int, default=1,
                        help='number of optimization steps for the \
                        dagger classifier (4)')
    parser.add_argument('--init-dagger-optimizer', action='store_true', default=False,
                        help='anneal the probability of using the expert')
    parser.add_argument('--scale-weights', action='store_true', default=False,
                        help='scale weights before each training loop')
    parser.add_argument('--weight-scale-factor', type=float, default=0.5,
                        help='factor for scaling the weights before each \
                        training loop (0.5)')
    parser.add_argument('--num-episodes-dagger', type=int, default=10,
                        help='number of episodes to collect data from (10)')
    parser.add_argument('--stop-grads-value', action='store_true', default=False,
                        help='do not backprop the value loss through the \
                        shared params of the policy and value networks if True')
    parser.add_argument('--use-value-loss', action='store_true', default=False,
                        help='use the value loss to update the network params \
                        when training the dagger agent')

    # for QMIX
    parser.add_argument('--buffer-size', type=int, default=int(1e6),
                        help='size of the replay buffer (default 1e6)')
    parser.add_argument('--eps-max', type=float, default=1.0,
                        help='maximum epsilon for epsilon-greedy action selection (default 1.0)')
    parser.add_argument('--eps-min', type=float, default=0.1,
                        help='minimum epsilon for epsilon-greedy action selection (default 0.1)')
    parser.add_argument('--eps-max-steps', type=int, default=100000,
                        help='maximum number of steps to linearly anneal epsilon over (default 100000)')
    parser.add_argument('--target-update-steps', type=int, default=2,
                        help='number of gradient steps to update target network after (default 2)')

    # for Generating Game Data
    parser.add_argument('--num-episodes', type=int, default=2,
                        help='number of episodes for which to generate data.')
    parser.add_argument('--prob-optim', type=float, default=1.0,
                        help='probability that the expert (used for replay trajectories) \
                        takes the optimal action.')
    parser.add_argument('--num-more-optimal', type=float, default=0,
                        help='the number greater than optimal that we accept')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
