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
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (0.2)')
    parser.add_argument('--num-stack', type=int, default=2,
                        help='number of frames to stack (2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (10)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='save interval, one save per n updates (10)')
    parser.add_argument('--num-frames', type=int, default=10e6,
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

    # specific to Pommerman
    parser.add_argument('--board_size', type=int, default=13,
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
    parser.add_argument('--how-train', type=str, default='simple',
                        help='how to train: simple, homogenous, heterogenous.')
    parser.add_argument('--num-channels', type=int, default=256,
                        help='number of channels in the convolutional layers')
    parser.add_argument('--render', default=False, action='store_true',
                        help='whether to render the first process.')

    parser.add_argument('--cims-address', type=str, default='',
                        help='cims address, e.g. resnick@access.cims.nyu.edu. '
                        'if empty, then assumed that we are using local.')
    parser.add_argument('--cims-password', type=str, default='',
                        help='cims password to copy over the model.')
    parser.add_argument('--cims-save-model-local', type=str, default='',
                        help='directory where to save a cims model locally.')
    parser.add_argument('--num-steps-eval', type=int, default=1000,
                        help='number of steps to run for evaluation')
    parser.add_argument('--target-eval-paths', type=str, default='',
                        help='path to saved models being tested, should be in '
                        'saved-paths as well. if >1, then assumes team.')

    parser.add_argument('--expert-prob', type=float, default=0.5,
                        help='probability that the agent will act using the experts action')
    parser.add_argument('--minibatch-size', type=int, default=100,
                        help='size of the minibatch for training on the aggregated dataset')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
