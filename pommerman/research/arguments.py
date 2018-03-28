import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # general for PPO
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=6, # TODO: Change back to 16.
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=3, # TODO: Change back to more.
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--num-layers', type=int, default=13,
                        help='number of layers in the Resnet')
    parser.add_argument('--model', type=str, default='convnet',
                        help='neural net architecture of the policy')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=20,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--num-stack', type=int, default=3,
                        help='number of frames to stack (default: 4)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10000,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='Pommerman',
                        help='environment to train on (default: Pommerman) other options: PongNoFrameskip-v4)')
    parser.add_argument('--log_dir', default='/home/roberta/pomme_logs/stats',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='/home/roberta/pomme_logs/trained_models',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')

    # specific to Pommerman
    parser.add_argument('--config', type=str, default='ffa_v0',
                        help='game configuration such as Free For All (ffa_v0), Team Random (team_v0), Team Radio \
                        (default: ffa_v0) options: ffa_v0 | ffa_v0_fast | ffa_v1 | team_v0 | radio_v2')
    parser.add_argument('--nagents', type=int, default=1,
                        help='number of agents to train. this is independent of the number of agents in a battle.')
    parser.add_argument('--saved-models', type=str, default='',
                        help='comma delineated paths to where the nagent # of agents are saved.')
    parser.add_argument('--game-state-file', type=str, default='',
                        help='a game state file from which to load.')
    parser.add_argument('--how-train', type=str, default='simple',
                        help='how to train agents: simple, homogenous, heterogenous.')
    parser.add_argument('--num-channels', type=int, default=256,
                        help='number of channels in the convolutional layers')
    # TODO: Remove this. It's always 13.
    parser.add_argument('--board_size', type=int, default=13,
                        help='size of the board')

    parser.add_argument('--server', type=str, default='http://216.165.70.24',
                        help='how to train agents: simple, homogenous, heterogenous.')

    parser.add_argument('--num-steps-eval', type=int, default=1000,
                        help='number of steps to run for evaluation')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    return args
