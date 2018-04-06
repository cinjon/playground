import inspect
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical


# A temporary solution from the master branch.
# https://github.com/pytorch/pytorch/blob/7752fe5d4e50052b3b0bbc9109e599f8157febc0/torch/nn/init.py#L312
# Remove after the next version of PyTorch gets release.
def orthogonal(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor


def get_actor_critic(model):
    """Gets an actor critic from this """
    actor_critics = ['PommeCNNPolicySmall']

    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if name not in actor_critics:
            continue

        if name == model:
            return obj
    return None


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class _FFPolicy(nn.Module):
    def __init__(self):
        super(_FFPolicy, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, x, states = self(inputs, states, masks)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs, states

    def evaluate_actions(self, inputs, states, masks, actions):
        value, x, states = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy, states

    def get_action_scores(self, inputs, states, masks, deterministic=False):
        value, x, states = self(inputs, states, masks)
        return self.dist(x)


class PommeCNNPolicySmall(_FFPolicy):
    """Class implementing a policy.

    Args:
      state_dict: The state dict from which we are loading. If this is None,
        then initializes anew.
      num_inputs: The int number of inputs to the convnet.
      action_space: The action space from the environment.
      board_size: The size of the game board (13).
      num_channels: The number of channels to use in the convnet.
    """
    def __init__(self, state_dict, num_inputs, action_space, board_size,
                 num_channels):
        super(PommeCNNPolicySmall, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_inputs, self.num_channels, 3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)

        # NOTE: should it go straight to 512?
        self.fc1 = nn.Linear(
            self.num_channels*(self.board_size)*(self.board_size), 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, 1)

        self.dist = Categorical(512, action_space.n)
        self.train()
        self.reset_parameters()
        if state_dict:
            self.load_state_dict(state_dict)

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(_weights_init)

        relu_gain = nn.init.calculate_gain('relu')

        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        self.fc1.weight.data.mul_(relu_gain)
        self.fc2.weight.data.mul_(relu_gain)

    def forward(self, inputs, states, masks):
        x = F.relu(self.conv1(inputs)) # 2x256x13x13
        x = F.relu(self.conv2(x)) # 2x256x13x13
        x = F.relu(self.conv3(x)) # 2x256x13x13
        x = F.relu(self.conv4(x)) # 2x256x13x13

        x = x.view(-1, self.num_channels * self.board_size**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.critic_linear(x), x, states
