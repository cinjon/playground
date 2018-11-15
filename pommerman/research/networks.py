import inspect
import sys

import numpy as np
import pommerman
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical as TorchCategorical
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
    actor_critics = ['PommeCNNPolicySmall',
                     'PommeCNNPolicySmaller',
                     'PommeCNNPolicySmallNonlinCritic',
                     'PommeCNNPolicySmallerNonlinCritic',
                     'GridCNNPolicy',
                     'GridCNNPolicyNonlinCritic']

    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if name not in actor_critics:
            continue

        if name == model:
            return obj
    return None


def get_q_network(model):
    """Gets an actor critic from this """
    q_nets = ['QMIXNet']

    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if name not in q_nets:
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
        action_log_probs, dist_entropy, probs, log_probs = \
            self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs, states, probs, log_probs

    def evaluate_actions(self, inputs, states, masks, actions):
        value, x, states = self(inputs, states, masks)
        action_log_probs, dist_entropy, _, _ = self.dist.logprobs_and_entropy(
            x, actions)
        return value, action_log_probs, dist_entropy, states

    def get_action_scores(self, inputs, states, masks, deterministic=False):
        value, x, states = self(inputs, states, masks)
        return self.dist(x)

    def get_values_action_scores(self, inputs, states, masks, deterministic=False):
        value, x, states = self(inputs, states, masks)
        return value, self.dist(x)


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
                 num_channels, use_gru=False):
        super(PommeCNNPolicySmall, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.use_gru = use_gru
        self._init_network(num_inputs)
        self.dist = Categorical(512, action_space.n)
        self.train()
        self.reset_parameters()
        if state_dict:
            self.load_state_dict(state_dict)

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def _init_network(self, num_inputs):
        self.conv1 = nn.Conv2d(num_inputs, self.num_channels, 3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.fc1 = nn.Linear(
            self.num_channels*(self.board_size)*(self.board_size), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, 1)

        if self.use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal(self.gru.weight_ih.data)
            nn.init.orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

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
        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
        return self.critic_linear(x), x, states


class PommeCNNPolicySmaller(_FFPolicy):
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
                 num_channels, use_gru=False):
        super(PommeCNNPolicySmaller, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.use_gru = use_gru
        self._init_network(num_inputs)
        self.dist = Categorical(512, action_space.n)
        self.train()
        self.reset_parameters()
        if state_dict:
            self.load_state_dict(state_dict)

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def _init_network(self, num_inputs):
        self.conv1 = nn.Conv2d(num_inputs, self.num_channels, 3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.fc1 = nn.Linear(
            self.num_channels*(self.board_size)*(self.board_size), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, 1)

        if self.use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal(self.gru.weight_ih.data)
            nn.init.orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    def reset_parameters(self):
        self.apply(_weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.fc1.weight.data.mul_(relu_gain)
        self.fc2.weight.data.mul_(relu_gain)

    def forward(self, inputs, states, masks):
        x = F.relu(self.conv1(inputs)) # 2x256x13x13
        x = F.relu(self.conv2(x)) # 2x256x13x13
        x = x.view(-1, self.num_channels * self.board_size**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        return self.critic_linear(x), x, states


class PommeCNNPolicySmallNonlinCritic(_FFPolicy):
    """Class implementing a policy that adds extra nonlinearity to the value head.
    Args:
      state_dict: The state dict from which we are loading. If this is None,
        then initializes anew.
      num_inputs: The int number of inputs to the convnet.
      action_space: The action space from the environment.
      board_size: The size of the game board (13).
      num_channels: The number of channels to use in the convnet.
    """
    def __init__(self, state_dict, num_inputs, action_space, board_size,
                 num_channels, use_gru=False):
        super(PommeCNNPolicySmallNonlinCritic, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.use_gru = use_gru
        self._init_network(num_inputs)
        self.dist = Categorical(512, action_space.n)
        self.train()
        self.reset_parameters()
        if state_dict:
            self.load_state_dict(state_dict)

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def _init_network(self, num_inputs):
        self.conv1 = nn.Conv2d(num_inputs, self.num_channels, 3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.fc1 = nn.Linear(
            self.num_channels*(self.board_size)*(self.board_size), 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.fc_critic = nn.Linear(512, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, 1)

        if self.use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal(self.gru.weight_ih.data)
            nn.init.orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

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

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        y = F.tanh(self.fc_critic(x))

        return self.critic_linear(y), x, states

class PommeCNNPolicySmallerNonlinCritic(_FFPolicy):
    """Class implementing a policy that adds extra nonlinearity to the value head.
    Args:
      state_dict: The state dict from which we are loading. If this is None,
        then initializes anew.
      num_inputs: The int number of inputs to the convnet.
      action_space: The action space from the environment.
      board_size: The size of the game board (13).
      num_channels: The number of channels to use in the convnet.
    """
    def __init__(self, state_dict, num_inputs, action_space, board_size,
                 num_channels, use_gru=False):
        super(PommeCNNPolicySmallerNonlinCritic, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.use_gru = use_gru
        self._init_network(num_inputs)
        self.dist = Categorical(512, action_space.n)
        self.train()
        self.reset_parameters()
        if state_dict:
            self.load_state_dict(state_dict)

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def _init_network(self, num_inputs):
        self.conv1 = nn.Conv2d(num_inputs, self.num_channels, 3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.fc1 = nn.Linear(
            self.num_channels*(self.board_size)*(self.board_size), 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.fc_critic = nn.Linear(512, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, 1)

        if self.use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal(self.gru.weight_ih.data)
            nn.init.orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    def reset_parameters(self):
        self.apply(_weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.fc1.weight.data.mul_(relu_gain)
        self.fc2.weight.data.mul_(relu_gain)

    def forward(self, inputs, states, masks):
        x = F.relu(self.conv1(inputs)) # 2x256x13x13
        x = F.relu(self.conv2(x)) # 2x256x13x13
        x = x.view(-1, self.num_channels * self.board_size**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        y = F.tanh(self.fc_critic(x))

        return self.critic_linear(y), x, states


class QMIXNet(nn.Module):
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
                 num_channels, num_agents):
        super(QMIXNet, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.num_agents = num_agents
        self.action_space = action_space
        self.mixing_hidden_size = 32

        self.conv1 = nn.Conv2d(num_inputs, self.num_channels, 3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)

        self.fc1 = nn.Linear(
            self.num_channels*self.board_size*self.board_size, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.agent_q_net = nn.Linear(512, action_space.n)

        # @TODO need global state shape, until then manual entry for first two operands
        self.hyper_net1 = nn.Linear(
            (num_inputs // 2) * 4 * self.board_size**2,
            self.num_agents * self.mixing_hidden_size)
        self.hyper_net2 = nn.Linear(
            (num_inputs // 2) * 4 * self.board_size ** 2,
            self.mixing_hidden_size)

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

    def _epsilon_greedy(self, num_actions, max_actions, eps):
        batch_size = max_actions.shape[0]
        dist = np.ones((batch_size, num_actions), dtype=np.float32) * eps / num_actions
        dist[np.arange(batch_size), max_actions.cpu().data.numpy()] += 1.0 - eps
        dist = TorchCategorical(Variable(torch.from_numpy(dist).float()))
        sample = dist.sample()
        if max_actions.is_cuda:
            sample = sample.cuda()
        return sample

    def forward(self, global_state, agent_state, eps=-1.0):
        """
        :param global_state: batch_size x 4 x 19 x 13 x 13
        :param agent_state: batch_size x num_agents x 38 x 13 x 13
        :param eps: parameter for the epsilon greedy action selection, negative if don't want it
        :return:
        """
        batch_size, num_agents, *agent_obs_shape = agent_state.shape
        agents_flattened = agent_state.view(-1, *agent_obs_shape)  # (batch_size*num_agents) x 38 x 13 x 13
        global_state_flattened = global_state.view(batch_size, -1) # batch_size x (4*19*13*13)
        x = F.relu(self.conv1(agents_flattened)) # (batch_size*num_agents) x 256 x 13 x 13
        x = F.relu(self.conv2(x)) # (batch_size*num_agents) x 256 x 13 x 13
        x = F.relu(self.conv3(x)) # (batch_size*num_agents) x 256 x 13 x 13
        x = F.relu(self.conv4(x)) # (batch_size*num_agents) x 256 x 13 x 13

        x = x.view(-1, self.num_channels * self.board_size**2) # (batch_size*num_agents) x (256*13*13)
        x = F.relu(self.fc1(x)) # (batch_size*num_agents) x 1024
        x = F.relu(self.fc2(x)) # (batch_size*num_agents) x 512

        agent_q_n = self.agent_q_net(x) # (batch_size*num_agents) x num_actions
        max_q, max_actions = agent_q_n.max(dim=1)

        # During DQN batch update, we shouldn't do epsilon greedy and take what the network gives
        if eps > 0.0:
            max_actions = self._epsilon_greedy(self.action_space.n, max_actions, eps)
            max_q = agent_q_n.gather(1, max_actions.unsqueeze(1))

        # Q-values for all agents in each batch
        batched_max_q = max_q.view(batch_size, -1).unsqueeze(1) # batch_size x 1 x num_agents
        batched_actions = max_actions.view(batch_size, -1) # batch_size x num_agents

        # Weights for the Mixing Network (absolute for monotonicity)
        w1 = self.hyper_net1(global_state_flattened).abs() # batch_size x (num_agents*32)
        w2 = self.hyper_net2(global_state_flattened).abs() # batch_size x 32

        # Reshape for Mixing Network
        w1 = w1.view(batch_size, self.num_agents, self.mixing_hidden_size) # batch_size x num_agents x 32
        w2 = w2.view(batch_size, self.mixing_hidden_size, 1) # batch_size x 32 x 1

        # Calculate mixing of agent values for q_tot
        batched_q_tot = F.elu(torch.bmm(batched_max_q, w1)) # batch_size x 1 x 32
        batched_q_tot = F.elu(torch.bmm(batched_q_tot, w2)) # batch_size x 1 x 1

        return batched_q_tot.squeeze(2), batched_actions # batch_size x 1, batch_size x num_agents


def featurize3D(obs, use_step=True):
    """Create 3D Feature Maps for Pommerman.
    Args:
        obs: The observation input. Should be for a single agent.
        use_step: Whether to include the step as an argument. We need this for
          the old dagger_agents. Can remove when updated.
    Returns:
        A 3D Feature Map where each map is bsXbs. The 19 features are:
        - (2) Bomb blast strength and Bomb life.
        - (1) Agent position, a single 1 on the map specifying the agent's
            location. This is all zeros if the agent is dead.
        - (3) Agent ammo, blast strength, can_kick.
        - (1) Whether has teammate.
        - (1 / 0) If teammate, then the teammate's position.
        - (2 / 3) Enemies' positions.
        - (8) Positions for:
            Passage/Rigid/Wood/Flames/ExtraBomb/IncrRange/Kick/Skull
        - (1) Step: Integer map in range [0, max_game_length=2500)
    """
    agent_dummy = pommerman.constants.Item.AgentDummy
    board = obs["board"]
    map_size = len(board)

    feature_maps = []

    # feature maps with ints for bomb blast strength and life.
    if "bomb_blast_strength" in obs:
        bomb_blast_strength = obs["bomb_blast_strength"] \
                              .astype(np.float32) \
                              .reshape(1, map_size, map_size)
        bomb_life = obs["bomb_life"].astype(np.float32) \
                                    .reshape(1, map_size, map_size)
        feature_maps.extend([bomb_blast_strength, bomb_life])

    # position of self. If the agent is dead, then this is all zeros.
    position = np.zeros((1, map_size, map_size)).astype(np.float32)
    if obs["is_alive"]:
        position[0, obs["position"][0], obs["position"][1]] = 1
    feature_maps.append(position)

    if "goal_position" in obs:
        # This is GridWorld.
        gx, gy = obs["goal_position"]
        mx, my = obs["position"]
        goal_position = np.zeros((1, map_size, map_size)).astype(np.float32)
        goal_position[0, gx, gy] = 1

        passages = board.copy()[None, :, :]
        passages[0, gx, gy] = 1
        passages[0, mx, my] = 1
        passages = 1 - passages

        walls = board.copy()[None, :, :]
        walls[0, gx, gy] = 0
        walls[0, mx, my] = 0
        feature_maps.extend([goal_position, passages, walls])

    # ammo of self agent: constant feature map.
    if "ammo" in obs:
        ammo = np.ones((1, map_size, map_size)).astype(np.float32) * obs["ammo"]
        feature_maps.append(ammo)

    # blast strength of self agent: constant feature map
    if "blast_strength" in obs:
        blast_strength = np.ones((1, map_size, map_size)).astype(np.float32)
        blast_strength *= obs["blast_strength"]
        feature_maps.append(blast_strength)

    # whether the agent can kick: constant feature map of 1 or 0.
    if "can_kick" in obs:
        can_kick = np.ones((1, map_size, map_size)).astype(np.float32)
        can_kick *= float(obs["can_kick"])
        feature_maps.append(can_kick)

    if "teammate" in obs:
        items = np.zeros((8, map_size, map_size))
        for num, i in enumerate([0, 1, 2, 4, 6, 7, 8, 9]):
            items[num][np.where(obs["board"] == i)] = 1
        feature_maps.append(items)

        if obs["teammate"] == agent_dummy:
            has_teammate = np.zeros((1, map_size, map_size)) \
                             .astype(np.float32)
            teammate = None
        else:
            has_teammate = np.ones((1, map_size, map_size)) \
                             .astype(np.float32)
            teammate = np.zeros((map_size, map_size)).astype(np.float32)
            teammate[np.where(obs["board"] == obs["teammate"].value)] = 1
            teammate = teammate.reshape(1, map_size, map_size)
        feature_maps.append(has_teammate)

        # Enemy feature maps.
        _enemies = [e for e in obs["enemies"] if e != agent_dummy]
        enemies = np.zeros((len(_enemies), map_size, map_size)) \
                    .astype(np.float32)
        for i in range(len(_enemies)):
            enemies[i][np.where(obs["board"] == _enemies[i].value)] = 1
        feature_maps.append(enemies)
    else:
        teammate = None

    # step count
    step = np.ones((1, map_size, map_size)).astype(np.float32) * obs["step"]
    if use_step:
        feature_maps.append(step)
    if teammate is not None:
        feature_maps.append(teammate)

    return np.concatenate(feature_maps)


class GridCNNPolicy(_FFPolicy):
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
                 num_channels, use_gru=False):
        super(GridCNNPolicy, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.use_gru = use_gru
        self._init_network(num_inputs)
        self.dist = Categorical(128, action_space.n)
        self.train()
        self.reset_parameters()
        if state_dict:
            self.load_state_dict(state_dict)

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 128
        else:
            return 1

    @property
    def output_size(self):
        return 128

    def _init_network(self, num_inputs):
        self.conv1 = nn.Conv2d(num_inputs, self.num_channels, 3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.fc1 = nn.Linear(
            self.num_channels*(self.board_size)*(self.board_size), 128)
        self.fc2 = nn.Linear(128, 128)
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, 1)

        if self.use_gru:
            self.gru = nn.GRUCell(128, 128)
            nn.init.orthogonal(self.gru.weight_ih.data)
            nn.init.orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    def reset_parameters(self):
        self.apply(_weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.fc1.weight.data.mul_(relu_gain)
        self.fc2.weight.data.mul_(relu_gain)

    def forward(self, inputs, states, masks):
        x = F.relu(self.conv1(inputs)) # 2x256x13x13
        x = F.relu(self.conv2(x)) # 2x256x13x13
        x = x.view(-1, self.num_channels * self.board_size**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        return self.critic_linear(x), x, states


class GridCNNPolicyNonlinCritic(_FFPolicy):
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
                 num_channels, use_gru=False):
        super(GridCNNPolicyNonlinCritic, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.use_gru = use_gru
        self._init_network(num_inputs)
        self.dist = Categorical(128, action_space.n)
        self.train()
        self.reset_parameters()
        if state_dict:
            self.load_state_dict(state_dict)

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 128
        else:
            return 1

    @property
    def output_size(self):
        return 128

    def _init_network(self, num_inputs):
        self.conv1 = nn.Conv2d(num_inputs, self.num_channels, 3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3,
                               stride=1, padding=1)
        self.fc1 = nn.Linear(
            self.num_channels*(self.board_size)*(self.board_size), 128)
        self.fc2 = nn.Linear(128, 128)

        self.fc_critic = nn.Linear(128, 128)
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, 1)


        if self.use_gru:
            self.gru = nn.GRUCell(128, 128)
            nn.init.orthogonal(self.gru.weight_ih.data)
            nn.init.orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    def reset_parameters(self):
        self.apply(_weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.fc1.weight.data.mul_(relu_gain)
        self.fc2.weight.data.mul_(relu_gain)

    def forward(self, inputs, states, masks):
        x = F.relu(self.conv1(inputs)) # 2x256x13x13
        x = F.relu(self.conv2(x)) # 2x256x13x13
        x = x.view(-1, self.num_channels * self.board_size**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        y = F.tanh(self.fc_critic(x))

        return self.critic_linear(y), x, states
