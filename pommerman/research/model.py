import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import orthogonal
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

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
        return self.dist.linear(x)


# TODO: what's the min number of layers for this to work?
class PommeCNNPolicySmall(FFPolicy):
    def __init__(self, num_inputs, action_space, args):
        super(PommeCNNPolicySmall, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(num_inputs, args.num_channels, 3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3,
                               stride=1, padding=1)

        # TODO: should it go straight to 512?
        self.fc1 = nn.Linear(
            args.num_channels*(args.board_size)*(args.board_size), 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, 1)

        self.dist = Categorical(512, action_space.n)
        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')

        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        self.fc1.weight.data.mul_(relu_gain)
        self.fc2.weight.data.mul_(relu_gain)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = F.relu(self.conv1(inputs)) # 2x256x13x13
        x = F.relu(self.conv2(x)) # 2x256x13x13
        x = F.relu(self.conv3(x)) # 2x256x13x13
        x = F.relu(self.conv4(x)) # 2x256x13x13

        board_size = self.args.board_size
        num_channels = self.args.num_channels
        x = x.view(-1, num_channels * (board_size)*(board_size))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.critic_linear(x), x, states


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(FFPolicy):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(FFPolicy):
    def __init__(self, block, layers, num_inputs, action_space, args):
        super(ResNet, self).__init__()

        self.args = args
        self.inplanes = args.num_channels

        self.conv1 = nn.Conv2d(num_inputs, args.num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, args.num_channels, layers[0])
        self.layer2 = self._make_layer(block, args.num_channels, layers[1])
        self.layer3 = self._make_layer(block, args.num_channels, layers[2])
        self.layer4 = self._make_layer(block, args.num_channels, layers[3])
        self.layer5 = self._make_layer(block, args.num_channels, layers[4])
        self.layer6 = self._make_layer(block, args.num_channels, layers[5])
        self.layer7 = self._make_layer(block, args.num_channels, layers[6])
        self.layer8 = self._make_layer(block, args.num_channels, layers[7])
        self.layer9 = self._make_layer(block, args.num_channels, layers[8])
        self.layer10 = self._make_layer(block, args.num_channels, layers[9])
        self.layer11 = self._make_layer(block, args.num_channels, layers[10])
        self.layer12 = self._make_layer(block, args.num_channels, layers[11])
        self.layer13 = self._make_layer(block, args.num_channels, layers[12])

        # policy head
        self.policy_conv = nn.Conv2d(args.num_channels, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU()
        # self.policy_linear = nn.Linear(2*(args.board_size)*(args.board_size), action_space.n)

        # value head
        self.value_conv = nn.Conv2d(args.num_channels, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm1d(1)
        self.value_relu1 = nn.ReLU()
        self.value_linear1 = nn.Linear(args.board_size*args.board_size, args.num_channels)
        self.value_relu2 = nn.ReLU()
        self.value_linear2 = nn.Linear(args.num_channels, 1)

        self.dist = Categorical(2*(args.board_size)*(args.board_size), action_space.n)

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.train()
        self.reset_parameters()   # XXX: do we need this or not?

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')

        self.conv1.weight.data.mul_(relu_gain)
        self.policy_conv.weight.data.mul_(relu_gain)
        self.value_conv.weight.data.mul_(relu_gain)

        # self.policy_linear.weight.data.mul_(relu_gain)
        self.value_linear1.weight.data.mul_(relu_gain)
        self.value_linear1.weight.data.mul_(relu_gain)

        if hasattr(self, 'gru'):
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = x.view(-1, self.args.num_channels*(self.args.board_size)*(self.args.board_size))

        # XXX: if we have recurrency, should we still have policy_linear or stop at relu here?
        pi = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        pi = pi.view(-1, 2*(self.args.board_size)*(self.args.board_size))
        # pi = self.policy_linear(pi)


        v = self.value_relu1(self.value_bn(self.value_conv(x)))
        v = v.view(-1, self.args.board_size*self.args.board_size)
        v = self.value_linear2(self.value_relu2(self.value_linear1(v)))


        # XXX: can we include the gru? what is the right way to do it?
        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                pi = states = self.gru(pi, states * masks)
            else:
                pi = pi.view(-1, states.size(0), pi.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(pi.size(0)):
                    hx = states = self.gru(pi[i], states * masks[i])
                    outputs.append(hx)
                pi = torch.cat(outputs, 0)

        return F.tanh(v), pi, states


def PommeResnetPolicy(num_inputs, action_space, args):
    model = ResNet(BasicBlock,
                   [2 for i in range(args.num_layers)],
                   num_inputs,
                   action_space,
                   args)
    return model
