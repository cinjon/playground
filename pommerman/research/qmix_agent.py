from research_agent import ResearchAgent
from pommerman import characters

import torch
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy


class QMIXMetaAgent(ResearchAgent):
    """
    This agent acts as a wrapper for multiple QMIX-trained agents. The agent networks
    are shared and easier to keep in one place
    """
    def __init__(self, qmix_net, character=characters.Bomber, **kwargs):
        self._actor_critic = None # @TODO this is hack to fix nomenclature
        super(QMIXMetaAgent, self).__init__(character, **kwargs)
        self.qmix_net = qmix_net
        self.target_qmix_net = deepcopy(qmix_net)

    def cuda(self):
        self.qmix_net.cuda()
        self.target_qmix_net.cuda()

    @property
    def model(self):
        return self.qmix_net

    @property
    def optimizer(self):
        return self._optimizer

    def set_eval(self):
        self.qmix_net.eval()
        self.target_qmix_net.eval()

    def set_train(self):
        self.qmix_net.train()
        self.target_qmix_net.train()

    def act(self, global_obs, agent_obs, eps=-1.0):
        """
        :param global_obs: Batch of corresponding global obs, batch_size x global_obs_shape
        :param agent_obs: A batch of corresponding agent observations, batch_size x num_agents x obs_shape
        :param eps: value for epsilon greedy action, negative to disable epsilon greedy action selection
        :return:
        """
        critic_value, actions = self.qmix_net(Variable(global_obs), Variable(agent_obs), eps)
        return critic_value, actions

    def initialize(self, args, obs_shape, action_space,
                   num_training_per_episode, num_episodes, total_steps,
                   num_epoch, optimizer_state_dict):
        params = self.qmix_net.parameters()
        self._optimizer = optim.Adam(params, lr=args.lr, eps=args.eps)
        if optimizer_state_dict:
            self.optimizer.load_state_dict(optimizer_state_dict)
        self.num_episodes = num_episodes
        self.total_steps = total_steps
        self.num_epoch = num_epoch
