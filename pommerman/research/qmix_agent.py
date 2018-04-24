from research_agent import ResearchAgent
from pommerman import characters

import torch
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy
import networks


class QMIXMetaAgent(ResearchAgent):
    """
    This agent acts as a wrapper for multiple QMIX-trained agents. The agent networks
    are shared and easier to keep in one place
    """
    def __init__(self, qmix_net, character=characters.Bomber, **kwargs):
        super(QMIXMetaAgent, self).__init__(character, **kwargs)
        self.qmix_net = qmix_net
        self.target_qmix_net = deepcopy(qmix_net)

    def cuda(self):
        self.qmix_net.cuda()
        self.target_qmix_net.cuda()

    @property
    def model(self):
        return self.qmix_net

    def set_eval(self):
        self.qmix_net.eval()
        self.target_qmix_net.eval()

    def set_train(self):
        self.qmix_net.train()
        self.target_qmix_net.train()

    def act(self, global_obs, agent_obs):
        """
        :param global_obs: Batch of corresponding global obs, batch_size x global_obs_shape
        :param agent_obs: A batch of corresponding agent observations, batch_size x num_agents x obs_shape
        :return:
        """
        obs = networks.featurize3D(obs)
        obs = torch.from_numpy(obs)
        if self._cuda:
            obs = obs.cuda()
        self._obs_stack.append(obs)
        stacked_obs = list(self._obs_stack)
        if len(stacked_obs) < self._num_stack:
            prepend = [stacked_obs[0]]*(self._num_stack - len(stacked_obs))
            stacked_obs = prepend + stacked_obs
        stacked_obs = torch.cat(stacked_obs).unsqueeze(0).float()
        masks = torch.ones(1, 1)
        value, action, _, states = self._actor_critic.act(
            Variable(stacked_obs, volatile=True),
            Variable(self._states, volatile=True),
            Variable(masks, volatile=True),
            deterministic=True)
        self._states = states.data
        action = action.data.squeeze(1).cpu().numpy()[0]
        return action

    def initialize(self, args, obs_shape, action_space,
                   num_training_per_episode, num_episodes, total_steps,
                   num_epoch, optimizer_state_dict):
        params = self.qmix_net.parameters()
        self.optimizer = optim.Adam(params, lr=args.lr, eps=args.eps)
        if optimizer_state_dict:
            self.optimizer.load_state_dict(optimizer_state_dict)
        self.num_episodes = num_episodes
        self.total_steps = total_steps
        self.num_epoch = num_epoch
