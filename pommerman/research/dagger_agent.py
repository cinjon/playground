"""
Dagger Agent.
"""
from pommerman import characters
import torch.nn as nn
import torch.optim as optim

from research_agent import ResearchAgent


class DaggerAgent(ResearchAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""
    def __init__(self, actor_critic, character=characters.Bomber, **kwargs):
        self._actor_critic = actor_critic
        super(DaggerAgent, self).__init__(character, **kwargs)

    def cuda(self):
        self._actor_critic.cuda()

    @property
    def model(self):
        return self._actor_critic

    @property
    def optimizer(self):
        return self._optimizer

    def act_on_data(self, observations, states, masks, deterministic=False):
        return self._actor_critic.act(observations, states, masks,
                                      deterministic)

    def _evaluate_actions(self, observations, states, masks, actions):
        return self._actor_critic.evaluate_actions(observations, states, masks,
                                                   actions)

    def set_eval(self):
        self._actor_critic.eval()

    def set_train(self):
        self._actor_critic.train()

    def get_action_scores(self, observations, states, masks):
        return self._actor_critic.get_action_scores(observations, states,
                                                    masks)

    def get_values_action_scores(self, observations, states, masks):
        return self._actor_critic.get_values_action_scores(observations, states,
                                                    masks)

    # TODO: debug the stop_grads_value implementation
    def optimize(self, action_classification_loss, value_loss, max_grad_norm,
                 use_value_loss=False, stop_grads_value=False, add_nonlin=False):
        self._optimizer.zero_grad()
        if use_value_loss:
            if stop_grads_value:
                action_classification_loss.backward(retain_graph=True)
                for p in self._actor_critic.parameters():
                    p.requires_grad = False
                self._actor_critic.critic_linear.requires_grad = True
                if add_nonlin:
                    self._actor_critic.fc_critic.requires_grad = True
                value_loss.backward()
            else:
                (action_classification_loss + value_loss).backward()

        else:
            action_classification_loss.backward()

        nn.utils.clip_grad_norm(self._actor_critic.parameters(), max_grad_norm)
        self._optimizer.step()

        if use_value_loss:
            for p in self._actor_critic.parameters():
                p.requires_grad = True

    def initialize(self, args, obs_shape, action_space,
                   num_training_per_episode, num_episodes, total_steps,
                   num_epoch, optimizer_state_dict, num_steps, uniform_v,
                   uniform_v_prior):
        params = self._actor_critic.parameters()
        self._optimizer = optim.Adam(params, lr=args.lr, eps=args.eps)
        if optimizer_state_dict:
            self._optimizer.load_state_dict(optimizer_state_dict)
        self.num_episodes = num_episodes
        self.total_steps = total_steps
        self.num_epoch = num_epoch
        self.num_steps = num_steps
        self.uniform_v = uniform_v
        self.uniform_v_prior = uniform_v_prior
