"""
Dagger Agent.
"""
from pommerman.agents import BaseAgent
from pommerman import characters
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from storage import RolloutStorage


class DaggerAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""
    def __init__(self, actor_critic, character=characters.Bomber):
        self._actor_critic = actor_critic
        super(DaggerAgent, self).__init__(character)

    def cuda(self):
        self._actor_critic.cuda()
        self._rollout.cuda()

    @property
    def model(self):
        return self._actor_critic

    @property
    def optimizer(self):
        return self._optimizer

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions."""
        return None

    def set_eval(self):
        self._actor_critic.eval()

    def set_train(self):
        self._actor_critic.train()

    def run(self, step, num_agent=0, use_act=False):
        """Uses the actor_critic to take action.

        Args:
          step: The int timestep that we are acting.
          num_agent: Agent id that's running. Non-zero when agent has copies.
          is_act: Whether to call act or just run the actor_critic.

        Returns:
          See the actor_critic's act function in model.py.
        """
        observations = Variable(self._rollout.observations[step, num_agent],
                                volatile=True)
        states = Variable(self._rollout.states[step, num_agent], volatile=True)
        masks = Variable(self._rollout.masks[step, num_agent], volatile=True)
        if use_act:
            return self._actor_critic.act(observations, states, masks)
        else:
            return self._actor_critic(observations, states, masks)[0].data

    def get_action_scores(self, observations, states, masks):
        return self._actor_critic.get_action_scores(observations, states, masks)

    def optimize(self, action_classification_loss, max_grad_norm):
        self._optimizer.zero_grad()
        action_classification_loss.backward()
        nn.utils.clip_grad_norm(self._actor_critic.parameters(), max_grad_norm)
        self._optimizer.step()

    def initialize(self, args, obs_shape, action_space,
                   num_training_per_episode, num_episodes, total_steps,
                   num_epoch, optimizer_state_dict):
        params = self._actor_critic.parameters()
        self._optimizer = optim.Adam(params, lr=args.lr, eps=args.eps)
        if optimizer_state_dict:
            self._optimizer.load_state_dict(optimizer_state_dict)
        # TODO: Do we need the rollout storage here?
        self._rollout = RolloutStorage(
            args.num_steps, args.num_processes, obs_shape, action_space,
            self._actor_critic.state_size, num_training_per_episode
        )
        self.num_episodes = num_episodes
        self.total_steps = total_steps
        self.num_epoch = num_epoch

    def update_rollouts(self, obs, timestep):
        self._rollout.observations[timestep, :, :, :, :, :].copy_(obs)

    def insert_rollouts(self, step, current_obs, states, action,
                        action_log_prob, value, reward, mask):
        self._rollout.insert(step, current_obs, states, action,
                             action_log_prob, value, reward, mask)

    def feed_forward_generator(self, advantage, args):
        return self._rollout.feed_forward_generator(advantage, args)

    def copy(self):
        # NOTE: Ugh. This is bad.
        return PPOAgent(None, self._character)

    def after_update(self):
        self._rollout.after_update()
