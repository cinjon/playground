"""
PPO Agent using IKostrikov's approach for the ppo algorithm.
"""
from pommerman.agents import BaseAgent
from pommerman import characters
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from storage import RolloutStorage


class PPOAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""
    def __init__(self, actor_critic, character=characters.Bomber):
        self._actor_critic = actor_critic
        super(PPOAgent, self).__init__(character)

    def cuda(self):
        self._actor_critic.cuda()
        self._rollout.cuda()

    def get_model(self):
        return self._actor_critic

    def get_optimizer(self):
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

    def evaluate_actions(self, observations, states, masks, actions):
        return self._actor_critic.evaluate_actions(observations, states, masks,
                                                   actions)

    def optimize(self, value_loss, action_loss, dist_entropy, entropy_coef,
                 max_grad_norm):
        self._optimizer.zero_grad()
        (value_loss + action_loss - dist_entropy * entropy_coef).backward()
        nn.utils.clip_grad_norm(self._actor_critic.parameters(), max_grad_norm)
        self._optimizer.step()

    def compute_advantages(self, next_value_agents, use_gae, gamma, tau):
        for num_agent, next_value in enumerate(next_value_agents):
            self._rollout.compute_returns(next_value, use_gae, gamma, tau,
                                          num_agent)
        advantages = self._rollout.compute_advantages()
        diff = (advantages - advantages.mean()) 
        advantages = diff / (advantages.std() + 1e-5)
        return advantages

    def initialize(self, args, obs_shape, action_space,
                   num_training_per_episode):
        self._optimizer = optim.Adam(self._actor_critic.parameters(), args.lr,
                                     eps=args.eps)
        self._rollout = RolloutStorage(
            args.num_steps, args.num_processes, obs_shape, action_space,
            self._actor_critic.state_size, num_training_per_episode
        )

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
