"""
PPO Agent using IKostrikov's approach for the ppo algorithm.
"""
from pommerman.agents import BaseAgent
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from storage import RolloutStorage


class PPOAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""
    def __init__(self, agent, actor_critic):
        self._actor_critic = actor_critic
        self._agent = agent

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

    def act_pytorch(self, step, num_agent=0):
        """Uses the actor_critic to act.

        Args:
          step: The int timestep that we are acting.
          num_agent: The agent id that we are acting. This is non zero when this agent has copies.

        Returns:
          See the actor_critic's act function in model.py.
        """
        return self._actor_critic.act(
            Variable(self._rollout.observations[step, num_agent], volatile=True),
            Variable(self._rollout.states[step, num_agent], volatile=True),
            Variable(self._rollout.masks[step, num_agent], volatile=True)
        )

    def run_actor_critic(self, step, num_agent=0):
        return self._actor_critic(
            Variable(self._rollout.observations[step][num_agent], volatile=True),
            Variable(self._rollout.states[step][num_agent], volatile=True),
            Variable(self._rollout.masks[step][num_agent], volatile=True))[0].data

    def evaluate_actions(self, observations, states, masks, actions):
        return self._actor_critic.evaluate_actions(observations, states, masks, actions)

    def optimize(self, value_loss, action_loss, dist_entropy, entropy_coef, max_grad_norm):
        self._optimizer.zero_grad()
        (value_loss + action_loss - dist_entropy * entropy_coef).backward()
        nn.utils.clip_grad_norm(self._actor_critic.parameters(), max_grad_norm)
        self._optimizer.step()

    def compute_advantages(self, next_value_agents, use_gae, gamma, tau):
        for num_agent, next_value in enumerate(next_value_agents):
            self._rollout.compute_returns(next_value, use_gae, gamma, tau, num_agent)
        advantages = self._rollout.compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return advantages

    def initialize(self, args, obs_shape, action_space, num_training_per_episode):
        self._optimizer = optim.Adam(self._actor_critic.parameters(), args.lr, eps=args.eps)
        self._rollout = RolloutStorage(
            args.num_steps, args.num_processes, obs_shape, action_space,
            self._actor_critic.state_size, num_training_per_episode
        )

    def update_rollouts(self, obs, timestep):
        self._rollout.observations[timestep, :, :, :, :, :].copy_(obs)

    def insert_rollouts(self, step, current_obs, states, action, action_log_prob,
                        value, reward, mask):
        self._rollout.insert(step, current_obs, states, action, action_log_prob, value, reward, mask)

    def feed_forward_generator(self, advantage, args):
        return self._rollout.feed_forward_generator(advantage, args)

    def copy(self, agent):
        # NOTE: Ugh. This is bad.
        return PPOAgent(agent, None)

    def after_update(self):
        self._rollout.after_update()
