"""PPO Agent using IKostrikov's approach for the ppo algorithm."""
from pommerman import characters
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from research_agent import ResearchAgent
from storage import RolloutStorage


class PPOAgent(ResearchAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""
    def __init__(self, actor_critic, character=characters.Bomber, **kwargs):
        self._actor_critic = actor_critic
        super(PPOAgent, self).__init__(character, **kwargs)
        if self._actor_critic is not None:
            # This caveat is so that it works with both train and eval.
            self._states = torch.zeros(1, self._actor_critic.state_size)

    def cuda(self):
        self._actor_critic.cuda()
        if hasattr(self, "_rollout"):
            self._rollout.cuda()

    @property
    def model(self):
        return self._actor_critic

    @property
    def optimizer(self):
        return self._optimizer

    def set_eval(self):
        self._actor_critic.eval()

    def set_train(self):
        self._actor_critic.train()

    def _rollout_data(self, step, num_agent):
        observations = Variable(self._rollout.observations[step, num_agent],
                                volatile=True)
        states = Variable(self._rollout.states[step, num_agent], volatile=True)
        masks = Variable(self._rollout.masks[step, num_agent], volatile=True)
        return observations, states, masks

    def actor_critic_act(self, step, num_agent=0, use_act=False):
        """Uses the actor_critic to take action.

        Args:
          step: The int timestep that we are acting.
          num_agent: Agent id that's running. Non-zero when agent has copies.
          use_act: Whether to call act or just run the actor_critic.

        Returns:
          See the actor_critic's act function in model.py.
        """
        observations, states, masks = self._rollout_data(step, num_agent)
        return self._actor_critic.act(observations, states, masks)

    def actor_critic_call(self, step, num_agent=0):
        observations, states, masks = self._rollout_data(step, num_agent)
        return self._actor_critic(observations, states, masks)[0].data

    def _evaluate_actions(self, observations, states, masks, actions):
        return self._actor_critic.evaluate_actions(observations, states, masks,
                                                   actions)

    def _optimize(self, value_loss, action_loss, dist_entropy, entropy_coef,
                  max_grad_norm):
        self._optimizer.zero_grad()
        (value_loss + action_loss - dist_entropy * entropy_coef).backward()
        nn.utils.clip_grad_norm(self._actor_critic.parameters(), max_grad_norm)
        self._optimizer.step()

    def optimize_anneal(self, value_loss, action_loss, dist_entropy, entropy_coef,
                  max_grad_norm, lr_anneal, eps):
        params = self._actor_critic.parameters()
        self._optimizer = optim.Adam(params, lr=lr_anneal, eps=eps)
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
                   num_training_per_episode, num_episodes, total_steps,
                   num_epoch, optimizer_state_dict):
        params = self._actor_critic.parameters()
        self._optimizer = optim.Adam(params, lr=args.lr, eps=args.eps)
        if optimizer_state_dict:
            self._optimizer.load_state_dict(optimizer_state_dict)
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

    def ppo(self, advantages, num_mini_batch, num_steps, clip_param,
            entropy_coef, max_grad_norm, anneal=False, lr=1e-4, eps=1e-5):
        action_losses = []
        value_losses = []
        dist_entropies = []

        for sample in self._rollout.feed_forward_generator(
                advantages, num_mini_batch, num_steps):
            observations_batch, states_batch, actions_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = sample

            # Reshape to do in a single forward pass for all steps
            result = self._evaluate_actions(
                Variable(observations_batch),
                Variable(states_batch),
                Variable(masks_batch),
                Variable(actions_batch))
            values, action_log_probs, dist_entropy, states = result

            adv_targ = Variable(adv_targ)
            ratio = action_log_probs
            ratio -= Variable(old_action_log_probs_batch)
            ratio = torch.exp(ratio)

            surr1 = ratio * adv_targ
            surr2 = torch.clamp(
                ratio, 1.0 - clip_param, 1.0 + clip_param)
            surr2 *= adv_targ
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = (Variable(return_batch) - values) \
                         .pow(2).mean()

            if anneal:
                self.optimize_anneal(value_loss, action_loss, dist_entropy,
                               entropy_coef, max_grad_norm, lr, eps)
            else:
                self._optimize(value_loss, action_loss, dist_entropy,
                               entropy_coef, max_grad_norm)

            action_losses.append(action_loss.data[0])
            value_losses.append(value_loss.data[0])
            dist_entropies.append(dist_entropy.data[0])


        return action_losses, value_losses, dist_entropies

    def copy_ex_model(self):
        """Creates a copy that without the model.

        This is for operating with homogenous training.
        """
        return PPOAgent(None, self._character)

    def after_epoch(self):
        self._rollout.after_epoch()
