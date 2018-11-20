"""Vanilla Reinforce (Policy Gradient with Advantage)."""
from collections import deque

from pommerman import characters
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from research_agent import ResearchAgent
from storage import RolloutStorage


class ReinforceAgent(ResearchAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""
    def __init__(self, actor_critic, character=characters.Bomber, **kwargs):
        self._actor_critic = actor_critic
        super(ReinforceAgent, self).__init__(character, **kwargs)

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

    def _rollout_data(self, step, num_agent, num_agent_end=None):
        if num_agent_end is not None:
            assert(num_agent_end > num_agent)
            observations = Variable(
                self._rollout.observations[step, num_agent:num_agent_end],
                volatile=True)
            states = Variable(
                self._rollout.states[step, num_agent:num_agent_end],
                volatile=True)
            masks = Variable(
                self._rollout.masks[step, num_agent:num_agent_end],
                volatile=True)
        else:
            observations = Variable(
                self._rollout.observations[step, num_agent], volatile=True)
            states = Variable(self._rollout.states[step, num_agent],
                              volatile=True)
            masks = Variable(self._rollout.masks[step, num_agent],
                             volatile=True)
        return observations, states, masks

    def actor_critic_act(self, step, num_agent=0, deterministic=False):
        """Uses the actor_critic to take action.
        Args:
          step: The int timestep that we are acting.
          num_agent: Agent id that's running. Non-zero when agent has copies.
        Returns:
          See the actor_critic's act function in model.py.
        """
        return self._actor_critic.act(*self.get_rollout_data(step, num_agent),
                                        deterministic=deterministic)

    def get_rollout_data(self, step, num_agent, num_agent_end=None):
        return self._rollout_data(step, num_agent, num_agent_end)

    def actor_critic_call(self, step, num_agent=0):
        observations, states, masks = self._rollout_data(step, num_agent)
        return self._actor_critic(observations, states, masks)[0].data

    def _evaluate_actions(self, observations, states, masks, actions):
        return self._actor_critic.evaluate_actions(observations, states, masks,
                                                   actions)

    def _optimize(self, pg_loss, max_grad_norm, kl_loss=None, kl_factor=0, use_is=False):
        self._optimizer.zero_grad()
        loss = pg_loss
        if kl_factor > 0 and not use_is:
            loss += kl_factor * kl_loss
        loss.backward()
        nn.utils.clip_grad_norm(self._actor_critic.parameters(), max_grad_norm)
        self._optimizer.step()
        if hasattr(self, '_scheduler'):
            self._scheduler.step(loss)

    def halve_lr(self):
        for i, param_group in enumerate(self._optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * 0.5, 1e-7)
            param_group['lr'] = new_lr

    def compute_advantages(self, next_value_agents, use_gae, gamma, tau):
        for num_agent, next_value in enumerate(next_value_agents):
            self._rollout.compute_returns(next_value, use_gae, gamma, tau,
                                          num_agent)
        advantages = self._rollout.compute_advantages(reinforce=True)
        diff = (advantages - advantages.mean())
        advantages = diff / (advantages.std() + 1e-5)
        return advantages

    def initialize(self, args, obs_shape, action_space,
                   num_training_per_episode, num_episodes, total_steps,
                   num_epoch, optimizer_state_dict, num_steps, uniform_v,
                   uniform_v_prior):
        params = self._actor_critic.parameters()
        self._optimizer = optim.Adam(params, lr=args.lr, eps=args.eps)
        if optimizer_state_dict:
            self._optimizer.load_state_dict(optimizer_state_dict)
        if args.use_lr_scheduler:
            self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer, mode='min', verbose=True)

        self._rollout = RolloutStorage(
            num_steps, args.num_processes, obs_shape, action_space,
            self._actor_critic.state_size, num_training_per_episode
        )
        self.num_episodes = num_episodes
        self.total_steps = total_steps
        self.num_epoch = num_epoch
        self.uniform_v = uniform_v
        self.uniform_v_prior = uniform_v_prior

    def update_rollouts(self, obs, timestep):
        self._rollout.observations[timestep, :, :, :, :, :].copy_(obs)

    def insert_rollouts(self, step, current_obs, states, action,
                        action_log_prob, value, reward, mask,
                        action_log_prob_distr=None, dagger_prob_distr=None,
                        expert_action_log_prob=None, training_action_log_prob=None):
        self._rollout.insert(step, current_obs, states, action,
                             action_log_prob, value, reward, mask,
                             action_log_prob_distr, dagger_prob_distr,
                             expert_action_log_prob, training_action_log_prob)


    def reinforce(self, advantages, num_mini_batch, batch_size, num_steps,
                max_grad_norm, action_space, anneal=False, lr=1e-4, eps=1e-5,
                kl_factor=0, use_is=False):
        pg_losses = []
        kl_losses = []
        kl_loss = None
        total_losses = []

        for sample in self._rollout.feed_forward_generator(
                advantages, num_mini_batch, batch_size, num_steps, action_space,
                kl_factor, use_is):
            observations_batch, states_batch, actions_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ, \
                action_log_probs_distr_batch, dagger_probs_distr_batch, \
                expert_action_log_probs_batch, training_action_log_probs_batch = sample

            # Reshape to do in a single forward pass for all steps
            result = self._evaluate_actions(
                Variable(observations_batch),
                Variable(states_batch),
                Variable(masks_batch),
                Variable(actions_batch))
            values, action_log_probs, dist_entropy, states = result

            behavior_action_probs_batch = kl_factor * torch.exp(Variable(expert_action_log_probs_batch)) + \
                (1 - kl_factor) * torch.exp(Variable(training_action_log_probs_batch))

            training_action_probs_batch = torch.exp(Variable(training_action_log_probs_batch))

            training_single_action_probs_batch = training_action_probs_batch.gather(1, Variable(actions_batch))
            behavior_single_action_probs_batch = behavior_action_probs_batch.gather(1, Variable(actions_batch))

            adv_targ = Variable(adv_targ)
            if use_is:
                adv_targ *= training_single_action_probs_batch / behavior_single_action_probs_batch
            ratio = action_log_probs

            pg_loss = - (ratio * adv_targ).mean()
            total_loss = pg_loss

            if kl_factor > 0 and not use_is:
                criterion = nn.KLDivLoss()
                kl_loss = criterion(Variable(action_log_probs_distr_batch),
                                    Variable(dagger_probs_distr_batch))
                total_loss += kl_factor * kl_loss

            self._optimize(total_loss, max_grad_norm, kl_loss, kl_factor, use_is)
            lr = self._optimizer.param_groups[0]['lr']

            pg_losses.append(pg_loss.data[0])
            if kl_factor > 0 and not use_is:
                kl_losses.append(kl_loss.data[0])
            total_losses.append(total_loss.data[0])

        return pg_losses, kl_losses, total_losses, lr

    def copy_ex_model(self):
        """Creates a copy without the model. This is for operating with homogenous training."""
        return ReinforceAgent(None, self._character)

    def after_epoch(self):
        self._rollout.after_epoch()
