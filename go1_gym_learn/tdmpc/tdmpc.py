# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from common import math
from common.scale import RunningScale
from tdmpc.common.world_model import WorldModel
from tensordict import TensorDict

from go1_gym_learn.tdmpc import WorldModel
from go1_gym_learn.tdmpc import RolloutStorage
from go1_gym_learn.tdmpc import caches


class TDMPC_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 3.e-4*0.3 # 5.e-4 [Param]
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.


class TDMPC(torch.nn.Module):
    world_model: WorldModel      
    # [Changed Function]
    def __init__(self, world_model, cfg, device='cpu'):
        super().__init__()

        self.device = device
        self.model = world_model
        self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
        self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device))
        if cfg.compile:
            print('Compiling update function with torch.compile...')        
        self._update = torch.compile(self._update, mode="reduce-overhead")
        self.transition = RolloutStorage.Transition()


    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.world_model.test()

    def train_mode(self):
        self.world_model.train()

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)


    # [Changed Function]
    @torch.no_grad()
    def act(self, obs, privileged_obs, obs_history, t0=False, eval_mode=False, task=None):
        # Compute the actions and values

            obs = obs.to(self.device, non_blocking=True)

            if self.cfg.mpc:
                action = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
            else:
                # Encode observation into latent space
                z = self.model.encode(obs, task)
                action, info = self.model.pi(z, task)  # Get action from policy
                if eval_mode:
                    action = info["mean"]

            # [Store transition information like PPO]
            self.transition.actions = action[0].cpu().detach()  # Store action
            self.transition.values = self.model.evaluate(obs, task).cpu().detach()  # Store value estimate
            self.transition.actions_log_prob = self.model.get_actions_log_prob(action).cpu().detach()
            self.transition.action_mean = info["mean"].cpu().detach()
            self.transition.action_sigma = info["log_std"].exp().cpu().detach() if "log_std" in info else None

            # Store observations
            self.transition.observations = obs.cpu()
            self.transition.critic_observations = obs.cpu()
            self.transition.privileged_observations = None  # TDMPC may not use privileged obs
            self.transition.observation_histories = None  # TDMPC may not track history like PPO

            return action[0].cpu()

    
    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G = G + discount * reward
            discount_update = self.discount
            discount = discount * discount_update
        action, _ = self.model.pi(z, task)
        return G + discount * self.model.Q(z, action, task, return_type='avg')


    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.

        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """
        # Sample policy trajectories
        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[:,t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[:,t], task)
            pi_actions[:,-1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(1).values
            score = torch.exp(self.cfg.temperature*(elite_value - max_value.unsqueeze(1)))
            score = (score / score.sum(1, keepdim=True))
            mean = (score.unsqueeze(1) * elite_actions).sum(2) / (score.sum(1, keepdim=True) + 1e-9)
            std = ((score.unsqueeze(1) * (elite_actions - mean.unsqueeze(2)) ** 2).sum(2) / (score.sum(1, keepdim=True) + 1e-9)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        rand_idx = math.gumbel_softmax_sample(score.squeeze(2), dim=1)  # gumbel_softmax_sample is compatible with cuda graphs
        actions = elite_actions[torch.arange(self.cfg.num_envs), :, rand_idx]
        action, std = actions[:, 0], std[:, 0]
        if not eval_mode:
            action = action + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return action.clamp(-1, 1)

    def process_env_step(self, rewards, dones, infos, task='dog-run'):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.discount * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.world_model.reset(dones)

    # def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
    #     last_values = self.world_model.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
    #     self.storage.compute_returns(last_values, TDMPC_Args.gamma, TDMPC_Args.lam)

    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.

        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        action, info = self.model.pi(zs, task)
        qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        info = TensorDict({
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale": self.scale.value,
        })
        return info


    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: TD-target.
        """
        action, _ = self.model.pi(next_z, task)
        discount = self.discount
        return reward + discount * self.model.Q(next_z, action, task, return_type='min', target=True)

    def _update(self, obs, action, reward, task=None):
        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        # Prepare for update
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action, task)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
            reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update policy
        pi_info = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        info = TensorDict({
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
        })
        info.update(pi_info)
        return info.detach().mean()


    # [Needs changes to Replay Buffer]
    def update(self):
        """
        Main update function. Corresponds to one iteration of model learning.

        Returns:		
            dict: Dictionary of training statistics.
        """

        # Prepare batch for training
        obs, action, reward, task = self.storage.sample()

        kwargs = {}
        if task is not None:
            kwargs["task"] = task

        # Mark the step for CUDA graph optimization
        torch.compiler.cudagraph_mark_step_begin()

        # Call the internal update function
        return self._update(obs, action, reward, **kwargs)
