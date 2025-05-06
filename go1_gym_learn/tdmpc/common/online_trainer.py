from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from .base import Trainer
from time import time


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards = []
		for i in range(self.cfg.eval_episodes // self.cfg.num_envs):
			obs_dict, done, ep_reward, t = self.env.reset(), torch.tensor(False), 0, 0
			obs = obs_dict["obs"]
			# if self.cfg.save_video:
				# self.logger.video.init(self.env, enabled=(i==0))
			while not done.any():
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs_dict, reward, done, info = self.env.step(action)
				# print("Reward:", reward)
				# print(done)
				# print(self._step)
				obs = obs_dict["obs"]
				ep_reward += reward
				t += 1
				# if self.cfg.save_video:
					# self.logger.video.record(self.env)
			assert done.all(), 'Vectorized environments must reset all environments at once.'
			ep_rewards.append(ep_reward)
			# if self.cfg.save_video:
				# self.logger.video.save(self._step)
		return dict(
			episode_reward=torch.cat(ep_rewards).mean(),
			# episode_success=inFalsefo['success'].mean(),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan')).repeat(self.cfg.num_envs)
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1, self.cfg.num_envs,))
		return td

	def train(self):

		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, torch.tensor(True), False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % 128 ==0:
				print("Step:", self._step)
			if self._step % self.cfg.eval_freq == 0:
				eval_next = False

			# print(self._step, ":", done)
			# print("Train Metrics:", train_metrics)
			# Reset environment
			# print("Done:", done)
			if done.any():
				
				assert done.all(), 'Vectorized environments must reset all environments at once.'
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					print("Eval Metrics:", eval_metrics)
					# self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					device = torch.device("cuda:0")
					tds = torch.cat([td.to(device) for td in self._tds])
					# 
					train_metrics.update(
						episode_reward=tds['reward'].nansum(0).mean(),
						# episode_success=info['success'].nanmean(),
					)
					train_metrics.update(self.common_metrics())
					print("Train Metrics:", train_metrics)
					self._ep_idx = self.buffer.add(tds)

				obs_dict = self.env.reset()
				obs = obs_dict["obs"]
				self._tds = []

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==0)
			else:
				action = self.env.rand_act()
			#	action = self.agent.act(obs, t0=len(self._tds)==0)			
			obs_dict, reward, done, info = self.env.step(action)
			obs = obs_dict["obs"]
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					# num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
					num_updates = int(self.cfg.seed_steps / 4)
					num_updates = 1	
					print('Pretraining agent on seed data...')
				else:
					# num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))
					num_updates = max(1,int(self.cfg.num_envs / 4))	
				# print("Step:", self._step, num_updates)
				for i in range(num_updates):
					# print("Update step:", i)
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)
				# print("Step: ",self._step, train_metrics)


			self._step += self.cfg.num_envs
	
		# self.logger.finish(self.agent)

