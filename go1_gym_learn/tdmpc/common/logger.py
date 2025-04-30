import dataclasses
import os
import datetime
import re

import numpy as np
import pandas as pd
from termcolor import colored

# from common import TASK_SET



MODEL_SIZE = { # parameters (M)
	1:   {'enc_dim': 256,
		  'mlp_dim': 384,
		  'latent_dim': 128,
		  'num_enc_layers': 2,
		  'num_q': 2},
	5:   {'enc_dim': 256,
		  'mlp_dim': 512,
		  'latent_dim': 512,
		  'num_enc_layers': 2},
	19:  {'enc_dim': 1024,
		  'mlp_dim': 1024,
		  'latent_dim': 768,
		  'num_enc_layers': 3},
	48:  {'enc_dim': 1792,
		  'mlp_dim': 1792,
		  'latent_dim': 768,
		  'num_enc_layers': 4},
	317: {'enc_dim': 4096,
		  'mlp_dim': 4096,
		  'latent_dim': 1376,
		  'num_enc_layers': 5,
		  'num_q': 8},
}

TASK_SET = {
	'mt30': [
		# 19 original dmcontrol tasks
		'walker-stand', 'walker-walk', 'walker-run', 'cheetah-run', 'reacher-easy',
	    'reacher-hard', 'acrobot-swingup', 'pendulum-swingup', 'cartpole-balance', 'cartpole-balance-sparse',
		'cartpole-swingup', 'cartpole-swingup-sparse', 'cup-catch', 'finger-spin', 'finger-turn-easy',
		'finger-turn-hard', 'fish-swim', 'hopper-stand', 'hopper-hop',
		# 11 custom dmcontrol tasks
		'walker-walk-backwards', 'walker-run-backwards', 'cheetah-run-backwards', 'cheetah-run-front', 'cheetah-run-back',
		'cheetah-jump', 'hopper-hop-backwards', 'reacher-three-easy', 'reacher-three-hard', 'cup-spin',
		'pendulum-spin',
	],
	'mt80': [
		# 19 original dmcontrol tasks
		'walker-stand', 'walker-walk', 'walker-run', 'cheetah-run', 'reacher-easy',
	    'reacher-hard', 'acrobot-swingup', 'pendulum-swingup', 'cartpole-balance', 'cartpole-balance-sparse',
		'cartpole-swingup', 'cartpole-swingup-sparse', 'cup-catch', 'finger-spin', 'finger-turn-easy',
		'finger-turn-hard', 'fish-swim', 'hopper-stand', 'hopper-hop',
		# 11 custom dmcontrol tasks
		'walker-walk-backwards', 'walker-run-backwards', 'cheetah-run-backwards', 'cheetah-run-front', 'cheetah-run-back',
		'cheetah-jump', 'hopper-hop-backwards', 'reacher-three-easy', 'reacher-three-hard', 'cup-spin',
		'pendulum-spin',
		# meta-world mt50
		'mw-assembly', 'mw-basketball', 'mw-button-press-topdown', 'mw-button-press-topdown-wall', 'mw-button-press',
		'mw-button-press-wall', 'mw-coffee-button', 'mw-coffee-pull', 'mw-coffee-push', 'mw-dial-turn',
		'mw-disassemble', 'mw-door-open', 'mw-door-close', 'mw-drawer-close', 'mw-drawer-open',
		'mw-faucet-open', 'mw-faucet-close', 'mw-hammer', 'mw-handle-press-side', 'mw-handle-press',
		'mw-handle-pull-side', 'mw-handle-pull', 'mw-lever-pull', 'mw-peg-insert-side', 'mw-peg-unplug-side',
		'mw-pick-out-of-hole', 'mw-pick-place', 'mw-pick-place-wall', 'mw-plate-slide', 'mw-plate-slide-side',
		'mw-plate-slide-back', 'mw-plate-slide-back-side', 'mw-push-back', 'mw-push', 'mw-push-wall',
		'mw-reach', 'mw-reach-wall', 'mw-shelf-place', 'mw-soccer', 'mw-stick-push',
		'mw-stick-pull', 'mw-sweep-into', 'mw-sweep', 'mw-window-open', 'mw-window-close',
		'mw-bin-picking', 'mw-box-close', 'mw-door-lock', 'mw-door-unlock', 'mw-hand-insert',
	],
}


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("total_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self._save_dir = make_dir(cfg.work_dir / 'eval_video')
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 0:
			frames = np.stack(self.frames)
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = False
			self._wandb = None
			self._video = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb

		wandb.init(
			project=self.project,
			entity=self.entity,
			name=str(cfg.seed),
			group=self._group,
			tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
			dir=self._log_dir,
			config=dataclasses.asdict(cfg),
		)
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		self._video = (
			VideoRecorder(cfg, self._wandb)
			if self._wandb and cfg.save_video
			else None
		)

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self._group + '-' + str(self._seed) + '-' + str(identifier),
					type='model',
				)
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)

	def finish(self, agent=None):
		try:
			self.save_agent(agent)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def pprint_multitask(self, d, cfg):
		"""Pretty-print evaluation metrics for multi-task training."""
		print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		dmcontrol_reward = []
		metaworld_reward = []
		metaworld_success = []
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if task in TASK_SET['mt30'] and k.startswith('episode_reward'): # DMControl
				dmcontrol_reward.append(v)
				print(colored(f'  {task:<22}\tR: {v:.01f}', 'yellow'))
			elif task in TASK_SET['mt80'] and task not in TASK_SET['mt30']: # Meta-World
				if k.startswith('episode_reward'):
					metaworld_reward.append(v)
				elif k.startswith('episode_success'):
					metaworld_success.append(v)
					print(colored(f'  {task:<22}\tS: {v:.02f}', 'yellow'))
		dmcontrol_reward = np.nanmean(dmcontrol_reward)
		d['episode_reward+avg_dmcontrol'] = dmcontrol_reward
		print(colored(f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}', 'yellow', attrs=['bold']))
		if cfg.task == 'mt80':
			metaworld_reward = np.nanmean(metaworld_reward)
			metaworld_success = np.nanmean(metaworld_success)
			d['episode_reward+avg_metaworld'] = metaworld_reward
			d['episode_success+avg_metaworld'] = metaworld_success
			print(colored(f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}', 'yellow', attrs=['bold']))
			print(colored(f'  {"metaworld":<22}\tS: {metaworld_success:.02f}', 'yellow', attrs=['bold']))

	def log(self, d, category="train"):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			if category in {"train", "eval"}:
				xkey = "step"
			elif category == "pretrain":
				xkey = "iteration"
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			self._wandb.log(_d, step=d[xkey])
		if category == "eval" and self._save_csv:
			keys = ["step", "episode_reward"]
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				self._log_dir / "eval.csv", header=keys, index=None
			)
		self._print(d, category)
