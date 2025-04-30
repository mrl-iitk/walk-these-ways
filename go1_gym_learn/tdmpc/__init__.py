# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import time
from collections import deque
import hydra

import torch
from ml_logger import logger
from params_proto import PrefixProto
import os
import copy
from termcolor import colored

from .common.world_model import WorldModel
from .common.online_trainer import OnlineTrainer
from .rollout_storage import RolloutStorage
from .common.buffer import Buffer
from .common.parser import parse_cfg
from .common.logger import Logger
from go1_gym import MINI_GYM_ROOT_DIR
# def par(cfg: dict):
    # print(cfg, "p")
    # assert cfg.steps > 0, 'Must train for at least 1 step.'
    # cfgr = parse_cfg(cfg)
    # print(cfgr)
    # return cfgr

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from go1_gym_learn.ppo.metrics_caches import DistCache, SlotCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'TD-MPC2'
    num_steps_per_env = 24  # per iteration [unchanged]
    max_iterations = 1500  # number of policy updates [unchanged]

    # logging
    save_interval = 400  # check for potential saves every this many iterations [unchanged]
    save_video_interval = 100 # [unchanged]
    log_freq = 10 # [unchanged]

    # load and resume [Section unchanged]
    resume = False 
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt


class Runner:
    
    def __init__(self, env, conf, device='cpu'):
        from .tdmpc import TDMPC # [need to change tdmpc.py]
        
        self.device = device
        self.env = env
        cfg = conf
        cfg.obs_shape = {'state': (70,)}
        cfg.num_envs = 4
        # print(type(cfg),
        # 'Insider Runner')
        # print(cfg)
        # cfg = par()
        # print(conf, "-p-")
        # cfg = parse_cfg(conf)
        print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
        self.cfg = cfg
        # assert cfg.steps > 0, 'Must train for at least 1 step.'

        # [need to replace ActorCritic with WorldModel]
        # world_model = WorldModel(cfg).to(self.device) 
        world_model = WorldModel(cfg).to(self.device)
        
        self.alg = TDMPC(world_model, cfg, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        self.env.reset()

    def train(self):
        trainer = OnlineTrainer(
            cfg=self.cfg,
            env=self.env,
            agent=self.alg,
            buffer=Buffer(self.cfg),
            logger=Logger(self.cfg)
        )
        trainer.train()
        print('\nTraining completed successfully')
        
    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, eval_expert=False):
        from ml_logger import logger
        # initialize writer
        assert logger.prefix, "you will overwrite the entire instrument server"

        logger.start('start', 'epoch', 'episode', 'run', 'step')

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.model.train()

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        
        if hasattr(self.env, "curriculum"):
            caches.__init__(curriculum_bins=len(self.env.curriculum))

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(obs[:num_train_envs])
                    # if eval_expert:
                    #     actions_eval = self.alg.world_model.act_teacher(obs[num_train_envs:],
                    #                                                      privileged_obs[num_train_envs:])
                    # else:
                    #     actions_eval = self.alg.world_model.act_student(obs[num_train_envs:],           # [Prio]
                    #                                                      obs_history[num_train_envs:])
                    # ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))      
                    ret = self.env.step(actions_train)         
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    # self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)  # [Prio]

                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])

                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])

                    if 'curriculum' in infos:
                        curr_bins_train = infos['curriculum']['reset_train_env_bins']
                        curr_bins_eval = infos['curriculum']['reset_eval_env_bins']

                        caches.slot_cache.log(curr_bins_train, **{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('slot/train')
                        })
                        caches.slot_cache.log(curr_bins_eval, **{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('slot/eval')
                        })
                        caches.dist_cache.log(**{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('dist/train')
                        })
                        caches.dist_cache.log(**{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('dist/eval')
                        })

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                # Learning step 
                # self.alg.compute_returns(obs[:num_train_envs], privileged_obs[:num_train_envs]) # [Calculates Advantage Function so commented]

                # if it % eval_freq == 0:
                #     self.env.reset_evaluation_envs()

                # if it % eval_freq == 0:
                    # logger.save_pkl({"iteration": it,
                    #                  **caches.slot_cache.get_summary(),
                    #                  **caches.dist_cache.get_summary()},
                    #                 path=f"curriculum/info.pkl", append=True)
            
            # [Might have to take average of loss in the actual update function]
            print("Actions", actions_train.shape)
            print(actions_train)
            print("Observations", obs.shape)
            print(obs)
            print("Rewards", rewards.shape)
            print(rewards)
            print("Dones", dones.shape)
            print(dones)
            
            info = self.alg.update()  # [This now returns a TensorDict]
            
            logger.store_metrics(
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                consistency_loss=info['consistency_loss'],
                reward_loss=info['reward_loss'],
                value_loss=info['value_loss'],
                total_loss=info['total_loss'],
                grad_norm=info['grad_norm']
            )

            # [If pi_info contains additional metrics, you can store those too]
            for key, value in info.items():
                if key not in ['consistency_loss', 'reward_loss', 'value_loss', 'total_loss', 'grad_norm']:
                    logger.store_metrics(**{key: value})
            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                logger.job_running()

            if it % RunnerArgs.save_interval == 0:
                with logger.Sync():
                    logger.torch_save(self.alg.world_model.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = f'{MINI_GYM_ROOT_DIR}/tmp/legged_data'

                    os.makedirs(path, exist_ok=True)

                    adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                    adaptation_module = copy.deepcopy(self.alg.world_model.adaptation_module).to('cpu') # [Prio]
                    traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    traced_script_adaptation_module.save(adaptation_module_path)

                    body_path = f'{path}/body_latest.jit'
                    body_model = copy.deepcopy(self.alg.world_model.actor_body).to('cpu') # [Prio]
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(body_path)

                    logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
                    logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

            self.current_learning_iteration += num_learning_iterations

        with logger.Sync():
            logger.torch_save(self.alg.world_model.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            path = f'{MINI_GYM_ROOT_DIR}/tmp/legged_data'

            os.makedirs(path, exist_ok=True)

            adaptation_module_path = f'{path}/adaptation_module_latest.jit'
            adaptation_module = copy.deepcopy(self.alg.world_model.adaptation_module).to('cpu') # [Prio] [Understand]
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            traced_script_adaptation_module.save(adaptation_module_path)

            body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.world_model.actor_body).to('cpu') # [Prio] [Understand]
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

            logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)

    def get_inference_policy(self, device=None):
        self.alg.world_model.eval() # [Prio]
        if device is not None:
            self.alg.world_model.to(device)
        return self.alg.world_model.act_inference # [Prio]

    def get_expert_policy(self, device=None):
        self.alg.world_model.eval() # [Prio]
        if device is not None:
            self.alg.world_model.to(device)
        return self.alg.world_model.act_expert # [Prio]
