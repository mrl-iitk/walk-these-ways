import torch
import numpy as np
from m2_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi
from isaacgym.torch_utils import quat_to_euler_xyz

class MIMOCRewards:
    # initialize the class with environment, ref_traj as input
    def __init__(self, env, ref_traj=None):
        self.env = env
        self.ref_traj = ref_traj
        
        # Widths (αβ) - will be computed based on reference trajectory range
        self.widths = {}
        
        # Initialize widths if reference trajectory is provided
        if ref_traj is not None:
            self._compute_widths()
    
    def load_env(self, env):
        """If a new environment needs to be set."""
        self.env = env
    
    def set_ref_traj(self, ref_traj):
        """Set the reference trajectory and compute widths"""
        self.ref_traj = ref_traj
        self._compute_widths()
    
    def _compute_widths(self):
        """Compute widths αβ based on equation (4) of paper: αβ = 2 / (max(βref) - min(βref))²"""
        ref_traj = self.ref_traj
        
        # Compute widths for each state variable as key-value in dict
        for key in ref_traj.keys():
            if key in ['joint_position', 'joint_velocity', 'joint_torque', 'base_position', 
                      'base_body_frame_lin_vel', 'base_rpy', 'angular_velocity', 'foot_position', 'foot_velocity']:
                data = ref_traj[key]
                data_range = torch.max(data) - torch.min(data)
                self.widths[key] = (2.0 / data_range) ** 2
                
    
    def get_current_reference(self, timesteps):
        """Get reference values for specified timesteps
        Input - timesteps: tensor of shape (num_envs,)"""
        ref_values = {}
        for key, traj in self.ref_traj.items():
            ref_values[key] = traj[timesteps] 
        return ref_values
    
    def _generic_tracking_reward(self, current_state, reference_state, width):
        """Generic reward function: wβ * exp(-αβ * ||βref - β||²)"""
        if reference_state is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        
        # Ensure tensors are on the same device
        if not isinstance(current_state, torch.Tensor):
            current_state = torch.tensor(current_state, device=self.env.device)
        if not isinstance(reference_state, torch.Tensor):
            reference_state = torch.tensor(reference_state, device=self.env.device)
        
        # Compute squared error
        squared_error = (reference_state - current_state)**2  # (num_envs, dim)
        weighted_error = torch.sum(width * squared_error, dim=-1)
 
        # exponential reward as used in paper
        reward = torch.exp(-weighted_error)
        
        return reward
    
    # ------------ Trajectory tracking reward functions ----------------
    
    def _reward_track_joint_pos(self, timesteps=None):
        """Track reference joint positions (rq)"""
        ref_values = self.get_current_reference(timesteps)
        return self._generic_tracking_reward(
            self.env.dof_pos,
            ref_values['joint_position'],
            self.widths.get('joint_pos', 1.0) # gives 1 by default
        )
    
    def _reward_track_joint_vel(self, timesteps=None):
        """Track reference joint velocities (rq̇)"""
        ref_values = self.get_current_reference(timesteps)
        return self._generic_tracking_reward(
            self.env.dof_vel,
            ref_values['joint_velocity'],
            self.widths.get('joint_vel', 1.0)
        )
    
    def _reward_track_joint_torque(self, timesteps=None):
        """Track reference joint torques (rτ)"""
        ref_values = self.get_current_reference(timesteps)
        return self._generic_tracking_reward(
            self.env.torques,
            ref_values['joint_torque'],
            self.widths.get('joint_torque', 1.0)
        )
    
    def _reward_track_body_pos(self, timesteps=None):
        """Track reference body position (rx)"""
        ref_values = self.get_current_reference(timesteps)
        return self._generic_tracking_reward(
            self.env.base_pos,
            ref_values['base_position'],
            self.widths.get('body_pos', 1.0)
        )
    
    def _reward_track_body_lin_vel(self, timesteps=None):
        """Track reference body linear velocity (rẋ)"""
        ref_values = self.get_current_reference(timesteps)
        return self._generic_tracking_reward(
            self.env.base_lin_vel,
            ref_values['base_body_frame_lin_vel'],
            self.widths.get('body_lin_vel', 1.0)
        )
    
    def _reward_track_body_orient(self, timesteps=None):
        """Track reference body orientation (θ) - Roll, Pitch, Yaw"""
        ref_values = self.get_current_reference(timesteps) 
        base_quat = self.env.root_states[3:7]
        base_RPY = quat_to_euler_xyz(base_quat)
        return self._generic_tracking_reward(
            base_RPY,
            ref_values['base_rpy'],
            self.widths.get('body_orient', 1.0)
        )
    
    def _reward_track_body_ang_vel(self, timesteps=None):
        """Track reference body angular velocity (rω)"""
        ref_values = self.get_current_reference(timesteps)
        return self._generic_tracking_reward(
            self.env.base_ang_vel,
            ref_values['angular_velocity'],
            self.widths.get('body_ang_vel', 1.0)
        )
    
    def _reward_track_ee_pos(self, timesteps=None):
        """Track reference end-effector positions (re)"""
        ref_values = self.get_current_reference(timesteps)   
        return self._generic_tracking_reward(
            self.env.foot_positions,
            ref_values['foot_position'],
            self.widths.get('ee_pos', 1.0)
        )
    
    def _reward_track_ee_vel(self, timesteps=None):
        """Track reference end-effector velocities (rė)"""
        ref_values = self.get_current_reference(timesteps)       
        return self._generic_tracking_reward(
            self.env.foot_velocities,
            ref_values['foot_velocity'],
            self.widths.get('ee_vel', 1.0)
        )
    


