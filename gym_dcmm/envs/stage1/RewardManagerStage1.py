"""
Reward computation for DcmmVecEnv.
Handles all reward calculation logic.
"""

import numpy as np
import torch
import os
import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.utils.quat_utils import quat_rotate_vector

# AVP Imports
from gym_dcmm.algs.ppo_dcmm.stage2.ModelsStage2 import ActorCritic as CriticStage2
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd


class RewardManagerStage1:
    """Manages reward computation for the environment."""

    def __init__(self, env):
        """
        Initialize reward manager.

        Args:
            env: Reference to the parent DcmmVecEnv instance
        """
        self.env = env
        self.prev_action_reward = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # --- AVP Configuration from DcmmCfg ---
        self.use_avp = getattr(DcmmCfg, 'avp', None) is not None and DcmmCfg.avp.enabled
        if self.use_avp:
            # Curriculum decay: lambda decays from start to end over training
            self.avp_lambda_start = DcmmCfg.avp.lambda_weight_start
            self.avp_lambda_end = DcmmCfg.avp.lambda_weight_end
            self.avp_lambda = self.avp_lambda_start  # Will be updated by curriculum
            
            self.avp_gate_distance = DcmmCfg.avp.gate_distance
            self.avp_checkpoint_path = DcmmCfg.avp.checkpoint_path
            self.avp_ready_pose = DcmmCfg.avp.ready_pose
            self.avp_state_dim = DcmmCfg.avp.state_dim
            self.avp_img_size = DcmmCfg.avp.img_size
            
            # AVP logging stats (for WandB)
            self.avp_stats = {
                'reward_sum': 0.0,
                'critic_value_sum': 0.0,
                'count': 0,
                'gated_count': 0,  # Times AVP was skipped due to distance
            }
            
            # Load Stage 2 Critic
            self._load_stage2_critic()
        else:
            print(">>> AVP: Disabled by config (DcmmCfg.avp.enabled = False)")
            self.grasp_critic = None
            self.running_mean_std = None
            self.avp_stats = None
    
    def _load_stage2_critic(self):
        """Load Stage 2 Critic model for AVP reward computation."""
        # Find checkpoint path (relative to project root)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        ckpt_path = os.path.join(project_root, self.avp_checkpoint_path)
        
        if not os.path.exists(ckpt_path):
            print(f">>> AVP Warning: Stage 2 Checkpoint not found at {ckpt_path}")
            print(">>> AVP will be disabled.")
            self.grasp_critic = None
            self.running_mean_std = None
            return
        
        try:
            # Stage 2 network configuration (must match training config)
            net_config = {
                'actor_units': [256, 128],
                'actions_num': 20,  # 2 base + 6 arm + 12 hand (full action space)
                'input_shape': (self.avp_state_dim + self.avp_img_size * self.avp_img_size,),
                'state_dim': self.avp_state_dim,
                'depth_pixels': self.avp_img_size * self.avp_img_size,
                'img_size': self.avp_img_size,
                'separate_value_mlp': True,
            }
            
            # Instantiate model
            self.grasp_critic = CriticStage2(net_config).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
            # Load model weights
            self.grasp_critic.load_state_dict(checkpoint['model'])
            
            # Load RunningMeanStd for normalization
            self.running_mean_std = RunningMeanStd((self.avp_state_dim,)).to(self.device)
            if 'running_mean_std' in checkpoint:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            else:
                print(">>> AVP Warning: No running_mean_std in checkpoint, using default normalization")
            
            # Freeze model and set to eval mode
            self.grasp_critic.eval()
            for param in self.grasp_critic.parameters():
                param.requires_grad = False
            self.running_mean_std.eval()
            
            print(f">>> AVP: Stage 2 Critic Loaded Successfully from {ckpt_path}")
            print(f">>> AVP Config: lambda={self.avp_lambda}, gate_dist={self.avp_gate_distance}m")
            
        except Exception as e:
            print(f">>> AVP Error loading Stage 2 Critic: {e}")
            import traceback
            traceback.print_exc()
            self.grasp_critic = None
            self.running_mean_std = None
    
    def construct_virtual_obs(self, obs):
        """
        Construct virtual observation for Stage 2 Critic.
        
        The virtual observation simulates what the state would look like
        if the arm was already in a ready-to-grasp pose, while using
        the actual current object position and depth image.
        
        Stage 2 State Vector (35 dim):
        [ee_pos(3), ee_quat(4), ee_vel(3), arm_joints(6), obj_pos(3), hand_joints(12), touch(4)]
        
        Args:
            obs: Current observation dict from Stage 1
            
        Returns:
            dict: Input dict for Stage 2 Critic
        """
        # 1. Virtual EE position (relative to arm base, in base frame)
        # Use a typical ready-to-grasp position
        virtual_ee_pos = np.array([0.3, 0.0, 0.2], dtype=np.float32)
        
        # 2. Virtual EE quaternion (palm facing forward)
        # Identity quaternion with slight forward tilt
        virtual_ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # 3. Virtual EE velocity (static, zero)
        virtual_ee_vel = np.zeros(3, dtype=np.float32)
        
        # 4. Virtual arm joints (Ready Pose)
        virtual_arm_joints = self.avp_ready_pose.astype(np.float32)
        
        # 5. REAL object position (relative to arm base, in base frame)
        # This is the key input - the actual object location
        real_obj_pos = self.env.obs_manager.get_relative_object_pos3d().astype(np.float32)
        
        # 6. Virtual hand joints (open posture)
        virtual_hand_joints = self.env.hand_open_angles.astype(np.float32)
        
        # 7. Virtual touch (no contact yet)
        virtual_touch = np.zeros(4, dtype=np.float32)
        
        # Concatenate state vector (35 dim)
        state_vec = np.concatenate([
            virtual_ee_pos,       # 3
            virtual_ee_quat,      # 4
            virtual_ee_vel,       # 3
            virtual_arm_joints,   # 6
            real_obj_pos,         # 3 - REAL!
            virtual_hand_joints,  # 12
            virtual_touch         # 4
        ])
        
        # Convert to tensor
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Apply normalization
        state_normalized = self.running_mean_std(state_tensor)
        
        # 8. Get REAL depth image
        # Use render_manager to get properly formatted depth
        depth_obs = self.env.render_manager.get_depth_obs(
            width=self.avp_img_size, 
            height=self.avp_img_size,
            add_noise=False,  # Don't add noise for Critic evaluation
            add_holes=False
        )
        
        # Flatten depth: (1, 84, 84) -> (1, 7056)
        depth_flat = depth_obs.flatten()
        depth_tensor = torch.tensor(depth_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Concatenate state and depth
        obs_combined = torch.cat([state_normalized, depth_tensor], dim=1)
        
        return {'obs': obs_combined}
    
    def compute_avp_reward(self, obs, info):
        """
        Compute AVP (Asymmetric Value Propagation) reward using Stage 2 Critic.
        
        Features:
        - Uses ee_distance for gating (more accurate than base_distance)
        - Curriculum decay: lambda decreases as training progresses
        - Collects stats for WandB logging
        
        Args:
            obs: Current observation dict
            info: Current info dict
            
        Returns:
            float: AVP reward value
        """
        if not self.use_avp or self.grasp_critic is None:
            return 0.0
        
        # Update AVP lambda based on curriculum (uses same difficulty as environment)
        if hasattr(self.env, 'curriculum_difficulty'):
            difficulty = self.env.curriculum_difficulty  # 0.0 → 1.0
            self.avp_lambda = self.avp_lambda_start + (self.avp_lambda_end - self.avp_lambda_start) * difficulty
        
        # Distance gating: Use ee_distance (more accurate than base_distance)
        if info["ee_distance"] > self.avp_gate_distance:
            if self.avp_stats is not None:
                self.avp_stats['gated_count'] += 1
            return 0.0
        
        try:
            # Construct virtual observation
            input_dict = self.construct_virtual_obs(obs)
            
            # Critic inference
            with torch.no_grad():
                res = self.grasp_critic.act(input_dict)
                value_est = res['values'].item()
                if self.env.print_reward:
                    print(f">>> AVP Raw Value: {value_est:.4f}")
            
            # Scale the value to reasonable reward range
            avp_reward = self.avp_lambda * value_est
            
            # Clip to prevent extreme values
            avp_reward = np.clip(avp_reward, -5.0, 5.0)
            
            # Collect stats for logging
            if self.avp_stats is not None:
                self.avp_stats['reward_sum'] += avp_reward
                self.avp_stats['critic_value_sum'] += value_est
                self.avp_stats['count'] += 1
            
            return avp_reward
            
        except Exception as e:
            if self.env.print_reward:
                print(f">>> AVP Error during inference: {e}")
            return 0.0
    
    def get_avp_stats_and_reset(self):
        """
        Get AVP statistics for WandB logging and reset counters.
        
        Returns:
            dict: AVP statistics (mean reward, mean value, lambda, gate ratio)
        """
        if self.avp_stats is None or self.avp_stats['count'] == 0:
            return None
        
        total = self.avp_stats['count'] + self.avp_stats['gated_count']
        stats = {
            'avp/reward_mean': self.avp_stats['reward_sum'] / self.avp_stats['count'],
            'avp/critic_value_mean': self.avp_stats['critic_value_sum'] / self.avp_stats['count'],
            'avp/lambda': self.avp_lambda,
            'avp/gate_ratio': self.avp_stats['gated_count'] / total if total > 0 else 0,
            'avp/count': self.avp_stats['count'],
        }
        
        # Reset counters
        self.avp_stats = {
            'reward_sum': 0.0,
            'critic_value_sum': 0.0,
            'count': 0,
            'gated_count': 0,
        }
        
        return stats

    def norm_ctrl(self, ctrl, components):
        """
        Convert the ctrl (dict type) to the numpy array and return its norm value.

        Args:
            ctrl: dict, control actions
            components: list of component names to include

        Returns:
            float: norm value
        """
        ctrl_array = np.concatenate([ctrl[component]*DcmmCfg.reward_weights['r_ctrl'][component]
                                    for component in components])
        return np.linalg.norm(ctrl_array)

    def compute_reward(self, obs, info, ctrl):
        """
        Compute total reward based on observations, info, and control.

        Optimized Rewards:
        - Reaching Reward (EE approaching target)
        - Base Approach Reward (Vehicle moving closer)
        - Orientation Reward (Hand facing target)
        - Touch Reward (Contact with target)
        - Regularization Penalty
        - Collision Penalties

        Args:
            obs: Current observation dict
            info: Current info dict
            ctrl: Control action dict

        Returns:
            float: Total reward
        """
        # 1. EE Reaching Reward: Normalized tanh (0.0 to 1.0)
        # When d=0, reward=1.0; when d is large, reward -> 0.0
        reward_reaching = 1.0 * (1.0 - np.tanh(2.0 * info["ee_distance"]))

        # 2. Base Approach Reward: Sweet Spot at 0.8m
        optimal_dist = 0.8
        dist_error = abs(info["base_distance"] - optimal_dist)
        reward_base_approach = np.exp(-5.0 * dist_error**2)

        # 3. Orientation Reward: Palm should face target (Stricter 4th power)
        reward_orientation = 0.0
        if info["ee_distance"] < 2.0:
            # Get positions
            ee_pos = self.env.Dcmm.data.body("link6").xpos
            obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos

            # Direction from hand to target
            ee_to_obj = obj_pos - ee_pos
            ee_to_obj_norm = ee_to_obj / (np.linalg.norm(ee_to_obj) + 1e-6)

            # Get palm forward direction from quaternion
            ee_quat = self.env.Dcmm.data.body("link6").xquat
            # Palm forward = negative Z-axis of EE frame
            palm_forward = quat_rotate_vector(ee_quat, np.array([0, 0, -1]))

            # Alignment: 1.0 = perfect, 0.0 = perpendicular, -1.0 = backwards
            alignment = np.dot(palm_forward, ee_to_obj_norm)
            
            # Stricter alignment reward (Dynamic power)
            reward_orientation = max(0, alignment) ** self.env.current_orient_power * 2.0

        # 4. Touch Reward: SIGNIFICANTLY increased (core objective)
        # Modified to penalize high-velocity impacts (Gentle Touch)
        reward_touch = 0.0
        reward_impact = 0.0
        if self.env.step_touch:
            # Calculate global velocity of End Effector
            ee_vel_global = self.env.Dcmm.data.body("link6").cvel[3:6]
            impact_speed = np.linalg.norm(ee_vel_global)

            # Base reward
            base_touch_reward = 10.0

            # Penalty for high speed impact (Encourage speed < 0.5 m/s)
            reward_impact = -4.0 * impact_speed

            reward_touch = base_touch_reward + reward_impact

        # 5. Regularization: Keep control smooth
        reward_regularization = -self.norm_ctrl(ctrl, ['base', 'arm']) * 0.01

        # 6. Collision Penalty (catastrophic failure)
        reward_collision = 0.0
        if self.env.terminated and not self.env.step_touch:
             reward_collision = DcmmCfg.reward_weights["r_collision"]  # -10.0

        # 7. Plant Collision Penalty: Differentiated
        reward_plant_collision = 0.0

        # Stem Collision (Rigid, Avoid! - Dynamic Penalty)
        if self.env.contacts['plant_contacts'].size != 0:
            reward_plant_collision += self.env.current_w_stem

        # Leaf Collision (Soft, Gentle interaction allowed - Velocity dependent)
        if self.env.contacts['leaf_contacts'].size != 0:
            ee_vel = np.linalg.norm(self.env.Dcmm.data.body("link6").cvel[3:6])
            reward_plant_collision += -0.5 * (1.0 + ee_vel)

        # 8. Action Rate Penalty (Smoothness)
        # Flatten current action dict
        current_action_vec = np.concatenate([ctrl['base'], ctrl['arm'], ctrl['hand']])

        # Initialize prev_action_reward if needed
        if self.prev_action_reward is None:
            self.prev_action_reward = np.zeros_like(current_action_vec)

        action_diff = current_action_vec - self.prev_action_reward
        reward_action_rate = -np.linalg.norm(action_diff) * 0.05
        self.prev_action_reward = current_action_vec.copy()

        # 9. AVP Reward: Learned graspability signal from Stage 2 Critic
        reward_avp = self.compute_avp_reward(obs, info)

        rewards = (reward_reaching + reward_base_approach + reward_orientation +
                  reward_touch + reward_regularization + reward_collision +
                  reward_plant_collision + reward_action_rate + reward_avp)

        if self.env.print_reward:
            print(f"reward_reaching: {reward_reaching:.3f}")
            print(f"reward_base_approach: {reward_base_approach:.3f}")
            print(f"reward_orientation: {reward_orientation:.3f}")
            print(f"reward_touch: {reward_touch:.3f} (Impact Penalty: {reward_impact:.3f})")
            print(f"reward_regularization: {reward_regularization:.3f}")
            print(f"reward_collision: {reward_collision:.3f}")
            print(f"reward_plant_collision: {reward_plant_collision:.3f}")
            print(f"reward_action_rate: {reward_action_rate:.3f}")
            print(f"reward_avp: {reward_avp:.3f}")
            print(f"total reward: {rewards:.3f}\n")

        return rewards

