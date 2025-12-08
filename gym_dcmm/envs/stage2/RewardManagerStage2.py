"""
Reward computation for DcmmVecEnvCatch.
Handles all reward calculation logic for Stage 2 (Catch).
"""

import numpy as np
import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.utils.quat_utils import quat_rotate_vector


class RewardManagerStage2:
    """Manages reward computation for the environment (Stage 2 Catch)."""

    def __init__(self, env):
        """
        Initialize reward manager.

        Args:
            env: Reference to the parent DcmmVecEnvCatch instance
        """
        self.env = env
        self.prev_action_reward = None
        
        # Perturbation Test State
        self.perturbation_active = False
        self.initial_grasp_pos = None  # Object position when force threshold met
        self.perturbation_timer = 0.0
        self.perturbation_force_mag = 0.0
        self.perturbation_direction = np.zeros(3)

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

    def apply_perturbation_force(self):
        """
        Apply random external force to the object to test grasp stability.
        Simulates real-world disturbances (wind, pulling, etc.)
        """
        # Generate random force direction (uniformly on sphere)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        
        self.perturbation_direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Random force magnitude (reduced for easier training)
        self.perturbation_force_mag = np.random.uniform(0.5, 1.5)
        
        # Apply force to object via MuJoCo's external force array
        # xfrc_applied is [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        object_body_id = self.env.Dcmm.data.body(self.env.object_name).id
        force_vector = self.perturbation_direction * self.perturbation_force_mag
        self.env.Dcmm.data.xfrc_applied[object_body_id, :3] = force_vector
        
    def compute_slippage(self):
        """
        Measure object displacement from initial grasp position.
        Returns slippage distance in meters.
        """
        current_obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos
        if self.initial_grasp_pos is None:
            return 0.0
        slippage = np.linalg.norm(current_obj_pos - self.initial_grasp_pos)
        return slippage
    
    def evaluate_grasp_stability(self, total_contact_force):
        """
        Orchestrate perturbation test and return stability reward.
        
        Args:
            total_contact_force: Sum of all touch sensor readings
            
        Returns:
            float: Perturbation reward (+10.0 for stable, -5.0 for slip)
        """
        reward_perturbation = 0.0
        dt = self.env.Dcmm.model.opt.timestep * self.env.steps_per_policy
        
        # State machine: Idle -> Testing -> Evaluate
        if not self.perturbation_active:
            # Check if conditions met to enter testing phase
            if total_contact_force >= 1.0:
                # Enter testing mode
                self.perturbation_active = True
                self.initial_grasp_pos = self.env.Dcmm.data.body(self.env.object_name).xpos.copy()
                self.perturbation_timer = 0.0
                # Apply initial perturbation force
                self.apply_perturbation_force()
        else:
            # Testing phase: accumulate time and check slippage
            self.perturbation_timer += dt
            
            # Continuously apply force during test window (0.5 seconds)
            if self.perturbation_timer < 0.5:
                # Refresh force application
                object_body_id = self.env.Dcmm.data.body(self.env.object_name).id
                force_vector = self.perturbation_direction * self.perturbation_force_mag
                self.env.Dcmm.data.xfrc_applied[object_body_id, :3] = force_vector
            else:
                # Test complete: evaluate slippage
                slippage = self.compute_slippage()
                threshold = 0.01  # 1cm
                
                if slippage < threshold:
                    # Success: Resisted perturbation
                    reward_perturbation = 10.0
                else:
                    # Failure: Object slipped
                    reward_perturbation = -5.0
                
                # Reset for next test
                self.perturbation_active = False
                self.initial_grasp_pos = None
                # Clear external force
                object_body_id = self.env.Dcmm.data.body(self.env.object_name).id
                self.env.Dcmm.data.xfrc_applied[object_body_id, :] = 0.0
                
        return reward_perturbation

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
        
        # 1b. Distance Shaping Reward (milestone bonuses for approaching)
        reward_distance_shaping = 0.0
        ee_dist = info["ee_distance"]
        if ee_dist < 0.30:
            reward_distance_shaping += 1.0
        if ee_dist < 0.15:
            reward_distance_shaping += 2.0
        if ee_dist < 0.08:
            reward_distance_shaping += 3.0
        if ee_dist < 0.05:  # Stage 1 target distance
            reward_distance_shaping += 5.0

        # 2. Base Approach Reward: REMOVED (Base is locked)
        reward_base_approach = 0.0

        # 3. Orientation Reward: Palm should face target (Stricter 4th power)
        reward_orientation = 0.0
        if info["ee_distance"] < 0.5: # Only active when close
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

        # 4. Soft Grasp Reward (Force Feedback) - Relaxed thresholds
        reward_grasp = 0.0
        total_contact_force = np.sum(obs['touch'])
        force_threshold_low = 0.5   # Reduced from 1.0
        force_threshold_high = 4.0   # Increased from 2.5
        
        if total_contact_force > 0.01: # If touching
            if total_contact_force < force_threshold_low:
                # Any contact gets base reward + bonus for force
                reward_grasp = 2.0 + 1.0 * total_contact_force
            elif force_threshold_low <= total_contact_force <= force_threshold_high:
                # Perfect range: High dense reward
                reward_grasp = 5.0
                # Bonus for wrapping (more fingers touching)
                fingers_touching = np.count_nonzero(obs['touch'] > 0.1)
                reward_grasp += fingers_touching * 1.0
            else: # > force_threshold_high
                # Too strong: Gentle penalty (reduced from -10 to -3)
                reward_grasp = max(-3.0, 3.0 - 0.5 * (total_contact_force - force_threshold_high))
        
        # 4b. Perturbation Test Reward (Grasp Stability Validation)
        reward_perturbation = self.evaluate_grasp_stability(total_contact_force)
        
        # 5. Impact Velocity Penalty (Exponential) - STRENGTHENED
        reward_impact = 0.0
        # Check if any touch sensor is active or general contact
        if total_contact_force > 0.01 or self.env.step_touch:
             ee_vel_global = self.env.Dcmm.data.body("link6").cvel[3:6]
             impact_speed = np.linalg.norm(ee_vel_global)
             
             # Reduced exponential penalty with lower cap
             # Speed < 0.3m/s -> penalty ~ 0
             # Speed = 0.5m/s -> penalty ~ -3.4
             # Speed >= 1.0m/s -> penalty capped at -10.0 (reduced from -30)
             raw_impact_penalty = -2.0 * (np.exp(3.0 * max(0, impact_speed - 0.3)) - 1.0)
             reward_impact = max(-10.0, raw_impact_penalty)  # Reduced cap

        # 6. Regularization: Keep control smooth
        reward_regularization = -self.norm_ctrl(ctrl, ['arm', 'hand']) * 0.01

        # 7. Collision Penalty (reduced severity)
        reward_collision = 0.0
        # If terminated but NOT success -> Failure
        if self.env.terminated and not info.get('is_success', False):
             reward_collision = -5.0  # Reduced from -10.0

        # 8. Plant Collision Penalty: Differentiated
        reward_plant_collision = 0.0

        # Stem Collision (Rigid, Avoid! - Dynamic Penalty)
        if self.env.contacts['plant_contacts'].size != 0:
            reward_plant_collision += self.env.current_w_stem # e.g. -20.0

        # Leaf Collision (Soft, Gentle interaction allowed - Velocity dependent)
        if self.env.contacts['leaf_contacts'].size != 0:
            ee_vel = np.linalg.norm(self.env.Dcmm.data.body("link6").cvel[3:6])
            # Small penalty encouraging slow movement through leaves
            reward_plant_collision += -0.1 * (1.0 + ee_vel)

        # 9. Action Rate Penalty (Smoothness)
        # Flatten current action dict
        current_action_vec = np.concatenate([ctrl['base'], ctrl['arm'], ctrl['hand']])

        # Initialize prev_action_reward if needed
        if self.prev_action_reward is None:
            self.prev_action_reward = np.zeros_like(current_action_vec)

        action_diff = current_action_vec - self.prev_action_reward
        reward_action_rate = -np.linalg.norm(action_diff) * 0.05
        self.prev_action_reward = current_action_vec.copy()
        
        # 10. Success Reward
        reward_success = 0.0
        if info.get('is_success', False):
            reward_success = 50.0 # Large sparse reward

        rewards = (reward_reaching + reward_distance_shaping + reward_orientation +
                  reward_grasp + reward_perturbation + reward_impact + 
                  reward_regularization + reward_collision +
                  reward_plant_collision + reward_action_rate + reward_success)

        # Track reward components for WandB logging
        if not hasattr(self, 'reward_stats'):
            self._init_reward_stats()
        self.reward_stats['reaching_sum'] += reward_reaching
        self.reward_stats['distance_shaping_sum'] += reward_distance_shaping
        self.reward_stats['orientation_sum'] += reward_orientation
        self.reward_stats['grasp_sum'] += reward_grasp
        self.reward_stats['perturbation_sum'] += reward_perturbation
        self.reward_stats['impact_sum'] += reward_impact
        self.reward_stats['collision_sum'] += reward_collision
        self.reward_stats['success_sum'] += reward_success
        self.reward_stats['contact_force_sum'] += total_contact_force
        self.reward_stats['fingers_touching_sum'] += np.count_nonzero(obs['touch'] > 0.1)
        self.reward_stats['count'] += 1

        if self.env.print_reward:
            print(f"reward_reaching: {reward_reaching:.3f}")
            print(f"reward_orientation: {reward_orientation:.3f}")
            print(f"reward_grasp: {reward_grasp:.3f} (Force: {total_contact_force:.2f}N)")
            print(f"reward_perturbation: {reward_perturbation:.3f}")
            print(f"reward_impact: {reward_impact:.3f}")
            print(f"reward_regularization: {reward_regularization:.3f}")
            print(f"reward_collision: {reward_collision:.3f}")
            print(f"reward_plant_collision: {reward_plant_collision:.3f}")
            print(f"reward_success: {reward_success:.3f}")
            print(f"total reward: {rewards:.3f}\n")

        return rewards

    def _init_reward_stats(self):
        """Initialize reward statistics for WandB logging."""
        self.reward_stats = {
            'reaching_sum': 0.0,
            'distance_shaping_sum': 0.0,
            'orientation_sum': 0.0,
            'grasp_sum': 0.0,
            'perturbation_sum': 0.0,
            'impact_sum': 0.0,
            'collision_sum': 0.0,
            'success_sum': 0.0,
            'contact_force_sum': 0.0,
            'fingers_touching_sum': 0,
            'count': 0,
        }
    
    def get_reward_stats_and_reset(self):
        """
        Get reward statistics for WandB logging and reset counters.
        
        Returns:
            dict: Reward component averages
        """
        if not hasattr(self, 'reward_stats') or self.reward_stats['count'] == 0:
            return None
        
        count = self.reward_stats['count']
        stats = {
            'rewards/reaching_mean': self.reward_stats['reaching_sum'] / count,
            'rewards/distance_shaping_mean': self.reward_stats['distance_shaping_sum'] / count,
            'rewards/orientation_mean': self.reward_stats['orientation_sum'] / count,
            'rewards/grasp_mean': self.reward_stats['grasp_sum'] / count,
            'rewards/perturbation_mean': self.reward_stats['perturbation_sum'] / count,
            'rewards/impact_mean': self.reward_stats['impact_sum'] / count,
            'rewards/collision_mean': self.reward_stats['collision_sum'] / count,
            'rewards/success_mean': self.reward_stats['success_sum'] / count,
            'grasp/contact_force_mean': self.reward_stats['contact_force_sum'] / count,
            'grasp/fingers_touching_mean': self.reward_stats['fingers_touching_sum'] / count,
        }
        
        # Reset counters
        self._init_reward_stats()
        
        return stats

