"""
Reward computation for DcmmVecEnv.
Handles all reward calculation logic.
"""

import numpy as np
import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.utils.quat_utils import quat_rotate_vector


class RewardManager:
    """Manages reward computation for the environment."""

    def __init__(self, env):
        """
        Initialize reward manager.

        Args:
            env: Reference to the parent DcmmVecEnv instance
        """
        self.env = env
        self.prev_action_reward = None

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
        # 1. EE Reaching Reward: Linear positive shaping
        reward_reaching = max(0.0, 5.0 - info["ee_distance"])

        # 2. Base Approach Reward: Encourage vehicle to move closer
        reward_base_approach = max(0.0, 2.0 - info["base_distance"])

        # 3. Orientation Reward: Palm should face target (extended trigger range)
        reward_orientation = 0.0
        if info["ee_distance"] < 2.0:  # Extended from 1.0m to 2.0m
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
            # Increased from 1.0 to 2.0 for stronger signal
            reward_orientation = max(0, alignment) * 2.0

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
            # Weight = 4.0:
            # 0.1 m/s -> -0.4 (Reward 9.6)
            # 1.0 m/s -> -4.0 (Reward 6.0)
            # 2.5 m/s -> -10.0 (Reward 0.0)
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

        # Stem Collision (Rigid, Avoid!)
        if self.env.contacts['plant_contacts'].size != 0:
            reward_plant_collision += -2.0

        # Leaf Collision (Soft, Gentle interaction allowed)
        if self.env.contacts['leaf_contacts'].size != 0:
            # Penalize based on velocity to encourage gentle pushing
            ee_vel = np.linalg.norm(self.env.Dcmm.data.body("link6").cvel[3:6])
            reward_plant_collision += -0.1 * (1.0 + ee_vel)

        # 8. Action Rate Penalty (Smoothness)
        # Flatten current action dict
        current_action_vec = np.concatenate([ctrl['base'], ctrl['arm'], ctrl['hand']])

        # Initialize prev_action_reward if needed
        if self.prev_action_reward is None:
            self.prev_action_reward = np.zeros_like(current_action_vec)

        action_diff = current_action_vec - self.prev_action_reward
        reward_action_rate = -np.linalg.norm(action_diff) * 0.05
        self.prev_action_reward = current_action_vec.copy()

        rewards = (reward_reaching + reward_base_approach + reward_orientation +
                  reward_touch + reward_regularization + reward_collision +
                  reward_plant_collision + reward_action_rate)

        if self.env.print_reward:
            print(f"reward_reaching: {reward_reaching:.3f}")
            print(f"reward_base_approach: {reward_base_approach:.3f}")
            print(f"reward_orientation: {reward_orientation:.3f}")
            print(f"reward_touch: {reward_touch:.3f} (Impact Penalty: {reward_impact:.3f})")
            print(f"reward_regularization: {reward_regularization:.3f}")
            print(f"reward_collision: {reward_collision:.3f}")
            print(f"reward_plant_collision: {reward_plant_collision:.3f}")
            print(f"reward_action_rate: {reward_action_rate:.3f}")
            print(f"total reward: {rewards:.3f}\n")

        return rewards

