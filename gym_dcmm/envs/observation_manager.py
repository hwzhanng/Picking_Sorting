"""
Observation management for DcmmVecEnv.
Handles all observation collection and processing.
"""

import numpy as np
from gym_dcmm.utils.quat_utils import quat_rotate_vector
import cv2 as cv


class ObservationManager:
    """Manages observation collection for the environment."""

    def __init__(self, env):
        """
        Initialize observation manager.

        Args:
            env: Reference to the parent DcmmVecEnv instance
        """
        self.env = env

    def get_base_vel(self):
        """Get the velocity of the mobile base in base frame (2D)."""
        base_pos_world = self.env.Dcmm.data.body("arm_base").xpos[0:2]
        base_quat = self.env.Dcmm.data.body("arm_base").xquat
        base_vel_world = self.env.Dcmm.data.body("arm_base").cvel[3:5]

        # Transform to base frame
        base_vel_world_3d = np.array([base_vel_world[0], base_vel_world[1], 0.0])
        base_quat_inv = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]])
        base_vel_base = quat_rotate_vector(base_quat_inv, base_vel_world_3d)

        return base_vel_base[0:2]

    def get_relative_ee_pos3d(self):
        """Get end-effector position relative to base in base frame."""
        ee_pos_world = self.env.Dcmm.data.body("link6").xpos
        base_pos_world = self.env.Dcmm.data.body("arm_base").xpos
        base_quat = self.env.Dcmm.data.body("arm_base").xquat

        # Vector from base to EE in world frame
        ee_rel_world = ee_pos_world - base_pos_world

        # Transform to base frame
        base_quat_inv = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]])
        ee_rel_base = quat_rotate_vector(base_quat_inv, ee_rel_world)

        return ee_rel_base

    def get_relative_ee_quat(self):
        """Get end-effector orientation relative to base."""
        ee_quat = self.env.Dcmm.data.body("link6").xquat
        return ee_quat

    def get_relative_ee_v_lin_3d(self):
        """Get end-effector linear velocity in base frame."""
        ee_vel_world = self.env.Dcmm.data.body("link6").cvel[3:6]
        base_quat = self.env.Dcmm.data.body("arm_base").xquat

        # Transform to base frame
        base_quat_inv = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]])
        ee_vel_base = quat_rotate_vector(base_quat_inv, ee_vel_world)

        return ee_vel_base

    def get_relative_object_pos3d(self):
        """Get object position relative to base in base frame."""
        obj_pos_world = self.env.Dcmm.data.body(self.env.object_name).xpos
        base_pos_world = self.env.Dcmm.data.body("arm_base").xpos
        base_quat = self.env.Dcmm.data.body("arm_base").xquat

        # Vector from base to object in world frame
        obj_rel_world = obj_pos_world - base_pos_world

        # Transform to base frame
        base_quat_inv = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]])
        obj_rel_base = quat_rotate_vector(base_quat_inv, obj_rel_world)

        return obj_rel_base

    def get_relative_object_v_lin_3d(self):
        """Get object linear velocity in base frame."""
        obj_vel_world = self.env.Dcmm.data.body(self.env.object_name).cvel[3:6]
        base_quat = self.env.Dcmm.data.body("arm_base").xquat

        # Transform to base frame
        base_quat_inv = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]])
        obj_vel_base = quat_rotate_vector(base_quat_inv, obj_vel_world)

        return obj_vel_base

    def get_obs(self):
        """
        Collect all observations for the current state.

        Returns:
            dict: Observation dictionary with base, arm, object, and depth info
        """
        obs = {
            "base": {
                "v_lin_2d": self.get_base_vel().astype(np.float32),
            },
            "arm": {
                "ee_pos3d": self.get_relative_ee_pos3d().astype(np.float32),
                "ee_quat": self.get_relative_ee_quat().astype(np.float32),
                "ee_v_lin_3d": self.get_relative_ee_v_lin_3d().astype(np.float32),
                "joint_pos": self.env.Dcmm.data.qpos[15:21].astype(np.float32),
            },
            "object": {
                "pos3d": self.get_relative_object_pos3d().astype(np.float32),
            },
        }

        # Add depth image if render mode is set
        if self.env.render_mode is not None:
            imgs = self.env.render()
            obs["depth"] = imgs.astype(np.float32)
        else:
            obs["depth"] = np.zeros((1, self.env.img_size[0], self.env.img_size[1]), dtype=np.float32)

        if self.env.print_obs:
            print(f"##### Observation #####")
            print(f"base_vel: {obs['base']['v_lin_2d']}")
            print(f"ee_pos3d: {obs['arm']['ee_pos3d']}")
            print(f"ee_quat: {obs['arm']['ee_quat']}")
            print(f"ee_v_lin_3d: {obs['arm']['ee_v_lin_3d']}")
            print(f"joint_pos: {obs['arm']['joint_pos']}")
            print(f"object_pos3d: {obs['object']['pos3d']}")
            print(f"depth shape: {obs['depth'].shape}\n")

        return obs

    def get_hand_obs(self):
        """
        Get hand joint positions.

        Returns:
            np.ndarray: Hand joint positions
        """
        import configs.env.DcmmCfg as DcmmCfg
        hand_joint_indices = np.where(DcmmCfg.hand_mask == 1)[0] + 15
        hand_obs = self.env.Dcmm.data.qpos[hand_joint_indices].astype(np.float32)
        return hand_obs

