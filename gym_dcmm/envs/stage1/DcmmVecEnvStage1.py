"""
Main DcmmVecEnv environment class (Refactored).
This file now imports functionality from modular manager classes.
"""

import os
import sys
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../gym_dcmm/'))

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from gym_dcmm.agents.MujocoDcmm import MJ_DCMM
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from collections import deque
import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.utils.util import get_total_dimension, DynamicDelayBuffer

# Import manager classes
from gym_dcmm.envs.constants import env_key_callback, cmd_lin_x, cmd_lin_y, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
from gym_dcmm.envs.observation_manager import ObservationManager
from gym_dcmm.envs.stage1.RewardManagerStage1 import RewardManagerStage1
from gym_dcmm.envs.randomization_manager import RandomizationManager
from gym_dcmm.envs.control_manager import ControlManager
from gym_dcmm.envs.render_manager import RenderManager

np.set_printoptions(precision=8)


class DcmmVecEnvStage1(gym.Env):
    """
    DCMM Vectorized Environment for mobile manipulation tasks.

    Supports two tasks:
    - Tracking: Touch target with palm (gentle contact)
    - Catching: Grasp moving object with hand

    Args:
        task (str): Task type - "tracking" or "catching"
        render_mode (str): Rendering mode - "rgb_array", "depth_array", "depth_rgb_array"
        render_per_step (bool): Whether to render every simulation step
        viewer (bool): Whether to show MuJoCo viewer
        imshow_cam (bool): Whether to display camera images
        object_eval (bool): Use evaluation object set
        camera_name (list): List of camera names to use
        object_name (str): Name of the target object
        env_time (float): Maximum episode time
        steps_per_policy (int): Simulation steps per policy action
        img_size (tuple): Camera image size (height, width)
        device (str): Device for computation
        print_* (bool): Various debug printing flags
    """

    metadata = {"render_modes": ["rgb_array", "depth_array", "depth_rgb_array"]}

    def __init__(
        self,
        task="tracking",
        render_mode="depth_array",
        render_per_step=False,
        viewer=False,
        imshow_cam=False,
        object_eval=False,
        camera_name=["wrist"],
        object_name="object",
        env_time=2.5,
        steps_per_policy=20,
        img_size=(480, 640),
        device='cuda:0',
        print_obs=False,
        print_reward=False,
        print_ctrl=False,
        print_info=False,
        print_contacts=False,
    ):
        task = task.capitalize()
        if task not in ["Tracking", "Catching"]:
            raise ValueError("Invalid task: {}".format(task))

        # Force depth_array for vision-based training
        self.render_mode = "depth_array"
        self.camera_name = camera_name
        self.object_name = object_name
        self.imshow_cam = imshow_cam
        self.task = task
        self.img_size = img_size
        self.device = device
        self.steps_per_policy = steps_per_policy
        self.render_per_step = render_per_step

        # Print Settings
        self.print_obs = print_obs
        self.print_reward = print_reward
        self.print_ctrl = print_ctrl
        self.print_info = print_info
        self.print_contacts = print_contacts

        # Initialize the DCMM robot
        self.Dcmm = MJ_DCMM(viewer=viewer, object_name=object_name, object_eval=object_eval)
        self.fps = 1 / (self.steps_per_policy * self.Dcmm.model.opt.timestep)

        # Initialize manager classes
        self.obs_manager = ObservationManager(self)
        self.reward_manager = RewardManagerStage1(self)
        self.random_manager = RandomizationManager(self)
        self.control_manager = ControlManager(self, stage=1)  # Stage 1: hand locked
        self.render_manager = RenderManager(self)

        # Randomize the Object Info
        self.random_mass = 0.25
        self.object_static_time = 0.75
        self.object_throw = False
        self.object_train = True
        if object_eval:
            self.set_object_eval()

        # Reset object properties
        self.Dcmm.model_xml_string = self.random_manager.reset_object()
        self.Dcmm.model = mujoco.MjModel.from_xml_string(self.Dcmm.model_xml_string)
        self.Dcmm.data = mujoco.MjData(self.Dcmm.model)

        # Get important geom IDs
        self.hand_start_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'mcp_joint') - 1
        print("self.hand_start_id: ", self.hand_start_id)
        self.floor_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.object_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)
        self.base_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'ranger_base')

        # Set camera configuration
        self.Dcmm.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]
        self.Dcmm.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        self.mujoco_renderer = MujocoRenderer(self.Dcmm.model, self.Dcmm.data)

        # Setup viewer if needed
        if self.Dcmm.open_viewer:
            if self.Dcmm.viewer:
                print("Close the previous viewer")
                self.Dcmm.viewer.close()
            self.Dcmm.viewer = mujoco.viewer.launch_passive(
                self.Dcmm.model, self.Dcmm.data, key_callback=env_key_callback)
            # Modify the view position and orientation
            self.Dcmm.viewer.cam.lookat[0:2] = [0, 1]
            self.Dcmm.viewer.cam.distance = 5.0
            self.Dcmm.viewer.cam.azimuth = 180
        else:
            self.Dcmm.viewer = None

        # Define observation space
        hand_joint_indices = np.where(DcmmCfg.hand_mask == 1)[0] + 15
        self.observation_space = spaces.Dict(
            {
                "base": spaces.Dict({
                    "v_lin_2d": spaces.Box(-4, 4, shape=(2,), dtype=np.float32),
                }),
                "arm": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                    "joint_pos": spaces.Box(
                        low = np.array([self.Dcmm.model.jnt_range[i][0] for i in range(9, 15)]),
                        high = np.array([self.Dcmm.model.jnt_range[i][1] for i in range(9, 15)]),
                        dtype=np.float32),
                }),
                "object": spaces.Dict({
                    "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                }),
                "depth": spaces.Box(low=0, high=255, shape=(1, self.img_size[0], self.img_size[1]), dtype=np.uint8),
            }
        )

        # Define action space
        base_low = np.array([-4, -4])
        base_high = np.array([4, 4])
        arm_low = -0.05*np.ones(6)
        arm_high = 0.05*np.ones(6)
        hand_low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices])
        hand_high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices])

        # Get initial positions
        self.init_pos = True
        self.initial_ee_pos3d = self.obs_manager.get_relative_ee_pos3d()
        self.initial_obj_pos3d = self.obs_manager.get_relative_object_pos3d()
        self.prev_ee_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_obj_pos3d = np.array([0.0, 0.0, 0.0])

        # Fixed open hand posture for Stage 1 (12 DOF hand)
        self.hand_open_angles = np.array([
            0.0, 0.0, 0.0,   # Index finger extended
            0.0, 0.0, 0.0,   # Middle finger extended
            0.0, 0.0, 0.0,   # Ring finger extended
            0.0, 0.3, 0.0    # Thumb slightly abducted
        ])
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d[:] = self.initial_obj_pos3d[:]

        self.action_space = spaces.Dict(
            {
                "base": spaces.Box(base_low, base_high, shape=(2,), dtype=np.float32),
                "arm": spaces.Box(arm_low, arm_high, shape=(6,), dtype=np.float32),
                "hand": spaces.Box(hand_low, hand_high, shape=(12,), dtype=np.float32),
            }
        )

        self.action_buffer = {
            "base": DynamicDelayBuffer(maxlen=2),
            "arm": DynamicDelayBuffer(maxlen=2),
            "hand": DynamicDelayBuffer(maxlen=2),
        }

        # Combine the limits of the action space
        self.actions_low = np.concatenate([base_low, arm_low, hand_low])
        self.actions_high = np.concatenate([base_high, arm_high, hand_high])

        self.obs_dim = get_total_dimension(self.observation_space)
        self.act_dim = get_total_dimension(self.action_space)
        # Dimension for training
        self.obs_t_dim = 2 + 3 + 4 + 3 + 3  # Total: 15
        self.act_t_dim = 8 # 2 base + 6 arm
        self.obs_c_dim = self.obs_dim - 6  # dim = 30
        self.act_c_dim = 20 # 2 base + 6 arm + 12 hand
        print("##### Tracking Task \n obs_dim: {}, act_dim: {}".format(self.obs_t_dim, self.act_t_dim))
        print("##### Catching Task \n obs_dim: {}, act_dim: {}\n".format(self.obs_c_dim, self.act_c_dim))

        # Init env params
        self.arm_limit = True
        self.terminated = False
        self.start_time = self.Dcmm.data.time
        self.catch_time = self.Dcmm.data.time - self.start_time
        self.reward_touch = 0
        self.reward_stability = 0
        self.env_time = env_time
        self.stage_list = ["tracking", "grasping"]
        self.stage = self.stage_list[0]
        self.steps = 0

        self.prev_ctrl = np.zeros(20)
        self.init_ctrl = True
        self.vel_init = False
        self.vel_history = deque(maxlen=4)

        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("link6").xpos -
                                          self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                            self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "env_time": self.Dcmm.data.time - self.start_time,
            "imgs": {}
        }
        self.contacts = {
            "object_contacts": np.array([]),
            "hand_contacts": np.array([]),
            "base_contacts": np.array([]),
            "plant_contacts": np.array([]),
            "leaf_contacts": np.array([]),
        }

        self.object_q = np.array([1, 0, 0, 0])
        self.object_pos3d = np.array([0, 0, 1.5])
        self.object_vel6d = np.array([0., 0., 1.25, 0.0, 0.0, 0.0])
        self.step_touch = False

        self.imgs = np.zeros((0, self.img_size[0], self.img_size[1], 1))

        # Random PID Params
        self.k_arm = np.ones(6)
        self.k_drive = np.ones(4)
        self.k_steer = np.ones(4)
        self.k_hand = np.ones(1)

        # Random Obs & Act Params
        self.k_obs_base = DcmmCfg.k_obs_base
        self.k_obs_arm = DcmmCfg.k_obs_arm
        self.k_obs_hand = DcmmCfg.k_obs_hand
        self.k_obs_object = DcmmCfg.k_obs_object
        self.k_act = DcmmCfg.k_act

        # Curriculum Learning Params
        self.global_step = 0
        self.current_w_stem = DcmmCfg.curriculum.collision_stem_start
        self.current_w_stem = DcmmCfg.curriculum.collision_stem_start
        self.current_orient_power = DcmmCfg.curriculum.orient_power_start
        self.last_debug_step = -1

    def set_object_eval(self):
        """Set environment to use evaluation objects."""
        self.object_train = False

    def update_render_state(self, render_per_step):
        """Update rendering state."""
        self.render_per_step = render_per_step

    def update_stage(self, stage):
        """Update current task stage."""
        if stage in self.stage_list:
            self.stage = stage
        else:
            raise ValueError("Invalid stage: {}".format(stage))

    def render(self):
        """Render the current state (delegates to RenderManager)."""
        return self.render_manager.render()

    def _get_info(self):
        """Get current environment info."""
        env_time = self.Dcmm.data.time - self.start_time
        ee_distance = np.linalg.norm(self.Dcmm.data.body("link6").xpos -
                                    self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3])
        base_distance = np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                        self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2])
        if self.print_info:
            print("##### print info")
            print("env_time: ", env_time)
            print("ee_distance: ", ee_distance)
        return {
            "env_time": env_time,
            "ee_distance": ee_distance,
            "base_distance": base_distance,
        }

    def _reset_simulation(self):
        """Reset the MuJoCo simulation (uses RandomizationManager)."""
        # Reset the data in Mujoco Simulation
        mujoco.mj_resetData(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_resetData(self.Dcmm.model_arm, self.Dcmm.data_arm)
        if self.Dcmm.model.na == 0:
            self.Dcmm.data.act[:] = None
        if self.Dcmm.model_arm.na == 0:
            self.Dcmm.data_arm.act[:] = None
        self.Dcmm.data.ctrl = np.zeros(self.Dcmm.model.nu)
        self.Dcmm.data_arm.ctrl = np.zeros(self.Dcmm.model_arm.nu)
        self.Dcmm.data.qpos[15:21] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.qpos[21:37] = DcmmCfg.hand_joints[:]
        self.Dcmm.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.body("object").xpos[0:3] = np.array([2, 2, 1])

        # Randomize plants and fruit
        self.random_manager.randomize_plants()
        self.random_manager.randomize_fruit_and_occlusion()
        
        # Apply visual domain randomization (lighting + ground color)
        self.random_manager.apply_full_visual_dr()

        # Set the object position (fruit is mocap)
        object_body_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        if object_body_id != -1:
            object_mocap_id = self.Dcmm.model.body_mocapid[object_body_id]
            if object_mocap_id != -1:
                self.Dcmm.data.mocap_pos[object_mocap_id] = self.object_pos3d
                self.Dcmm.data.mocap_quat[object_mocap_id] = self.object_q

        # Random Gravity
        self.Dcmm.model.opt.gravity[2] = -9.81 + 0.5*np.random.uniform(-1, 1)
        # Random PID
        self.random_manager.random_PID()
        # Random Delay
        self.random_manager.random_delay()
        # Forward Kinematics
        mujoco.mj_forward(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_forward(self.Dcmm.model_arm, self.Dcmm.data_arm)

    def reset(self):
        """Reset the environment to initial state."""
        # Reset the basic simulation
        self._reset_simulation()
        self.init_ctrl = True
        self.init_pos = True
        self.vel_init = False
        self.object_throw = False
        self.steps = 0

        # Reset Action Filter
        self.prev_action = np.zeros(20) # 2 base + 6 arm + 12 hand
        self.alpha_lpf = 0.3 # Smoothing factor

        # Reset the time
        self.start_time = self.Dcmm.data.time
        self.catch_time = self.Dcmm.data.time - self.start_time

        ## Reset target states
        self.Dcmm.target_base_vel = np.array([0.0, 0.0, 0.0])
        self.Dcmm.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
        self.Dcmm.target_hand_qpos[:] = DcmmCfg.hand_joints[:]

        # [Fix 2025-12-09] Ensure arm qpos matches target to prevent PID fighting
        self.Dcmm.data.qpos[15:21] = self.Dcmm.target_arm_qpos.copy()

        # [Fix 2025-12-09] Initialize action buffers with valid initial values
        # This prevents IndexError when get_ctrl() accesses empty buffers
        for _ in range(self.action_buffer["arm"].maxlen):
            self.action_buffer["base"].append(np.zeros(3))
            self.action_buffer["arm"].append(self.Dcmm.target_arm_qpos.copy())
            self.action_buffer["hand"].append(self.Dcmm.target_hand_qpos.copy())

        # Set hand to fixed open posture
        self.Dcmm.data.qpos[21:33] = self.hand_open_angles
        mujoco.mj_forward(self.Dcmm.model, self.Dcmm.data)

        ## Reset the reward
        self.stage = "tracking"
        self.terminated = False
        self.reward_touch = 0
        self.contact_count = 0  # Reset contact counter
        self.reward_stability = 0

        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("link6").xpos -
                                       self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "evn_time": self.Dcmm.data.time - self.start_time,
        }

        # Get the observation and info
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d = self.obs_manager.get_relative_object_pos3d()
        observation = self.obs_manager.get_obs()
        info = self._get_info()

        # Rendering
        if self.render_mode is not None:
            imgs = self.render()
            info['imgs'] = imgs
        else:
            info['imgs'] = None

        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))

        return observation, info

    def update_curriculum_difficulty(self):
        # Calculate curriculum coefficient alpha (0 to 1)
        current_step = self.global_step
        
        # Simple linear interpolation logic
        # 0 ~ 6M steps: Difficulty increases from 0.0 to 1.0
        # > 6M steps: Difficulty locked at 1.0 (Full)
        max_steps = DcmmCfg.curriculum.stage2_steps
        difficulty = min(max(current_step / max_steps, 0.0), 1.0)
        
        # Dynamic adjustment of Stem Collision Penalty
        w_stem_start = DcmmCfg.curriculum.collision_stem_start
        w_stem_end = DcmmCfg.curriculum.collision_stem_end
        self.current_w_stem = w_stem_start + (w_stem_end - w_stem_start) * difficulty
        
        # Dynamic adjustment of Orientation Strictness
        p_start = DcmmCfg.curriculum.orient_power_start
        p_end = DcmmCfg.curriculum.orient_power_end
        self.current_orient_power = p_start + (p_end - p_start) * difficulty
        
        # Debug print (ensure it prints roughly every 10k steps, handling batch jumps)
        if self.global_step // 10000 > self.last_debug_step // 10000:
            print(f"[Curriculum] Step: {self.global_step}, Stem Penalty: {self.current_w_stem:.2f}, Orient Power: {self.current_orient_power:.2f}")
            self.last_debug_step = self.global_step

        return self.current_w_stem, self.current_orient_power

    def set_global_step(self, step):
        """Set the global training step for curriculum learning."""
        self.global_step = step
    
    @property
    def curriculum_difficulty(self):
        """Get current curriculum difficulty (0.0 to 1.0) for AVP lambda decay."""
        max_steps = DcmmCfg.curriculum.stage2_steps
        return min(max(self.global_step / max_steps, 0.0), 1.0)
    
    def get_avp_stats(self):
        """Get AVP statistics for WandB logging (called via env_method from PPO)."""
        if hasattr(self, 'reward_manager') and hasattr(self.reward_manager, 'get_avp_stats_and_reset'):
            return self.reward_manager.get_avp_stats_and_reset()
        return None

    def get_reward_stats(self):
        """Get reward decomposition statistics for WandB logging."""
        if hasattr(self, 'reward_manager') and hasattr(self.reward_manager, 'get_reward_stats_and_reset'):
            return self.reward_manager.get_reward_stats_and_reset()
        return None


    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: Action dictionary with 'base' and 'arm' keys

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Update curriculum parameters based on global step
        self.update_curriculum_difficulty()

        try:
            self.steps += 1
            self.control_manager.step_mujoco_simulation(action)

            # Get the obs and info
            obs = self.obs_manager.get_obs()
            info = self._get_info()

            if self.task == 'Catching':
                if info['ee_distance'] < DcmmCfg.distance_thresh and self.stage == "tracking":
                    self.stage = "grasping"
                elif info['ee_distance'] >= DcmmCfg.distance_thresh and self.stage == "grasping":
                    self.terminated = True

            # Design the reward function
            reward = self.reward_manager.compute_reward(obs, info, action)
            self.info["base_distance"] = info["base_distance"]
            self.info["ee_distance"] = info["ee_distance"]

            # Rendering
            if self.render_mode is not None and self.imshow_cam:
                imgs = self.render()
                info['imgs'] = imgs
            else:
                info['imgs'] = None

            ctrl_delay = np.array([len(self.action_buffer['base']),
                                len(self.action_buffer['arm']),
                                len(self.action_buffer['hand'])])
            info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))

            # Check truncation conditions
            if self.task == "Catching":
                if info["env_time"] > self.env_time:
                    truncated = True
                else:
                    truncated = False
            elif self.task == "Tracking":
                # Require sustained contact (10 steps) before success
                if self.step_touch:
                    self.contact_count += 1
                    if self.contact_count >= 10:
                        truncated = True
                    else:
                        truncated = False
                else:
                    # Lost contact, reset counter
                    self.contact_count = 0
                    truncated = False

            terminated = self.terminated
            done = terminated or truncated

            return obs, reward, terminated, truncated, info
        except Exception as e:
            print(f"Exception in env step: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def close(self):
        """Close the environment and cleanup resources."""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        if self.Dcmm.viewer != None:
            self.Dcmm.viewer.close()

    def run_test(self):
        """Run a manual test with keyboard control."""
        import gym_dcmm.envs.constants as constants
        self.reset()
        action = np.zeros(20)
        while True:
            # Keyboard control
            action[0:2] = np.array([constants.cmd_lin_x, constants.cmd_lin_y])
            if constants.trigger_delta:
                print("delta_xyz: ", constants.delta_xyz)
                action[2:8] = np.array([constants.delta_xyz]*6)
                constants.trigger_delta = False
            else:
                action[2:8] = np.zeros(6)
            if constants.trigger_delta_hand:
                print("delta_xyz_hand: ", constants.delta_xyz_hand)
                action[8:20] = np.ones(12)*constants.delta_xyz_hand
                constants.trigger_delta_hand = False
            else:
                action[8:20] = np.zeros(12)

            actions_dict = {
                'arm': action[2:8],
                'base': action[:2],
                'hand': action[8:20]
            }
            observation, reward, terminated, truncated, info = self.step(actions_dict)


if __name__ == "__main__":
    os.chdir('../../')
    parser = argparse.ArgumentParser(description="Args for DcmmVecEnv")
    parser.add_argument('--viewer', action='store_true', help="open the mujoco.viewer or not")
    parser.add_argument('--imshow_cam', action='store_true', help="imshow the camera image or not")
    args = parser.parse_args()
    print("args: ", args)
    env = DcmmVecEnvStage1(task='Catching', object_name='object', render_per_step=False,
                    print_reward=False, print_info=False,
                    print_contacts=False, print_ctrl=False,
                    print_obs=False, camera_name = ["top"],
                    render_mode="rgb_array", imshow_cam=args.imshow_cam,
                    viewer = args.viewer, object_eval=False,
                    env_time = 2.5, steps_per_policy=20)
    env.run_test()

