import os
import numpy as np
from pathlib import Path

## Define the model path
path = os.path.realpath(__file__)
root = str(Path(path).parent)
ASSET_PATH = os.path.join(root, "../../assets")
# print("ASSET_PATH: ", ASSET_PATH)
# Use Leap Hand
XML_DCMM_LEAP_OBJECT_PATH = "urdf/x1_xarm6_leap_right_object.xml"
XML_DCMM_LEAP_UNSEEN_OBJECT_PATH = "urdf/x1_xarm6_leap_right_unseen_object.xml"
XML_ARM_PATH = "urdf/xarm6_right.xml"
## Weight Saved Path
WEIGHT_PATH = os.path.join(ASSET_PATH, "weights")

## The distance threshold to change the stage from 'tracking' to 'grasping'
distance_thresh = 0.25

## Define the initial joint positions of the arm and the hand
# Pre-grasp pose for maximum flexibility (Stage 2 optimized)
arm_joints = np.array([
   0.0, 0.0, 0.0, 0.8, 0.0, 0.0 
])

hand_joints = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
])

## Define the reward weights
reward_weights = {
    "r_base_pos": 0.0,
    "r_ee_pos": 1.0,
    "r_precision": 10.0,
    "r_orient": 1.0,
    "r_touch": {
        'Tracking': 10,
        'Catching': 0.1
    },
    "r_constraint": 1.0,
    "r_stability": 20.0,
    "r_ctrl": {
        'base': 0.2,
        'arm': 1.0,
        'hand': 0.2,
    },
    "r_collision": -10.0,
    "r_finger_approach": 1.0,
    "r_force_closure": 5.0,
    "r_regularization": 0.05,
}

## Define the camera params for the MujocoRenderer.
cam_config = {
    "name": "top",
    "width": 640,
    "height": 480,
}

## Define the params of the Double Ackerman model.
RangerMiniV2Params = { 
  'wheel_radius': 0.1,                  # in meter //ranger-mini 0.1
  'steer_track': 0.364,                 # in meter (left & right wheel distance) //ranger-mini 0.364
  'wheel_base': 0.494,                   # in meter (front & rear wheel distance) //ranger-mini 0.494
  'max_linear_speed': 1.5,              # in m/s
  'max_angular_speed': 4.8,             # in rad/s
  'max_speed_cmd': 10.0,                # in rad/s
  'max_steer_angle_ackermann': 0.6981,  # 40 degree
  'max_steer_angle_parallel': 1.570,    # 180 degree
  'max_round_angle': 0.935671,
  'min_turn_radius': 0.47644,
}

## Define IK
ik_config = {
    "solver_type": "QP", 
    "ps": 0.001, 
    "λΣ": 12.5, 
    "ilimit": 100, 
    "ee_tol": 1e-4
}

# Define the Randomization Params
## Wheel Drive
k_drive = np.array([0.75, 1.25])
## Wheel Steer
k_steer = np.array([0.75, 1.25])
## Arm Joints
k_arm = np.array([0.75, 1.25])
## Hand Joints
k_hand = np.array([0.75, 1.25])
## Object Shape and Size
object_shape = ["box", "cylinder", "sphere", "ellipsoid", "capsule"]
object_mesh = ["bottle_mesh", "bread_mesh", "bowl_mesh", "cup_mesh", "winnercup_mesh"]
object_size = {
    "sphere": np.array([[0.035, 0.045]]),
    "capsule": np.array([[0.025, 0.035], [0.025, 0.04]]),
    "cylinder": np.array([[0.025, 0.035], [0.025, 0.035]]),
    "box": np.array([[0.025, 0.035], [0.025, 0.035], [0.025, 0.035]]),
    "ellipsoid": np.array([[0.03, 0.03], [0.045, 0.045], [0.045, 0.045]]),
}
object_mass = np.array([0.035, 0.075])
object_damping = np.array([5e-3, 2e-2])
object_static = np.array([0.5, 0.75])
## Observation Noise
k_obs_base = 0.01
k_obs_arm = 0.001
k_obs_object = 0.025
k_obs_hand = 0.01
## Actions Noise
k_act = 0.025
## Action Delay
act_delay = {
    'base': [1, 2, 3],
    'arm': [1, 2, 3],
    'hand': [1, 2, 3],
}

## Define PID params for wheel drive and steering. 
# driving
Kp_drive = 5
Ki_drive = 1e-3
Kd_drive = 1e-1
llim_drive = -200
ulim_drive = 200
# steering
Kp_steer = 50.0
Ki_steer = 2.5
Kd_steer = 7.5
llim_steer = -50
ulim_steer = 50

## Define PID params for the arm and hand. 
Kp_arm = np.array([300.0, 400.0, 400.0, 50.0, 200.0, 20.0])
Ki_arm = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3])
Kd_arm = np.array([40.0, 40.0, 40.0, 5.0, 10.0, 1])
llim_arm = np.array([-300.0, -300.0, -300.0, -50.0, -50.0, -20.0])
ulim_arm = np.array([300.0, 300.0, 300.0, 50.0, 50.0, 20.0])

Kp_hand = np.array([4e-1, 1e-2, 2e-1, 2e-1,
                      4e-1, 1e-2, 2e-1, 2e-1,
                      4e-1, 1e-2, 2e-1, 2e-1,
                      1e-1, 1e-1, 1e-1, 1e-2,])
Ki_hand = 1e-2
Kd_hand = np.array([3e-2, 1e-3, 2e-3, 1e-3,
                      3e-2, 1e-3, 2e-3, 1e-3,
                      3e-2, 1e-3, 2e-3, 1e-3,
                      1e-2, 1e-2, 2e-2, 1e-3,])
llim_hand = -5.0
ulim_hand = 5.0
hand_mask = np.array([1, 0, 1, 1,
                      1, 0, 1, 1,
                      1, 0, 1, 1,
                      0, 1, 1, 1])

class curriculum:
    # Define stage switching thresholds (steps)
    stage1_steps = 2e6  # First 2M steps
    stage2_steps = 10e6  # Extended curriculum period
    
    # Two-phase training configuration
    phase1_steps = 5e6  # Phase 1: Learn grasping (Actor + Critic)
    phase2_steps = 3e6  # Phase 2: Learn value discrimination (Critic only)
    
    # Stem collision penalty changes (reduced severity)
    collision_stem_start = -0.5
    collision_stem_end = -5.0
    
    # Orientation strictness changes (reduced severity)
    orient_power_start = 1.0
    orient_power_end = 2.0
    
    # Adaptive curriculum parameters
    success_rate_threshold = 0.3
    difficulty_decay_rate = 0.1

## AVP (Asymmetric Value Propagation) Configuration
## Toggle for ablation studies
class avp:
    # Master switch - set False for baseline experiments
    enabled = True
    
    # Reward weight range (decays with curriculum)
    lambda_weight_start = 0.8   # Early training: strong AVP guidance
    lambda_weight_end = 0.2     # Late training: rely more on original rewards
    
    # Distance gate - only compute AVP when EE is closer than this (meters)
    gate_distance = 1.5
    
    # Checkpoint path for Stage 2 Critic (relative to project root)
    checkpoint_path = "assets/checkpoints/avp/stage2_critic.pth"
    
    # Virtual observation configuration (pre-grasp posture, matches Stage 2)
    ready_pose = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0])
    
    # Network dimensions (must match Stage 2 training)
    state_dim = 35
    img_size = 84

## Depth Noise Configuration (for sim-to-real transfer)
## Simulates RealSense D435i artifacts in outdoor/agricultural scenes
class depth_noise:
    # Master switch
    enabled = True
    
    # Cutout noise (rectangular block dropout - simulates depth camera holes)
    cutout_enabled = True
    cutout_num = (1, 3)          # Range of number of cutout blocks
    cutout_size = (0.05, 0.15)   # Ratio of image dimensions
    
    # Salt-Pepper noise (extreme depth values - simulates specular reflections)
    salt_pepper_enabled = True
    salt_ratio = (0.01, 0.03)    # Pixels set to max depth
    pepper_ratio = (0.02, 0.05)  # Pixels set to 0
    
    # Pixel dropout (scattered missing pixels)
    dropout_rate = (0.05, 0.10)
    
    # Depth-dependent Gaussian noise (far objects noisier)
    depth_dependent = True
    base_std = 0.01              # Base noise for near objects
    scale_factor = 0.03          # Additional noise scaling with depth
    
    # Edge blur (depth discontinuity artifacts)
    edge_blur_enabled = True
    edge_noise_std = 0.1

## Lighting Domain Randomization Configuration
class lighting_dr:
    # Master switch
    enabled = True
    
    # Ambient light intensity range (simulates overcast vs clear sky)
    ambient_range = (0.1, 0.5)
    
    # Diffuse light intensity range (simulates sun intensity)
    diffuse_range = (0.3, 0.8)
    
    # Light direction noise (simulates different sun positions)
    dir_noise = 0.3

## Ground Color Domain Randomization
class ground_dr:
    # Master switch
    enabled = True
    
    # Ground types: soil_dry, soil_wet, grass, gravel
    ground_types = ['soil_dry', 'soil_wet', 'grass', 'gravel']