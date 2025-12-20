"""
JAX/MJX-compatible robot wrapper for DCMM mobile manipulation.

This module provides pure-functional interfaces for robot control
using MuJoCo MJX (XLA-accelerated physics simulation).

Migration Notes:
- Removed class-based stateful design  
- All functions are pure and JAX-compatible
- Uses mjx.Data and mjx.Model instead of mujoco.MjData/MjModel
- PID control integrated via pid_jax.py
- IK base control via ik_base_jax.py
- Designed for vmap across parallel environments

Key Differences from OOP version:
- No self references - model/data passed explicitly
- State containers for PID and other controllers
- Returns updated data instead of modifying in place
- Can be JIT-compiled and vmapped

TODO Notes:
- Full IK (inverse kinematics) is NOT supported in JAX JIT
  Use direct joint position control instead
- Viewer functionality is not available (use CPU MuJoCo for visualization)
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional, Dict, Any
from functools import partial

# Note: mujoco.mjx requires mujoco >= 3.0
try:
    import mujoco
    from mujoco import mjx
    MJX_AVAILABLE = True
except ImportError:
    MJX_AVAILABLE = False
    print("Warning: MJX not available. Install mujoco>=3.0 for GPU acceleration.")

from gym_dcmm.utils.pid_jax import (
    PIDState, PIDParams,
    create_pid_state, create_pid_params,
    pid_step
)
from gym_dcmm.utils.ik_pkg.ik_base_jax import (
    MobileBaseParams, get_default_params as get_default_base_params,
    ik_base_pure
)


# ============================================
# Body ID Mapping (for FK lookup)
# ============================================

class BodyIDMapping(NamedTuple):
    """Pre-computed body IDs for FK lookup.
    
    These IDs must be computed from the CPU model BEFORE JIT compilation.
    Use create_body_id_mapping() to create this from an MjModel.
    
    Attributes:
        ee_body_id: End-effector body ID (e.g., 'link6')
        base_body_id: Mobile base body ID (e.g., 'arm_base')
        object_body_id: Target object body ID (e.g., 'object')
        hand_body_ids: Hand finger body IDs for contact detection
    """
    ee_body_id: int
    base_body_id: int
    object_body_id: int
    hand_body_ids: Tuple[int, ...]


def create_body_id_mapping(mj_model, body_names: Dict[str, str] = None) -> BodyIDMapping:
    """Create body ID mapping from CPU MuJoCo model.
    
    This must be called BEFORE converting to MJX and JIT compilation.
    The IDs are used for FK lookup in reward computation.
    
    Args:
        mj_model: CPU MuJoCo model (mujoco.MjModel)
        body_names: Optional custom body name mapping:
            {
                'ee': 'link6',           # End-effector body name
                'base': 'arm_base',       # Base body name
                'object': 'object',       # Target object body name
                'hand': ['finger1', ...], # Hand body names (list)
            }
            
    Returns:
        BodyIDMapping with pre-computed IDs
        
    Example:
        mj_model = mujoco.MjModel.from_xml_path('robot.xml')
        body_ids = create_body_id_mapping(mj_model)
        
        # Now convert to MJX
        mx_model = mjx.put_model(mj_model)
        mx_data = mjx.put_data(mj_model, mujoco.MjData(mj_model))
        
        # Use in reward computation (body_ids.ee_body_id is static)
        ee_pos = get_ee_position(mx_data, body_ids.ee_body_id)
    """
    if body_names is None:
        body_names = {
            'ee': 'link6',
            'base': 'arm_base',
            'object': 'object',
            'hand': [],
        }
    
    def get_id(name):
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            print(f"Warning: Body '{name}' not found in model")
        return bid
    
    ee_id = get_id(body_names.get('ee', 'link6'))
    base_id = get_id(body_names.get('base', 'arm_base'))
    obj_id = get_id(body_names.get('object', 'object'))
    
    hand_names = body_names.get('hand', [])
    hand_ids = tuple(get_id(n) for n in hand_names)
    
    return BodyIDMapping(
        ee_body_id=ee_id,
        base_body_id=base_id,
        object_body_id=obj_id,
        hand_body_ids=hand_ids
    )


# ============================================
# State Containers
# ============================================

class RobotControlState(NamedTuple):
    """State container for robot controllers.
    
    Attributes:
        drive_pid_state: PID state for drive motors
        steer_pid_state: PID state for steer motors
        arm_pid_state: PID state for arm joints
        hand_pid_state: PID state for hand joints
        target_base_vel: Target base velocity [vx, vy, vyaw]
        target_arm_qpos: Target arm joint positions
        target_hand_qpos: Target hand joint positions
    """
    drive_pid_state: PIDState
    steer_pid_state: PIDState
    arm_pid_state: PIDState
    hand_pid_state: PIDState
    target_base_vel: jnp.ndarray  # Shape: (3,)
    target_arm_qpos: jnp.ndarray  # Shape: (6,)
    target_hand_qpos: jnp.ndarray  # Shape: (16,)


class RobotConfig(NamedTuple):
    """Configuration for robot control.
    
    Attributes:
        drive_pid: PID params for drive motors
        steer_pid: PID params for steer motors
        arm_pid: PID params for arm joints
        hand_pid: PID params for hand joints
        base_params: Mobile base kinematic params
        arm_joint_limits: Joint limits for arm [lower, upper]
        hand_joint_limits: Joint limits for hand [lower, upper]
        initial_arm_joints: Default arm joint positions
        initial_hand_joints: Default hand joint positions
    """
    drive_pid: PIDParams
    steer_pid: PIDParams
    arm_pid: PIDParams
    hand_pid: PIDParams
    base_params: MobileBaseParams
    arm_joint_limits: Tuple[jnp.ndarray, jnp.ndarray]
    hand_joint_limits: Tuple[jnp.ndarray, jnp.ndarray]
    initial_arm_joints: jnp.ndarray
    initial_hand_joints: jnp.ndarray


# ============================================
# Initialization Functions
# ============================================

def create_robot_config(
    # Drive PID
    kp_drive: float = 5.0,
    ki_drive: float = 1e-3,
    kd_drive: float = 0.1,
    # Steer PID
    kp_steer: float = 50.0,
    ki_steer: float = 2.5,
    kd_steer: float = 7.5,
    # Arm PID (can be array for different gains per joint)
    kp_arm: jnp.ndarray = None,
    ki_arm: jnp.ndarray = None,
    kd_arm: jnp.ndarray = None,
    # Hand PID
    kp_hand: jnp.ndarray = None,
    ki_hand: float = 1e-2,
    kd_hand: jnp.ndarray = None,
    # Limits (should come from model)
    arm_lower: jnp.ndarray = None,
    arm_upper: jnp.ndarray = None,
    hand_lower: jnp.ndarray = None,
    hand_upper: jnp.ndarray = None,
    # Initial positions
    initial_arm: jnp.ndarray = None,
    initial_hand: jnp.ndarray = None,
) -> RobotConfig:
    """Create robot configuration with default or custom parameters.
    
    All parameters have sensible defaults matching DcmmCfg.
    """
    # Default PID gains from DcmmCfg
    if kp_arm is None:
        kp_arm = jnp.array([300.0, 400.0, 400.0, 50.0, 200.0, 20.0])
    if ki_arm is None:
        ki_arm = jnp.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3])
    if kd_arm is None:
        kd_arm = jnp.array([40.0, 40.0, 40.0, 5.0, 10.0, 1.0])
    
    if kp_hand is None:
        kp_hand = jnp.array([
            4e-1, 1e-2, 2e-1, 2e-1,
            4e-1, 1e-2, 2e-1, 2e-1,
            4e-1, 1e-2, 2e-1, 2e-1,
            1e-1, 1e-1, 1e-1, 1e-2
        ])
    if kd_hand is None:
        kd_hand = jnp.array([
            3e-2, 1e-3, 2e-3, 1e-3,
            3e-2, 1e-3, 2e-3, 1e-3,
            3e-2, 1e-3, 2e-3, 1e-3,
            1e-2, 1e-2, 2e-2, 1e-3
        ])
    
    # Default limits (approximate, should be loaded from model)
    if arm_lower is None:
        arm_lower = jnp.array([-3.14, -2.0, -3.14, 1.8, 0.0, -2.35])
    if arm_upper is None:
        arm_upper = jnp.array([3.14, 2.0, 3.14, 4.14, 2.65, -0.785])
    if hand_lower is None:
        hand_lower = jnp.zeros(16)
    if hand_upper is None:
        hand_upper = jnp.ones(16) * 1.5
    
    # Default initial positions
    if initial_arm is None:
        initial_arm = jnp.array([0.0, 0.0, 0.0, 1.8, 0.0, -0.785])
    if initial_hand is None:
        initial_hand = jnp.zeros(16)
    
    return RobotConfig(
        drive_pid=create_pid_params(kp_drive, ki_drive, kd_drive, 4, llim=-200, ulim=200),
        steer_pid=create_pid_params(kp_steer, ki_steer, kd_steer, 4, llim=-50, ulim=50),
        arm_pid=PIDParams(
            Kp=kp_arm, Ki=ki_arm, Kd=kd_arm,
            llim=jnp.array([-300.0, -300.0, -300.0, -50.0, -50.0, -20.0]),
            ulim=jnp.array([300.0, 300.0, 300.0, 50.0, 50.0, 20.0]),
            offset=0.0
        ),
        hand_pid=PIDParams(
            Kp=kp_hand, Ki=jnp.full(16, ki_hand), Kd=kd_hand,
            llim=jnp.full(16, -5.0), ulim=jnp.full(16, 5.0),
            offset=0.0
        ),
        base_params=get_default_base_params(),
        arm_joint_limits=(arm_lower, arm_upper),
        hand_joint_limits=(hand_lower, hand_upper),
        initial_arm_joints=initial_arm,
        initial_hand_joints=initial_hand,
    )


def create_robot_control_state(config: RobotConfig) -> RobotControlState:
    """Create initial robot control state.
    
    Args:
        config: Robot configuration
        
    Returns:
        Initial control state with zeroed PID states
    """
    return RobotControlState(
        drive_pid_state=create_pid_state(4),
        steer_pid_state=create_pid_state(4),
        arm_pid_state=create_pid_state(6),
        hand_pid_state=create_pid_state(16),
        target_base_vel=jnp.zeros(3),
        target_arm_qpos=config.initial_arm_joints,
        target_hand_qpos=config.initial_hand_joints,
    )


# ============================================
# Mobile Base Control (Pure Functions)
# ============================================

@jax.jit
def compute_base_control(
    target_vel: jnp.ndarray,
    current_steer_pos: jnp.ndarray,
    current_drive_vel: jnp.ndarray,
    time: float,
    steer_state: PIDState,
    drive_state: PIDState,
    config: RobotConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, PIDState, PIDState]:
    """Compute mobile base control signals.
    
    Converts target velocity to wheel steering angles and drive velocities
    using inverse kinematics, then applies PID control.
    
    Args:
        target_vel: Target velocity [vx, vy, vyaw] (3,)
        current_steer_pos: Current steering positions [fl, fr, rl, rr] (4,)
        current_drive_vel: Current drive velocities [fl, fr, rl, rr] (4,)
        time: Current simulation time
        steer_state: Steering PID state
        drive_state: Drive PID state
        config: Robot configuration
        
    Returns:
        Tuple of:
            - steer_ctrl: Steering control signals (4,)
            - drive_ctrl: Drive control signals (4,)
            - new_steer_state: Updated steering PID state
            - new_drive_state: Updated drive PID state
    """
    # Compute target wheel states via IK
    target_steer, target_drive = ik_base_pure(
        target_vel[0], target_vel[1], target_vel[2],
        config.base_params
    )
    
    # Apply PID control for steering
    steer_ctrl, new_steer_state = pid_step(
        target_steer, current_steer_pos, time,
        steer_state, config.steer_pid
    )
    
    # Apply PID control for driving
    drive_ctrl, new_drive_state = pid_step(
        target_drive, current_drive_vel, time,
        drive_state, config.drive_pid
    )
    
    return steer_ctrl, drive_ctrl, new_steer_state, new_drive_state


# ============================================
# Arm Control (Pure Functions)
# ============================================

@jax.jit
def compute_arm_control(
    target_qpos: jnp.ndarray,
    current_qpos: jnp.ndarray,
    time: float,
    arm_state: PIDState,
    config: RobotConfig,
) -> Tuple[jnp.ndarray, PIDState]:
    """Compute arm joint control signals.
    
    Uses joint position control with PID.
    
    Args:
        target_qpos: Target joint positions (6,)
        current_qpos: Current joint positions (6,)
        time: Current simulation time
        arm_state: Arm PID state
        config: Robot configuration
        
    Returns:
        Tuple of (control_signal, new_pid_state)
    """
    ctrl, new_state = pid_step(
        target_qpos, current_qpos, time,
        arm_state, config.arm_pid
    )
    return ctrl, new_state


@jax.jit
def update_arm_target(
    current_target: jnp.ndarray,
    action_delta: jnp.ndarray,
    config: RobotConfig,
) -> jnp.ndarray:
    """Update arm target position with delta action.
    
    Applies delta and clips to joint limits.
    
    Args:
        current_target: Current target positions (6,)
        action_delta: Action delta to apply (6,)
        config: Robot configuration
        
    Returns:
        New target positions clipped to limits
    """
    new_target = current_target + action_delta
    return jnp.clip(
        new_target,
        config.arm_joint_limits[0],
        config.arm_joint_limits[1]
    )


# ============================================
# Hand Control (Pure Functions)
# ============================================

@jax.jit
def compute_hand_control(
    target_qpos: jnp.ndarray,
    current_qpos: jnp.ndarray,
    time: float,
    hand_state: PIDState,
    config: RobotConfig,
) -> Tuple[jnp.ndarray, PIDState]:
    """Compute hand joint control signals.
    
    Args:
        target_qpos: Target joint positions (16,)
        current_qpos: Current joint positions (16,)
        time: Current simulation time
        hand_state: Hand PID state
        config: Robot configuration
        
    Returns:
        Tuple of (control_signal, new_pid_state)
    """
    ctrl, new_state = pid_step(
        target_qpos, current_qpos, time,
        hand_state, config.hand_pid
    )
    return ctrl, new_state


@jax.jit  
def update_hand_target(
    current_target: jnp.ndarray,
    action_delta: jnp.ndarray,
    hand_mask: jnp.ndarray,
    config: RobotConfig,
) -> jnp.ndarray:
    """Update hand target position with masked delta action.
    
    Only updates joints where mask is 1.
    
    Args:
        current_target: Current target positions (16,)
        action_delta: Action delta for masked joints (12,)
        hand_mask: Which joints to update (16,) with 12 ones
        config: Robot configuration
        
    Returns:
        New target positions clipped to limits
    """
    # Expand action_delta to full 16 dims using mask
    masked_indices = jnp.where(hand_mask == 1)[0]
    delta_full = jnp.zeros(16)
    delta_full = delta_full.at[masked_indices].set(action_delta)
    
    new_target = current_target + delta_full
    return jnp.clip(
        new_target,
        config.hand_joint_limits[0],
        config.hand_joint_limits[1]
    )


# ============================================
# Full Robot Step Function
# ============================================

class RobotStepOutput(NamedTuple):
    """Output from robot control step.
    
    Attributes:
        ctrl: Full control vector to apply to MuJoCo
        control_state: Updated control state
    """
    ctrl: jnp.ndarray  # Shape: (30,) - steer(4) + drive(4) + arm(6) + hand(16)
    control_state: RobotControlState


# Note: robot_control_step is NOT JIT-compiled because:
# 1. It has conditional logic that depends on lock_base/lock_hand booleans
# 2. For maximum performance, JIT the entire environment step function instead
# 3. The individual control functions (compute_*_control) ARE JIT-compiled

def robot_control_step(
    # Current state from MJX data
    time: float,
    steer_qpos: jnp.ndarray,  # (4,) current steering positions
    drive_qvel: jnp.ndarray,  # (4,) current drive velocities
    arm_qpos: jnp.ndarray,    # (6,) current arm positions
    hand_qpos: jnp.ndarray,   # (16,) current hand positions
    # Actions
    base_action: jnp.ndarray,  # (2,) vx, vy
    arm_action: jnp.ndarray,   # (6,) delta joint positions
    hand_action: jnp.ndarray,  # (12,) delta joint positions for masked joints
    # State
    control_state: RobotControlState,
    # Config
    config: RobotConfig,
    # Options
    lock_base: bool = False,  # For Stage 2
    lock_hand: bool = False,  # For Stage 1
    hand_mask: jnp.ndarray = None,
) -> RobotStepOutput:
    """Compute full robot control for one step.
    
    This is the main entry point for robot control computation.
    
    Args:
        time: Current simulation time
        steer_qpos: Current steering joint positions (4,)
        drive_qvel: Current drive joint velocities (4,)
        arm_qpos: Current arm joint positions (6,)
        hand_qpos: Current hand joint positions (16,)
        base_action: Base velocity command [vx, vy] (2,)
        arm_action: Arm delta action (6,)
        hand_action: Hand delta action (12,)
        control_state: Previous control state
        config: Robot configuration
        lock_base: If True, zero base control (Stage 2)
        lock_hand: If True, keep hand at open position (Stage 1)
        hand_mask: Which hand joints to control (16,)
        
    Returns:
        RobotStepOutput with control signals and updated state
    """
    if hand_mask is None:
        hand_mask = jnp.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1])
    
    # Update targets
    new_target_base_vel = jax.lax.cond(
        lock_base,
        lambda _: jnp.zeros(3),
        lambda _: jnp.array([base_action[0], base_action[1], 0.0]),
        operand=None
    )
    
    new_target_arm = update_arm_target(
        control_state.target_arm_qpos,
        arm_action,
        config
    )
    
    new_target_hand = jax.lax.cond(
        lock_hand,
        lambda _: config.initial_hand_joints,
        lambda _: update_hand_target(
            control_state.target_hand_qpos,
            hand_action,
            hand_mask,
            config
        ),
        operand=None
    )
    
    # Compute base control
    steer_ctrl, drive_ctrl, new_steer_state, new_drive_state = jax.lax.cond(
        lock_base,
        lambda _: (
            jnp.zeros(4), jnp.zeros(4),
            control_state.steer_pid_state,
            control_state.drive_pid_state
        ),
        lambda _: compute_base_control(
            new_target_base_vel,
            steer_qpos, drive_qvel, time,
            control_state.steer_pid_state,
            control_state.drive_pid_state,
            config
        ),
        operand=None
    )
    
    # Compute arm control
    arm_ctrl, new_arm_state = compute_arm_control(
        new_target_arm, arm_qpos, time,
        control_state.arm_pid_state, config
    )
    
    # Compute hand control
    hand_ctrl, new_hand_state = compute_hand_control(
        new_target_hand, hand_qpos, time,
        control_state.hand_pid_state, config
    )
    
    # Concatenate control vector
    ctrl = jnp.concatenate([steer_ctrl, drive_ctrl, arm_ctrl, hand_ctrl])
    
    # Update state
    new_control_state = RobotControlState(
        drive_pid_state=new_drive_state,
        steer_pid_state=new_steer_state,
        arm_pid_state=new_arm_state,
        hand_pid_state=new_hand_state,
        target_base_vel=new_target_base_vel,
        target_arm_qpos=new_target_arm,
        target_hand_qpos=new_target_hand,
    )
    
    return RobotStepOutput(ctrl=ctrl, control_state=new_control_state)


# ============================================
# MJX Integration Functions
# ============================================

# ============================================
# Forward Kinematics (FK) for Reward Computation
# ============================================

# IMPORTANT: For reward computation, you need to know WHERE the end-effector is
# in Cartesian space. MJX provides this via mx_data.xpos (body positions) and
# mx_data.site_xpos (site positions after mj_forward/mj_step).
#
# Usage Pattern:
#   After mjx.step(), the forward kinematics are automatically computed.
#   Access end-effector position via:
#     ee_pos = mx_data.xpos[ee_body_id]  # Body position
#   or
#     ee_pos = mx_data.site_xpos[ee_site_id]  # Site position (more precise)
#
# The body/site IDs must be pre-computed from the CPU model before JIT:
#   ee_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
#   ee_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')

@jax.jit
def get_ee_position(mx_data, ee_body_id: int) -> jnp.ndarray:
    """Get end-effector position from MJX data.
    
    Forward kinematics are automatically computed by mjx.step().
    This function simply extracts the position for a given body.
    
    Args:
        mx_data: MJX data after physics step (FK already computed)
        ee_body_id: Body ID for end-effector (pre-computed from CPU model)
        
    Returns:
        End-effector position in world frame (3,)
        
    Note: Body IDs are integers that must be obtained from the CPU model:
        ee_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
    """
    return mx_data.xpos[ee_body_id]


@jax.jit
def get_ee_orientation(mx_data, ee_body_id: int) -> jnp.ndarray:
    """Get end-effector orientation (quaternion) from MJX data.
    
    Args:
        mx_data: MJX data after physics step
        ee_body_id: Body ID for end-effector
        
    Returns:
        End-effector quaternion [w, x, y, z] in world frame (4,)
    """
    return mx_data.xquat[ee_body_id]


@jax.jit
def get_ee_velocity(mx_data, ee_body_id: int) -> jnp.ndarray:
    """Get end-effector linear velocity from MJX data.
    
    Args:
        mx_data: MJX data after physics step
        ee_body_id: Body ID for end-effector
        
    Returns:
        End-effector linear velocity in world frame (3,)
        
    Note: cvel is [angular_vel(3), linear_vel(3)], we extract linear part
    """
    return mx_data.cvel[ee_body_id, 3:6]


@jax.jit
def get_object_position(mx_data, obj_body_id: int) -> jnp.ndarray:
    """Get object/target position from MJX data.
    
    Args:
        mx_data: MJX data after physics step
        obj_body_id: Body ID for object/target
        
    Returns:
        Object position in world frame (3,)
    """
    return mx_data.xpos[obj_body_id]


@jax.jit
def compute_ee_to_target_distance(
    mx_data,
    ee_body_id: int,
    target_body_id: int
) -> jnp.ndarray:
    """Compute distance from end-effector to target.
    
    This is the key metric for reward computation.
    
    Args:
        mx_data: MJX data after physics step
        ee_body_id: Body ID for end-effector
        target_body_id: Body ID for target object
        
    Returns:
        Euclidean distance (scalar)
    """
    ee_pos = mx_data.xpos[ee_body_id]
    target_pos = mx_data.xpos[target_body_id]
    return jnp.linalg.norm(ee_pos - target_pos)


# Batched versions for parallel environments
get_ee_position_batched = jax.vmap(get_ee_position, in_axes=(0, None))
get_ee_orientation_batched = jax.vmap(get_ee_orientation, in_axes=(0, None))
get_ee_velocity_batched = jax.vmap(get_ee_velocity, in_axes=(0, None))
get_object_position_batched = jax.vmap(get_object_position, in_axes=(0, None))
compute_ee_to_target_distance_batched = jax.vmap(
    compute_ee_to_target_distance, in_axes=(0, None, None)
)


if MJX_AVAILABLE:
    
    def load_model_to_mjx(xml_string: str) -> Tuple[mjx.Model, mjx.Data]:
        """Load MuJoCo model from XML string and convert to MJX.
        
        Args:
            xml_string: MuJoCo XML model string
            
        Returns:
            Tuple of (mjx_model, mjx_data)
        """
        # Load CPU model first
        mj_model = mujoco.MjModel.from_xml_string(xml_string)
        mj_data = mujoco.MjData(mj_model)
        
        # Convert to MJX
        mx_model = mjx.put_model(mj_model)
        mx_data = mjx.put_data(mj_model, mj_data)
        
        return mx_model, mx_data
    
    
    @partial(jax.jit, static_argnums=(0,))
    def mjx_step_with_ctrl(
        mx_model: mjx.Model,
        mx_data: mjx.Data,
        ctrl: jnp.ndarray
    ) -> mjx.Data:
        """Execute one MJX physics step with control.
        
        Args:
            mx_model: MJX model (static)
            mx_data: MJX data state
            ctrl: Control vector
            
        Returns:
            Updated MJX data
        """
        # Set control
        mx_data = mx_data.replace(ctrl=ctrl)
        # Step physics
        mx_data = mjx.step(mx_model, mx_data)
        return mx_data
    
    
    def get_robot_state_from_mjx(
        mx_data: mjx.Data
    ) -> Dict[str, jnp.ndarray]:
        """Extract robot state from MJX data.
        
        Assumes standard DCMM joint ordering:
        - qpos[0:9]: base joints (steer + drive positions)
        - qpos[9:15]: arm joints (NOTE: indices shifted from original 15:21)
        - qpos[15:31]: hand joints (NOTE: indices shifted from original 21:37)
        
        Args:
            mx_data: MJX data state
            
        Returns:
            Dictionary with robot state components
            
        IMPORTANT: Joint indices must be configured based on your MJCF model.
        The DCMM model uses the following joint ordering:
        - qpos indices 0-8: Base joints (steer: 0,1,2,3; free joint for body: 4-8)
        - qpos indices 9-14: Not used (depends on model)
        - qpos indices 15-20: Arm joints (6 DOF)
        - qpos indices 21-36: Hand joints (16 DOF)
        
        Use mujoco.mj_name2id() on the CPU model to get correct indices.
        Example:
            steer_fl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'steer_fl')
        """
        # Joint indices for DCMM model (verify against your specific model)
        # These indices are based on the original MujocoDcmm.py implementation
        STEER_QPOS_INDICES = jnp.array([0, 1, 2, 3])  # steer_fl, steer_fr, steer_rl, steer_rr
        DRIVE_QVEL_INDICES = jnp.array([4, 5, 6, 7])  # drive_fl, drive_fr, drive_rl, drive_rr
        ARM_QPOS_START = 15
        ARM_QPOS_END = 21
        HAND_QPOS_START = 21
        HAND_QPOS_END = 37
        
        return {
            'steer_qpos': mx_data.qpos[STEER_QPOS_INDICES],
            'drive_qvel': mx_data.qvel[DRIVE_QVEL_INDICES],
            'arm_qpos': mx_data.qpos[ARM_QPOS_START:ARM_QPOS_END],
            'hand_qpos': mx_data.qpos[HAND_QPOS_START:HAND_QPOS_END],
            'time': mx_data.time,
        }
    
    
    # Batched MJX step for vectorized environments
    mjx_step_batched = jax.vmap(
        lambda model, data, ctrl: mjx_step_with_ctrl(model, data, ctrl),
        in_axes=(None, 0, 0),  # model shared, data and ctrl batched
    )


# ============================================
# Utility Functions
# ============================================

def reset_control_state(config: RobotConfig) -> RobotControlState:
    """Reset control state to initial values.
    
    Use this at episode reset.
    """
    return create_robot_control_state(config)


@jax.jit
def apply_action_noise(
    ctrl: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
    noise_scale: float = 0.025
) -> jnp.ndarray:
    """Apply multiplicative noise to control signal.
    
    Simulates actuator uncertainty for domain randomization.
    
    Args:
        ctrl: Control signal
        rng_key: PRNG key
        noise_scale: Standard deviation of multiplicative noise
        
    Returns:
        Noisy control signal
    """
    noise = jax.random.normal(rng_key, ctrl.shape) * noise_scale + 1.0
    return ctrl * noise
