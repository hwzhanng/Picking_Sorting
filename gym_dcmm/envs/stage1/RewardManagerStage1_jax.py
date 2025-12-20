"""
JAX-compatible Reward Manager for Stage 1 (Tracking Task).

This module provides stateless, pure-functional reward computation
that can be JIT-compiled with JAX and vectorized with vmap for
massively parallel environment execution.

Migration Notes:
- Removed class-based stateful design
- All state must be passed explicitly as function arguments
- Uses jax.numpy instead of numpy
- AVP rewards computed via Flax critic (see ModelsStage2_flax.py)
- All functions can be JIT-compiled

Key Differences from OOP version:
- No self references - all data passed as arguments
- Returns tuple of (reward, new_state, stats) instead of modifying self
- State containers use NamedTuples for JAX compatibility
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Tuple, Optional, Any
from functools import partial

from gym_dcmm.utils.quat_utils_jax import quat_rotate_vector


# ============================================
# State and Configuration Containers
# ============================================

class RewardState(NamedTuple):
    """State container for reward computation.
    
    Attributes:
        prev_action: Previous action for action rate penalty
        contact_count: Counter for sustained contact
    """
    prev_action: jnp.ndarray  # Shape: (20,) - base(2) + arm(6) + hand(12)
    contact_count: int


class RewardConfig(NamedTuple):
    """Configuration for reward weights and parameters.
    
    All weights are configurable to match DcmmCfg.reward_weights.
    """
    # Reaching rewards
    w_arm_reaching: float = 2.0
    w_global_reaching: float = 0.5
    w_arm_motion: float = 0.5
    w_arm_action: float = 0.2
    
    # Base approach
    optimal_base_dist: float = 0.8
    
    # Orientation
    orientation_max_dist: float = 2.0
    
    # Touch rewards
    w_touch_base: float = 10.0
    w_touch_impact: float = -4.0
    
    # Penalties
    w_base_ctrl: float = 0.005
    w_arm_ctrl: float = 0.001
    w_action_rate: float = 0.02
    w_collision: float = -10.0
    
    # Plant collision (curriculum-adjusted)
    w_stem: float = -0.1  # Will be updated by curriculum
    w_leaf_base: float = -0.5


class CurriculumState(NamedTuple):
    """State for curriculum learning progression.
    
    Attributes:
        global_step: Current training step
        difficulty: Current difficulty [0, 1]
        w_stem: Current stem collision penalty
        orient_power: Current orientation strictness
    """
    global_step: int
    difficulty: float
    w_stem: float
    orient_power: float


class AVPConfig(NamedTuple):
    """Configuration for AVP (Asymmetric Value Propagation).
    
    Attributes:
        enabled: Whether AVP is active
        lambda_weight: Current AVP reward weight
        gate_distance: Max EE distance for AVP computation
        state_dim: Stage 2 state dimension
        img_size: Depth image size
    """
    enabled: bool = True
    lambda_weight: float = 0.8
    gate_distance: float = 1.5
    state_dim: int = 35
    img_size: int = 84


# ============================================
# Initialization Functions
# ============================================

def create_reward_state() -> RewardState:
    """Create initial reward computation state."""
    return RewardState(
        prev_action=jnp.zeros(20),
        contact_count=0
    )


def create_reward_config() -> RewardConfig:
    """Create default reward configuration."""
    return RewardConfig()


def create_curriculum_state(
    collision_stem_start: float = -0.1,
    orient_power_start: float = 1.0
) -> CurriculumState:
    """Create initial curriculum state."""
    return CurriculumState(
        global_step=0,
        difficulty=0.0,
        w_stem=collision_stem_start,
        orient_power=orient_power_start
    )


# ============================================
# Core Reward Components (Pure Functions)
# ============================================

@jax.jit
def compute_arm_reaching_reward(
    ee_pos_rel: jnp.ndarray,
    obj_pos_rel: jnp.ndarray,
    weight: float = 2.0
) -> Tuple[float, float]:
    """Compute arm reaching reward in robot's local frame.
    
    This isolates the arm's contribution to reaching.
    
    Args:
        ee_pos_rel: End-effector position in base frame (3,)
        obj_pos_rel: Object position in base frame (3,)
        weight: Reward weight
        
    Returns:
        Tuple of (reward, distance)
    """
    distance = jnp.linalg.norm(ee_pos_rel - obj_pos_rel)
    reward = weight * (1.0 - jnp.tanh(3.0 * distance))
    return reward, distance


@jax.jit
def compute_global_reaching_reward(
    ee_distance: float,
    weight: float = 0.5
) -> float:
    """Compute global EE-to-target reaching reward.
    
    Args:
        ee_distance: Distance from EE to target (scalar)
        weight: Reward weight
        
    Returns:
        Reward value
    """
    return weight * (1.0 - jnp.tanh(2.0 * ee_distance))


@jax.jit
def compute_base_approach_reward(
    base_distance: float,
    optimal_dist: float = 0.8
) -> float:
    """Compute base approach reward with optimal distance sweet spot.
    
    Args:
        base_distance: Distance from base to target (scalar)
        optimal_dist: Optimal distance to maintain
        
    Returns:
        Reward value (1.0 at optimal distance)
    """
    dist_error = jnp.abs(base_distance - optimal_dist)
    return jnp.exp(-5.0 * dist_error**2)


@jax.jit
def compute_arm_motion_reward(
    current_arm_joints: jnp.ndarray,
    initial_arm_joints: jnp.ndarray,
    weight: float = 0.5
) -> Tuple[float, float]:
    """Compute reward for arm joint deviation from initial pose.
    
    Encourages the agent to explore arm movements.
    
    Args:
        current_arm_joints: Current arm joint angles (6,)
        initial_arm_joints: Initial arm joint angles (6,)
        weight: Reward weight
        
    Returns:
        Tuple of (reward, joint_deviation)
    """
    deviation = jnp.linalg.norm(current_arm_joints - initial_arm_joints)
    reward = weight * jnp.tanh(3.0 * deviation)
    return reward, deviation


@jax.jit
def compute_arm_action_reward(
    arm_action: jnp.ndarray,
    weight: float = 0.2
) -> float:
    """Compute reward for arm action magnitude.
    
    Encourages the agent to use arm controls.
    
    Args:
        arm_action: Arm action vector (6,)
        weight: Reward weight
        
    Returns:
        Reward value
    """
    return weight * jnp.linalg.norm(arm_action)


@jax.jit
def compute_orientation_reward(
    ee_pos: jnp.ndarray,
    obj_pos: jnp.ndarray,
    ee_quat: jnp.ndarray,
    ee_distance: float,
    orient_power: float = 1.0,
    max_distance: float = 2.0
) -> float:
    """Compute reward for palm facing the target.
    
    Only computed when EE is within max_distance of target.
    
    Args:
        ee_pos: End-effector world position (3,)
        obj_pos: Object world position (3,)
        ee_quat: End-effector quaternion [w,x,y,z] (4,)
        ee_distance: Current EE-to-target distance
        orient_power: Power for strictness (curriculum)
        max_distance: Max distance to compute orientation reward
        
    Returns:
        Orientation alignment reward
    """
    def compute_orientation(_):
        # Direction from EE to object
        ee_to_obj = obj_pos - ee_pos
        ee_to_obj_norm = ee_to_obj / (jnp.linalg.norm(ee_to_obj) + 1e-6)
        
        # Palm forward direction (negative Z-axis of EE frame)
        palm_forward = quat_rotate_vector(ee_quat, jnp.array([0.0, 0.0, -1.0]))
        
        # Alignment: 1.0 = perfect, -1.0 = backwards
        alignment = jnp.dot(palm_forward, ee_to_obj_norm)
        
        # Apply power function for stricter alignment
        return jnp.maximum(0.0, alignment) ** orient_power * 2.0
    
    return jax.lax.cond(
        ee_distance < max_distance,
        compute_orientation,
        lambda _: 0.0,
        operand=None
    )


@jax.jit
def compute_touch_reward(
    step_touch: bool,
    ee_velocity: jnp.ndarray,
    base_reward: float = 10.0,
    impact_penalty_scale: float = -4.0
) -> Tuple[float, float]:
    """Compute reward for touching the target.
    
    Includes penalty for high-speed impacts.
    
    Args:
        step_touch: Whether contact occurred this step
        ee_velocity: End-effector linear velocity (3,)
        base_reward: Base touch reward
        impact_penalty_scale: Scale for impact speed penalty
        
    Returns:
        Tuple of (total_reward, impact_penalty)
    """
    def compute_touch(_):
        impact_speed = jnp.linalg.norm(ee_velocity)
        impact_penalty = impact_penalty_scale * impact_speed
        return base_reward + impact_penalty, impact_penalty
    
    return jax.lax.cond(
        step_touch,
        compute_touch,
        lambda _: (0.0, 0.0),
        operand=None
    )


@jax.jit
def compute_regularization_penalty(
    base_action: jnp.ndarray,
    arm_action: jnp.ndarray,
    base_scale: float = 0.005,
    arm_scale: float = 0.001
) -> float:
    """Compute control regularization penalty.
    
    Penalizes base control more than arm to encourage arm usage.
    
    Args:
        base_action: Base velocity command (2,)
        arm_action: Arm joint command (6,)
        base_scale: Base control penalty scale
        arm_scale: Arm control penalty scale
        
    Returns:
        Negative penalty value
    """
    base_penalty = -jnp.linalg.norm(base_action) * base_scale
    arm_penalty = -jnp.linalg.norm(arm_action) * arm_scale
    return base_penalty + arm_penalty


@jax.jit
def compute_collision_penalty(
    terminated: bool,
    step_touch: bool,
    penalty: float = -10.0
) -> float:
    """Compute catastrophic collision penalty.
    
    Args:
        terminated: Whether episode terminated
        step_touch: Whether touch occurred (success)
        penalty: Collision penalty value
        
    Returns:
        Penalty (negative) if terminated without touch, else 0
    """
    return jax.lax.cond(
        terminated & ~step_touch,
        lambda _: penalty,
        lambda _: 0.0,
        operand=None
    )


@jax.jit
def compute_plant_collision_penalty(
    plant_contact: bool,
    leaf_contact: bool,
    ee_velocity: jnp.ndarray,
    w_stem: float = -0.1,
    w_leaf_base: float = -0.5
) -> float:
    """Compute plant collision penalties.
    
    Stem collisions penalized more severely.
    Leaf penalties are velocity-dependent.
    
    Args:
        plant_contact: Whether stem contact occurred
        leaf_contact: Whether leaf contact occurred
        ee_velocity: EE velocity for leaf penalty scaling
        w_stem: Stem collision penalty (curriculum-adjusted)
        w_leaf_base: Base leaf collision penalty
        
    Returns:
        Total plant collision penalty
    """
    stem_penalty = jax.lax.cond(
        plant_contact,
        lambda _: w_stem,
        lambda _: 0.0,
        operand=None
    )
    
    def leaf_penalty_fn(_):
        ee_vel_norm = jnp.linalg.norm(ee_velocity)
        return w_leaf_base * (1.0 + ee_vel_norm)
    
    leaf_penalty = jax.lax.cond(
        leaf_contact,
        leaf_penalty_fn,
        lambda _: 0.0,
        operand=None
    )
    
    return stem_penalty + leaf_penalty


@jax.jit
def compute_action_rate_penalty(
    current_action: jnp.ndarray,
    prev_action: jnp.ndarray,
    scale: float = 0.02
) -> float:
    """Compute penalty for rapid action changes.
    
    Args:
        current_action: Current full action vector
        prev_action: Previous full action vector
        scale: Penalty scale
        
    Returns:
        Negative penalty for action changes
    """
    action_diff = current_action - prev_action
    return -jnp.linalg.norm(action_diff) * scale


# ============================================
# Curriculum Learning
# ============================================

@jax.jit
def update_curriculum(
    state: CurriculumState,
    collision_stem_start: float = -0.1,
    collision_stem_end: float = -2.0,
    orient_power_start: float = 1.0,
    orient_power_end: float = 1.5,
    max_steps: float = 2e6
) -> CurriculumState:
    """Update curriculum difficulty based on training progress.
    
    Args:
        state: Current curriculum state
        collision_stem_start: Initial stem penalty
        collision_stem_end: Final stem penalty
        orient_power_start: Initial orientation power
        orient_power_end: Final orientation power
        max_steps: Steps to reach full difficulty
        
    Returns:
        Updated curriculum state
    """
    difficulty = jnp.clip(state.global_step / max_steps, 0.0, 1.0)
    w_stem = collision_stem_start + (collision_stem_end - collision_stem_start) * difficulty
    orient_power = orient_power_start + (orient_power_end - orient_power_start) * difficulty
    
    return CurriculumState(
        global_step=state.global_step,
        difficulty=difficulty,
        w_stem=w_stem,
        orient_power=orient_power
    )


# ============================================
# AVP Reward Computation
# ============================================

@jax.jit
def compute_avp_reward_pure(
    critic_value: float,
    ee_distance: float,
    avp_config: AVPConfig
) -> float:
    """Compute AVP reward from critic value.
    
    This function takes a pre-computed critic value.
    The actual critic inference should be done separately.
    
    Args:
        critic_value: Value estimate from Stage 2 critic
        ee_distance: Current EE-to-target distance
        avp_config: AVP configuration
        
    Returns:
        AVP reward (0 if disabled or gated)
    """
    def compute_avp(_):
        return jnp.clip(avp_config.lambda_weight * critic_value, -5.0, 5.0)
    
    # Check if AVP should be computed
    should_compute = avp_config.enabled & (ee_distance <= avp_config.gate_distance)
    
    return jax.lax.cond(
        should_compute,
        compute_avp,
        lambda _: 0.0,
        operand=None
    )


def construct_virtual_obs_for_avp(
    real_obj_pos_rel: jnp.ndarray,
    depth_image: jnp.ndarray,
    avp_config: AVPConfig,
    ready_pose: jnp.ndarray = jnp.array([0.0, 0.0, 0.0, 1.8, 0.0, -0.785]),
    hand_open_angles: jnp.ndarray = None
) -> jnp.ndarray:
    """Construct virtual observation for AVP critic evaluation.
    
    Creates a hypothetical Stage 2 observation with:
    - Virtual "ready" arm pose
    - Real object position
    - Real depth image
    - Open hand posture
    
    Args:
        real_obj_pos_rel: Real object position in base frame (3,)
        depth_image: Real depth image (H, W) or (H*W,)
        avp_config: AVP configuration
        ready_pose: Virtual arm joint angles (6,)
        hand_open_angles: Open hand joint angles (12,)
        
    Returns:
        Flattened observation for Stage 2 critic (state_dim + img_pixels,)
    """
    if hand_open_angles is None:
        hand_open_angles = jnp.zeros(12)
    
    # Virtual state components (35 dim total for Stage 2)
    virtual_ee_pos = jnp.array([0.3, 0.0, 0.2])  # Default ready position
    virtual_ee_quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity
    virtual_ee_vel = jnp.zeros(3)
    virtual_arm_joints = ready_pose
    virtual_touch = jnp.zeros(4)
    
    # Concatenate state (35 dim)
    state_vec = jnp.concatenate([
        virtual_ee_pos,      # 3
        virtual_ee_quat,     # 4
        virtual_ee_vel,      # 3
        virtual_arm_joints,  # 6
        real_obj_pos_rel,    # 3
        hand_open_angles,    # 12
        virtual_touch        # 4
    ])  # Total: 35
    
    # Flatten depth if needed
    depth_flat = depth_image.flatten()
    
    # Combine state and depth
    return jnp.concatenate([state_vec, depth_flat])


# ============================================
# Main Reward Computation Function
# ============================================

class RewardOutput(NamedTuple):
    """Output from reward computation.
    
    Attributes:
        total_reward: Sum of all reward components
        state: Updated reward state
        components: Dictionary of individual reward components
    """
    total_reward: float
    state: RewardState
    components: Dict[str, float]


def compute_reward_stage1(
    # Observations
    ee_pos_rel: jnp.ndarray,
    obj_pos_rel: jnp.ndarray,
    ee_pos_world: jnp.ndarray,
    obj_pos_world: jnp.ndarray,
    ee_quat: jnp.ndarray,
    ee_velocity: jnp.ndarray,
    arm_joints: jnp.ndarray,
    initial_arm_joints: jnp.ndarray,
    ee_distance: float,
    base_distance: float,
    # Actions
    base_action: jnp.ndarray,
    arm_action: jnp.ndarray,
    hand_action: jnp.ndarray,
    # State flags
    step_touch: bool,
    terminated: bool,
    plant_contact: bool,
    leaf_contact: bool,
    # State
    reward_state: RewardState,
    # Configuration
    curriculum_state: CurriculumState,
    config: RewardConfig,
    # Optional AVP
    avp_reward: float = 0.0,
) -> RewardOutput:
    """Compute total reward for Stage 1 (Tracking).
    
    This is the main entry point for reward computation.
    All inputs are explicit - no hidden state.
    
    Args:
        ee_pos_rel: EE position in base frame (3,)
        obj_pos_rel: Object position in base frame (3,)
        ee_pos_world: EE world position (3,)
        obj_pos_world: Object world position (3,)
        ee_quat: EE quaternion [w,x,y,z] (4,)
        ee_velocity: EE linear velocity (3,)
        arm_joints: Current arm joints (6,)
        initial_arm_joints: Initial arm joints (6,)
        ee_distance: EE-to-target distance
        base_distance: Base-to-target distance
        base_action: Base velocity command (2,)
        arm_action: Arm action (6,)
        hand_action: Hand action (12,)
        step_touch: Whether touch occurred
        terminated: Whether episode ended
        plant_contact: Whether stem contact occurred
        leaf_contact: Whether leaf contact occurred
        reward_state: Previous reward state
        curriculum_state: Current curriculum state
        config: Reward configuration
        avp_reward: Pre-computed AVP reward
        
    Returns:
        RewardOutput with total reward, updated state, and components
    """
    # Individual reward components
    r_arm_reaching, arm_reach_dist = compute_arm_reaching_reward(
        ee_pos_rel, obj_pos_rel, config.w_arm_reaching
    )
    
    r_global_reaching = compute_global_reaching_reward(
        ee_distance, config.w_global_reaching
    )
    
    r_base_approach = compute_base_approach_reward(
        base_distance, config.optimal_base_dist
    )
    
    r_arm_motion, arm_joint_dev = compute_arm_motion_reward(
        arm_joints, initial_arm_joints, config.w_arm_motion
    )
    
    r_arm_action = compute_arm_action_reward(
        arm_action, config.w_arm_action
    )
    
    r_orientation = compute_orientation_reward(
        ee_pos_world, obj_pos_world, ee_quat, ee_distance,
        curriculum_state.orient_power, config.orientation_max_dist
    )
    
    r_touch, r_impact = compute_touch_reward(
        step_touch, ee_velocity, config.w_touch_base, config.w_touch_impact
    )
    
    r_regularization = compute_regularization_penalty(
        base_action, arm_action, config.w_base_ctrl, config.w_arm_ctrl
    )
    
    r_collision = compute_collision_penalty(
        terminated, step_touch, config.w_collision
    )
    
    r_plant_collision = compute_plant_collision_penalty(
        plant_contact, leaf_contact, ee_velocity,
        curriculum_state.w_stem, config.w_leaf_base
    )
    
    current_action = jnp.concatenate([base_action, arm_action, hand_action])
    r_action_rate = compute_action_rate_penalty(
        current_action, reward_state.prev_action, config.w_action_rate
    )
    
    # Sum all rewards
    total_reward = (
        r_arm_reaching + r_global_reaching + r_base_approach +
        r_arm_motion + r_arm_action + r_orientation +
        r_touch + r_regularization + r_collision +
        r_plant_collision + r_action_rate + avp_reward
    )
    
    # Update state
    new_state = RewardState(
        prev_action=current_action,
        contact_count=jax.lax.cond(
            step_touch,
            lambda c: c + 1,
            lambda c: 0,
            reward_state.contact_count
        )
    )
    
    # Collect components for logging
    components = {
        'arm_reaching': r_arm_reaching,
        'global_reaching': r_global_reaching,
        'base_approach': r_base_approach,
        'arm_motion': r_arm_motion,
        'arm_action': r_arm_action,
        'orientation': r_orientation,
        'touch': r_touch,
        'regularization': r_regularization,
        'collision': r_collision,
        'plant_collision': r_plant_collision,
        'action_rate': r_action_rate,
        'avp': avp_reward,
        'arm_reach_dist': arm_reach_dist,
        'arm_joint_dev': arm_joint_dev,
    }
    
    return RewardOutput(
        total_reward=total_reward,
        state=new_state,
        components=components
    )


# ============================================
# Batched Version for Vectorized Environments
# ============================================

# Use vmap to create batched reward computation
# This processes multiple environments in parallel
compute_reward_stage1_batched = jax.vmap(
    compute_reward_stage1,
    in_axes=(
        0, 0, 0, 0, 0, 0, 0, None,  # Observations (batch) / initial_arm_joints (shared)
        0, 0,  # Distances (batch)
        0, 0, 0,  # Actions (batch)
        0, 0, 0, 0,  # Flags (batch)
        0,  # Reward state (batch)
        None, None,  # Curriculum and config (shared)
        0,  # AVP reward (batch)
    ),
    out_axes=0
)
