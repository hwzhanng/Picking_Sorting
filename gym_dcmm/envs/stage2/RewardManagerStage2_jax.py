"""
JAX-compatible Reward Manager for Stage 2 (Catching Task).

This module provides stateless, pure-functional reward computation
for Stage 2 (grasp) training that can be JIT-compiled with JAX
and vectorized with vmap for massively parallel environment execution.

Migration Notes:
- Removed class-based stateful design
- All state must be passed explicitly as function arguments
- Uses jax.numpy instead of numpy
- Perturbation testing is stateless (state passed through)
- All functions can be JIT-compiled
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Tuple, Optional
from functools import partial

from gym_dcmm.utils.quat_utils_jax import quat_rotate_vector


# ============================================
# State and Configuration Containers
# ============================================

class RewardStateStage2(NamedTuple):
    """State container for Stage 2 reward computation.
    
    Attributes:
        prev_action: Previous action for action rate penalty
        perturbation_active: Whether perturbation test is ongoing
        initial_grasp_pos: Object position when test started
        perturbation_timer: Time elapsed in perturbation test
        perturbation_force_dir: Direction of perturbation force
        perturbation_force_mag: Magnitude of perturbation force
        success_held_steps: Steps of successful grasp hold
    """
    prev_action: jnp.ndarray  # Shape: (20,)
    perturbation_active: bool
    initial_grasp_pos: jnp.ndarray  # Shape: (3,)
    perturbation_timer: float
    perturbation_force_dir: jnp.ndarray  # Shape: (3,)
    perturbation_force_mag: float
    success_held_steps: int


class RewardConfigStage2(NamedTuple):
    """Configuration for Stage 2 reward weights.
    
    All weights are configurable to match DcmmCfg settings.
    """
    # Reaching rewards
    w_reaching: float = 2.0
    
    # Distance milestones
    milestone_distances: Tuple[float, ...] = (0.30, 0.15, 0.08, 0.05)
    milestone_rewards: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    
    # Orientation
    orientation_dist_thresh: float = 0.5
    w_orientation: float = 1.0
    
    # Grasp
    w_grasp_base: float = 1.0
    w_grasp_per_finger: float = 0.5
    w_grasp_force_bonus: float = 2.0
    force_range_min: float = 0.5
    force_range_max: float = 3.0
    
    # Perturbation
    perturbation_test_duration: float = 0.5
    perturbation_force_min: float = 0.5
    perturbation_force_max: float = 1.5
    perturbation_force_thresh: float = 1.0
    perturbation_slip_thresh: float = 0.01
    w_perturbation_success: float = 10.0
    w_perturbation_fail: float = -5.0
    
    # Penalties
    w_impact: float = -2.0
    impact_speed_thresh: float = 0.3
    w_regularization: float = 0.005
    w_collision: float = -2.0
    w_action_rate: float = 0.02
    
    # Plant collision (curriculum-adjusted)
    w_stem: float = -0.5
    w_leaf: float = -0.05
    
    # Success
    w_success: float = 20.0
    success_hold_steps: int = 50  # ~1 second at 50Hz


class CurriculumStateStage2(NamedTuple):
    """Curriculum state for Stage 2.
    
    Attributes:
        global_step: Current training step
        phase: Training phase (1 or 2)
        difficulty: Current difficulty [0, 1]
        w_stem: Current stem collision penalty
        perturbation_enabled: Whether perturbation test is active
    """
    global_step: int
    phase: int
    difficulty: float
    w_stem: float
    perturbation_enabled: bool


# ============================================
# Initialization Functions
# ============================================

def create_reward_state_stage2() -> RewardStateStage2:
    """Create initial Stage 2 reward state."""
    return RewardStateStage2(
        prev_action=jnp.zeros(20),
        perturbation_active=False,
        initial_grasp_pos=jnp.zeros(3),
        perturbation_timer=0.0,
        perturbation_force_dir=jnp.array([1.0, 0.0, 0.0]),
        perturbation_force_mag=0.0,
        success_held_steps=0
    )


def create_reward_config_stage2() -> RewardConfigStage2:
    """Create default Stage 2 reward configuration."""
    return RewardConfigStage2()


def create_curriculum_state_stage2(
    collision_stem_start: float = -0.5
) -> CurriculumStateStage2:
    """Create initial Stage 2 curriculum state."""
    return CurriculumStateStage2(
        global_step=0,
        phase=1,
        difficulty=0.0,
        w_stem=collision_stem_start,
        perturbation_enabled=False
    )


# ============================================
# Core Reward Components (Pure Functions)
# ============================================

@jax.jit
def compute_reaching_reward_stage2(
    ee_distance: float,
    weight: float = 2.0
) -> float:
    """Compute EE reaching reward with exponential decay.
    
    Args:
        ee_distance: Distance from EE to target
        weight: Reward weight
        
    Returns:
        Reaching reward
    """
    return weight * jnp.exp(-2.0 * ee_distance)


@jax.jit
def compute_distance_milestone_reward(
    ee_distance: float,
    distances: Tuple[float, ...] = (0.30, 0.15, 0.08, 0.05),
    rewards: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
) -> float:
    """Compute cumulative distance milestone bonuses.
    
    Args:
        ee_distance: Current EE-to-target distance
        distances: Distance thresholds (descending)
        rewards: Rewards for each threshold
        
    Returns:
        Cumulative milestone reward
    """
    total = 0.0
    for dist, rew in zip(distances, rewards):
        total += jax.lax.cond(
            ee_distance < dist,
            lambda _: rew,
            lambda _: 0.0,
            operand=None
        )
    return total


@jax.jit
def compute_orientation_reward_stage2(
    ee_pos: jnp.ndarray,
    obj_pos: jnp.ndarray,
    ee_quat: jnp.ndarray,
    ee_distance: float,
    dist_thresh: float = 0.5,
    weight: float = 1.0
) -> float:
    """Compute orientation reward for Stage 2.
    
    Only computed when close to target.
    Uses linear alignment (gentler than Stage 1).
    
    Args:
        ee_pos: End-effector world position (3,)
        obj_pos: Object world position (3,)
        ee_quat: EE quaternion [w,x,y,z] (4,)
        ee_distance: Current distance
        dist_thresh: Distance threshold for orientation reward
        weight: Reward weight
        
    Returns:
        Orientation reward
    """
    def compute_orient(_):
        ee_to_obj = obj_pos - ee_pos
        ee_to_obj_norm = ee_to_obj / (jnp.linalg.norm(ee_to_obj) + 1e-6)
        palm_forward = quat_rotate_vector(ee_quat, jnp.array([0.0, 0.0, -1.0]))
        alignment = jnp.dot(palm_forward, ee_to_obj_norm)
        return jnp.maximum(0.0, alignment) * weight
    
    return jax.lax.cond(
        ee_distance < dist_thresh,
        compute_orient,
        lambda _: 0.0,
        operand=None
    )


@jax.jit
def compute_grasp_reward(
    touch_sensors: jnp.ndarray,
    config: RewardConfigStage2
) -> Tuple[float, int, float]:
    """Compute grasp reward based on touch feedback.
    
    Args:
        touch_sensors: Touch sensor readings (4,) for fingertips
        config: Reward configuration
        
    Returns:
        Tuple of (reward, fingers_touching, total_force)
    """
    total_force = jnp.sum(touch_sensors)
    fingers_touching = jnp.sum(touch_sensors > 0.05)
    
    def has_contact(_):
        # Base reward for any contact
        base = config.w_grasp_base
        # Bonus per finger
        finger_bonus = config.w_grasp_per_finger * fingers_touching
        # Bonus for good force range
        force_bonus = jax.lax.cond(
            (total_force >= config.force_range_min) & (total_force <= config.force_range_max),
            lambda _: config.w_grasp_force_bonus,
            lambda _: jax.lax.cond(
                total_force > config.force_range_max,
                lambda _: -0.5 * jnp.minimum(total_force - config.force_range_max, 2.0),
                lambda _: 0.0,
                operand=None
            ),
            operand=None
        )
        return base + finger_bonus + force_bonus
    
    reward = jax.lax.cond(
        total_force > 0.01,
        has_contact,
        lambda _: 0.0,
        operand=None
    )
    
    return reward, fingers_touching, total_force


@jax.jit
def compute_perturbation_reward(
    state: RewardStateStage2,
    obj_pos: jnp.ndarray,
    total_contact_force: float,
    dt: float,
    rng_key: jax.random.PRNGKey,
    config: RewardConfigStage2,
    curriculum: CurriculumStateStage2
) -> Tuple[float, RewardStateStage2, jnp.ndarray]:
    """Evaluate grasp stability under perturbation.
    
    State machine:
    - Idle: Wait for sufficient contact force
    - Testing: Apply force and measure slippage
    - Evaluate: Return reward based on slippage
    
    Args:
        state: Current perturbation state
        obj_pos: Current object position (3,)
        total_contact_force: Sum of touch sensors
        dt: Time step
        rng_key: PRNG key for force randomization
        config: Reward configuration
        curriculum: Curriculum state (for enabling)
        
    Returns:
        Tuple of (reward, new_state, force_to_apply)
    """
    # Return early if perturbation not enabled
    def disabled_path(_):
        return 0.0, state, jnp.zeros(3)
    
    def enabled_path(_):
        def idle_state(_):
            # Check if should start perturbation test
            def start_test(_):
                # Generate random force direction
                k1, k2 = jax.random.split(rng_key)
                theta = jax.random.uniform(k1, minval=0, maxval=jnp.pi)
                phi = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
                direction = jnp.array([
                    jnp.sin(theta) * jnp.cos(phi),
                    jnp.sin(theta) * jnp.sin(phi),
                    jnp.cos(theta)
                ])
                magnitude = jax.random.uniform(
                    rng_key,
                    minval=config.perturbation_force_min,
                    maxval=config.perturbation_force_max
                )
                new_state = RewardStateStage2(
                    prev_action=state.prev_action,
                    perturbation_active=True,
                    initial_grasp_pos=obj_pos,
                    perturbation_timer=0.0,
                    perturbation_force_dir=direction,
                    perturbation_force_mag=magnitude,
                    success_held_steps=state.success_held_steps
                )
                force = direction * magnitude
                return 0.0, new_state, force
            
            def stay_idle(_):
                return 0.0, state, jnp.zeros(3)
            
            return jax.lax.cond(
                total_contact_force >= config.perturbation_force_thresh,
                start_test,
                stay_idle,
                operand=None
            )
        
        def testing_state(_):
            new_timer = state.perturbation_timer + dt
            
            def continue_test(_):
                force = state.perturbation_force_dir * state.perturbation_force_mag
                new_state = RewardStateStage2(
                    prev_action=state.prev_action,
                    perturbation_active=True,
                    initial_grasp_pos=state.initial_grasp_pos,
                    perturbation_timer=new_timer,
                    perturbation_force_dir=state.perturbation_force_dir,
                    perturbation_force_mag=state.perturbation_force_mag,
                    success_held_steps=state.success_held_steps
                )
                return 0.0, new_state, force
            
            def evaluate_test(_):
                # Compute slippage
                slippage = jnp.linalg.norm(obj_pos - state.initial_grasp_pos)
                reward = jax.lax.cond(
                    slippage < config.perturbation_slip_thresh,
                    lambda _: config.w_perturbation_success,
                    lambda _: config.w_perturbation_fail,
                    operand=None
                )
                # Reset state
                new_state = RewardStateStage2(
                    prev_action=state.prev_action,
                    perturbation_active=False,
                    initial_grasp_pos=jnp.zeros(3),
                    perturbation_timer=0.0,
                    perturbation_force_dir=jnp.array([1.0, 0.0, 0.0]),
                    perturbation_force_mag=0.0,
                    success_held_steps=state.success_held_steps
                )
                return reward, new_state, jnp.zeros(3)
            
            return jax.lax.cond(
                new_timer < config.perturbation_test_duration,
                continue_test,
                evaluate_test,
                operand=None
            )
        
        return jax.lax.cond(
            state.perturbation_active,
            testing_state,
            idle_state,
            operand=None
        )
    
    return jax.lax.cond(
        curriculum.perturbation_enabled,
        enabled_path,
        disabled_path,
        operand=None
    )


@jax.jit
def compute_impact_penalty_stage2(
    touch_sensors: jnp.ndarray,
    step_touch: bool,
    ee_velocity: jnp.ndarray,
    config: RewardConfigStage2
) -> float:
    """Compute impact velocity penalty.
    
    Args:
        touch_sensors: Touch sensor readings
        step_touch: Whether contact occurred
        ee_velocity: EE linear velocity (3,)
        config: Reward configuration
        
    Returns:
        Impact penalty (negative or zero)
    """
    total_force = jnp.sum(touch_sensors)
    has_contact = (total_force > 0.01) | step_touch
    impact_speed = jnp.linalg.norm(ee_velocity)
    
    def compute_penalty(_):
        excess_speed = impact_speed - config.impact_speed_thresh
        return jax.lax.cond(
            excess_speed > 0,
            lambda _: config.w_impact * jnp.minimum(excess_speed, 1.5),
            lambda _: 0.0,
            operand=None
        )
    
    return jax.lax.cond(
        has_contact,
        compute_penalty,
        lambda _: 0.0,
        operand=None
    )


@jax.jit
def compute_regularization_penalty_stage2(
    arm_action: jnp.ndarray,
    hand_action: jnp.ndarray,
    weight: float = 0.005
) -> float:
    """Compute control regularization penalty.
    
    Args:
        arm_action: Arm action (6,)
        hand_action: Hand action (12,)
        weight: Penalty scale
        
    Returns:
        Negative penalty
    """
    ctrl_norm = jnp.linalg.norm(jnp.concatenate([arm_action, hand_action]))
    return -ctrl_norm * weight


@jax.jit
def compute_plant_collision_penalty_stage2(
    plant_contact: bool,
    leaf_contact: bool,
    w_stem: float = -0.5,
    w_leaf: float = -0.05
) -> float:
    """Compute plant collision penalty for Stage 2.
    
    Args:
        plant_contact: Whether stem contact occurred
        leaf_contact: Whether leaf contact occurred
        w_stem: Stem penalty (curriculum-adjusted)
        w_leaf: Leaf penalty (small)
        
    Returns:
        Total plant collision penalty
    """
    stem_pen = jax.lax.cond(plant_contact, lambda _: w_stem, lambda _: 0.0, operand=None)
    leaf_pen = jax.lax.cond(leaf_contact, lambda _: w_leaf, lambda _: 0.0, operand=None)
    return stem_pen + leaf_pen


@jax.jit
def compute_success_reward(
    state: RewardStateStage2,
    touch_sensors: jnp.ndarray,
    config: RewardConfigStage2
) -> Tuple[float, bool, RewardStateStage2]:
    """Check for successful stable grasp.
    
    Success requires sustained multi-finger contact.
    
    Args:
        state: Current reward state
        touch_sensors: Touch sensor readings (4,)
        config: Reward configuration
        
    Returns:
        Tuple of (success_reward, is_success, new_state)
    """
    fingers_touching = jnp.sum(touch_sensors > 0.1)
    good_grasp = fingers_touching >= 3
    
    def update_held(_):
        new_held = state.success_held_steps + 1
        is_success = new_held >= config.success_hold_steps
        reward = jax.lax.cond(
            is_success,
            lambda _: config.w_success,
            lambda _: 0.0,
            operand=None
        )
        new_state = RewardStateStage2(
            prev_action=state.prev_action,
            perturbation_active=state.perturbation_active,
            initial_grasp_pos=state.initial_grasp_pos,
            perturbation_timer=state.perturbation_timer,
            perturbation_force_dir=state.perturbation_force_dir,
            perturbation_force_mag=state.perturbation_force_mag,
            success_held_steps=new_held
        )
        return reward, is_success, new_state
    
    def reset_held(_):
        new_state = RewardStateStage2(
            prev_action=state.prev_action,
            perturbation_active=state.perturbation_active,
            initial_grasp_pos=state.initial_grasp_pos,
            perturbation_timer=state.perturbation_timer,
            perturbation_force_dir=state.perturbation_force_dir,
            perturbation_force_mag=state.perturbation_force_mag,
            success_held_steps=0
        )
        return 0.0, False, new_state
    
    return jax.lax.cond(
        good_grasp,
        update_held,
        reset_held,
        operand=None
    )


# ============================================
# Curriculum Learning
# ============================================

@jax.jit
def update_curriculum_stage2(
    state: CurriculumStateStage2,
    phase1_steps: float = 15e6,
    phase2_steps: float = 10e6,
    collision_stem_start: float = -0.5,
    collision_stem_end: float = -5.0,
    perturbation_enable_ratio: float = 0.33
) -> CurriculumStateStage2:
    """Update Stage 2 curriculum based on training progress.
    
    Args:
        state: Current curriculum state
        phase1_steps: Steps in Phase 1 (Actor + Critic)
        phase2_steps: Steps in Phase 2 (Critic only)
        collision_stem_start: Initial stem penalty
        collision_stem_end: Final stem penalty
        perturbation_enable_ratio: Ratio of phase1 to enable perturbation
        
    Returns:
        Updated curriculum state
    """
    total_steps = phase1_steps + phase2_steps
    
    # Determine phase
    phase = jax.lax.cond(
        state.global_step < phase1_steps,
        lambda _: 1,
        lambda _: 2,
        operand=None
    )
    
    # Compute difficulty (0 to 1)
    difficulty = jnp.clip(state.global_step / total_steps, 0.0, 1.0)
    
    # Stem penalty
    w_stem = collision_stem_start + (collision_stem_end - collision_stem_start) * difficulty
    
    # Perturbation enabling
    perturbation_enabled = state.global_step > (phase1_steps * perturbation_enable_ratio)
    
    return CurriculumStateStage2(
        global_step=state.global_step,
        phase=phase,
        difficulty=difficulty,
        w_stem=w_stem,
        perturbation_enabled=perturbation_enabled
    )


# ============================================
# Main Reward Computation Function
# ============================================

class RewardOutputStage2(NamedTuple):
    """Output from Stage 2 reward computation.
    
    Attributes:
        total_reward: Sum of all reward components
        state: Updated reward state
        perturbation_force: Force to apply to object
        is_success: Whether task completed successfully
        components: Dictionary of individual reward components
    """
    total_reward: float
    state: RewardStateStage2
    perturbation_force: jnp.ndarray
    is_success: bool
    components: Dict[str, float]


def compute_reward_stage2(
    # Observations
    ee_pos: jnp.ndarray,
    obj_pos: jnp.ndarray,
    ee_quat: jnp.ndarray,
    ee_velocity: jnp.ndarray,
    touch_sensors: jnp.ndarray,
    ee_distance: float,
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
    reward_state: RewardStateStage2,
    # Configuration
    curriculum_state: CurriculumStateStage2,
    config: RewardConfigStage2,
    # Physics
    dt: float,
    rng_key: jax.random.PRNGKey,
) -> RewardOutputStage2:
    """Compute total reward for Stage 2 (Catching).
    
    Args:
        ee_pos: EE world position (3,)
        obj_pos: Object world position (3,)
        ee_quat: EE quaternion [w,x,y,z] (4,)
        ee_velocity: EE linear velocity (3,)
        touch_sensors: Touch sensor readings (4,)
        ee_distance: EE-to-target distance
        base_action: Base action (2,) - should be locked to zero in Stage 2
        arm_action: Arm action (6,)
        hand_action: Hand action (12,)
        step_touch: Whether touch occurred
        terminated: Whether episode ended
        plant_contact: Whether stem contact occurred
        leaf_contact: Whether leaf contact occurred
        reward_state: Previous reward state
        curriculum_state: Current curriculum state
        config: Reward configuration
        dt: Physics timestep
        rng_key: PRNG key for perturbation
        
    Returns:
        RewardOutputStage2 with total reward, updated state, and components
    """
    # 1. Reaching reward
    r_reaching = compute_reaching_reward_stage2(ee_distance, config.w_reaching)
    
    # 2. Distance milestones
    r_milestones = compute_distance_milestone_reward(
        ee_distance, config.milestone_distances, config.milestone_rewards
    )
    
    # 3. Orientation reward
    r_orientation = compute_orientation_reward_stage2(
        ee_pos, obj_pos, ee_quat, ee_distance,
        config.orientation_dist_thresh, config.w_orientation
    )
    
    # 4. Grasp reward
    r_grasp, fingers_touching, total_force = compute_grasp_reward(touch_sensors, config)
    
    # 5. Perturbation test
    r_perturbation, perturb_state, perturb_force = compute_perturbation_reward(
        reward_state, obj_pos, total_force, dt, rng_key, config, curriculum_state
    )
    
    # 6. Impact penalty
    r_impact = compute_impact_penalty_stage2(
        touch_sensors, step_touch, ee_velocity, config
    )
    
    # 7. Regularization
    r_regularization = compute_regularization_penalty_stage2(
        arm_action, hand_action, config.w_regularization
    )
    
    # 8. Collision penalty
    r_collision = jax.lax.cond(
        terminated & ~step_touch,
        lambda _: config.w_collision,
        lambda _: 0.0,
        operand=None
    )
    
    # 9. Plant collision
    r_plant = compute_plant_collision_penalty_stage2(
        plant_contact, leaf_contact, curriculum_state.w_stem, config.w_leaf
    )
    
    # 10. Action rate penalty
    current_action = jnp.concatenate([base_action, arm_action, hand_action])
    action_diff = current_action - reward_state.prev_action
    r_action_rate = -jnp.linalg.norm(action_diff) * config.w_action_rate
    
    # 11. Success check
    r_success, is_success, success_state = compute_success_reward(
        perturb_state, touch_sensors, config
    )
    
    # Total reward
    total_reward = (
        r_reaching + r_milestones + r_orientation + r_grasp +
        r_perturbation + r_impact + r_regularization + r_collision +
        r_plant + r_action_rate + r_success
    )
    
    # Update state
    new_state = RewardStateStage2(
        prev_action=current_action,
        perturbation_active=success_state.perturbation_active,
        initial_grasp_pos=success_state.initial_grasp_pos,
        perturbation_timer=success_state.perturbation_timer,
        perturbation_force_dir=success_state.perturbation_force_dir,
        perturbation_force_mag=success_state.perturbation_force_mag,
        success_held_steps=success_state.success_held_steps
    )
    
    # Collect components
    components = {
        'reaching': r_reaching,
        'milestones': r_milestones,
        'orientation': r_orientation,
        'grasp': r_grasp,
        'perturbation': r_perturbation,
        'impact': r_impact,
        'regularization': r_regularization,
        'collision': r_collision,
        'plant': r_plant,
        'action_rate': r_action_rate,
        'success': r_success,
        'fingers_touching': fingers_touching,
        'total_force': total_force,
    }
    
    return RewardOutputStage2(
        total_reward=total_reward,
        state=new_state,
        perturbation_force=perturb_force,
        is_success=is_success,
        components=components
    )


# ============================================
# Batched Version for Vectorized Environments
# ============================================

# Note: For batched version, we need separate PRNG keys per environment
compute_reward_stage2_batched = jax.vmap(
    compute_reward_stage2,
    in_axes=(
        0, 0, 0, 0, 0, 0,  # Observations (batch)
        0, 0, 0,  # Actions (batch)
        0, 0, 0, 0,  # Flags (batch)
        0,  # Reward state (batch)
        None, None,  # Curriculum and config (shared)
        None, 0,  # dt (shared), rng_keys (batch)
    ),
    out_axes=0
)
