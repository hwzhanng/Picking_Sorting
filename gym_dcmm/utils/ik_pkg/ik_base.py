"""
JAX-compatible inverse kinematics functions for mobile base control.

This module provides stateless, pure-functional implementations of the
4-wheel drive mobile base inverse kinematics that can be JIT-compiled
with JAX and vectorized with vmap.

Migration Notes:
- Replaced numpy with jax.numpy for GPU/TPU compatibility
- Replaced math module with jax equivalents
- All functions are pure and can be JIT-compiled
- Uses jax.lax.cond for JAX-compatible conditionals
"""

import jax
import jax.numpy as jnp
from typing import Tuple


# Mobile base configuration (Ranger Mini V2)
# These should match DcmmCfg.RangerMiniV2Params
@jax.tree_util.register_pytree_node_class
class MobileBaseParams:
    """Configuration parameters for mobile base IK.
    
    Attributes:
        wheel_radius: Wheel radius in meters
        steer_track: Left-right wheel distance in meters
        wheel_base: Front-rear wheel distance in meters
        max_linear_speed: Maximum linear speed in m/s
        max_angular_speed: Maximum angular speed in rad/s
        max_steer_angle_parallel: Max steer angle for parallel mode
        max_steer_angle_ackermann: Max steer angle for Ackermann mode
        min_turn_radius: Minimum turn radius in meters
    """
    def __init__(
        self,
        wheel_radius: float = 0.1,
        steer_track: float = 0.364,
        wheel_base: float = 0.494,
        max_linear_speed: float = 1.5,
        max_angular_speed: float = 4.8,
        max_steer_angle_parallel: float = 1.570,
        max_steer_angle_ackermann: float = 0.6981,
        min_turn_radius: float = 0.47644,
    ):
        self.wheel_radius = wheel_radius
        self.steer_track = steer_track
        self.wheel_base = wheel_base
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.max_steer_angle_parallel = max_steer_angle_parallel
        self.max_steer_angle_ackermann = max_steer_angle_ackermann
        self.min_turn_radius = min_turn_radius
    
    def tree_flatten(self):
        """For JAX pytree compatibility."""
        children = (
            self.wheel_radius,
            self.steer_track,
            self.wheel_base,
            self.max_linear_speed,
            self.max_angular_speed,
            self.max_steer_angle_parallel,
            self.max_steer_angle_ackermann,
            self.min_turn_radius,
        )
        aux_data = None
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """For JAX pytree compatibility."""
        return cls(*children)


def get_default_params() -> MobileBaseParams:
    """Get default Ranger Mini V2 parameters."""
    return MobileBaseParams(
        wheel_radius=0.1,
        steer_track=0.364,
        wheel_base=0.494,
        max_linear_speed=1.5,
        max_angular_speed=4.8,
        max_steer_angle_parallel=1.570,
        max_steer_angle_ackermann=0.6981,
        min_turn_radius=0.47644,
    )


@jax.jit
def damper(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to [min_val, max_val] range.
    
    Args:
        value: Input value to clamp
        min_val: Lower bound
        max_val: Upper bound
    
    Returns:
        Clamped value
    """
    return jnp.clip(value, min_val, max_val)


@jax.jit
def ik_base_pure(
    v_lin_x: float,
    v_lin_y: float,
    v_yaw: float,
    params: MobileBaseParams
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate inverse kinematics for 4-wheel drive mobile base.
    
    This is a pure, JIT-compiled function with explicit parameters.
    
    Args:
        v_lin_x: Linear x velocity of mobile base (base_link frame)
        v_lin_y: Linear y velocity of mobile base (base_link frame)
        v_yaw: Angular velocity of mobile base (base_link frame)
        params: Mobile base configuration parameters
    
    Returns:
        Tuple of:
            - steer_ang: Steering angles [fl, fr, rl, rr] (shape: [4])
            - drive_vel: Drive velocities [fl, fr, rl, rr] (shape: [4])
    """
    # Apply dead zone for small velocities
    v_lin_x = jnp.where(jnp.abs(v_lin_x) < 0.01, 0.0, v_lin_x)
    v_lin_y = jnp.where(jnp.abs(v_lin_y) < 0.01, 0.0, v_lin_y)
    v_yaw = jnp.where(jnp.abs(v_yaw) < 0.01, 0.0, v_yaw)
    
    # Check if all velocities are zero
    all_zero = (jnp.abs(v_lin_x) < 0.01) & (jnp.abs(v_lin_y) < 0.01) & (jnp.abs(v_yaw) < 0.01)
    
    def zero_case(_):
        return jnp.zeros(4), jnp.zeros(4)
    
    def nonzero_case(_):
        sign = jnp.sign(v_lin_y)
        # Handle zero v_lin_y
        sign = jnp.where(jnp.abs(v_lin_y) < 1e-6, 1.0, sign)
        
        # Check if in parallel motion mode (v_lin_x != 0)
        def parallel_mode(_):
            # Parallel motion: all wheels steer same direction
            steer_cmd = -jnp.arctan2(v_lin_x, v_lin_y + 1e-5)
            steer_cmd = damper(
                steer_cmd,
                -params.max_steer_angle_parallel,
                params.max_steer_angle_parallel
            )
            vel_cmd = sign * jnp.sqrt(v_lin_y**2 + v_lin_x**2) / params.wheel_radius
            return jnp.array([steer_cmd, steer_cmd, steer_cmd, steer_cmd]), \
                   jnp.array([vel_cmd, vel_cmd, vel_cmd, vel_cmd])
        
        def ackermann_or_spin_mode(_):
            # Calculate turn radius
            radius = jnp.where(
                jnp.abs(v_yaw) < 1e-6,
                jnp.inf,
                jnp.abs(v_lin_y / v_yaw)
            )
            
            # Calculate individual wheel velocities
            half_track = params.steer_track / 2.0
            half_base = params.wheel_base / 2.0
            
            vel_fl = sign * jnp.sqrt(
                (v_lin_y - v_yaw * half_track)**2 + (v_yaw * half_base)**2
            ) / params.wheel_radius
            
            vel_fr = sign * jnp.sqrt(
                (v_lin_y + v_yaw * half_track)**2 + (v_yaw * half_base)**2
            ) / params.wheel_radius
            
            vel_rl = sign * jnp.sqrt(
                (v_lin_y - v_yaw * half_track)**2 + (v_yaw * half_base)**2
            ) / params.wheel_radius
            
            vel_rr = sign * jnp.sqrt(
                (v_lin_y + v_yaw * half_track)**2 + (v_yaw * half_base)**2
            ) / params.wheel_radius
            
            # Check if spin mode (radius < min_turn_radius)
            def spin_mode(_):
                # Pure rotation mode
                fl_steer = jnp.sign(v_yaw) * jnp.pi / 2
                fr_steer = jnp.sign(v_yaw) * jnp.pi / 2
                rl_steer = -fl_steer
                rr_steer = -fr_steer
                return jnp.array([fl_steer, fr_steer, rl_steer, rr_steer])
            
            def ackermann_mode(_):
                # Ackermann steering geometry
                fl_steer = jnp.arctan2(
                    v_yaw * params.wheel_base,
                    2.0 * v_lin_y - v_yaw * params.steer_track
                )
                fr_steer = jnp.arctan2(
                    v_yaw * params.wheel_base,
                    2.0 * v_lin_y + v_yaw * params.steer_track
                )
                rl_steer = -fl_steer
                rr_steer = -fr_steer
                return jnp.array([fl_steer, fr_steer, rl_steer, rr_steer])
            
            steer_ang = jax.lax.cond(
                radius < params.min_turn_radius,
                spin_mode,
                ackermann_mode,
                operand=None
            )
            
            drive_vel = jnp.array([vel_fl, vel_fr, vel_rl, vel_rr])
            return steer_ang, drive_vel
        
        return jax.lax.cond(
            jnp.abs(v_lin_x) > 0.01,
            parallel_mode,
            ackermann_or_spin_mode,
            operand=None
        )
    
    return jax.lax.cond(
        all_zero,
        zero_case,
        nonzero_case,
        operand=None
    )


# JIT-compiled version with default parameters
@jax.jit
def ik_base(v_lin_x: float, v_lin_y: float, v_yaw: float = 0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate inverse kinematics for mobile base with default parameters.
    
    This is a convenience wrapper using default Ranger Mini V2 parameters.
    
    Args:
        v_lin_x: Linear x velocity (m/s)
        v_lin_y: Linear y velocity (m/s)
        v_yaw: Angular velocity (rad/s), default 0.0
    
    Returns:
        Tuple of:
            - steer_ang: Steering angles [fl, fr, rl, rr] (shape: [4])
            - drive_vel: Drive velocities [fl, fr, rl, rr] (shape: [4])
    """
    params = get_default_params()
    return ik_base_pure(v_lin_x, v_lin_y, v_yaw, params)


# ============================================
# Batched operations for vectorized environments
# ============================================

def ik_base_batched_inner(velocities: jnp.ndarray, params: MobileBaseParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Batched IK for multiple environments.
    
    Args:
        velocities: Shape (batch, 3) array of [v_lin_x, v_lin_y, v_yaw]
        params: Mobile base parameters
    
    Returns:
        Tuple of:
            - steer_ang: Shape (batch, 4) steering angles
            - drive_vel: Shape (batch, 4) drive velocities
    """
    def single_ik(vel):
        return ik_base_pure(vel[0], vel[1], vel[2], params)
    
    # Use vmap for batched execution
    return jax.vmap(single_ik)(velocities)


# Create vmapped version
ik_base_batched = jax.jit(ik_base_batched_inner, static_argnums=(1,))


# ============================================
# Forward Kinematics (for completeness)
# ============================================

@jax.jit
def fk_base(
    steer_angles: jnp.ndarray,
    wheel_velocities: jnp.ndarray,
    params: MobileBaseParams
) -> Tuple[float, float, float]:
    """Forward kinematics: compute base velocity from wheel states.
    
    This is approximate and assumes Ackermann geometry.
    
    Args:
        steer_angles: Wheel steering angles [fl, fr, rl, rr] (shape: [4])
        wheel_velocities: Wheel angular velocities [fl, fr, rl, rr] (shape: [4])
        params: Mobile base parameters
    
    Returns:
        Tuple of (v_x, v_y, v_yaw) base velocities
    """
    # Average wheel linear velocities
    avg_vel = jnp.mean(wheel_velocities) * params.wheel_radius
    
    # Average front steering angle
    avg_steer = (steer_angles[0] + steer_angles[1]) / 2.0
    
    # Decompose into v_x, v_y
    v_y = avg_vel * jnp.cos(avg_steer)
    v_x = -avg_vel * jnp.sin(avg_steer)
    
    # Estimate yaw rate from differential steering
    steer_diff = steer_angles[1] - steer_angles[0]  # fr - fl
    v_yaw = jnp.where(
        jnp.abs(steer_diff) > 0.01,
        2.0 * v_y * jnp.tan(avg_steer) / params.wheel_base,
        0.0
    )
    
    return v_x, v_y, v_yaw
