"""
JAX-compatible PID controller functions.

This module provides stateless, pure-functional PID control implementations
that can be JIT-compiled with JAX and vectorized with vmap.

Migration Notes:
- Removed class-based stateful design
- All state (integral error, previous error, time) must be passed explicitly
- Returns tuples of (output, new_state) for JAX functional paradigm
- Uses jax.numpy instead of numpy for GPU/TPU compatibility
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple


class PIDState(NamedTuple):
    """State container for PID controller.
    
    Attributes:
        integral: Accumulated integral error (shape: [dim])
        e_prev: Previous error for derivative calculation (shape: [dim])
        time_prev: Previous timestamp (scalar)
        initialized: Whether the controller has been initialized (bool)
    """
    integral: jnp.ndarray
    e_prev: jnp.ndarray
    time_prev: float
    initialized: bool


class PIDParams(NamedTuple):
    """Parameters for PID controller.
    
    Attributes:
        Kp: Proportional gain (scalar or array matching dim)
        Ki: Integral gain (scalar or array matching dim)
        Kd: Derivative gain (scalar or array matching dim)
        llim: Lower limit for velocity damping (scalar or array)
        ulim: Upper limit for velocity damping (scalar or array)
        offset: Output offset (scalar, default 0.0)
    """
    Kp: jnp.ndarray
    Ki: jnp.ndarray
    Kd: jnp.ndarray
    llim: jnp.ndarray
    ulim: jnp.ndarray
    offset: float = 0.0


def create_pid_state(dim: int) -> PIDState:
    """Create initial PID state.
    
    Args:
        dim: Dimension of the control signal
        
    Returns:
        PIDState with zeroed integral, error, and uninitialized flag
    """
    return PIDState(
        integral=jnp.zeros(dim),
        e_prev=jnp.zeros(dim),
        time_prev=0.0,
        initialized=False
    )


def create_pid_params(
    Kp: float,
    Ki: float,
    Kd: float,
    dim: int,
    llim: float = -25.0,
    ulim: float = 25.0,
    offset: float = 0.0
) -> PIDParams:
    """Create PID parameters.
    
    Args:
        Kp: Proportional gain
        Ki: Integral gain
        Kd: Derivative gain
        dim: Dimension for broadcasting
        llim: Lower velocity damper limit
        ulim: Upper velocity damper limit
        offset: Output offset
        
    Returns:
        PIDParams namedtuple
    """
    return PIDParams(
        Kp=jnp.asarray(Kp),
        Ki=jnp.asarray(Ki),
        Kd=jnp.asarray(Kd),
        llim=jnp.asarray(llim),
        ulim=jnp.asarray(ulim),
        offset=offset
    )


@jax.jit
def pid_step(
    setpoint: jnp.ndarray,
    measurement: jnp.ndarray,
    time: float,
    state: PIDState,
    params: PIDParams
) -> Tuple[jnp.ndarray, PIDState]:
    """Execute one step of PID control.
    
    This is a pure function - all state is passed in and returned.
    
    Args:
        setpoint: Desired value (shape: [dim])
        measurement: Current measured value (shape: [dim])
        time: Current timestamp (scalar)
        state: Previous PID state
        params: PID parameters
        
    Returns:
        Tuple of:
            - control_output: Manipulated variable (shape: [dim])
            - new_state: Updated PID state
    """
    # Calculate error
    e = setpoint - measurement
    
    # Proportional term
    P = params.Kp * e
    
    # Time delta (avoid division by zero)
    dt = jnp.maximum(time - state.time_prev, 1e-6)
    
    # Derivative term (only if initialized)
    D = jax.lax.cond(
        state.initialized,
        lambda _: params.Kd * (e - state.e_prev) / dt,
        lambda _: jnp.zeros_like(e),
        operand=None
    )
    
    # Apply velocity damping to D
    D = jnp.clip(D, params.llim, params.ulim)
    
    # Integral term
    delta_I = params.Ki * e * dt
    new_integral = state.integral + delta_I
    
    # Calculate output
    MV = params.offset + P + new_integral + D
    
    # Create new state
    new_state = PIDState(
        integral=new_integral,
        e_prev=e,
        time_prev=time,
        initialized=True
    )
    
    return MV, new_state


@jax.jit
def pid_reset(dim: int, k: float = 1.0, params: PIDParams = None) -> Tuple[PIDState, PIDParams]:
    """Reset PID state and optionally scale parameters.
    
    Args:
        dim: Dimension of control signal
        k: Scaling factor for gains (default 1.0)
        params: Original parameters to scale (optional)
        
    Returns:
        Tuple of (reset_state, scaled_params)
    """
    state = create_pid_state(dim)
    
    if params is not None:
        scaled_params = PIDParams(
            Kp=params.Kp * k,
            Ki=params.Ki,
            Kd=params.Kd,
            llim=params.llim * k,
            ulim=params.ulim * k,
            offset=params.offset
        )
        return state, scaled_params
    
    return state, params


# ============================================
# Incremental PID (Position-Velocity form)
# ============================================

class IncrementalPIDState(NamedTuple):
    """State container for Incremental PID controller.
    
    Attributes:
        e_prev: Previous error (shape: [dim])
        e_prev2: Error from two steps ago (shape: [dim])
    """
    e_prev: jnp.ndarray
    e_prev2: jnp.ndarray


def create_incremental_pid_state(dim: int) -> IncrementalPIDState:
    """Create initial state for incremental PID.
    
    Args:
        dim: Dimension of control signal
        
    Returns:
        IncrementalPIDState with zeroed errors
    """
    return IncrementalPIDState(
        e_prev=jnp.zeros(dim),
        e_prev2=jnp.zeros(dim)
    )


@jax.jit
def incremental_pid_step(
    setpoint: jnp.ndarray,
    measurement: jnp.ndarray,
    state: IncrementalPIDState,
    params: PIDParams
) -> Tuple[jnp.ndarray, IncrementalPIDState]:
    """Execute one step of incremental PID control.
    
    Incremental PID outputs the change in control signal (delta_u)
    rather than the absolute value. Useful for position control.
    
    Args:
        setpoint: Desired value (shape: [dim])
        measurement: Current measured value (shape: [dim])
        state: Previous incremental PID state
        params: PID parameters
        
    Returns:
        Tuple of:
            - delta_MV: Change in manipulated variable (shape: [dim])
            - new_state: Updated state
    """
    # Calculate error
    e = setpoint - measurement
    
    # Incremental PID formula:
    # delta_u = Kp*(e - e_prev) + Ki*e + Kd*(e - 2*e_prev + e_prev2)
    P = params.Kp * (e - state.e_prev)
    I = params.Ki * e
    D = params.Kd * (e - 2 * state.e_prev + state.e_prev2)
    
    delta_MV = params.offset + P + I + D
    
    # Update state
    new_state = IncrementalPIDState(
        e_prev=e,
        e_prev2=state.e_prev
    )
    
    return delta_MV, new_state


# ============================================
# Batched operations for vectorized environments
# ============================================

# Create vmapped versions for parallel environment execution
pid_step_batched = jax.vmap(
    pid_step,
    in_axes=(0, 0, 0, 0, None),  # Batch over setpoint, measurement, time, state; share params
    out_axes=(0, 0)
)

incremental_pid_step_batched = jax.vmap(
    incremental_pid_step,
    in_axes=(0, 0, 0, None),  # Batch over setpoint, measurement, state; share params
    out_axes=(0, 0)
)
