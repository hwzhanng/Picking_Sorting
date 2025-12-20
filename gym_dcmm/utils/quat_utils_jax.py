"""
JAX-compatible quaternion utility functions for orientation calculations.

This module provides pure functional implementations of quaternion operations
that can be JIT-compiled with JAX and vectorized with vmap.

Migration Notes:
- Replaced numpy with jax.numpy for GPU/TPU compatibility
- All functions are pure and can be JIT-compiled
- Follows MuJoCo quaternion convention [w, x, y, z]
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def quat_rotate_vector(quat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    """Rotate a vector by a quaternion using MuJoCo convention.
    
    Uses the formula: v' = q * v * q^-1 (Hamilton product)
    Optimized form: v' = v + 2*w*(xyz × v) + 2*(xyz × (xyz × v))
    
    Args:
        quat: Quaternion in MuJoCo format [w, x, y, z] (shape: [4])
        vec: 3D vector [x, y, z] (shape: [3])
    
    Returns:
        Rotated 3D vector (shape: [3])
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    xyz = jnp.array([x, y, z])
    
    # Efficient quaternion rotation using cross products
    # v' = v + 2*w*(xyz × v) + 2*(xyz × (xyz × v))
    t = 2.0 * jnp.cross(xyz, vec)
    rotated = vec + w * t + jnp.cross(xyz, t)
    
    return rotated


@jax.jit
def quat_rotate_inverse(quat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    """Rotate a vector by the inverse of a quaternion.
    
    Equivalent to transforming from world frame to local frame.
    
    Args:
        quat: Quaternion in MuJoCo format [w, x, y, z] (shape: [4])
        vec: 3D vector [x, y, z] (shape: [3])
    
    Returns:
        Rotated 3D vector (in local frame) (shape: [3])
    """
    # Inverse quaternion [w, -x, -y, -z]
    quat_inv = jnp.array([quat[0], -quat[1], -quat[2], -quat[3]])
    return quat_rotate_vector(quat_inv, vec)


@jax.jit
def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Multiply two quaternions (Hamilton product).
    
    Args:
        q1: First quaternion [w, x, y, z] (shape: [4])
        q2: Second quaternion [w, x, y, z] (shape: [4])
    
    Returns:
        Product quaternion [w, x, y, z] (shape: [4])
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return jnp.array([w, x, y, z])


@jax.jit
def quat_conjugate(quat: jnp.ndarray) -> jnp.ndarray:
    """Compute conjugate of a quaternion.
    
    For unit quaternions, conjugate equals inverse.
    
    Args:
        quat: Quaternion [w, x, y, z] (shape: [4])
    
    Returns:
        Conjugate quaternion [w, -x, -y, -z] (shape: [4])
    """
    return jnp.array([quat[0], -quat[1], -quat[2], -quat[3]])


@jax.jit
def quat_normalize(quat: jnp.ndarray) -> jnp.ndarray:
    """Normalize a quaternion to unit length.
    
    Args:
        quat: Quaternion [w, x, y, z] (shape: [4])
    
    Returns:
        Normalized quaternion (shape: [4])
    """
    norm = jnp.linalg.norm(quat)
    return quat / jnp.maximum(norm, 1e-8)


@jax.jit
def quat_to_rotation_matrix(quat: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion to 3x3 rotation matrix.
    
    Args:
        quat: Quaternion [w, x, y, z] (shape: [4])
    
    Returns:
        Rotation matrix (shape: [3, 3])
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Precompute common products
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    return jnp.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz), 2.0*(xz + wy)],
        [2.0*(xy + wz), 1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy), 2.0*(yz + wx), 1.0 - 2.0*(xx + yy)]
    ])


@jax.jit
def rotation_matrix_to_quat(R: jnp.ndarray) -> jnp.ndarray:
    """Convert 3x3 rotation matrix to quaternion.
    
    Uses Shepperd's method for numerical stability.
    
    Args:
        R: Rotation matrix (shape: [3, 3])
    
    Returns:
        Quaternion [w, x, y, z] (shape: [4])
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    def case_trace(R):
        s = jnp.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
        return jnp.array([w, x, y, z])
    
    def case_x(R):
        s = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
        return jnp.array([w, x, y, z])
    
    def case_y(R):
        s = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
        return jnp.array([w, x, y, z])
    
    def case_z(R):
        s = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
        return jnp.array([w, x, y, z])
    
    # Use lax.cond for JAX-compatible branching
    quat = jax.lax.cond(
        trace > 0,
        case_trace,
        lambda R: jax.lax.cond(
            R[0, 0] > R[1, 1],
            lambda R: jax.lax.cond(
                R[0, 0] > R[2, 2],
                case_x,
                case_z,
                R
            ),
            lambda R: jax.lax.cond(
                R[1, 1] > R[2, 2],
                case_y,
                case_z,
                R
            ),
            R
        ),
        R
    )
    
    return quat_normalize(quat)


@jax.jit
def quat_angle_axis(quat: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
    """Extract rotation angle and axis from quaternion.
    
    Args:
        quat: Quaternion [w, x, y, z] (shape: [4])
    
    Returns:
        Tuple of:
            - angle: Rotation angle in radians (scalar)
            - axis: Rotation axis (shape: [3])
    """
    # Normalize to ensure valid
    quat = quat_normalize(quat)
    w = quat[0]
    xyz = quat[1:4]
    
    # angle = 2 * arccos(w)
    # Need to handle w close to 1 (small angle)
    angle = 2.0 * jnp.arccos(jnp.clip(w, -1.0, 1.0))
    
    # axis = xyz / sin(angle/2)
    sin_half = jnp.sqrt(1.0 - w*w)
    axis = jax.lax.cond(
        sin_half > 1e-6,
        lambda _: xyz / sin_half,
        lambda _: jnp.array([1.0, 0.0, 0.0]),  # Default axis for near-zero rotation
        operand=None
    )
    
    return angle, axis


@jax.jit
def quat_from_angle_axis(angle: float, axis: jnp.ndarray) -> jnp.ndarray:
    """Create quaternion from rotation angle and axis.
    
    Args:
        angle: Rotation angle in radians (scalar)
        axis: Rotation axis (shape: [3]), will be normalized
    
    Returns:
        Quaternion [w, x, y, z] (shape: [4])
    """
    # Normalize axis
    axis = axis / jnp.maximum(jnp.linalg.norm(axis), 1e-8)
    
    half_angle = angle * 0.5
    w = jnp.cos(half_angle)
    xyz = axis * jnp.sin(half_angle)
    
    return jnp.array([w, xyz[0], xyz[1], xyz[2]])


@jax.jit
def quat_slerp(q1: jnp.ndarray, q2: jnp.ndarray, t: float) -> jnp.ndarray:
    """Spherical linear interpolation between two quaternions.
    
    Args:
        q1: Start quaternion [w, x, y, z] (shape: [4])
        q2: End quaternion [w, x, y, z] (shape: [4])
        t: Interpolation parameter [0, 1] (scalar)
    
    Returns:
        Interpolated quaternion (shape: [4])
    """
    # Normalize inputs
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    
    # Compute dot product
    dot = jnp.sum(q1 * q2)
    
    # If dot < 0, negate q2 to take shorter path
    q2 = jax.lax.cond(
        dot < 0,
        lambda q: -q,
        lambda q: q,
        q2
    )
    dot = jnp.abs(dot)
    
    # If very close, use linear interpolation
    def lerp_case(_):
        result = q1 + t * (q2 - q1)
        return quat_normalize(result)
    
    def slerp_case(_):
        theta = jnp.arccos(jnp.clip(dot, -1.0, 1.0))
        sin_theta = jnp.sin(theta)
        s1 = jnp.sin((1.0 - t) * theta) / sin_theta
        s2 = jnp.sin(t * theta) / sin_theta
        return s1 * q1 + s2 * q2
    
    return jax.lax.cond(
        dot > 0.9995,
        lerp_case,
        slerp_case,
        operand=None
    )


@jax.jit  
def quat_relative(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Compute relative quaternion: q_rel = q1^-1 * q2.
    
    This gives the rotation from q1's frame to q2's frame.
    
    Args:
        q1: Reference quaternion [w, x, y, z] (shape: [4])
        q2: Target quaternion [w, x, y, z] (shape: [4])
    
    Returns:
        Relative quaternion (shape: [4])
    """
    q1_inv = quat_conjugate(q1)  # For unit quaternions, conjugate = inverse
    return quat_multiply(q1_inv, q2)


@jax.jit
def quat2theta(qw: float, qz: float) -> float:
    """Extract yaw angle assuming only Z-axis rotation.
    
    Args:
        qw: W component of quaternion
        qz: Z component of quaternion
    
    Returns:
        Yaw angle in radians [-pi, pi]
    """
    a = 2.0 * jnp.arctan2(qz, qw)
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


# ============================================
# Batched operations for vectorized environments
# ============================================

# Vectorized quaternion rotation
quat_rotate_vector_batched = jax.vmap(
    quat_rotate_vector,
    in_axes=(0, 0),
    out_axes=0
)

quat_rotate_inverse_batched = jax.vmap(
    quat_rotate_inverse,
    in_axes=(0, 0),
    out_axes=0
)

quat_to_rotation_matrix_batched = jax.vmap(
    quat_to_rotation_matrix,
    in_axes=0,
    out_axes=0
)
