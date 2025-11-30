"""
Quaternion utility functions for orientation calculations.
"""
import numpy as np


def quat_rotate_vector(quat, vec):
    """
    Rotate a vector by a quaternion using MuJoCo convention.
    
    Args:
        quat: quaternion in MuJoCo format [w, x, y, z]
        vec: 3D vector [x, y, z]
    
    Returns:
        Rotated 3D vector
    """
    w, x, y, z = quat
    vx, vy, vz = vec
    
    # Quaternion rotation: v' = q * v * q^-1
    # Expanded form (Hamilton product)
    t = 2 * np.cross([x, y, z], vec)
    rotated = vec + w * t + np.cross([x, y, z], t)
    
    return rotated
