"""
JAX/MJX Migration Utilities.

This module provides utilities for migrating from the original
NumPy/PyTorch/MuJoCo implementation to JAX/Flax/MJX.

Contents:
1. Weight conversion from PyTorch to Flax
2. State conversion helpers
3. Common patterns for JAX-based RL
4. Example usage patterns

Migration Overview:
==================

The key changes when migrating to JAX/MJX are:

1. **Stateless Functions**: All state must be passed explicitly.
   - Old: `self.integral += delta`
   - New: `(output, new_state) = function(input, state)`

2. **Physics Engine**:
   - Old: `mujoco.mj_step(model, data)`
   - New: `data = mjx.step(sys, data)`

3. **Math Library**:
   - Old: `import numpy as np`
   - New: `import jax.numpy as jnp`

4. **Neural Networks**:
   - Old: PyTorch `nn.Module`
   - New: Flax `nn.Module`

5. **Parallelization**:
   - Old: Python loops or multiprocessing
   - New: `jax.vmap` for automatic vectorization
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional
import numpy as np


# ============================================
# Weight Conversion: PyTorch → Flax
# ============================================

def convert_torch_state_dict_to_numpy(torch_state_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert PyTorch state dict to numpy arrays.
    
    This is the first step in PyTorch → Flax weight conversion.
    
    Args:
        torch_state_dict: PyTorch model.state_dict()
        
    Returns:
        Dictionary with numpy arrays
    """
    import torch
    
    numpy_dict = {}
    for key, value in torch_state_dict.items():
        if isinstance(value, torch.Tensor):
            numpy_dict[key] = value.detach().cpu().numpy()
        else:
            numpy_dict[key] = np.array(value)
    
    return numpy_dict


def transpose_linear_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose linear layer weight from PyTorch to Flax format.
    
    PyTorch: (out_features, in_features)
    Flax: (in_features, out_features)
    """
    return weight.T


def transpose_conv_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose conv layer weight from PyTorch to Flax format.
    
    PyTorch: (out_channels, in_channels, H, W)
    Flax: (H, W, in_channels, out_channels)
    """
    return np.transpose(weight, (2, 3, 1, 0))


# ============================================
# RunningMeanStd Conversion
# ============================================

def convert_running_mean_std(torch_rms_state: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    """Convert PyTorch RunningMeanStd state to JAX format.
    
    Args:
        torch_rms_state: PyTorch RunningMeanStd.state_dict()
        
    Returns:
        Dictionary compatible with Flax RunningMeanStd
    """
    import torch
    
    mean = torch_rms_state.get('running_mean', torch_rms_state.get('mean'))
    var = torch_rms_state.get('running_var', torch_rms_state.get('var'))
    count = torch_rms_state.get('count', np.array(1e-4))
    
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(var, torch.Tensor):
        var = var.cpu().numpy()
    if isinstance(count, torch.Tensor):
        count = count.cpu().numpy()
    
    return {
        'mean': jnp.array(mean),
        'var': jnp.array(var),
        'count': jnp.array(count),
    }


# ============================================
# JAX-compatible Data Structures
# ============================================

def tree_stack(trees):
    """Stack a list of PyTrees along a new first axis.
    
    Useful for batching multiple environment states.
    
    Args:
        trees: List of PyTrees with the same structure
        
    Returns:
        Single PyTree with arrays stacked along axis 0
    """
    return jax.tree_map(lambda *xs: jnp.stack(xs), *trees)


def tree_unstack(tree):
    """Unstack a batched PyTree into a list of PyTrees.
    
    Args:
        tree: PyTree with batched arrays (first axis is batch)
        
    Returns:
        List of PyTrees, one per batch element
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    n = leaves[0].shape[0]
    return [treedef.unflatten([leaf[i] for leaf in leaves]) for i in range(n)]


# ============================================
# Common JAX Patterns for RL
# ============================================

def create_batched_rng_keys(key: jax.random.PRNGKey, n: int) -> jax.random.PRNGKey:
    """Create n independent PRNG keys for parallel environments.
    
    Args:
        key: Base PRNG key
        n: Number of keys needed
        
    Returns:
        Array of shape (n, 2) containing n independent keys
    """
    return jax.random.split(key, n)


@jax.jit
def normalize_observation(
    obs: jnp.ndarray,
    mean: jnp.ndarray,
    var: jnp.ndarray,
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """Normalize observation using running statistics.
    
    Args:
        obs: Observation to normalize
        mean: Running mean
        var: Running variance
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized observation
    """
    return (obs - mean) / jnp.sqrt(var + epsilon)


@jax.jit
def clip_action(
    action: jnp.ndarray,
    low: jnp.ndarray,
    high: jnp.ndarray
) -> jnp.ndarray:
    """Clip action to valid range.
    
    Args:
        action: Raw action
        low: Lower bounds
        high: Upper bounds
        
    Returns:
        Clipped action
    """
    return jnp.clip(action, low, high)


@jax.jit
def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation.
    
    JAX-compatible GAE computation.
    
    Args:
        rewards: Reward sequence (T,)
        values: Value estimates (T,)
        dones: Done flags (T,)
        next_value: Value of final state
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    T = rewards.shape[0]
    
    def scan_fn(carry, t):
        gae, next_val = carry
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        return (gae, values[t]), gae
    
    # Scan backwards
    _, advantages = jax.lax.scan(
        scan_fn,
        (0.0, next_value),
        jnp.arange(T - 1, -1, -1)
    )
    
    # Reverse to get correct order
    advantages = jnp.flip(advantages)
    returns = advantages + values
    
    return advantages, returns


# ============================================
# MJX-specific Utilities
# ============================================

def get_body_position(
    data,  # mjx.Data
    body_name: str,
    model  # mjx.Model or mujoco.MjModel for name lookup
) -> jnp.ndarray:
    """Get body position from MJX data.
    
    Note: MJX doesn't support name lookup at runtime.
    You need to pre-compute body IDs from the CPU model.
    
    Args:
        data: MJX data
        body_name: Name of body (for documentation)
        model: Model for body ID lookup
        
    Returns:
        Body position (3,)
    """
    import mujoco
    
    # Get body ID from CPU model
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id]


def get_body_quat(
    data,  # mjx.Data
    body_name: str,
    model
) -> jnp.ndarray:
    """Get body quaternion from MJX data.
    
    Args:
        data: MJX data
        body_name: Name of body
        model: Model for body ID lookup
        
    Returns:
        Body quaternion [w, x, y, z] (4,)
    """
    import mujoco
    
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xquat[body_id]


# ============================================
# Example: Converting a Complete Model
# ============================================

def example_convert_stage2_model():
    """Example showing how to convert Stage 2 model.
    
    This is a complete example of loading a PyTorch checkpoint
    and converting it to Flax for use in MJX environments.
    """
    import torch
    from gym_dcmm.algs.ppo_dcmm.stage2.ModelsStage2 import (
        ActorCriticFlax,
        convert_pytorch_to_flax,
        RunningMeanStdFlax
    )
    
    # 1. Load PyTorch checkpoint
    checkpoint_path = "assets/checkpoints/avp/stage2_critic.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 2. Create Flax model
    model = ActorCriticFlax(
        state_dim=35,
        action_dim=20,
        units=(256, 128),
        img_size=84
    )
    
    # 3. Initialize with dummy input
    dummy_obs = jnp.zeros((1, 35 + 84*84))
    params = model.init(jax.random.PRNGKey(0), dummy_obs)
    
    # 4. Convert weights
    params = convert_pytorch_to_flax(checkpoint['model'], params)
    
    # 5. Convert RunningMeanStd if present
    if 'running_mean_std' in checkpoint:
        rms_state = RunningMeanStdFlax.from_pytorch(checkpoint['running_mean_std'])
    else:
        rms_state = RunningMeanStdFlax.create((35,))
    
    return model, params, rms_state


# ============================================
# Example: JAX Environment Step
# ============================================

def example_env_step_pattern():
    """Example showing the JAX environment step pattern.
    
    This demonstrates the functional programming style
    required for JAX compatibility.
    """
    
    # The key pattern is:
    # new_state = step_fn(state, action)
    # 
    # Instead of:
    # self.state = mutate(self.state, action)
    
    from typing import NamedTuple
    
    class EnvState(NamedTuple):
        """All environment state must be explicit."""
        physics_data: Any  # mjx.Data
        reward_state: Any  # RewardState
        control_state: Any  # RobotControlState
        step_count: int
        done: bool
    
    def env_reset(rng: jax.random.PRNGKey) -> EnvState:
        """Reset returns initial state."""
        # ... initialization logic ...
        pass
    
    def env_step(state: EnvState, action: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray, float, bool]:
        """Step returns (new_state, obs, reward, done).
        
        Note: No mutation of state! Return new state instead.
        """
        # ... physics step, reward computation, etc. ...
        pass
    
    # Vectorize for parallel environments:
    # batched_step = jax.vmap(env_step)
    
    # JIT compile for speed:
    # jit_step = jax.jit(env_step)


# ============================================
# Migration Checklist
# ============================================

MIGRATION_CHECKLIST = """
JAX/MJX GPU-Accelerated Branch
==============================

This is a dedicated GPU-accelerated branch using JAX/MJX.
All modules have been converted to pure-functional JAX code.

✓ Core Utilities
  ✓ PID Controller (pid.py)
  ✓ Quaternion Utils (quat_utils.py)
  ✓ IK Base (ik_base.py)

✓ Neural Networks
  ✓ Stage 2 ActorCritic (ModelsStage2.py) - Flax implementation
  ✓ Weight Conversion Functions (PyTorch → Flax)
  ✓ Numerical Verification Test
  ✓ RunningMeanStd JAX version

✓ Reward Functions
  ✓ Stage 1 Reward Manager (RewardManagerStage1.py)
  ✓ Stage 2 Reward Manager (RewardManagerStage2.py)

✓ Robot Control
  ✓ Robot Wrapper (MujocoDcmm.py)
  ✓ Forward Kinematics (FK) functions
  ✓ Control State Containers
  ✓ MJX Integration Functions

□ TODO: Environment Classes
  ☐ DcmmVecEnvStage1 JAX version
  ☐ DcmmVecEnvStage2 JAX version
  ☐ Observation Manager JAX version
  ☐ Randomization Manager JAX version

□ TODO: Training Loop
  ☐ PPO implementation in JAX
  ☐ Experience buffer
  ☐ Policy/Value network updates

Notes:
------
- IK (inverse kinematics) cannot be JIT-compiled due to iterative solvers
  → Use direct joint position control instead
- Viewer functionality not available in MJX
  → Use CPU MuJoCo for visualization, MJX for training
- scipy functions not JAX-compatible
  → Re-implement with jax.numpy or use jax.scipy where available
"""

if __name__ == "__main__":
    print(MIGRATION_CHECKLIST)
