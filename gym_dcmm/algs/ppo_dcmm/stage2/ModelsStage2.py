"""
Flax (JAX) neural network models for Stage 2 (Catching).

This module provides JAX/Flax implementations of the neural network architectures
used in Stage 2 training, replacing the PyTorch implementations for GPU-accelerated
inference within MJX environments.

Migration Notes:
- Replaced PyTorch nn.Module with Flax linen nn.Module
- Uses JAX PRNG for initialization
- Provides weight conversion utilities from PyTorch to Flax
- All operations are JIT-compilable

Usage:
    # Initialize model
    model = ActorCriticFlax(config)
    params = model.init(jax.random.PRNGKey(0), dummy_obs)
    
    # Load PyTorch weights
    params = convert_pytorch_to_flax(torch_state_dict, params)
    
    # Inference
    output = model.apply(params, obs)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
from typing import Dict, Tuple, Any, Sequence
import numpy as np


class MLP(nn.Module):
    """Multi-layer perceptron with ELU activations.
    
    Attributes:
        features: Sequence of output dimensions for each layer
    """
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for feat in self.features:
            x = nn.Dense(
                feat,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0))
            )(x)
            x = nn.elu(x)
        return x


class DepthCNN(nn.Module):
    """CNN for processing depth images.
    
    Architecture matches the PyTorch version:
    - Conv2d(1→32, k=8, s=4) + ReLU
    - Conv2d(32→64, k=4, s=2) + ReLU  
    - Conv2d(64→32, k=3, s=1) + ReLU
    - Flatten → Linear(1568 → 256) + ReLU
    
    Input: (B, H, W, 1) or (B, 1, H, W) depending on data format
    Output: (B, 256)
    """
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Ensure NHWC format (Flax default)
        # Input should be (B, H, W, C) or (B, C, H, W)
        if x.ndim == 4 and x.shape[1] == 1:
            # NCHW -> NHWC
            x = jnp.transpose(x, (0, 2, 3, 1))
        elif x.ndim == 3:
            # (B, H, W) -> (B, H, W, 1)
            x = x[..., None]
        
        # Normalize to [0, 1] assuming 0-255 input
        # Note: Using static division (faster than conditional)
        # If your data is already normalized, you can skip this
        x = x.astype(jnp.float32) / 255.0
        
        # Conv layers with ReLU
        x = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding='VALID',
            kernel_init=nn.initializers.variance_scaling(
                2.0, 'fan_out', 'normal'
            )
        )(x)
        x = nn.relu(x)
        
        x = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='VALID',
            kernel_init=nn.initializers.variance_scaling(
                2.0, 'fan_out', 'normal'
            )
        )(x)
        x = nn.relu(x)
        
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='VALID',
            kernel_init=nn.initializers.variance_scaling(
                2.0, 'fan_out', 'normal'
            )
        )(x)
        x = nn.relu(x)
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # Linear layer to 256
        x = nn.Dense(
            256,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0))
        )(x)
        x = nn.relu(x)
        
        return x


class ActorCriticFlax(nn.Module):
    """Actor-Critic network for Stage 2 (Catching) using Flax.
    
    Architecture:
    - Actor: State-only MLP → action mean
    - Critic: State MLP + Depth CNN → concatenate → value head
    
    Attributes:
        state_dim: Dimension of state input
        action_dim: Number of actions
        units: MLP hidden layer sizes
        img_size: Depth image size (assumes square)
        log_sigma_min: Minimum log std for exploration
        log_sigma_max: Maximum log std for exploration
    """
    state_dim: int = 35
    action_dim: int = 20
    units: Sequence[int] = (256, 128)
    img_size: int = 84
    log_sigma_min: float = -5.0
    log_sigma_max: float = 0.0
    
    def setup(self):
        # Actor MLP (state only)
        self.actor_mlp = MLP(features=self.units)
        self.mu_head = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01)
        )
        
        # Critic components
        self.critic_state_mlp = MLP(features=self.units)
        self.critic_cnn = DepthCNN()
        self.value_head = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0)
        )
        
        # Learnable log_sigma (initialized to small exploration)
        # In Flax, we use self.param() instead of nn.Parameter
    
    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        deterministic: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass through actor-critic.
        
        Args:
            obs: Combined observation (B, state_dim + img_pixels)
            deterministic: If True, return mean action without sampling
            
        Returns:
            Dictionary with 'action', 'value', 'mu', 'sigma', 'log_prob'
        """
        # Split observation
        state = obs[:, :self.state_dim]
        depth_flat = obs[:, self.state_dim:]
        
        # Reshape depth to image
        batch_size = obs.shape[0]
        depth_img = depth_flat.reshape(batch_size, self.img_size, self.img_size, 1)
        
        # Actor forward (state only)
        actor_feat = self.actor_mlp(state)
        mu = self.mu_head(actor_feat)
        mu = jnp.tanh(mu)  # Bound to [-1, 1]
        
        # Log sigma (learnable parameter)
        log_sigma = self.param(
            'log_sigma',
            nn.initializers.constant(-2.0),
            (self.action_dim,)
        )
        log_sigma = jnp.clip(log_sigma, self.log_sigma_min, self.log_sigma_max)
        sigma = jnp.exp(log_sigma)
        
        # Critic forward (state + depth)
        state_feat = self.critic_state_mlp(state)
        vis_feat = self.critic_cnn(depth_img)
        combined = jnp.concatenate([state_feat, vis_feat], axis=-1)
        value = self.value_head(combined)
        
        result = {
            'mu': mu,
            'sigma': jnp.broadcast_to(sigma, mu.shape),
            'value': value.squeeze(-1),
        }
        
        if deterministic:
            result['action'] = mu
        else:
            # Sample from Gaussian
            # Note: For actual training, you'd pass in the RNG key
            result['action'] = mu  # Placeholder - actual sampling done externally
        
        return result
    
    def get_value(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Get only the value estimate (for AVP reward computation).
        
        Args:
            obs: Combined observation (B, state_dim + img_pixels)
            
        Returns:
            Value estimate (B,)
        """
        result = self(obs, deterministic=True)
        return result['value']


def create_actor_critic(config: Dict[str, Any]) -> ActorCriticFlax:
    """Factory function to create ActorCritic model from config.
    
    Args:
        config: Dictionary with keys:
            - state_dim: int
            - actions_num: int
            - actor_units: Sequence[int]
            - img_size: int
            
    Returns:
        ActorCriticFlax model instance
    """
    return ActorCriticFlax(
        state_dim=config.get('state_dim', 35),
        action_dim=config.get('actions_num', 20),
        units=tuple(config.get('actor_units', [256, 128])),
        img_size=config.get('img_size', 84),
    )


# ============================================
# Weight Conversion: PyTorch → Flax
# ============================================

def convert_pytorch_to_flax(
    torch_state_dict: Dict[str, Any],
    flax_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert PyTorch state dict to Flax parameter structure.
    
    This function maps PyTorch weight names to Flax parameter names
    and transposes weight matrices as needed (PyTorch uses row-major,
    Flax uses column-major for Dense layers).
    
    Args:
        torch_state_dict: PyTorch model state_dict
        flax_params: Initialized Flax parameter dict (for structure reference)
        
    Returns:
        Flax parameter dict with converted weights
    
    Example:
        # Load PyTorch checkpoint
        checkpoint = torch.load('model.pth')
        torch_state_dict = checkpoint['model']
        
        # Initialize Flax model
        model = ActorCriticFlax(...)
        dummy_obs = jnp.zeros((1, 35 + 84*84))
        flax_params = model.init(jax.random.PRNGKey(0), dummy_obs)
        
        # Convert weights
        flax_params = convert_pytorch_to_flax(torch_state_dict, flax_params)
    """
    import torch
    
    flax_params = unfreeze(flax_params)
    
    # Mapping from PyTorch names to Flax names
    # Format: pytorch_prefix -> (flax_path, needs_transpose)
    name_mapping = {
        # Actor MLP
        'actor_mlp_c.mlp.0': ('params', 'actor_mlp', 'Dense_0'),
        'actor_mlp_c.mlp.2': ('params', 'actor_mlp', 'Dense_1'),
        'mu_c': ('params', 'mu_head'),
        
        # Critic State MLP  
        'value_mlp.mlp.0': ('params', 'critic_state_mlp', 'Dense_0'),
        'value_mlp.mlp.2': ('params', 'critic_state_mlp', 'Dense_1'),
        'value_head': ('params', 'value_head'),
        
        # Critic CNN
        'critic_cnn.main.0': ('params', 'critic_cnn', 'Conv_0'),
        'critic_cnn.main.2': ('params', 'critic_cnn', 'Conv_1'),
        'critic_cnn.main.4': ('params', 'critic_cnn', 'Conv_2'),
        'critic_cnn.linear.0': ('params', 'critic_cnn', 'Dense_0'),
        
        # Log sigma
        'sigma_c': ('params', 'log_sigma'),
    }
    
    def set_nested(d, path, value):
        """Set value in nested dict using tuple path."""
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = value
    
    def get_nested(d, path):
        """Get value from nested dict using tuple path."""
        for key in path:
            d = d[key]
        return d
    
    for torch_name, torch_tensor in torch_state_dict.items():
        # Convert to numpy
        if isinstance(torch_tensor, torch.Tensor):
            np_array = torch_tensor.detach().cpu().numpy()
        else:
            np_array = np.array(torch_tensor)
        
        # Find mapping
        matched = False
        for prefix, flax_path in name_mapping.items():
            if torch_name.startswith(prefix):
                suffix = torch_name[len(prefix):]
                
                if suffix == '.weight':
                    # Linear/Dense layer weight
                    if 'Conv' in str(flax_path):
                        # Conv weight: PyTorch (out, in, H, W) -> Flax (H, W, in, out)
                        np_array = np.transpose(np_array, (2, 3, 1, 0))
                    else:
                        # Dense weight: PyTorch (out, in) -> Flax (in, out)
                        np_array = np_array.T
                    
                    full_path = flax_path + ('kernel',)
                    try:
                        set_nested(flax_params, full_path, jnp.array(np_array))
                        matched = True
                    except KeyError:
                        print(f"Warning: Could not set {full_path}")
                        
                elif suffix == '.bias':
                    full_path = flax_path + ('bias',)
                    try:
                        set_nested(flax_params, full_path, jnp.array(np_array))
                        matched = True
                    except KeyError:
                        print(f"Warning: Could not set {full_path}")
                        
                elif suffix == '' and 'sigma' in torch_name:
                    # Log sigma parameter
                    try:
                        set_nested(flax_params, flax_path, jnp.array(np_array))
                        matched = True
                    except KeyError:
                        print(f"Warning: Could not set {flax_path}")
                
                break
        
        if not matched and 'num_batches_tracked' not in torch_name:
            print(f"Warning: Unmatched PyTorch weight: {torch_name}")
    
    return freeze(flax_params)


def load_pytorch_checkpoint_to_flax(
    checkpoint_path: str,
    model: ActorCriticFlax,
    dummy_obs: jnp.ndarray
) -> Dict[str, Any]:
    """Load PyTorch checkpoint and convert to Flax parameters.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint file
        model: Flax model instance
        dummy_obs: Dummy observation for initialization
        
    Returns:
        Flax parameter dict
    
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        KeyError: If checkpoint doesn't have 'model' key
    """
    import torch
    
    # Load PyTorch checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        torch_state_dict = checkpoint['model']
    else:
        # Assume checkpoint is the state dict directly
        torch_state_dict = checkpoint
    
    # Initialize Flax model
    rng = jax.random.PRNGKey(0)
    flax_params = model.init(rng, dummy_obs)
    
    # Convert weights
    flax_params = convert_pytorch_to_flax(torch_state_dict, flax_params)
    
    return flax_params


# ============================================
# JIT-compiled inference functions
# ============================================

# Note: For JIT compilation with model as argument, use functools.partial
# to bind the model, or use model.apply directly inside a jitted function.

def create_value_fn(model: ActorCriticFlax):
    """Create a JIT-compiled value function for a specific model.
    
    Usage:
        model = ActorCriticFlax(...)
        get_value = create_value_fn(model)
        value = get_value(params, obs)
    """
    @jax.jit
    def get_value(params: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        result = model.apply(params, obs, deterministic=True)
        return result['value']
    return get_value


def create_action_fn(model: ActorCriticFlax):
    """Create a JIT-compiled action function for a specific model.
    
    Usage:
        model = ActorCriticFlax(...)
        get_action = create_action_fn(model)
        action = get_action(params, obs)
    """
    @jax.jit
    def get_action(params: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        result = model.apply(params, obs, deterministic=True)
        return result['mu']
    return get_action


# Legacy functions (for backward compatibility)
# Note: Passing model instance to JIT is inefficient - use create_*_fn instead

def get_value_jit(
    params: Dict[str, Any],
    obs: jnp.ndarray,
    model: ActorCriticFlax
) -> jnp.ndarray:
    """Value estimation (not JIT-compiled, use create_value_fn for JIT).
    
    Args:
        params: Flax parameters
        obs: Observation array
        model: Model instance
        
    Returns:
        Value estimate
    """
    result = model.apply(params, obs, deterministic=True)
    return result['value']


def get_action_jit(
    params: Dict[str, Any],
    obs: jnp.ndarray,
    model: ActorCriticFlax
) -> jnp.ndarray:
    """Deterministic action (not JIT-compiled, use create_action_fn for JIT).
    
    Args:
        params: Flax parameters
        obs: Observation array
        model: Model instance
        
    Returns:
        Action mean (deterministic)
    """
    result = model.apply(params, obs, deterministic=True)
    return result['mu']


# ============================================
# Running Mean/Std for observation normalization
# ============================================

class RunningMeanStdFlax:
    """Running mean and standard deviation tracker.
    
    JAX-compatible implementation of RunningMeanStd for observation
    normalization. State is stored as a dictionary.
    """
    
    @staticmethod
    def create(shape: Tuple[int, ...]) -> Dict[str, jnp.ndarray]:
        """Create initial state dict.
        
        Args:
            shape: Shape of the statistics to track
            
        Returns:
            State dict with 'mean', 'var', 'count'
        """
        return {
            'mean': jnp.zeros(shape),
            'var': jnp.ones(shape),
            'count': jnp.array(1e-4),
        }
    
    @staticmethod
    @jax.jit
    def normalize(
        x: jnp.ndarray,
        state: Dict[str, jnp.ndarray],
        eps: float = 1e-8
    ) -> jnp.ndarray:
        """Normalize input using running statistics.
        
        Args:
            x: Input array
            state: Running statistics state dict
            eps: Small constant for numerical stability
            
        Returns:
            Normalized array
        """
        return (x - state['mean']) / jnp.sqrt(state['var'] + eps)
    
    @staticmethod
    def update(
        state: Dict[str, jnp.ndarray],
        x: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Update running statistics with new batch.
        
        Args:
            state: Current state dict
            x: New batch of observations
            
        Returns:
            Updated state dict
        """
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]
        
        return RunningMeanStdFlax.update_from_moments(
            state, batch_mean, batch_var, batch_count
        )
    
    @staticmethod
    def update_from_moments(
        state: Dict[str, jnp.ndarray],
        batch_mean: jnp.ndarray,
        batch_var: jnp.ndarray,
        batch_count: int
    ) -> Dict[str, jnp.ndarray]:
        """Update using pre-computed moments (Welford's algorithm).
        
        Args:
            state: Current state dict
            batch_mean: Mean of new batch
            batch_var: Variance of new batch
            batch_count: Size of new batch
            
        Returns:
            Updated state dict
        """
        delta = batch_mean - state['mean']
        tot_count = state['count'] + batch_count
        
        new_mean = state['mean'] + delta * batch_count / tot_count
        
        m_a = state['var'] * state['count']
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * state['count'] * batch_count / tot_count
        new_var = m_2 / tot_count
        
        return {
            'mean': new_mean,
            'var': new_var,
            'count': tot_count,
        }
    
    @staticmethod
    def from_pytorch(torch_rms) -> Dict[str, jnp.ndarray]:
        """Convert PyTorch RunningMeanStd to Flax state dict.
        
        Args:
            torch_rms: PyTorch RunningMeanStd instance or state_dict
            
        Returns:
            Flax state dict
        """
        import torch
        
        if hasattr(torch_rms, 'state_dict'):
            state_dict = torch_rms.state_dict()
        else:
            state_dict = torch_rms
        
        return {
            'mean': jnp.array(state_dict['running_mean'].cpu().numpy()),
            'var': jnp.array(state_dict['running_var'].cpu().numpy()),
            'count': jnp.array(state_dict.get('count', 1e-4)),
        }


# ============================================
# Numerical Verification Test
# ============================================

def verify_pytorch_flax_conversion(
    torch_model,
    flax_model: ActorCriticFlax,
    flax_params: Dict[str, Any],
    test_input: np.ndarray = None,
    atol: float = 1e-5
) -> bool:
    """Verify PyTorch → Flax weight conversion is numerically correct.
    
    This test ensures the weight transposition was done correctly by
    comparing outputs from both models on the same input.
    
    Args:
        torch_model: Original PyTorch model (in eval mode)
        flax_model: Flax model instance
        flax_params: Converted Flax parameters
        test_input: Fixed test input (if None, creates random input)
        atol: Absolute tolerance for numerical comparison
        
    Returns:
        True if outputs match within tolerance
        
    Raises:
        AssertionError: If outputs don't match (with detailed error message)
        
    Example:
        import torch
        from gym_dcmm.algs.ppo_dcmm.stage2.ModelsStage2 import ActorCritic
        
        # Load PyTorch model
        torch_model = ActorCritic(config)
        torch_model.load_state_dict(checkpoint['model'])
        torch_model.eval()
        
        # Create Flax model and convert weights
        flax_model = ActorCriticFlax(...)
        flax_params = convert_pytorch_to_flax(checkpoint['model'], flax_params)
        
        # Verify conversion
        verify_pytorch_flax_conversion(torch_model, flax_model, flax_params)
    """
    import torch
    
    # Create test input if not provided
    if test_input is None:
        # Fixed seed for reproducibility
        np.random.seed(42)
        state_dim = flax_model.state_dim
        img_size = flax_model.img_size
        test_input = np.random.randn(1, state_dim + img_size * img_size).astype(np.float32)
    
    # Run PyTorch model
    torch_model.eval()
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input)
        torch_output = torch_model(torch_input)
        
        # Extract outputs (handle dict or tuple)
        if isinstance(torch_output, dict):
            y_torch_mu = torch_output.get('mu', torch_output.get('action'))
            y_torch_value = torch_output.get('value')
        else:
            y_torch_mu = torch_output[0] if len(torch_output) > 0 else None
            y_torch_value = torch_output[1] if len(torch_output) > 1 else None
        
        if y_torch_mu is not None:
            y_torch_mu = y_torch_mu.numpy()
        if y_torch_value is not None:
            y_torch_value = y_torch_value.numpy()
    
    # Run Flax model
    jax_input = jnp.array(test_input)
    flax_output = flax_model.apply(flax_params, jax_input, deterministic=True)
    y_flax_mu = np.array(flax_output['mu'])
    y_flax_value = np.array(flax_output['value'])
    
    # Compare outputs
    errors = []
    
    if y_torch_mu is not None:
        if not np.allclose(y_torch_mu, y_flax_mu, atol=atol):
            max_diff = np.max(np.abs(y_torch_mu - y_flax_mu))
            errors.append(f"Action (mu) mismatch: max_diff={max_diff:.2e} > atol={atol}")
    
    if y_torch_value is not None:
        if not np.allclose(y_torch_value, y_flax_value, atol=atol):
            max_diff = np.max(np.abs(y_torch_value - y_flax_value))
            errors.append(f"Value mismatch: max_diff={max_diff:.2e} > atol={atol}")
    
    if errors:
        error_msg = "PyTorch → Flax conversion verification FAILED:\n" + "\n".join(errors)
        error_msg += "\n\nPossible causes:"
        error_msg += "\n  1. Dense layer weight not transposed (PyTorch: out×in, Flax: in×out)"
        error_msg += "\n  2. Conv layer weight not transposed (PyTorch: out×in×H×W, Flax: H×W×in×out)"
        error_msg += "\n  3. Weight name mapping is incorrect"
        raise AssertionError(error_msg)
    
    print("✓ PyTorch → Flax conversion verification PASSED")
    print(f"  - Action (mu) shape: {y_flax_mu.shape}")
    print(f"  - Value shape: {y_flax_value.shape}")
    
    return True
