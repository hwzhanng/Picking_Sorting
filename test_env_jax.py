#!/usr/bin/env python3
"""
Test script for JAX/MJX environment setup.

Run this after installation to verify everything works:
    python test_env_jax.py
"""

import sys


def test_jax():
    """Test JAX installation and GPU detection."""
    print("=" * 60)
    print("Testing JAX...")
    print("=" * 60)
    
    try:
        import jax
        import jax.numpy as jnp
        
        print(f"✓ JAX version: {jax.__version__}")
        print(f"✓ JAX devices: {jax.devices()}")
        
        # Test basic computation
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x ** 2)
        print(f"✓ JAX computation test: sum([1,2,3]²) = {y}")
        
        # Test JIT compilation
        @jax.jit
        def test_fn(x):
            return jnp.sin(x) + jnp.cos(x)
        
        result = test_fn(jnp.array(0.5))
        print(f"✓ JIT compilation test: sin(0.5)+cos(0.5) = {result:.4f}")
        
        # Check if GPU is available
        gpu_devices = [d for d in jax.devices() if 'gpu' in str(d).lower()]
        if gpu_devices:
            print(f"✓ GPU detected: {gpu_devices}")
        else:
            print("⚠ No GPU detected - running on CPU (slower)")
        
        return True
    except ImportError as e:
        print(f"✗ JAX import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        return False


def test_flax():
    """Test Flax installation."""
    print("\n" + "=" * 60)
    print("Testing Flax...")
    print("=" * 60)
    
    try:
        import flax
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
        
        print(f"✓ Flax version: {flax.__version__}")
        
        # Test simple MLP
        class TestMLP(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(64)(x)
                x = nn.relu(x)
                x = nn.Dense(1)(x)
                return x
        
        model = TestMLP()
        params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 10)))
        output = model.apply(params, jnp.ones((1, 10)))
        print(f"✓ Flax MLP test: input(1,10) -> output{output.shape}")
        
        return True
    except ImportError as e:
        print(f"✗ Flax import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Flax test failed: {e}")
        return False


def test_mujoco():
    """Test MuJoCo installation."""
    print("\n" + "=" * 60)
    print("Testing MuJoCo...")
    print("=" * 60)
    
    try:
        import mujoco
        
        print(f"✓ MuJoCo version: {mujoco.__version__}")
        
        # Test basic model loading
        xml = """
        <mujoco>
            <worldbody>
                <body name="test_body">
                    <geom type="sphere" size="0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        print(f"✓ MuJoCo physics test: Created model with {model.nbody} bodies")
        
        return True, model, data
    except ImportError as e:
        print(f"✗ MuJoCo import failed: {e}")
        return False, None, None
    except Exception as e:
        print(f"✗ MuJoCo test failed: {e}")
        return False, None, None


def test_mjx(mj_model=None, mj_data=None):
    """Test MuJoCo MJX (GPU acceleration)."""
    print("\n" + "=" * 60)
    print("Testing MuJoCo MJX (GPU acceleration)...")
    print("=" * 60)
    
    try:
        import mujoco
        from mujoco import mjx
        import jax
        
        print("✓ MJX imported successfully")
        
        if mj_model is None:
            xml = """
            <mujoco>
                <worldbody>
                    <body name="test_body">
                        <joint type="slide" axis="0 0 1"/>
                        <geom type="sphere" size="0.1" mass="1"/>
                    </body>
                </worldbody>
            </mujoco>
            """
            mj_model = mujoco.MjModel.from_xml_string(xml)
            mj_data = mujoco.MjData(mj_model)
        
        # Convert to MJX
        mx_model = mjx.put_model(mj_model)
        mx_data = mjx.put_data(mj_model, mj_data)
        print("✓ Model converted to MJX")
        
        # Test MJX step
        @jax.jit
        def mjx_step(mx_model, mx_data):
            return mjx.step(mx_model, mx_data)
        
        mx_data = mjx_step(mx_model, mx_data)
        print("✓ MJX step executed successfully")
        
        # Test vmap for parallel environments
        import jax.numpy as jnp
        
        @jax.jit
        @jax.vmap
        def batched_step(mx_data):
            return mjx.step(mx_model, mx_data)
        
        # Create batch of data
        batch_size = 4
        batched_data = jax.tree_map(
            lambda x: jnp.stack([x] * batch_size),
            mx_data
        )
        batched_data = batched_step(batched_data)
        print(f"✓ Batched MJX step ({batch_size} envs) executed successfully")
        
        return True
    except ImportError as e:
        print(f"✗ MJX import failed: {e}")
        print("  Note: MJX requires mujoco >= 3.0")
        return False
    except Exception as e:
        print(f"✗ MJX test failed: {e}")
        return False


def test_gym_dcmm():
    """Test gym_dcmm package."""
    print("\n" + "=" * 60)
    print("Testing gym_dcmm package...")
    print("=" * 60)
    
    try:
        # Test imports
        from gym_dcmm.utils.pid import PIDState, PIDParams, pid_step
        print("✓ PID controller imported")
        
        from gym_dcmm.utils.quat_utils import quat_multiply, quat_rotate_vector
        print("✓ Quaternion utils imported")
        
        from gym_dcmm.agents.MujocoDcmm import (
            BodyIDMapping, SiteIDMapping,
            get_ee_position, get_site_position
        )
        print("✓ MujocoDcmm (MJX wrapper) imported")
        
        # Test PID
        import jax.numpy as jnp
        state = PIDState(
            integral=jnp.zeros(3),
            prev_error=jnp.zeros(3),
            prev_time=0.0
        )
        params = PIDParams(
            Kp=1.0, Ki=0.01, Kd=0.1,
            llim=-1.0, ulim=1.0
        )
        error = jnp.array([0.1, 0.2, 0.3])
        output, new_state = pid_step(error, 0.01, state, params)
        print(f"✓ PID step test: error={error} -> output={output}")
        
        return True
    except ImportError as e:
        print(f"✗ gym_dcmm import failed: {e}")
        print("  Run: pip install -e .")
        return False
    except Exception as e:
        print(f"✗ gym_dcmm test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("  JAX/MJX Environment Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results['jax'] = test_jax()
    results['flax'] = test_flax()
    mujoco_ok, mj_model, mj_data = test_mujoco()
    results['mujoco'] = mujoco_ok
    results['mjx'] = test_mjx(mj_model, mj_data) if mujoco_ok else False
    results['gym_dcmm'] = test_gym_dcmm()
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:12} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Environment is ready for training.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
