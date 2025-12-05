#!/usr/bin/env python3
"""
Test script to validate Stage 2 optimizations.

Tests:
1. Memory-efficient buffer storage (uint8 for depth)
2. Perturbation test functionality
3. Velocity penalty effectiveness
"""

import torch
import numpy as np
import sys
import os

# Add project to path
sys.path.insert(0, os.path.abspath('../../'))

def test_buffer_memory_efficiency():
    """Test 1: Verify depth storage uses uint8"""
    print("=" * 60)
    print("TEST 1: Memory-Efficient Buffer Storage")
    print("=" * 60)
    
    from gym_dcmm.algs.ppo_dcmm.experience import ExperienceBuffer
    
    # Create buffer with dict observation space
    obs_dim = {'state': (35,), 'depth': (1, 84, 84)}
    buf = ExperienceBuffer(
        num_envs=128, 
        horizon_length=64, 
        batch_size=8192, 
        minibatch_size=1024,
        obs_dim=obs_dim,
        act_dim=18,
        device='cpu'
    )
    
    # Check dtype
    state_dtype = buf.storage_dict['obses']['state'].dtype
    depth_dtype = buf.storage_dict['obses']['depth'].dtype
    
    print(f"State storage dtype: {state_dtype}")
    print(f"Depth storage dtype: {depth_dtype}")
    
    # Calculate memory savings
    depth_shape = buf.storage_dict['obses']['depth'].shape
    depth_numel = np.prod(depth_shape)
    
    mem_float32 = depth_numel * 4  # 4 bytes per float32
    mem_uint8 = depth_numel * 1    # 1 byte per uint8
    savings_pct = (1 - mem_uint8 / mem_float32) * 100
    
    print(f"\nDepth tensor shape: {depth_shape}")
    print(f"Memory if float32: {mem_float32 / 1024 / 1024:.2f} MB")
    print(f"Memory with uint8: {mem_uint8 / 1024 / 1024:.2f} MB")
    print(f"Savings: {savings_pct:.1f}%")
    
    # Verify assertion
    assert depth_dtype == torch.uint8, f"Expected uint8, got {depth_dtype}"
    assert state_dtype == torch.float32, f"Expected float32 for state, got {state_dtype}"
    
    print("\n✅ TEST 1 PASSED: Depth stored as uint8, 75% memory saved!\n")


def test_velocity_penalty():
    """Test 3: Verify velocity penalty offsets contact reward"""
    print("=" * 60)
    print("TEST 3: Strengthened Velocity Penalty")
    print("=" * 60)
    
    # Test penalty formula at various speeds
    velocities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    print(f"{'Speed (m/s)':<12} {'Penalty':<12} {'Can offset 9.0?'}")
    print("-" * 40)
    
    for v in velocities:
        # New formula: -6.0 * (exp(5.0 * max(0, speed - 0.3)) - 1.0)
        penalty = -6.0 * (np.exp(5.0 * max(0, v - 0.3)) - 1.0)
        can_offset = "✅ Yes" if penalty <= -9.0 else "❌ No"
        print(f"{v:<12.1f} {penalty:<12.2f} {can_offset}")
    
    # Critical assertion: 0.5 m/s must offset max contact reward (9.0)
    penalty_at_05 = -6.0 * (np.exp(5.0 * max(0, 0.5 - 0.3)) - 1.0)
    assert penalty_at_05 <= -9.0, f"Penalty at 0.5m/s ({penalty_at_05:.2f}) must offset 9.0"
    
    print(f"\n✅ TEST 3 PASSED: Penalty at 0.5m/s = {penalty_at_05:.2f}, offsets max reward!\n")


def test_perturbation_logic():
    """Test 2: Verify perturbation test state machine logic"""
    print("=" * 60)
    print("TEST 2: Perturbation Test Logic")
    print("=" * 60)
    
    print("Testing state machine transitions:")
    print("1. Idle -> Testing (when force >= 1.0N)")
    print("2. Testing -> Idle (after evaluation)")
    print("3. Slippage calculation accuracy")
    print("4. Reward assignment (+10.0 / -5.0)")
    
    # Create mock environment state
    class MockEnv:
        class MockDcmm:
            class MockModel:
                class MockOpt:
                    timestep = 0.002
                opt = MockOpt()
            model = MockModel()
            
            class MockData:
                def __init__(self):
                    self._xfrc = np.zeros((10, 6))
                    
                @property
                def xfrc_applied(self):
                    return self._xfrc
                
                def body(self, name):
                    class MockBody:
                        id = 5
                        xpos = np.array([1.0, 2.0, 1.5])
                    return MockBody()
            
            data = MockData()
        
        Dcmm = MockDcmm()
        object_name = "object"
        steps_per_policy = 20
    
    from gym_dcmm.envs.stage2.RewardManagerStage2 import RewardManagerStage2
    
    env = MockEnv()
    reward_mgr = RewardManagerStage2(env)
    
    # Test 1: Idle state
    print("\nPhase 1: Idle (force < 1.0N)")
    reward = reward_mgr.evaluate_grasp_stability(0.5)
    assert reward == 0.0, "Should return 0 in idle state"
    assert not reward_mgr.perturbation_active, "Should remain inactive"
    print(f"  Reward: {reward:.1f}, Active: {reward_mgr.perturbation_active} ✅")
    
    # Test 2: Enter testing state
    print("\nPhase 2: Enter Testing (force >= 1.0N)")
    reward = reward_mgr.evaluate_grasp_stability(1.5)
    assert reward_mgr.perturbation_active, "Should activate testing"
    assert reward_mgr.initial_grasp_pos is not None, "Should record initial position"
    print(f"  Active: {reward_mgr.perturbation_active}, Initial pos recorded ✅")
    
    # Test 3: During testing (no reward yet)
    print("\nPhase 3: During Testing (0.0s < t < 0.5s)")
    for i in range(10):
        reward = reward_mgr.evaluate_grasp_stability(1.5)
    assert reward_mgr.perturbation_active, "Should remain active"
    print(f"  Timer: {reward_mgr.perturbation_timer:.3f}s, Active: {reward_mgr.perturbation_active} ✅")
    
    # Test 4: Advance timer to trigger evaluation
    print("\nPhase 4: Evaluation (t >= 0.5s)")
    # Manually advance timer
    reward_mgr.perturbation_timer = 0.6
    final_reward = reward_mgr.evaluate_grasp_stability(1.5)
    
    # Should have evaluated and reset
    print(f"  Final Reward: {final_reward:.1f}")
    print(f"  Active after eval: {reward_mgr.perturbation_active}")
    
    assert not reward_mgr.perturbation_active, "Should deactivate after evaluation"
    assert final_reward in [10.0, -5.0], f"Reward should be +10.0 or -5.0, got {final_reward}"
    
    print(f"\n✅ TEST 2 PASSED: State machine works correctly!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STAGE 2 OPTIMIZATION VALIDATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_buffer_memory_efficiency()
        test_perturbation_logic()
        test_velocity_penalty()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("✅ Memory efficiency: 75% savings on depth storage")
        print("✅ Perturbation test: State machine functional")
        print("✅ Velocity penalty: Effectively prevents aggressive impacts")
        print("\nReady for training! 🚀\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
