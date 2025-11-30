
import gymnasium as gym
import numpy as np
import cv2 as cv
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
import configs.env.DcmmCfg as DcmmCfg

def test_env():
    print("Initializing DcmmVecEnv...")
    try:
        env = DcmmVecEnv(
            task='Tracking', 
            object_name='object', 
            render_per_step=False, 
            print_reward=True, 
            print_info=False, 
            print_contacts=False, 
            print_ctrl=False, 
            print_obs=False, 
            camera_name=["wrist"],
            render_mode="depth_array", 
            imshow_cam=False, 
            viewer=False, 
            object_eval=False,
            env_time=2.5, 
            steps_per_policy=20,
            img_size=(112, 112)
        )
        print("Environment initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        raise e

    print("Resetting environment...")
    try:
        obs, info = env.reset()
        print("Reset successful.")
        print("Observation keys:", obs.keys())
        print("Depth shape:", obs['depth'].shape)
        print("Depth min/max:", obs['depth'].min(), obs['depth'].max())
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        raise e

    print("Stepping environment...")
    try:
        # Create dummy action
        action = {
            'base': np.zeros(2),
            'arm': np.zeros(4),
            'hand': np.zeros(12)
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        print("Step successful.")
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Depth shape after step:", obs['depth'].shape)
        
    except Exception as e:
        print(f"Failed to step environment: {e}")
        import traceback
        traceback.print_exc()
        raise e

    print("Test passed!")

if __name__ == "__main__":
    test_env()
