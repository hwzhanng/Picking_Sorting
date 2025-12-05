
import os
import sys
import numpy as np
import cv2
import gymnasium as gym
import gym_dcmm
from gym_dcmm.envs.stage2.DcmmVecEnvStage2 import DcmmVecEnvStage2
import configs.env.DcmmCfg as DcmmCfg

def visualize_randomization():
    print("Initializing Stage 2 Environment for Visualization...")
    # Initialize environment with RGB rendering for visualization
    env = DcmmVecEnvStage2(
        task='Catching',
        object_name='object',
        render_per_step=False,
        print_reward=False,
        print_info=True,
        print_contacts=False,
        print_ctrl=False,
        print_obs=False,
        camera_name=["top"],
        render_mode="rgb_array", # Use RGB for human-friendly visualization
        imshow_cam=False,
        viewer=False, # Headless
        object_eval=False,
        env_time=2.5,
        steps_per_policy=20
    )

    print("Starting Visualization Loop...")
    frames = []
    
    # Run 5 episodes to show different randomizations
    for episode in range(5):
        print(f"Episode {episode + 1}/5")
        obs, info = env.reset()
        
        # Capture initial frame
        img = env.render() # Returns rgb array
        
        # Add text to image
        # img is (H, W, 3)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        
        # Get robot and object positions for annotation
        robot_pos = env.Dcmm.data.body("arm_base").xpos
        object_pos = env.Dcmm.data.body("object").xpos
        dist = np.linalg.norm(robot_pos[0:2] - object_pos[0:2])
        
        cv2.putText(img, f"Ep {episode+1}: Dist {dist:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add to frames list (repeat for visibility)
        for _ in range(10): 
            frames.append(img)
            
        # Run a few steps to show it's live
        for _ in range(5):
            action = env.action_space.sample()
            # Zero out action to just see the static pose mostly
            # action['base'][:] = 0 # Base is removed from action space
            action['arm'][:] = 0
            action['hand'][:] = 0
            
            obs, reward, term, trunc, info = env.step(action)
            img = env.render()
            img = np.ascontiguousarray(img, dtype=np.uint8)
            frames.append(img)

    env.close()
    
    # Save as video
    height, width, layers = frames[0].shape
    video_path = os.path.abspath("stage2_randomization_viz.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
        
    video.release()
    print(f"Visualization saved to {video_path}")

if __name__ == "__main__":
    visualize_randomization()
