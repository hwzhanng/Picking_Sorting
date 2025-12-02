from __future__ import annotations

import hydra
import torch
import os
import random
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from gym_dcmm.utils.util import omegaconf_to_dict
from gym_dcmm.algs.ppo_dcmm.stage1.PPO_Stage1 import PPO_Stage1
import gymnasium as gym
import gym_dcmm
import datetime
import pytz
# os.environ['MUJOCO_GL'] = 'egl'
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_name='config_stage1', config_path='configs')
def main(config: DictConfig):
    torch.multiprocessing.set_start_method('spawn')
    config.test = config.test
    model_path = None
    if config.task == 'Tracking' and config.checkpoint_tracking:
        config.checkpoint_tracking = to_absolute_path(config.checkpoint_tracking)
        model_path = config.checkpoint_tracking
    elif (config.task == 'Catching_TwoStage' \
        or config.task == 'Catching_OneStage') \
        and config.checkpoint_catching:
        config.checkpoint_catching = to_absolute_path(config.checkpoint_catching)
        model_path = config.checkpoint_catching

    # use the device for rl
    config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
    config.seed = random.seed(config.seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    # Create and wrap the environment
    env_name = 'gym_dcmm/DcmmVecWorld-v0'
    task = 'Tracking' if config.task == 'Tracking' else 'Catching'
    
    if config.num_envs > 18:
        cprint(f"Warning: config.num_envs {config.num_envs} is too large for the available CPU cores. Capping at 18.", 'yellow')
        config.num_envs = 18
        
    print("config.num_envs: ", config.num_envs)
    env = gym.make_vec(env_name, num_envs=int(config.num_envs), 
                    task=task, camera_name=["top"],  # Use vehicle top camera
                    render_per_step=False, render_mode = "depth_array", # Use depth mode
                    object_name = "object",
                    img_size = config.train.ppo.img_dim,
                    imshow_cam = config.imshow_cam, 
                    viewer = config.viewer,
                    print_obs = False, print_info = False,
                    print_reward = False, print_ctrl = False,
                    print_contacts = False, object_eval = config.object_eval,
                    env_time = 5.0, steps_per_policy = 20)

    output_dif = os.path.join('outputs', config.output_name)
    # Get the local date and time
    local_tz = pytz.timezone('Asia/Shanghai')
    current_datetime = datetime.datetime.now().astimezone(local_tz)
    current_datetime_str = current_datetime.strftime("%Y-%m-%d/%H:%M:%S")
    output_dif = os.path.join(output_dif, current_datetime_str)
    os.makedirs(output_dif, exist_ok=True)

    agent = PPO_Stage1(env, output_dif, full_config=config)

    cprint('Start Training/Testing the Agent', 'green', attrs=['bold'])
    if config.test:
        if model_path:
            print("checkpoint loaded")
            agent.restore_test(model_path)
        print("testing")
        agent.test()
    else:
        # connect to wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.output_name,
            config=omegaconf_to_dict(config),
            mode=config.wandb_mode
        )

        agent.restore_train(model_path)
        agent.train()

        # close wandb
        wandb.finish()

if __name__ == '__main__':
    main()
