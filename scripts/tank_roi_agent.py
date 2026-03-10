# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import hanford_arm_isaaclab.tasks  # noqa: F401

def compute_ptz_action(ptz_pos_w: torch.Tensor, ee_pos_w: torch.Tensor) -> torch.Tensor:
    """Compute PTZ pan/tilt joints to point towards arm end effector
    
        PTZ initially spawns pointing in -Y direction
        
        Returns Joint targets in radians, Shape [num_envs, 2]
    """
    
    # make things right shapes, squeeze out second dimension if wrong
    if ptz_pos_w.ndim == 3:   # e.g., [N, 1, 3]
        ptz_pos_w = ptz_pos_w[:, 0, :]  # -> [N, 3]

    if ee_pos_w.ndim == 3:
        ee_pos_w = ee_pos_w[:, 0, :]
        
    # print("ee pos: ", ee_pos_w)
    # print("ptz root pose: ", ptz_pos_w)
    # Vector ptz -> ee
    delta = ee_pos_w - ptz_pos_w
    # print("ptz delta: ", delta)
    
    # get pan, pan is XY plane so atan (dy/dx), minus 90deg because ptz initially facing -Y direction
    pan = -torch.atan2(delta[:, 1], delta[:, 0]) - (math.pi) # [num_envs]
    # pan = wrap_to_pi(pan)
    
    # get horizontal distance (XY distance)
    horiz_dist = torch.norm(delta[:, :2], dim=1)  # [num_envs]
    
    tilt_lim = 140 * math.pi / 180.0
    # get tilt (tilt is aligned in the plane that is horiz_dist and Z)
    tilt = -torch.atan2(delta[:, 2], horiz_dist) # [num_envs]
    tilt = torch.clamp(tilt, -tilt_lim, tilt_lim) # clamp to limits
    
    # stack pan and tilt into action tensor [num_envs, 2], matches order of ptz joint names
    return torch.stack([pan, tilt], dim=1)

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi

def main():
    """Agent that sends pose commands from random sample inside tank.
    
    (Copied from zero_agent.py)
    """
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # get handles of articulations
    robot = env.unwrapped.scene["robot"]
    ptz = env.unwrapped.scene["ptz"]
    
    env.reset()
    
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            
            # get arm command [num_envs, 7]  →  [x, y, z, qw, qx, qy, qz]
            arm_command = env.unwrapped.command_manager.get_command("ee_pose")
            assert arm_command.shape == (env.unwrapped.num_envs, env.action_space.shape[-1]), \
                f"Command shape {arm_command.shape} doesn't match action space {env.action_space.shape}"
            
            # get ptz command
            ptz_pos_w = ptz.data.root_pos_w
            

            
            # get ee pos in world frame
            ee_body_idx = robot.find_bodies("end_effector")[0]
            ee_pos_w = robot.data.body_pos_w[:, ee_body_idx, :]
            
            # print("ptz joint pos:", ptz.data.joint_pos)
            # print("ptz joint vel:", ptz.data.joint_vel)
            # print("ptz pos target:", ptz.data.joint_pos_target)
            # convert to joint targets (rad)
            ptz_action = compute_ptz_action(
                ptz_pos_w=ptz_pos_w, 
                ee_pos_w=ee_pos_w
                )
            
            # print("PTZ action: ", ptz_action)
            
            # write directly to articulation buffer (no MDP terms)
            ptz.set_joint_position_target(ptz_action)
            
            # step (applies arm command + physics + ptz action)
            env.step(arm_command)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
