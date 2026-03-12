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
import isaacsim.core.prims as prims
import isaaclab.sim as sim_utils

from util.debug_viz import *
import hanford_arm_isaaclab.tasks  # noqa: F401

def compute_ptz_action(ptz_pos_w, ee_pos_w, **kwargs):
    if ptz_pos_w.ndim == 3:
        ptz_pos_w = ptz_pos_w[:, 0, :]
    if ee_pos_w.ndim == 3:
        ee_pos_w = ee_pos_w[:, 0, :]

    delta = ee_pos_w - ptz_pos_w

    # PTZ faces -Y in world frame, so pan=0 when EE is in -Y direction
    pan = torch.atan2(-delta[:, 0], -delta[:, 1])

    horiz_dist = torch.norm(delta[:, :2], dim=1)
    tilt = -torch.atan2(delta[:, 2], horiz_dist)
    tilt = torch.clamp(tilt, -50 * math.pi / 180.0, 230 * math.pi / 180.0)
    
    # # DEBUG (-Y is forward):
    # delta_zero = torch.zeros_like(delta)
    # delta_zero[:, 0] = 0.7
    # delta_zero[:, 1] = -0.7
    
    # pan = torch.atan2(-delta_zero[:, 0], -delta_zero[:, 1]) * 2.0

    # horiz_dist = torch.norm(delta_zero[:, :2], dim=1)
    # tilt = -torch.atan2(delta_zero[:, 2], horiz_dist)
    # tilt = torch.clamp(tilt, -50 * math.pi / 180.0, 230 * math.pi / 180.0)

    return torch.stack([pan, tilt], dim=1)

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
    ee_marker, ptz_marker, arrow_marker, actual_arrow_marker = make_ptz_debug_visualizers(args_cli.device)

    ptz_body_ids, _ = ptz.find_bodies("Tilt_Link")
    ptz_body_idx = int(ptz_body_ids[0])

    ee_body_ids, _ = robot.find_bodies("end_effector")
    ee_body_idx = int(ee_body_ids[0])

    print("PTZ body names:", ptz.body_names)
    print("PTZ joint names: ", ptz.joint_names)
    
    print("robot body names:", robot.body_names)
    print("robot joint names: ", robot.joint_names)
    
    while simulation_app.is_running():
        with torch.inference_mode():
            
            # get arm command [num_envs, 7]  →  [x, y, z, qw, qx, qy, qz]
            arm_command = env.unwrapped.command_manager.get_command("ee_pose")
            assert arm_command.shape == (env.unwrapped.num_envs, env.action_space.shape[-1]), \
                f"Command shape {arm_command.shape} doesn't match action space {env.action_space.shape}"
            
            # get ptz command
            ptz_pos_w = ptz.data.body_pos_w[:, ptz_body_idx, :]
            ptz_root_quat_w = ptz.data.root_quat_w
            
            # DEBUG print root pos
            # ptz_root_pos_w = ptz.data.root_pos_w
            # print("PTZ root pos: ", ptz_root_pos_w)
            # robot_root_pos_w = robot.data.root_pos_w
            # print("robot root pos: ", robot_root_pos_w)
            # for i, name in enumerate(ptz.body_names):
            #     print(f"{name}: {ptz.data.body_pos_w[0, i, :]}")
            
            # get ee pos in world frame
            ee_pos_w = robot.data.body_pos_w[:, ee_body_idx, :]
            
            ptz_pos_w = ptz.data.body_pos_w[:, ptz_body_idx, :]   # (N,3)
            ee_pos_w  = robot.data.body_pos_w[:, ee_body_idx, :]  # (N,3)
            
            ptz_action = compute_ptz_action(
                ptz_pos_w=ptz_pos_w, 
                ee_pos_w=ee_pos_w,
                ptz_root_quat_w=ptz_root_quat_w,
                )
            
            print("PTZ joint pos:", ptz.data.joint_pos)
            
            print("PTZ computed action: ", ptz_action)
    
            ptz_action = torch.zeros_like(ptz_action)
            
            # --- PTZ DIAGNOSTIC ---
            update_ptz_debug_vis(
                ee_marker, ptz_marker, arrow_marker, actual_arrow_marker,
                ee_pos_w=ee_pos_w,
                ptz_pos_w=ptz_pos_w,
                pan=ptz_action[:, 0],
                tilt=ptz_action[:, 1],
                ptz=ptz,
                device=args_cli.device,
            )
            
            # write directly to articulation buffer (no MDP terms)
            # ptz.set_joint_position_target(ptz_action)
            
            # ptz.write_data_to_sim()
            
            # step (applies arm command + physics + ptz action)
            env.step(arm_command)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
