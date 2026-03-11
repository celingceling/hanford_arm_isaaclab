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

PAN_OFFSET  =  1.89   # rad — joint value when camera is at physical zero (facing -Y world)
TILT_OFFSET =  0.0    # tilt looks close to zero at rest, may need minor tuning

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
    tilt = torch.clamp(tilt, -140 * math.pi / 180.0, 140 * math.pi / 180.0)
    
    # DEBUG: EE 0.5 m in front of PTZ (PTZ forward = -Y in world)
    delta_zero = torch.zeros_like(delta)
    delta_zero[:, 2] = +.5  # -Y forward
    
    pan = torch.atan2(-delta_zero[:, 0], -delta_zero[:, 1])

    horiz_dist = torch.norm(delta_zero[:, :2], dim=1)
    tilt = -torch.atan2(delta_zero[:, 2], horiz_dist)
    tilt = torch.clamp(tilt, -140 * math.pi / 180.0, 140 * math.pi / 180.0)

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
    print("PTZ joints at rest (zero command):", ptz.data.joint_pos[0])
    # simulate environment
    ee_marker, ptz_marker, arrow_marker, actual_arrow_marker = make_ptz_debug_visualizers(args_cli.device)

    ptz_body_ids, _ = ptz.find_bodies("Tilt_Link")
    ptz_body_idx = int(ptz_body_ids[0])

    ee_body_ids, _ = robot.find_bodies("end_effector")
    ee_body_idx = int(ee_body_ids[0])

    while simulation_app.is_running():
        with torch.inference_mode():
            
            # get arm command [num_envs, 7]  →  [x, y, z, qw, qx, qy, qz]
            arm_command = env.unwrapped.command_manager.get_command("ee_pose")
            assert arm_command.shape == (env.unwrapped.num_envs, env.action_space.shape[-1]), \
                f"Command shape {arm_command.shape} doesn't match action space {env.action_space.shape}"
            
            # get ptz command
            ptz_pos_w = ptz.data.body_pos_w[:, ptz_body_idx, :]
            ptz_root_quat_w = ptz.data.root_quat_w
            ptz_root_w = ptz.data.root_pos_w
            
            # get ee pos in world frame
            ee_pos_w = robot.data.body_pos_w[:, ee_body_idx, :]
            
            ptz_pos_w = ptz.data.body_pos_w[:, ptz_body_idx, :]   # (N,3)
            ee_pos_w  = robot.data.body_pos_w[:, ee_body_idx, :]  # (N,3)
            
            
            # DEBUG
            # ptz joints (names: J1=pan, J2=tilt)
            ptz_joint_ids = ptz.find_joints(["J1", "J2"])[0]  # ids
            pan_id, tilt_id = int(ptz_joint_ids[0]), int(ptz_joint_ids[1])

            # current positions (rad)
            q = ptz.data.joint_pos[:, [pan_id, tilt_id]]  # (N,2)

            # targets (rad) - if you set them via set_joint_position_target(...)
            q_tgt = ptz.data.joint_pos_target[:, [pan_id, tilt_id]]  # (N,2)

            # limits (rad)
            q_lo = ptz.data.joint_pos_limits[:, [pan_id, tilt_id], 0]  # (N,2)
            q_hi = ptz.data.joint_pos_limits[:, [pan_id, tilt_id], 1]  # (N,2)

            # print env0
            # print(
            #     f"PTZ env0 | q=[pan {q[0,0].item():+.4f}, tilt {q[0,1].item():+.4f}] "
            #     f"| tgt=[pan {q_tgt[0,0].item():+.4f}, tilt {q_tgt[0,1].item():+.4f}] "
            #     f"| diff=[pan {(q_tgt[0,0].item()-q[0,0].item()):+.4f}, tilt {(q_tgt[0,1].item()-q[0,1].item()):+.4f}] "
            #     # f"| lim=[pan ({q_lo[0,0].item():+.4f},{q_hi[0,0].item():+.4f}), "
            #     # f"tilt ({q_lo[0,1].item():+.4f},{q_hi[0,1].item():+.4f})]"
            # )
            
            
            
            ptz_action = compute_ptz_action(
                ptz_pos_w=ptz_pos_w, 
                ee_pos_w=ee_pos_w,
                ptz_root_quat_w=ptz_root_quat_w,
                )
            
            
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
            
            
            ptz_zero_action = torch.zeros_like(ptz_action)
            
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
