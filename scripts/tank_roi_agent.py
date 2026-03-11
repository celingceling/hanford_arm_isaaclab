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

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_angle_axis

import hanford_arm_isaaclab.tasks  # noqa: F401

def make_ptz_debug_visualizers(device: str):
    """Create visualization markers — call ONCE before the sim loop."""

    ee_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/ee_marker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
    )

    ptz_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/ptz_marker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
    )

    # Arrow: thin cylinder pointing along +X (long axis = X via scale)
    # CylinderGeom default long axis is Y, so we rotate 90° around Z in the marker itself
    # by making it very thin in Y/Z and long in X via a mesh offset — easiest is just
    # to use a flat cone pointing along +Z and accept we orient it via the quaternion.
    arrow_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/ptz_arrow",
            markers={
                "arrow": sim_utils.CylinderCfg(
                    radius=0.02,
                    height=0.6,
                    axis="X",  # long axis along X so quaternion rotation is intuitive
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 1.0)),
                ),
            },
        )
    )
    
    # Actual camera pointing direction — orange cylinder
    actual_arrow_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/ptz_actual_arrow",
            markers={
                "arrow": sim_utils.CylinderCfg(
                    radius=0.02,
                    height=0.6,
                    axis="X",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.0)),
                ),
            },
        )
    )

    return ee_marker, ptz_marker, arrow_marker, actual_arrow_marker


def update_ptz_debug_vis(ee_marker, 
                         ptz_marker, 
                         arrow_marker, 
                         actual_arrow_marker,
                         ee_pos_w, 
                         ptz_pos_w, 
                         pan, 
                         tilt, 
                         ptz, 
                         device):
    
    """Update marker positions/orientations each step."""
    
    N = ee_pos_w.shape[0]
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(N, -1)


    # -- EE sphere --
    ee_marker.visualize(translations=ee_pos_w, orientations=identity_quat)

    # -- PTZ sphere --
    ptz_marker.visualize(translations=ptz_pos_w, orientations=identity_quat)

    # -- Arrow: rotate to show desired pointing direction --
    # Arrow asset points along +X by default.
    # We need a quaternion that rotates +X toward the desired direction.
    # Desired direction from pan/tilt (PTZ faces -Y world at pan=0):
    #   forward_x = -sin(pan) * cos(tilt)   (pan rotates in XY around Z)
    #   forward_y = -cos(pan) * cos(tilt)
    #   forward_z = -sin(tilt)              (tilt sign: negative = up)
    cos_tilt = torch.cos(tilt)
    desired_dir = torch.stack([
        -torch.sin(pan) * cos_tilt,
        -torch.cos(pan) * cos_tilt,
        -torch.sin(tilt),
    ], dim=1)  # [N, 3], unit vector in world frame

    # Build quaternion from +X -> desired_dir via axis-angle
    x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(N, -1)
    # cross product gives rotation axis
    rot_axis = torch.linalg.cross(x_axis, desired_dir)  # [N, 3]
    rot_axis_norm = torch.norm(rot_axis, dim=1, keepdim=True).clamp(min=1e-6)
    rot_axis = rot_axis / rot_axis_norm
    # dot product gives cos(angle)
    dot = (x_axis * desired_dir).sum(dim=1).clamp(-1.0, 1.0)  # [N]
    angle = torch.acos(dot)  # [N]
    arrow_quat = quat_from_angle_axis(angle, rot_axis)  # [N, 4] wxyz

    arrow_marker.visualize(translations=ptz_pos_w, orientations=arrow_quat)
    
    # -- Actual camera direction from Tilt_Link body pose --
    cam_body_idx = ptz.find_bodies("Pan_Link")[0]
    cam_pos_w  = ptz.data.body_pos_w[:, cam_body_idx, :]   # (N,3)
    cam_quat_w = ptz.data.body_quat_w[:, cam_body_idx, :]  # (N,4)
    
    if cam_pos_w.ndim == 3:
        cam_pos_w = cam_pos_w[:, 0, :]
    if cam_quat_w.ndim == 3:
        cam_quat_w = cam_quat_w[:, 0, :]

    # Tilt_Link local forward axis — need to know which axis the camera faces.
    # Start with -Y (common convention); adjust if the orange arrow looks wrong.
    cam_forward_local = torch.tensor([[0.0, -1.0, 0.0]], device=device).expand(N, -1)
    from isaaclab.utils.math import quat_apply
    cam_forward_w = quat_apply(cam_quat_w, cam_forward_local)  # [N, 3]

    # Rotate +X -> cam_forward_w for the cylinder marker
    rot_axis2 = torch.linalg.cross(x_axis, cam_forward_w)
    rot_axis2 = rot_axis2 / torch.norm(rot_axis2, dim=1, keepdim=True).clamp(min=1e-6)
    dot2 = (x_axis * cam_forward_w).sum(dim=1).clamp(-1.0, 1.0)
    angle2 = torch.acos(dot2)
    actual_quat = quat_from_angle_axis(angle2, rot_axis2)

    actual_arrow_marker.visualize(translations=cam_pos_w, orientations=actual_quat)

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
    ee_marker, ptz_marker, arrow_marker, actual_arrow_marker = make_ptz_debug_visualizers(args_cli.device)

    ptz_body_ids, _ = ptz.find_bodies("Pan_Link")
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

            # print("ptz root pos: ", ptz_root_w)
            # print("ptz root quat: ", ptz_root_quat_w)
            
            # get ee pos in world frame
            ee_pos_w = robot.data.body_pos_w[:, ee_body_idx, :]
            
            ptz_pos_w = ptz.data.body_pos_w[:, ptz_body_idx, :]   # (N,3)
            ee_pos_w  = robot.data.body_pos_w[:, ee_body_idx, :]  # (N,3)
            
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

            # print("PTZ action: ", ptz_action)
            # print("EE pos: ", ee_pos_w)
            
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
