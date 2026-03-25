# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
from ..include.config import *

    
def reset_multi_from_3_spots(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    poses_w: torch.Tensor, # [3,7] (x,y,z,qw,qx,qy,qz)
    asset_names: list[str] = ["robot", "ptz"],
    ):
    
    """ Reset robot and ptz to one of 3 randomly sampled positions.
    
    ok claude reeealllyy wants me to reset ptz and robot joints to zero here instead of registering it as separate events
    but i don't wanna
    
    """
    
    # put poses on correct device (gpu)
    device = env_ids.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    poses_w = torch.as_tensor(poses_w, device=device, dtype=torch.float32)
    
    n = env_ids.numel()
    origins = env.scene.env_origins[env_ids]  # [n,3]
    ptz_offset = torch.tensor([0.0, 0.0, PTZ_OFFSET_Z], device=device, dtype=torch.float32)
    
    # define assets
    robot = env.scene[asset_names[0]]
    ptz= env.scene[asset_names[1]]
    
    # default state = what IsaacLab has cached at scene start-up
    default_root_robot = robot.data.default_root_state[env_ids]  # [n, 13]
    default_root_ptz = ptz.data.default_root_state[env_ids]  # from PTZ_CFG, init_state: rot=(0, 0, 1, 0)
    
    # pick random pose to reset to
    idx_robot = torch.randint(0, 3, (n,), device=device)
    idx_ptz = torch.randint(0, 2, (n,), device=device) # note only picks 0 or 1
    idx_ptz = idx_ptz + (idx_ptz >= idx_robot).to(idx_ptz.dtype) # if idx_1 is same or greater than idx_0, flags true and + 1
    # will never index out of bounds bc idx_1 only goes up to 2 

    # initialize as default state
    pose_robot = default_root_robot[:, :7].clone() # gets [x, y, z, w, i, j ,k]
    pose_ptz = default_root_ptz[:, :7].clone() 
    
    # keep per-env origin behavior (aka taking relative pos to global)
    # pose_robot[:, 0:3] = origins + poses_w[idx_robot, 0:3] # get env origin + [x,y,z] from desired poses
    # pose_ptz[:, 0:3] = origins + poses_w[idx_ptz, 0:3] + ptz_offset
    pose_robot[:, 0:3] = env_local_to_world(poses_w[idx_robot, 0:3], origins)
    pose_ptz[:, 0:3] = env_local_to_world(poses_w[idx_ptz, 0:3], origins) + ptz_offset

    # zero velocity
    vel = torch.zeros(n,6, device=device) # is this the right shape?
    
    # write states to sim
    robot.write_root_pose_to_sim(pose_robot,env_ids=env_ids)
    robot.write_root_velocity_to_sim(vel,env_ids=env_ids)

    ptz.write_root_pose_to_sim(pose_ptz,env_ids=env_ids)
    ptz.write_root_velocity_to_sim(vel,env_ids=env_ids)
 
def reset_coverage_buffer(env: "ManagerBasedRLEnv", env_ids: torch.Tensor):
    """
    Clears coverage grid cells and no-progress history for reset envs.
    
    """
    
    if hasattr(env, "coverage_grid"):
        env.coverage_grid.reset(env_ids)
    if hasattr(env, "_coverage_history"):
        env._coverage_history[env_ids] = 0.0
    

    
    
    
    
    
    
    
    
# UNUSED reset functions
def print_ptz_joints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ptz"),
):
    # define asset (ptz)
    asset = env.scene[asset_cfg.name]
    device = env_ids.device
    
    default_root = asset.data.default_root_state[env_ids]
    i=0
    
    print("\n--- PRINT_PTZ_JOINTS_EVENT ---")
    print("env_id:", env_ids[i].item())
    print("default_root_ptz:", default_root[i].detach().cpu())
    print('root velocity: ', asset.data.root_vel_w)
    
    print("ptz.data.root_quat_w:", asset.data.root_quat_w[env_ids[i]].detach().cpu())
    print("ptz.data.joint_pos:", asset.data.joint_pos[env_ids[i]].detach().cpu())
    
    # if hasattr(asset, "_joint_pos_target_sim"):
    #     print("joint_pos_target:", asset._joint_pos_target_sim[0].detach().cpu())

    # if hasattr(asset, "_joint_vel_target_sim"):
    #     print("joint_vel_target:", asset._joint_vel_target_sim[0].detach().cpu())

    # if hasattr(asset, "_joint_effort_target_sim"):
    #     print("joint_effort_target:", asset._joint_effort_target_sim[0].detach().cpu())

def reset_joints_uniform_within_limits(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """ Reset joints by sampling q within joint limits """
    asset = env.scene[asset_cfg.name]
    
    # get env_ids and reshape according to whatever the joint ids shape is
    # if the selection of joints isn't ALL of the joints (aka not a slice but a cherry picked list), 
    # add a new dimension because it would be in the wrong format
    iter_env_ids = env_ids[:, None] if asset_cfg.joint_ids != slice(None) else env_ids
    
    # get joint limits
    limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    q_min = limits[..., 0] # take index 0 of last dimension, last dimension is [min, max]
    q_max = limits[..., 1] # ... means keep everything else [:, :, 1]
    
    # uniform sample per joint per env
    q = q_min + (q_max - q_min) * torch.rand_like(q_min)
    
    # don't move
    qd = torch.zeros_like(q)
    
    # go sim
    asset.write_joint_state_to_sim(q, qd, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)    
    
def reset_robot_fixed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_w: torch.Tensor, # [7] (x,y,z,qw,qx,qy,qz)
    asset_name: str = "robot",
    ):
    
    """ Reset root to a fixed position """
    
    # define asset (robot)
    asset_cfg = SceneEntityCfg(asset_name)
    asset = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[env_ids]
    device = env_ids.device
    
    # move poses to gpu
    pose_w = torch.as_tensor(pose_w, device=device, dtype=torch.float32)
    
    # make pose and vel commands
    default_root = asset.data.default_root_state[env_ids]
    pose = default_root[:, :7].clone()
    pose[:, 0:3] = pose_w[0:3] + origins # keep per-env 
    vel = torch.zeros((len(env_ids),6), device=asset.device) # don't move
    
    asset.write_root_pose_to_sim(pose,env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel,env_ids=env_ids)
       
def reset_ptz_fixed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_w: torch.Tensor, # [7] (x,y,z,qw,qx,qy,qz)
    asset_name: str = "ptz",
    ):
    
    """ Reset root to a fixed position """
    
    # define asset (robot)
    asset_cfg = SceneEntityCfg(asset_name)
    asset = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[env_ids]
    device = env_ids.device
    # want ptz to be slightly higher
    ptz_offset = torch.tensor([0.0, 0.0, -0.15], device=device, dtype=torch.float32)
    
    # move poses to gpu
    pose_w = torch.as_tensor(pose_w, device=device, dtype=torch.float32)
    
    # make pose and vel commands
    default_root = asset.data.default_root_state[env_ids]
    pose = default_root[:, :7].clone()
    pose[:, 0:3] = pose_w[0:3] + origins + ptz_offset # keep per-env 
    vel = torch.zeros((len(env_ids),6), device=asset.device) # don't move
    
    asset.write_root_pose_to_sim(pose,env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel,env_ids=env_ids)

def reset_robot_from_3_spots(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    poses_w: torch.Tensor, # [3,7] (x,y,z,qw,qx,qy,qz)
    asset_name: str="robot",
    ):
    
    """ Reset root to one of 3 randomly sampled positions """
    
    # define asset (robot)
    asset_cfg = SceneEntityCfg(asset_name)
    asset = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[env_ids]
    device = env_ids.device
    
    # pick random pose to reset to via index
    idx = torch.randint(0, 3, (len(env_ids),), device=asset.device)
    poses_w = torch.as_tensor(poses_w, device=device, dtype=torch.float32) # move poses to gpu
    
    default_root = asset.data.default_root_state[env_ids]
    pose = default_root[:, :7].clone()
    pose[:, 0:3] = poses_w[idx, 0:3] + origins # keep per-env 
    
    vel = torch.zeros((len(env_ids),6), device=asset.device) # don't move
    
    asset.write_root_pose_to_sim(pose,env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel,env_ids=env_ids)
    
    # set ptz joint pos/vel to zero
    n = env_ids.numel()
    n_ptz_joints = asset.num_joints  # 2
    q_zero_ptz  = torch.zeros((n, n_ptz_joints), device=device)
    
    asset.set_joint_position_target(q_zero_ptz, env_ids=env_ids)
    asset.write_data_to_sim()
    
def reset_ptz_from_3_spots(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    poses_w: torch.Tensor, # [3,7] (x,y,z,qw,qx,qy,qz)
    asset_name: str="ptz",
    ):
    
    """ Reset root to one of 3 randomly sampled positions """
    
    # define asset (ptz)
    asset_cfg = SceneEntityCfg(asset_name)
    asset = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[env_ids]
    device = env_ids.device
    
    # want ptz to be slightly higher
    ptz_offset = torch.tensor([0.0, 0.0, -0.15], device=device, dtype=torch.float32)
    
    # pick random pose to reset to via index
    idx = torch.randint(0, 3, (len(env_ids),), device=asset.device)
    poses_w = torch.as_tensor(poses_w, device=device, dtype=torch.float32) # move poses to gpu
    
    default_root = asset.data.default_root_state[env_ids]

    pose = default_root[:, :7].clone()
    pose[:, 0:3] = poses_w[idx, 0:3] + origins + ptz_offset # keep per-env 
    vel = torch.zeros((len(env_ids),6), device=asset.device) # don't move
    
    # i = 0
    # print("\n--- BEFORE WRITE ---")
    # print("env_id:", env_ids[i].item())
    # print("default_root_ptz:", default_root[i].detach().cpu())
    # print("pose_written_for_ptz:", pose[i].detach().cpu())
    
    # print("ptz.data.root_quat_w:", asset.data.root_quat_w[env_ids[i]].detach().cpu())
    # print("ptz.data.joint_pos:", asset.data.joint_pos[env_ids[i]].detach().cpu())
    # print("PTZ joint pos:", asset.data.joint_pos)
    
    asset.write_root_pose_to_sim(pose,env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel,env_ids=env_ids)
    
    
    # set ptz joint pos/vel to zero
    n = env_ids.numel()
    n_ptz_joints = asset.num_joints  # 2
    q_zero_ptz  = torch.zeros((n, n_ptz_joints), device=device)
    
    asset.set_joint_position_target(q_zero_ptz, env_ids=env_ids)
    asset.write_data_to_sim()
    