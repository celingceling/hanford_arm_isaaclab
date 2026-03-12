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

POSITIONS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0)
)

def reset_from_3_spots(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    poses_w: torch.Tensor, # [3,7] (x,y,z,qw,qx,qy,qz)
    asset_name: str,
    ):
    
    """ Reset root to one of 3 randomly sampled positions """
    
    # define asset (robot)
    asset_cfg = SceneEntityCfg(asset_name)
    asset = env.scene[asset_cfg.name]
    
    # pick random pose to reset to via index
    idx = torch.randint(0, 3, (len(env_ids),), device=asset.device)
    poses_w = poses_w.to(device=asset.device) # move poses to gpu
    pose = poses_w[idx].clone()
    pose[:, 0:3] += env.scene.env_origins[env_ids] # keep per-env origin behavior (aka taking relative pos to global)
    # note: does this mean each env has the same random config, or is each env unique and random
    # answer: no because idx is a env_id length tensor (see above), so each is different index
    
    vel = torch.zeros((len(env_ids),6), device=asset.device) # don't move
    
    asset.write_root_pose_to_sim(pose,env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel,env_ids=env_ids)
    
def reset_multi_from_3_spots(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    poses_w: torch.Tensor, # [3,7] (x,y,z,qw,qx,qy,qz)
    asset_names: list[str] = ["robot", "ptz"],
    ):
    
    """ Reset root to one of 3 randomly sampled positions """
    
    device = env_ids.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    poses_w = poses_w.to(device=device, dtype=torch.float32)
    n = env_ids.numel()
    origins = env.scene.env_origins[env_ids]  # [n,3]
    ptz_offset = torch.tensor([0.0, 0.0, -0.15], device=device, dtype=torch.float32)
    
    # define assets
    robot = env.scene[asset_names[0]]
    ptz= env.scene[asset_names[1]]
    
    default_root_robot = robot.data.default_root_state[env_ids]  # [n, 13]
    default_root_ptz = ptz.data.default_root_state[env_ids]  # [n, 13]
    
    # pick random pose to reset to
    idx_robot = torch.randint(0, 3, (n,), device=device)
    idx_ptx = torch.randint(0, 2, (n,), device=device) # note only picks 0 or 1
    
    idx_ptx = idx_ptx + (idx_ptx >= idx_robot).to(idx_ptx.dtype) # if idx_1 is same or greater than idx_0, flags true and + 1
    # will never index out of bounds bc idx_1 only goes up to 2 

    # initialize as default state
    pose_robot = default_root_robot[:, :7].clone() 
    pose_ptz = default_root_ptz[:, :7].clone() 
    
    # keep per-env origin behavior (aka taking relative pos to global)
    pose_robot[:, 0:3] = origins + poses_w[idx_robot, 0:3]
    pose_ptz[:, 0:3] = origins + poses_w[idx_ptx, 0:3] + ptz_offset
    
    # zero velocity
    vel = torch.zeros(n,6, device=device)
    
    # set robot joint pos/vel to zero
    n_robot_joints = robot.num_joints  # 7 i think i forgot 
    q_zero_robot  = torch.zeros((n, n_robot_joints), device=device)
    qd_zero_robot = torch.zeros((n, n_robot_joints), device=device)
    
    # set ptz joint pos/vel to zero
    n_ptz_joints = ptz.num_joints  # 2
    q_zero_ptz  = torch.zeros((n, n_ptz_joints), device=device)
    qd_zero_ptz = torch.zeros((n, n_ptz_joints), device=device)
    
    # set TARGETS to zero so it doesn't generate corrective torque (reset actuators)
    robot.set_joint_position_target(q_zero_robot, env_ids=env_ids) 
    robot.write_data_to_sim() # send target buffer ^^^ to sim
    
    ptz.set_joint_position_target(q_zero_ptz, env_ids=env_ids)
    ptz.write_data_to_sim()
    
    # write states to sim
    robot.write_root_pose_to_sim(pose_robot,env_ids=env_ids)
    robot.write_root_velocity_to_sim(vel,env_ids=env_ids)
    # robot.write_joint_state_to_sim(q_zero_robot, qd_zero_robot, env_ids=env_ids)

    ptz.write_root_pose_to_sim(pose_ptz,env_ids=env_ids)
    ptz.write_root_velocity_to_sim(vel,env_ids=env_ids)
    # ptz.write_joint_state_to_sim(q_zero_ptz, qd_zero_ptz, env_ids=env_ids)
    


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
    