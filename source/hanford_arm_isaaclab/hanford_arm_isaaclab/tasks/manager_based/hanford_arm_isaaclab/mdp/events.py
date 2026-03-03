# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
    
    """ Reset root to one of 3 randomly sampled positions """
    
    # define asset (robot)
    asset = env.scene[asset_cfg.name]
    
    # pick random pose to reset to via index
    idx = torch.randint(0, 3, (len(env_ids),), device=asset.device)
    pose = poses_w[idx].clone()
    pose[:, 0:3] += env.scene.env_origins[env_ids] # keep per-env origin behavior (aka taking relative pos to global)
    # note: does this mean each env has the same random config, or is each env unique and random
    # answer: no because idx is a env_id length tensor (see above), so each is different index
    
    vel = torch.zeros((len(env_ids),6), device=asset.device) # don't move
    
    asset.write_root_pose_to_sim(pose,env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel,env_ids=env_ids)

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
    