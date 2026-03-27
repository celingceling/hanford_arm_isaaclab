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
    
from ..include.util import *


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    
    # get asset (robot arm)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    
    # compute the reward based on error
    return torch.sum(torch.square(joint_pos - target), dim=1)

def collision_reward(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor = None,
    asset_name: str = "robot",
    base_penalty: float =  -1.0,
    force_scale: float = -0.001,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """
    Returns a 1D tensor [N] of collision rewards (negative penalties).
    Reward = base_penalty * col_flag  + force_scale * max_force
    (force_scale should be negative to increase penalty with force)
    """
    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    asset = env.scene[SceneEntityCfg(asset_name).name]

    if not hasattr(asset, "data") or not hasattr(asset.data, "net_contact_forces_w"):
        # return per-env scalar reward, shape [N]
        return torch.zeros((env_ids.shape[0],), device=env.device, dtype=torch.float32)

    forces = asset.data.net_contact_forces_w[env_ids]  # [N, B, 3]
    mags = torch.linalg.norm(forces, dim=-1)          # [N, B]
    max_force, _ = mags.max(dim=-1)                   # [N]
    col_flag = (max_force > force_threshold).to(dtype=torch.float32)  # [N], if contact force exceeds max force, flag true

    # compute penalty
    penalties = (base_penalty * col_flag) + (force_scale * max_force.to(device=device))
    
    # ensure dtype float32
    return penalties.to(dtype=torch.float32)


def coverage_gain_placeholder(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reads LiDAR pointcloud, marks grid cells, returns [num_envs] float reward.

    LiDAR handoff point — when swapping to ZED depth, replace the
    mark_from_lidar() call with mark_from_depth(depth, cam_pose, intrinsics).
    Everything else (return shape, RewTerm wiring) stays the same.
    """
    if not hasattr(env, "coverage_grid"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    # Get LiDAR pointcloud — [num_envs, N, 3]
    lidar_data = env.scene["lidar"].data # this contains pos, orientation, hit points, distances, etc
    # ray_hits_w: [num_envs, N, 3]
    pts_w = lidar_data.ray_hits_w
    sensor_pos = lidar_data.pos_w               # [num_envs, 3]

    N = pts_w.shape[1]
    
    if not hasattr(env, "_dbg_step"):
        env._dbg_step = 0
    env._dbg_step += 1

    
    # ── Valid mask — filter max-range misses ─────
    diff       = pts_w - sensor_pos.unsqueeze(1)    # [num_envs, N, 3]
    dist_sq    = (diff * diff).sum(dim=-1)          # [num_envs, N]
    thresh_sq  = (LIDAR_MAX_DIST - 0.05) ** 2
    valid_mask = dist_sq < thresh_sq                # [num_envs, N] bool

    # if env._dbg_step % 50 == 0:
        
    #     i = 0
    #     print("\n \n ====== OUTPUT FROM ENV", i, " ====================================")
    #     print("\n--- LIDAR RAW ---")
    #     print("step:", env._dbg_step)
    #     print("pts_w.shape:", pts_w.shape)          # expect [num_envs, N, 3]
    #     print("sensor_pos.shape:", sensor_pos.shape) # expect [num_envs, 3]
    #     print("env0 sensor_pos:", sensor_pos[i].detach().cpu())
    #     print("env0 first 5 hits:", pts_w[i, :5].detach().cpu())
    #     print("env0 any nan in hits:", torch.isnan(pts_w[i]).any().item())
    #     print("env0 min hit:", pts_w[i].amin(dim=0).detach().cpu())
    #     print("env0 max hit:", pts_w[i].amax(dim=0).detach().cpu())
        
    #     dist = torch.sqrt(dist_sq[i])
    #     print("\n--- LIDAR VALID MASK ---")
    #     print("LIDAR_MAX_DIST:", LIDAR_MAX_DIST)
    #     print("thresh_sq:", thresh_sq)
    #     print("env0 valid rays:", valid_mask[i].sum().item(), "/", valid_mask[i].numel())
    #     print("env0 min dist:", dist.min().item())
    #     print("env0 max dist:", dist.max().item())
    #     print("env0 first 10 dist:", dist[:10].detach().cpu())
    #     print("env0 first 10 valid:", valid_mask[i, :10].detach().cpu())

    # ── Deterministic stride subsample to ~2000 pts ───────────────────────
    stride  = max(1, N // 2000)
    idx     = torch.arange(0, N, stride, device=env.device)
    pts_w   = pts_w[:, idx, :]                 # [num_envs, ~2000, 3]
    valid_mask = valid_mask[:, idx]            # [num_envs, ~2000]

    # ── Mark grid, get normalized new-voxel count ─────────────────────────
    # pts_local = pts_w - origins.unsqueeze(1) - tank_pos
    # new_count = env.coverage_grid.mark_from_lidar(pts_w, valid_mask)  # [num_envs] float
    
    pts_local = world_to_tank_local(pts_w, env.scene.env_origins)
    new_count = env.coverage_grid.mark_from_lidar(pts_local, valid_mask) # filters out invalid pts
    
    # ── DEBUG PRINT STATEMENTS ─────────────────────────
    if env._dbg_step % 50 == 0:
        
        i = 0
        # pts_local_debug = world_to_tank_local(pts_w[i], env.scene.env_origins[i])
        # print("\n--- LIDAR SUBSAMPLE ---")
        # print("original N:", N)
        # print("stride:", stride)
        # print("subsampled N:", pts_w.shape[1])
        # print("env0 valid after subsample:", valid_mask[i].sum().item(), "/", valid_mask[i].numel())
        
        # print("\n--- COVERAGE REWARD OUTPUT ---")
        # print("new_count shape:", new_count.shape)
        # print("new_count first 4:", new_count[:4].detach().cpu())
        # print("coverage pct first 4:", env.coverage_grid.coverage_pct()[:4].detach().cpu())
        # print("last_new_cells first 4:", env.coverage_grid._last_new_cells[:4].detach().cpu())
        # print("last_new_count first 4:", env.coverage_grid._last_new_count[:4].detach().cpu())
    
        
        # print("local min:", pts_local_debug.min(dim=0).values)
        # print("local max:", pts_local_debug.max(dim=0).values)
        # print("inside frac:", bounds_contains(
        #     pts_local_debug, TANK_MESH_LOCAL_MIN, TANK_MESH_LOCAL_MAX
        # ).float().mean().item())
        
        
        # if (new_count == 0).all():
        #     print("\n--- WARNING: ZERO COVERAGE GAIN ---")
        #     print("env0 valid rays:", valid_mask[i].sum().item())
        #     print("env0 coverage pct:", env.coverage_grid.coverage_pct()[i].item())
            # print("env0 sensor_pos:", sensor_pos[i].detach().cpu())
            # print("env0 first 10 subsampled hits:", pts_w[i, :10].detach().cpu())
    
    return new_count

def stagnation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns [num_envs] +1.0 when no new cell was found this step.
    RewTerm weight is negative (-0.5) → net effect is a penalty.

    Reads _last_new_cells set by coverage_gain_placeholder() — reward order matters.
    coverage_gain_placeholder must appear before stagnation in RewardsCfg.
    """
    if not hasattr(env, "coverage_grid") or \
       not hasattr(env.coverage_grid, "_last_new_cells"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    
    return (~env.coverage_grid._last_new_cells).float()   # +1.0 when stagnant, make negative in env_cfg


