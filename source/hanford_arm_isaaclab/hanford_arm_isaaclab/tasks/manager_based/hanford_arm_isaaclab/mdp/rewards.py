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


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def collision_reward(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor = None,
    asset_name: str = "robot",
    base_penalty: float =  -1.0,
    force_scale: float = -0.001,
    force_threshold: float = 1e-3,
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
    col_flag = (max_force > force_threshold).to(dtype=torch.float32)  # [N]

    # compute penalty
    penalties = (base_penalty * col_flag) + (force_scale * max_force.to(device=device))
    
    # ensure dtype float32
    return penalties.to(dtype=torch.float32)


def coverage_gain_placeholder(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Calls grid.mark() and returns [num_envs] float: 1.0 if new cell, else 0.0.
    (marks current EE cell)

    IMPORTANT: mark() is called here, not in observations.
    Isaac Lab step order: terminations → rewards → observations.
    Rewards run before observations — this is the correct place to update grid state.

    ZED handoff: replace the mark() call with mark_from_depth(depth, cam_pose, intrinsics).
    Everything else (return value, tensor shape, RewTerm wiring) stays the same.
    """
    
    if not hasattr(env, "coverage_grid"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    robot    = env.scene["robot"]
    body_idx = robot.data.body_names.index("end_effector")
    ee_pos_w = robot.data.body_pos_w[:, body_idx, :]   # [num_envs, 3] world frame

    # ── ZED handoff point ────────────────────────────────────────────────
    # NOW:   new_cells = env.coverage_grid.mark(ee_pos_w)
    # LATER: new_cells = env.coverage_grid.mark_from_depth(depth, cam_pose, intrinsics)
    new_cells = env.coverage_grid.mark(ee_pos_w)       # [num_envs] bool
    # ─────────────────────────────────────────────────────────────────────

    return new_cells.float()

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
    
    return (~env.coverage_grid._last_new_cells).float()   # +1.0 when stagnant


