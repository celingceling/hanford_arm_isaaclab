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
    env_ids: torch.Tensor,
    asset_name: str,
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
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=device)
    env_ids = env_ids.to(device=device, dtype=torch.long)

    asset_cfg = SceneEntityCfg(asset_name)
    asset = env.scene[asset_cfg.name]

    if not hasattr(asset, "data") or not hasattr(asset.data, "net_contact_forces_w"):
        raise RuntimeError(
            f"{asset_name} missing contact sensor data. Enable activate_contact_sensors=True."
        )

    forces = asset.data.net_contact_forces_w[env_ids]  # [N, B, 3]
    mags = torch.linalg.norm(forces, dim=-1)          # [N, B]
    max_force, _ = mags.max(dim=-1)                   # [N]
    col_flag = (max_force > force_threshold).to(dtype=torch.float32)  # [N]

    # compute penalty
    penalties = (base_penalty * col_flag) + (force_scale * max_force.to(device=device))
    
    # ensure dtype float32
    return penalties.to(dtype=torch.float32)