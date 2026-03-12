# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def object_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """
    Object observations (in world frame):
        object pos,
        object quat,
        left_eef to object,
        right_eef_to object,
    """

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    return torch.cat(
        (
            object_pos,
            object_quat,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )


def get_eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    return left_eef_pos


def get_eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat


def get_robot_joint_state(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
) -> torch.Tensor:
    # hand_joint_names is a list of regex, use find_joints
    indexes, _ = env.scene["robot"].find_joints(joint_names)
    indexes = torch.tensor(indexes, dtype=torch.long)
    robot_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    return robot_joint_states


def get_all_robot_link_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w

    return all_robot_link_pos

def collision_observation(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_name: str,
    force_threshold: float = 1e-3,
) -> torch.Tensor:
    """
    Returns a tensor [N, 2]:
      - col_flag: 1.0 if any monitored body has net contact magnitude > threshold, else 0.0
      - max_force: maximum net contact force magnitude across bodies (float)
    env_ids: 1D Long tensor of environment indices to sample.
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

    # forces: [num_envs_total, num_bodies, 3]
    forces = asset.data.net_contact_forces_w[env_ids]  # selects requested envs
    # compute per-body magnitudes -> [N, B]
    mags = torch.linalg.norm(forces, dim=-1)
    # per-env maximum magnitude
    max_force, _ = mags.max(dim=-1)                     # [N]
    col_flag = (max_force > force_threshold).to(dtype=torch.float32)  # [N]

    out = torch.stack([col_flag, max_force.to(dtype=torch.float32)], dim=-1)  # [N,2]
    
    return out