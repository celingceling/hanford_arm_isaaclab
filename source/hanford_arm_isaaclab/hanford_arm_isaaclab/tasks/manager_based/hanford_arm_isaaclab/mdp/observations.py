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


def get_ee_pose_world(env: ManagerBasedRLEnv, body_name: str = "end_effector") -> torch.Tensor:
    """
    Returns [num_envs, 7] EE pose in world frame: [x, y, z, qw, qx, qy, qz].
    Used directly in obs and as input to mark_from_depth() when ZED integrated.
    """
    # why the hell is it in world frame
    
    robot    = env.scene["robot"]
    idx      = robot.data.body_names.index(body_name)
    pos      = robot.data.body_pos_w[:, idx, :]    # [num_envs, 3]
    quat     = robot.data.body_quat_w[:, idx, :]   # [num_envs, 4]
    return torch.cat([pos, quat], dim=-1)           # [num_envs, 7]


def get_ptz_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns [num_envs, 2] PTZ joint positions: [pan (J1), tilt (J2)].
    Interface frozen now — PTZ enters RL decision loop in a later iteration.
    """
    
    return env.scene["ptz"].data.joint_pos[:, :2]   # [num_envs, 2]


def slam_state_placeholder(env: ManagerBasedRLEnv, state_dim: int = 64) -> torch.Tensor:
    """
    STUB — returns zeros until RTAB-Map.

    state_dim must be agreed before training starts — changing it invalidates checkpoints. # i have no idea what this means

    When implemented, expected content:
        - Occupancy map summary or frontier encoding
        - Localization confidence scalar
        - Unexplored volume estimate
    """
    return torch.zeros((env.num_envs, state_dim), device=env.device, dtype=torch.float32)

def get_coverage_grid(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    READ ONLY — returns [num_envs, 1000] flat coverage grid tensor.

    IMPORTANT: do NOT call mark() here.
    Isaac Lab step order: terminations → rewards → observations.
    Reward (coverage_gain_placeholder) runs first and calls mark().
    By the time this obs term runs, the grid is already updated.

    ZED handoff: mark_from_depth() is called in coverage_gain_placeholder()
    in rewards.py
    """
    
    if not hasattr(env, "coverage_grid") or env.coverage_grid is None:
        return torch.zeros((env.num_envs, 1000), device=env.device, dtype=torch.float32)
    
    return env.coverage_grid.as_tensor()


# UNUSED FUNCTIONS FROM OLD TESTS
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
    asset_name: str = "robot",
    force_threshold: float = 1e-3,
) -> torch.Tensor:
    """
    Returns a tensor [num_envs, 2]:
      - col_flag:  1.0 if any body has net contact magnitude > threshold, else 0.0
      - max_force: maximum net contact force magnitude across bodies
    """
    asset = env.scene[asset_name]

    if not hasattr(asset, "data") or not hasattr(asset.data, "net_contact_forces_w"):
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)

    forces   = asset.data.net_contact_forces_w          # [num_envs, num_bodies, 3]
    mags     = torch.linalg.norm(forces, dim=-1)         # [num_envs, num_bodies]
    max_force, _ = mags.max(dim=-1)                      # [num_envs]
    col_flag = (max_force > force_threshold).float()     # [num_envs]

    return torch.stack([col_flag, max_force.float()], dim=-1)  # [num_envs, 2]