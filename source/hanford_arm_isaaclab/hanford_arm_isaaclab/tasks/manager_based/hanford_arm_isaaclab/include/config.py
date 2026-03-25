from __future__ import annotations
from typing import Any

import torch

##
# Global Variables
##

# PROJECT_ROOT = "C:/Users/LICF/projects"
ARM_USD_PATH = "C:/Users/LICF/projects/hanford_wire_manipulator_with_camera_description/usd/robot_pit_end_effector/robot_pit_end_effector_2.usd" # hard coded
TANK_USD_PATH = "C:/Users/LICF/projects/hanford_wire_manipulator_with_camera_description/usd/tank.usd" # hard coded
PTZ_USD_PATH = "C:/Users/LICF/projects/scope89_ptz/usd/scope89_ptz/scope89_ptz.usd"

JOINT_NAMES=[ # list of joint names that the action will be mapped to
                "insert_into_pipe", "rotate_in_pipe", 
                "joint_1", "joint_2", "end_effector_joint",
                "joint_3_pulley_spin",
            ]

PTZ_JOINT_NAMES=["J1", "J2"]

EE_LIGHT_PRIM = "/World/envs/env_0/Robot/pulley_drive/SphereLight"
    
# reset root positions
POSES_W = [  # 3 poses, each [x,y,z,qw,qx,qy,qz] = 0
    [-0.55521,  0.01049,   1.10504, 1.0, 0.0, 0.0, 0.0],
    [ 1.0123,  0.41759,  1.10504, 1.0, 0.0, 0.0, 0.0],
    [ 1.671, -0.97459,  1.10504, 1.0, 0.0, 0.0, 0.0],
]

# ptz offset for reset
PTZ_OFFSET_Z = 0.9

CONTACT_BUFFER = 0.3
LIDAR_MAX_DIST = 5.0  # metres — single source of truth for sensor + reward filter

# Raw mesh extents from USD authoring space (meters)
# TANK_MESH_LOCAL_MIN = torch.tensor([-2.3989834, -1.3129291, -0.6194239], dtype=torch.float32)
# TANK_MESH_LOCAL_MAX = torch.tensor([ 2.3762163,  1.7096709,  1.4824264], dtype=torch.float32)
TANK_MESH_LOCAL_MIN = torch.tensor([-2.5, -1.4, -0.7], dtype=torch.float32)
TANK_MESH_LOCAL_MAX = torch.tensor([ 2.45,  1.8,  1.5], dtype=torch.float32)

# Tank spawn offset inside each env
TANK_SCENE_POS = torch.tensor([0.0, 0.0, 0.7], dtype=torch.float32)

# Tighter region for exploration / coverage
TANK_COVERAGE_LOCAL_MIN = TANK_MESH_LOCAL_MIN.clone()
TANK_COVERAGE_LOCAL_MAX = TANK_MESH_LOCAL_MAX.clone()

# Tank AABB local frame with 0.3 margin for CommandsCfg.ranges
TANK_LOCAL_MIN = torch.tensor([-1.682, -1.287, 0.381])
TANK_LOCAL_MAX = torch.tensor([ 2.293,  0.936, 1.882])

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _as_tensor_like(x: Any, *, device=None, dtype=None) -> torch.Tensor:
    """Convert x to a tensor on the requested device/dtype.

    If x is already a tensor, preserve it unless device/dtype override is given.
    """
    if isinstance(x, torch.Tensor):
        if device is None and dtype is None:
            return x
        return x.to(device=device if device is not None else x.device,
                    dtype=dtype if dtype is not None else x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _infer_device_dtype(*items: Any) -> tuple[torch.device | None, torch.dtype | None]:
    """Infer device/dtype from the first tensor-like argument that is a Tensor."""
    for item in items:
        if isinstance(item, torch.Tensor):
            return item.device, item.dtype
    return None, None


def _as_vec3_tensor(x: Any, *, device=None, dtype=None) -> torch.Tensor:
    """Return x as a tensor whose trailing dimension is 3.

    Accepts:
        [3]
        [N, 3]
        [E, N, 3]

    Raises:
        ValueError if the trailing dimension is not 3.
    """
    t = _as_tensor_like(x, device=device, dtype=dtype)
    if t.shape[-1] != 3:
        raise ValueError(f"Expected trailing dimension 3, got shape {tuple(t.shape)}")
    return t


def _expand_offset_like(offset: Any, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a 3-vector or compatible [..., 3] tensor to match ref.

    Examples:
        offset [3]     -> [1, 3] or [1, 1, 3] as needed
        offset [E, 3]  -> [E, 1, 3] if ref is [E, N, 3]
    """
    off = _as_vec3_tensor(offset, device=ref.device, dtype=ref.dtype)

    # Same rank already: let PyTorch broadcasting handle it
    if off.ndim == ref.ndim:
        return off

    # Common case: ref [E, N, 3], off [E, 3] -> [E, 1, 3]
    if off.ndim == ref.ndim - 1:
        return off.unsqueeze(-2)

    # Common case: off [3] -> prepend singleton dims
    while off.ndim < ref.ndim:
        off = off.unsqueeze(0)

    return off


# -----------------------------------------------------------------------------
# Generic frame transforms
# -----------------------------------------------------------------------------
# BTW NONE OF THESE DEAL WITH ROTATIONS IT'S JUST TRANSLATIONS

def world_to_env_local(pos_w: Any, env_origins: Any) -> torch.Tensor:
    """Convert world-frame positions to env-local positions.

    Args:
        pos_w:
            [3], [N,3], or [E,N,3] world-frame positions
        env_origins:
            [3] or [E,3] env origin offsets in world frame

    Returns:
        Tensor with same logical shape as pos_w, in env-local frame.
    """
    device, dtype = _infer_device_dtype(pos_w, env_origins)
    pos_w_t = _as_vec3_tensor(pos_w, device=device, dtype=dtype)
    origins_t = _expand_offset_like(env_origins, pos_w_t)
    return pos_w_t - origins_t


def env_local_to_world(pos_local: Any, env_origins: Any) -> torch.Tensor:
    """Convert env-local positions to world-frame positions."""
    device, dtype = _infer_device_dtype(pos_local, env_origins)
    pos_t = _as_vec3_tensor(pos_local, device=device, dtype=dtype)
    origins_t = _expand_offset_like(env_origins, pos_t)
    return pos_t + origins_t


def world_to_tank_local(pos_w: Any, env_origins: Any, tank_scene_pos: Any = TANK_SCENE_POS) -> torch.Tensor:
    """Convert world-frame positions to tank-local positions.

    Current assumption:
        tank has no rotation relative to env frame, only translation.

    tank-local = world - env_origin - tank_scene_pos
    """
    device, dtype = _infer_device_dtype(pos_w, env_origins, tank_scene_pos)
    pos_w_t = _as_vec3_tensor(pos_w, device=device, dtype=dtype)
    origins_t = _expand_offset_like(env_origins, pos_w_t)
    tank_t = _expand_offset_like(tank_scene_pos, pos_w_t)
    return pos_w_t - origins_t - tank_t


def tank_local_to_world(pos_local: Any, env_origins: Any, tank_scene_pos: Any = TANK_SCENE_POS) -> torch.Tensor:
    """Convert tank-local positions to world-frame positions.

    Current assumption:
        tank has no rotation relative to env frame, only translation.
    """
    device, dtype = _infer_device_dtype(pos_local, env_origins, tank_scene_pos)
    pos_t = _as_vec3_tensor(pos_local, device=device, dtype=dtype)
    origins_t = _expand_offset_like(env_origins, pos_t)
    tank_t = _expand_offset_like(tank_scene_pos, pos_t)
    return pos_t + origins_t + tank_t


# -----------------------------------------------------------------------------
# Convenience helpers for AABBs
# -----------------------------------------------------------------------------

def bounds_contains(points: Any, bounds_min: Any, bounds_max: Any) -> torch.Tensor:
    """Return bool mask for whether points lie inside an AABB.

    points can be [3], [N,3], [E,N,3], etc.
    Returns mask with trailing xyz dimension removed.
    """
    device, dtype = _infer_device_dtype(points, bounds_min, bounds_max)
    pts = _as_vec3_tensor(points, device=device, dtype=dtype)
    mn = _expand_offset_like(bounds_min, pts)
    mx = _expand_offset_like(bounds_max, pts)
    return ((pts >= mn) & (pts <= mx)).all(dim=-1)