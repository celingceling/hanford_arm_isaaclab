from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from isaaclab.envs import ManagerBasedRLEnv
    

def check_collision(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor = None,
    asset_name: str = "robot",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Returns [N] bool — True if any body contact force exceeds threshold.
    Set time_out=False in TerminationsCfg — collision is a failure, not a natural end.
    """    
    
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    asset = env.scene[SceneEntityCfg(asset_name).name]

    # IMPORTANT: during manager init, data may not exist yet. Don't raise.
    if not hasattr(asset, "data") or not hasattr(asset.data, "net_contact_forces_w"):
        return torch.zeros((env_ids.shape[0],), device=env.device, dtype=torch.bool)

    forces_w = asset.data.net_contact_forces_w[env_ids]     # [N, B, 3]
    mags = torch.linalg.norm(forces_w, dim=-1)              # [N, B]
    
    return mags.max(dim=-1).values > force_threshold        # [N]

def no_progress_termination(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor = None,
    min_coverage_gain: float = 0.005,
    window_steps: int = 1000,
) -> torch.Tensor:
    """
    Terminates envs that gained less than min_coverage_gain over the last window_steps.

    Warm-up gate: termination is suppressed until the rolling history buffer
    has been filled at least once (_coverage_step >= window_steps). This prevents
    spurious early terminations when the buffer is still mostly zeros.

    Registered with time_out=False — no progress is a soft failure.
    Tune min_coverage_gain and window_steps after observing training behavior.
    
    """
    
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    # Allocate rolling history buffer on first call
    if not hasattr(env, "_coverage_history"):
        env._coverage_history = torch.zeros(
            (env.num_envs, window_steps), device=env.device, dtype=torch.float32
        )
        env._coverage_step = 0

    # Read current coverage % from grid
    current = env.coverage_grid.coverage_pct() \
        if hasattr(env, "coverage_grid") \
        else torch.zeros(env.num_envs, device=env.device)

    # Write to rolling buffer
    slot = env._coverage_step % window_steps
    env._coverage_history[:, slot] = current
    env._coverage_step += 1
    
    
    # Warm-up gate: don't terminate until buffer has filled at least once
    if env._coverage_step < window_steps:
        return torch.zeros(env_ids.shape[0], device=env.device, dtype=torch.bool)


    # Terminate if gain over window is below threshold
    min_in_window = env._coverage_history[env_ids].min(dim=-1).values
    gain          = current[env_ids] - min_in_window
    
    return gain < min_coverage_gain   # [N] bool