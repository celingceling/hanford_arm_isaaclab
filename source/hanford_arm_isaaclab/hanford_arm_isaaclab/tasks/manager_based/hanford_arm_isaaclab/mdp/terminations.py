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
    force_threshold: float = 1e-3,
) -> torch.Tensor:
    # TerminationManager will pass env_ids, but keep default for direct calls
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