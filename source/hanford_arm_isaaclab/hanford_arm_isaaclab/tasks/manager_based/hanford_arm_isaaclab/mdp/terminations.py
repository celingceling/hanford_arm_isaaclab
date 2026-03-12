from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    

def check_collision(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_name: str,
    force_threshold: float = 1e-3,
    ) -> torch.Tensor:
    
    """Terminate when the asset is in contact (net contact force > threshold).

    Returns:
        torch.BoolTensor [len(env_ids)] True if collision detected in that env.
    """
    # normalize env_ids
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    
    
    asset_cfg = SceneEntityCfg(asset_name)
    asset = env.scene[asset_cfg.name]
    
    # check if contact sensor enabled
    if not hasattr(asset, "data") or not hasattr(asset.data, "net_contact_forces_w"):
        raise RuntimeError(
            f"{asset_name} has no net_contact_forces_w. "
            f"Enable contact sensors (activate_contact_sensors=True) on its spawn cfg."
        )

    forces_w = asset.data.net_contact_forces_w[env_ids]  # [n, B, 3]
    mags = torch.linalg.norm(forces_w, dim=-1)           # [n, B]
    in_contact = (mags.max(dim=-1).values > force_threshold)  # [n]
    
    return in_contact