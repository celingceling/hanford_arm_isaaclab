from isaaclab.envs import ManagerBasedRLEnv
from .hanford_arm_isaaclab_env_cfg import HanfordArmIsaaclabEnvCfg
from .include.config import TANK_COVERAGE_LOCAL_MIN, TANK_COVERAGE_LOCAL_MAX
from .include.coverage_grid import CoverageGrid


class HanfordArmCoverageEnv(ManagerBasedRLEnv):
    """
    Subclass of ManagerBasedRLEnv that attaches a CoverageGrid to the env instance.

    Why subclass:
        ManagerBasedRLEnv has no native hook for custom state initialization.
        The grid needs to live on the env so reward/obs/termination functions
        can access it via env.coverage_grid.

    Usage in your task registration:
        Replace ManagerBasedRLEnv with HanfordArmCoverageEnv in your task entry point.

    """

    def __init__(self, cfg: HanfordArmIsaaclabEnvCfg, **kwargs):
        self.coverage_grid = None
        self.slam_bridge   = None

        super().__init__(cfg, **kwargs)

        self.coverage_grid = CoverageGrid(
            bounds=(TANK_COVERAGE_LOCAL_MIN, TANK_COVERAGE_LOCAL_MAX),
            resolution=10,
            num_envs=self.num_envs,
            device=self.device,
        )