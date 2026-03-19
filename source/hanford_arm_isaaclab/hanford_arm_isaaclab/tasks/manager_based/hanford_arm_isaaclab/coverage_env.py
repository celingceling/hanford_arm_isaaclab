from isaaclab.envs import ManagerBasedRLEnv
from .hanford_arm_isaaclab_env_cfg import HanfordArmIsaaclabEnvCfg
from .mdp.commands import TANK_LOCAL_MIN, TANK_LOCAL_MAX
from .include.coverage_grid import CoverageGrid


class HanfordArmCoverageEnv(ManagerBasedRLEnv):
    """
    Subclass of ManagerBasedRLEnv that attaches a CoverageGrid to the env instance.

    Why subclass:
        ManagerBasedRLEnv has no native hook for custom state initialization.
        The grid needs to live on the env so reward/obs/termination functions
        can access it via env.coverage_grid.
        
    i mean ok i guess

    Usage in your task registration:
        Replace ManagerBasedRLEnv with HanfordArmCoverageEnv in your task entry point.
        
        ^ figure out what this means
    """

    def __init__(self, cfg: HanfordArmIsaaclabEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Attach coverage grid — bounds reuse validated TANK_AABB from env_cfg
        self.coverage_grid = CoverageGrid(
            bounds=(TANK_LOCAL_MIN, TANK_LOCAL_MAX),
            resolution=10,           # 10x10x10 = 1000 cells — MLP-compatible
            num_envs=self.num_envs,
            device=self.device,
        )

        # SLAM bridge — None until RTAB-Map
        # When ready: self.slam_bridge = SlamStateBridge()
        # Then replace slam_state_placeholder() with slam_state_real()
        # ^^ claude
        
        self.slam_bridge = None