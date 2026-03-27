from isaaclab.envs import ManagerBasedRLEnv
from .hanford_arm_isaaclab_env_cfg import HanfordArmIsaaclabEnvCfg
from .include.util import TANK_COVERAGE_LOCAL_MIN, TANK_COVERAGE_LOCAL_MAX
from .include.coverage_grid import CoverageGrid
from .include.ptz_follow import *

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
        print("HanfordArmCoverageEnv __init__ called")
        self.coverage_grid = None
        self.slam_bridge = None

        super().__init__(cfg, **kwargs)

        self.coverage_grid = CoverageGrid(
            bounds=(TANK_COVERAGE_LOCAL_MIN, TANK_COVERAGE_LOCAL_MAX),
            resolution=10,
            num_envs=self.num_envs,
            device=self.device,
        )

        ptz_body_ids, _ = self.scene["ptz"].find_bodies("Tilt_Link")
        self._ptz_body_idx = int(ptz_body_ids[0])

        ee_body_ids, _ = self.scene["robot"].find_bodies("end_effector")
        self._ee_body_idx = int(ee_body_ids[0])

    def step(self, action):
        # print("COVERAGE ENV STEP CALLED")
        self._update_ptz_tracking()
        return super().step(action)

    def _update_ptz_tracking(self):
        # print("PTZ tracking called")
        ptz = self.scene["ptz"]
        robot = self.scene["robot"]

        ptz_pos_w = ptz.data.body_pos_w[:, self._ptz_body_idx, :]
        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_idx, :]

        ptz_action = compute_ptz_action(ptz_pos_w, ee_pos_w)
        ptz.set_joint_position_target(ptz_action)
        
        