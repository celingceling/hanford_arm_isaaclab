# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

""" 
python scripts/skrl/train.py --task=Hanford-Arm-Isaaclab-v1 --num_envs=1 --headless
python scripts/skrl/play.py --task=Hanford-Arm-Isaaclab-v1 --num_envs=16

NOTE: this is modeled after reach_env_cfg.py from isaaclab_tasks

"""
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg

import isaacsim.core.prims as prims

from . import mdp
from .include.coverage_grid import CoverageGrid
from .include.util import *


##
# Configuration
##

"""Configuration for the Hanford Manipulator Arm"""
ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ARM_USD_PATH,
        activate_contact_sensors=True, # add contact sensors and set to true later
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"insert_into_pipe": 0.0,
                   "rotate_in_pipe": 0.0, 
                    "joint_1": 0.0, 
                    "joint_2": 0.0, 
                    "joint_3_pulley_spin": 0.0,
                    "end_effector_joint": 0.0,},
    ),
    actuators={ # joint names and properties taken from the usd in USD_PATH,ignore root joint bc don't intend to command
        "arm": ImplicitActuatorCfg(
            joint_names_expr=JOINT_NAMES,
            stiffness={ # stiffness and damping values tuned in GUI in robot usd and then copied here
                "insert_into_pipe": 20000.0,
                "rotate_in_pipe":  15000.0,
                "joint_1":         500.0,
                "joint_2":         10000.0,
                "joint_3_pulley_spin": 6.52348,
                "end_effector_joint":  144.0,
            },
            damping={
                "insert_into_pipe": 400.0,
                "rotate_in_pipe":  0.3,
                "joint_1":          100.0,
                "joint_2":          100.0,
                "joint_3_pulley_spin": 5,
                "end_effector_joint":  5,
            },
            effort_limit_sim={
                "insert_into_pipe": 100,
                "rotate_in_pipe":  100,
                "joint_1":          100,
                "joint_2":          100,
                "joint_3_pulley_spin": 100,
                "end_effector_joint":  100,
            }
        ),
    },
)

"""Configuration for the Scope89 Pan-Tilt-Zoom (PTZ) camera"""
PTZ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PTZ_USD_PATH,
        activate_contact_sensors=False, 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            enable_gyroscopic_forces=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "J1": 0.0, # Pan
            "J2": 0.0, # Tilt
            },
        rot=(0.0, 0.0, 1.0, 0.0)
    ),
    actuators={
        "ptz": ImplicitActuatorCfg(
            joint_names_expr=PTZ_JOINT_NAMES,
            stiffness={
                "J1": 1000.0, # Pan
                "J2": 1000.0, # Tilt
                },
            damping={
                "J1": 100.0, # Pan
                "J2": 100.0, # Tilt
                },
            effort_limit_sim={
                "J1": 100.0, # Pan
                "J2": 100.0, # Tilt
                },
        ),
    },
)

##
# Scene definition
##

@configclass
class HanfordArmIsaaclabSceneCfg(InteractiveSceneCfg):
    """Configuration for a HanfordArmIsaacLab scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0)
            ),
    )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9,0.9,0.9), 
            intensity=70.0
            ),
    )

    # assets
    robot: ArticulationCfg = ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ptz: ArticulationCfg = PTZ_CFG.replace(prim_path="{ENV_REGEX_NS}/PTZ")
    
    # tank
    tank_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tank",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TANK_USD_PATH,
            scale=(1.0,1.0,1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0,0.0,0.7),
        ),
    )
    
    # lidar -- mount at same link as ZED 
    # not really sure what most of the properties do exactly but this seems like a reasonable range :D
    lidar: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pulley_drive/camera_link",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32,
            vertical_fov_range=(-30.0, 30.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=2.0,
        ),
        max_distance=LIDAR_MAX_DIST,
        mesh_prim_paths=["{ENV_REGEX_NS}/Tank"],
        drift_range=(0.0, 0.0),
        debug_vis=False,
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """
    INTENTIONALLY EMPTY.
    ManagerBasedRLEnv unconditionally constructs CommandManager — do not delete this class.
    Coverage policy is not goal-conditioned; no command terms needed.
    
    ^^ at least according to claude but i don't think it hurts
    
    """
    pass



@configclass
class ActionsCfg:
    """
    Action specifications for the MDP.
    
    Policy outputs EE pose delta. Use Differential IK to solve for necessary joint positions.
    Can change IK method later but this is just easy rn
    
    """
    
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        body_name="end_effector",
        controller=DifferentialIKControllerCfg(
            command_type="pose",        # input is [x,y,z,qw,qx,qy,qz]
            use_relative_mode=True,     # want EE delta
            ik_method="dls",            # damped least squares - handles singularities
        ),
        scale=0.1, # step size
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        debug_vis=False,
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations.
        
        Dimensions (concatenated):
            joint_pos          : [num_envs, 6]
            joint_vel          : [num_envs, 6]
            actions            : [num_envs, 7]   (EE delta pose)
            arm_collision_flag : [num_envs, 2]
            ee_pose_world      : [num_envs, 7]   (xyz + wxyz)
            ptz_state          : [num_envs, 2]   (pan, tilt)
            slam_state         : [num_envs, 64]  (zeros until SLAM integrated)
            coverage_grid      : [num_envs, 1000]
            ──────────────────────────────────────
            Total              : [num_envs, 1095]
            
        """

        # Robot state
        joint_pos          = ObsTerm(func=base_mdp.joint_pos_rel,
                                     noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel          = ObsTerm(func=base_mdp.joint_vel_rel,
                                     noise=Unoise(n_min=-0.01, n_max=0.01))
        actions            = ObsTerm(func=base_mdp.last_action)
        arm_collision_flag = ObsTerm(func=mdp.collision_observation,
                                     params={"asset_name": "robot"})

        # EE world pose — needed for dense reward shaping and future ZED work
        ee_pose_world = ObsTerm(func=mdp.get_ee_pose_world,
                                params={"body_name": "end_effector"})

        # PTZ state — interface frozen now; used when PTZ enters decision loop
        ptz_state = ObsTerm(func=mdp.get_ptz_state)

        # SLAM state placeholder — zeros until RTAB-Map integrated
        # i do not know what state_dim means
        slam_state = ObsTerm(func=mdp.slam_state_placeholder,
                             params={"state_dim": 64})

        # Coverage grid — READ ONLY here.
        # mark() is called in coverage_gain_placeholder() (reward side).
        # Isaac Lab step order: terminations → rewards → observations.
        # Reward runs before obs — grid is already updated when this reads it.
        coverage_grid = ObsTerm(func=mdp.get_coverage_grid)

        # REMOVED: pose_command — env is no longer goal-conditioned
        # pose_command = ObsTerm(func=base_mdp.generated_commands, ...)

        def __post_init__(self):
            self.enable_corruption  = False
            self.concatenate_terms  = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """
    Events that happen upon reset.
    
    """
    
    # reset PTZ and robot roots randomly in one of the 3 ports and also joints back to default
    reset_roots = EventTerm(
        func=mdp.reset_multi_from_3_spots,
        mode="reset",
        params={
            "poses_w": POSES_W,
            "asset_names": ["robot","ptz"],
        },
    )
    
    # # reset joints to zero state
    # reset_robot_joints = EventTerm(
    #     func=base_mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    # reset_ptz_joints = EventTerm(
    #     func=base_mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("ptz"),
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    
    # Clear coverage grid and no-progress history each episode
    reset_coverage_buffer = EventTerm(
        func=mdp.reset_coverage_buffer,
        mode="reset",
    )

@configclass
class RewardsCfg:
    """
    Reward terms for the MDP.
    
    Removed alive and terminating rewards to encourage safe idling.
    
    """

    # reward new grid cell discovery, also marks the cell
    coverage_gain = RewTerm(
        func=mdp.coverage_gain_placeholder,
        weight = 0.4,
    )
    
    # smoothness penalties
    action_rate = RewTerm(
        func=base_mdp.action_rate_l2,
        weight = -0.005,
    )
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2, 
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot")
        }
    )
    
    # Collisions
    arm_collision = RewTerm(
        func=mdp.collision_reward, 
        weight = 1.0, # positive bc collision_reward() already returns negative
        params={"asset_name": "robot"}
    )
    
    # stagnation penalty (when no new cell is found)  # this kinda depends on coverage_gain firing first and that is kinda sketchy
    stagnation = RewTerm(
        func=mdp.stagnation_penalty,
        weight = -0.8,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(
        func=base_mdp.time_out, 
        time_out=True
    )
    
    # # (2) Collision # change to penalty instead of terminate
    # collided = DoneTerm(
    #     func=mdp.check_collision, 
    #     time_out=False,
    #     params={
    #         "force_threshold": 5.0,
    #     }
    # )
    
    # # (3) No progress there is something wrong with this, terminates too early. something wrong with window steps, does not reset upon reset
    # no_progress = DoneTerm(
    #     func=mdp.no_progress_termination,
    #     params={
    #         "min_coverage_gain": 0.005,
    #         "window_steps": 500,
    #     },
    #     time_out=False,
    # )



##
# Environment configuration
##


@configclass
class HanfordArmIsaaclabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: HanfordArmIsaaclabSceneCfg = HanfordArmIsaaclabSceneCfg(num_envs=16, env_spacing=6.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2 # number of physics steps per control, control (policy + env.step()) runs at 60 hz
        self.episode_length_s = 6.0
        # viewer settings
        self.viewer.eye = (3.20865, 4.14945, 9.11065)
        # simulation settings
        self.sim.dt = 1 / 120 # physics runs at 120 hz
        self.sim.render_interval = self.decimation