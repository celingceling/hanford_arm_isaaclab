# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Global Variables
##

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ARM_USD_PATH = str(PROJECT_ROOT / "hanford_wire_manipulator_with_camera_description" / "usd" / "robot_pit_end_effector" / "robot_pit_end_effector_2.usd")
TANK_USD_PATH = str(PROJECT_ROOT / "hanford_wire_manipulator_with_camera_description" / "usd" / "tank.usd")

JOINT_NAMES=[ # list of joint names that the action will be mapped to
                "insert_into_pipe", "rotate_in_pipe", 
                "joint_1", "joint_2", "joint_3_pulley_spin",
                "end_effector_joint",
            ]


##
# Configuration
##

"""Configuration for the Hanford Manipulator Arm"""
ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ARM_USD_PATH,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"insert_into_pipe": 0.0, 
                   "rotate_in_pipe": 0.0, 
                    "joint_1": 0.0, 
                    "joint_2": 0.0, 
                    "joint_3_pulley_spin": 0.0,
                    "end_effector_joint": 0.0,}, # fill with zeros
    ),
    actuators={ # joint names and properties taken from the usd in USD_PATH,ignore root joint bc don't intend to command
        "arm": ImplicitActuatorCfg(
            joint_names_expr=JOINT_NAMES,
        ), # don't set effort, stiffness, and damping, use usd defined
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
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9,0.9,0.9), 
            intensity=100.0
            ),
    )

    # robot
    robot: ArticulationCfg = ARM_CFG.spawn.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # tank
    tank_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/tank",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TANK_USD_PATH,
            scale=(1.0,1.0,1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0,0.0,0.0),
        ),
    )



##
# MDP settings
##


@configclass
class ActionsCfg:
    """
    Action specifications for the MDP.
    
    For this project, the action is the joint effort applied for each actuated joint.
    """

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot_pit_end_effector", # uhh double check this
        joint_names=JOINT_NAMES, 
        scale=100.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """
        Observations for policy group.
        
        joint_pos_vel(7x3=21), joint_vel_rel(7*3=21) = 42 obs terms
        """

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel) # extract EE position and vel from these
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset joint configuration in random config
    reset_joint_config = EventTerm(
        func=mdp.reset_from_3_spots,
        mode="reset",
        params={
            "poses_w": poses_w,
        },
    )

    # reset arm position randomly
    poses_w = torch.zeros((3, 7), dtype=torch.float32)  # 3 poses, each [x,y,z,qw,qx,qy,qz] = 0
    poses_w[:, 3] = 1.0  # no rotations, qw = 1, qx=qy=qz=0
    
    reset_joint_config = EventTerm(
        func=mdp.reset_joints_uniform_within_limits,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )


##
# Environment configuration
##


@configclass
class HanfordArmIsaaclabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: HanfordArmIsaaclabSceneCfg = HanfordArmIsaaclabSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation