# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

""" Usage:  %ISAACLAB%ISAACLAB_EXE% -p scripts/rsl_rl/train.py --task=Template-Hanford-Arm-Isaaclab-v0 --headless --num_envs=1"""

import math
from pathlib import Path
import torch
from pink.tasks import FrameTask

import carb

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
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
ARM_USD_PATH = "C:/Users/LICF/projects/hanford_wire_manipulator_with_camera_description/usd/robot_pit_end_effector/robot_pit_end_effector_2.usd"
TANK_USD_PATH = "C:/Users/LICF/projects/hanford_wire_manipulator_with_camera_description/usd/tank.usd"

JOINT_NAMES=[ # list of joint names that the action will be mapped to
                "insert_into_pipe", "rotate_in_pipe", 
                "joint_1", "joint_2", "joint_3_pulley_spin",
                "end_effector_joint",
            ]

    
# reset arm position randomly
# making these private (adding _) because otherwise it thinks it's an event term 
POSES_W = torch.zeros((3, 7), dtype=torch.float32)  # 3 poses, each [x,y,z,qw,qx,qy,qz] = 0
POSES_W[0, 0:3] = torch.tensor([-0.55, 0.0, 2.0], dtype=torch.float32)
POSES_W[1, 0:3] = torch.tensor([1.012, 0.414, 2.0], dtype=torch.float32)
POSES_W[2, 0:3] = torch.tensor([1.678, -0.976, 2.0], dtype=torch.float32)
POSES_W[:, 3] = 1.0  # no rotations, qw = 1, qx=qy=qz=0

##
# Configuration
##

"""Configuration for the Hanford Manipulator Arm"""
ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ARM_USD_PATH,
        activate_contact_sensors=True,
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
                    "end_effector_joint": 0.0,}, # fill with zeros
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
            effort_limit={
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
            intensity=2000.0
            ),
    )

    # robot
    robot: ArticulationCfg = ARM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        ) # squiggly but same as example usagef
    
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



##
# MDP settings
##


@configclass
class ActionsCfg:
    """
    Action specifications for the MDP.
    
    """

    # joint_effort = mdp.JointEffortActionCfg(
    #     asset_name="robot", # uhh double check this
    #     joint_names=JOINT_NAMES, 
    #     scale=100.0)
    
    pink_ik_cfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=JOINT_NAMES,
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pipe_entry",
            show_ik_warnings=False, # note: ?
            variable_input_tasks=[
                FrameTask( # change these values
                    "end_effector", 
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                NullSpacePostureTask( # change these values also what is this and frame task
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "end_effector",
                    ],
                    controlled_joints=JOINT_NAMES,
                    gain=0.3,
                ),
            ],
            fixed_input_tasks=[], # consider fixing insert into pipe...
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")), # idk
           ),
        target_eef_link_names={
           "camera": "end_effector",
           },
        enable_gravity_compensation=True,
    )


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
    reset_root = EventTerm(
        func=mdp.reset_from_3_spots,
        mode="reset",
        params={
            "poses_w": POSES_W,
        },
    )

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



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # # (2) Collision
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )


##
# Environment configuration
##


@configclass
class HanfordArmIsaaclabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: HanfordArmIsaaclabSceneCfg = HanfordArmIsaaclabSceneCfg(
        num_envs=4096, 
        env_spacing=6.0
        )
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
        self.episode_length_s = 1.5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation