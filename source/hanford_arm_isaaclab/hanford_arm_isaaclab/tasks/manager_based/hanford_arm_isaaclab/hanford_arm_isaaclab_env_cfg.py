# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

""" Usage:  %ISAACLAB%ISAACLAB_EXE% -p scripts/rsl_rl/train.py --task=Template-Hanford-Arm-Isaaclab-v0 --headless --num_envs=1"""
"""

NOTE: this is modeled after reach_env_cfg.py from isaaclab_tasks

"""
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

from . import mdp

##
# Global Variables
##

# PROJECT_ROOT = "C:/Users/LICF/projects"
ARM_USD_PATH = "C:/Users/LICF/projects/hanford_wire_manipulator_with_camera_description/usd/robot_pit_end_effector/robot_pit_end_effector_2.usd"
# ARM_URDF_PATH = "C:/Users/LICF/projects/hanford_wire_manipulator_with_camera_description/urdf/robot_pit_end_effector_edited.urdf"
TANK_USD_PATH = "C:/Users/LICF/projects/hanford_wire_manipulator_with_camera_description/usd/tank.usd"

JOINT_NAMES=[ # list of joint names that the action will be mapped to
                "insert_into_pipe", "rotate_in_pipe", 
                "joint_1", "joint_2", "end_effector_joint",
                "joint_3_pulley_spin",
            ]

# ALL_JOINT_NAMES = JOINT_NAMES + [ 
#                    "yaw_mount_to_base", "camera_link_to_zed_x_mini",
#                    "zed_base_to_left", "zed_base_to_right", "zed_left_to_optical", 
#                    "zed_base_to_imu", "zed_right_to_optical", 
#                    "camera_link_to_pulley"
#                    ]
    
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
        activate_contact_sensors=False, # add contact sensors and set to true later
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
class CommandsCfg:
    """
        Command terms for the MDP.
        Randomly samples EE target poses
    """

    ee_pose = base_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="end_effector", # end effector
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=base_mdp.UniformPoseCommandCfg.Ranges( # EDIT RANGES LATER
            pos_x=(0.35, 0.65), # ranges, figure out later
            pos_y=(-0.2, 0.2),
            pos_z=(-0.5, -0.15),
            roll=(0.0, 0.0), # also figure out axis stuff
            pitch=(0.0, 0.0),  # lock pitch and roll for now
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """
    Action specifications for the MDP.
    Use differential IK to solve joint positions
    
    """
    
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        body_name="end_effector",
        controller=DifferentialIKControllerCfg(
            command_type="pose",        # input is [x,y,z,qw,qx,qy,qz]
            use_relative_mode=False,    # absolute targets, not deltas
            ik_method="dls",            # damped least squares - handles singularities
        ),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations.
        
        joint_pos_rel: joint positions relative to their default (zero) positions.
                       Better than absolute because it's invariant to init state.
        joint_vel_rel: joint velocities. Helps policy damp oscillations.
        pose_command:  the target EE pose [x,y,z,qw,qx,qy,qz]. 
                       Critical - policy must see its goal.
        actions:       last action taken. Helps policy be temporally consistent.
        """

        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),  # small sensor noise
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        pose_command = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "ee_pose"},
        )
        actions = ObsTerm(func=base_mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = False # i think this adds some noise during training, set as false for now
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset joints to a random fraction of their limit range.
    
    Also respawn the root from one of 3 locations.
    """
    
    # reset joint configuration in random config
    reset_root = EventTerm(
        func=mdp.reset_from_3_spots,
        mode="reset",
        params={
            "poses_w": POSES_W,
        },
    )

    # reset_joint_config = EventTerm(
    #     func=mdp.reset_joints_uniform_within_limits,
    #     mode="reset",
    # )
    
    # ok maybe this function is better than the custom one made above
    
    # reset joints by offset of their default state 
    # use for now, switch to within custom limits later
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.7),  # adds ±0.3 rad offset to default (0.0)
            "velocity_range": (0.0, 0.0),
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP.""" # ignore for now

    # (1) Constant running reward
    alive = RewTerm(func=base_mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=base_mdp.is_terminated, weight=-2.0)

# Claude's RewardsCfg, figure out later
# class RewardsCfg:
#     """Three-part reward: coarse position, fine position, orientation.
#     Penalties on action rate and joint velocity encourage smooth motion.
    
#     Negative weights = penalty (we want small error).
#     Positive weights = bonus (we want this to be large).
#     """

#     # Penalize distance to target position (linear, always active)
#     end_effector_position_tracking = RewTerm(
#         func=mdp.position_command_error,
#         weight=-0.2,
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
#             "command_name": "ee_pose",
#         },
#     )
#     # Bonus when very close to target (tanh gives strong signal near goal)
#     end_effector_position_tracking_fine_grained = RewTerm(
#         func=mdp.position_command_error_tanh,
#         weight=0.1,
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
#             "std": 0.1,
#             "command_name": "ee_pose",
#         },
#     )
#     # Penalize orientation error
#     end_effector_orientation_tracking = RewTerm(
#         func=mdp.orientation_command_error,
#         weight=-0.1,
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
#             "command_name": "ee_pose",
#         },
#     )

#     # Penalize large action changes between steps (encourages smooth motion)
#     action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

#     # Penalize high joint velocities (encourages smooth motion)
#     joint_vel = RewTerm(
#         func=mdp.joint_vel_l2,
#         weight=-0.0001,
#         params={"asset_cfg": SceneEntityCfg("robot")},
#     )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    # # (2) Collision
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )


# reach gradually increases action rate and joint velocity penalty weights over num_steps. 
# first learns to reach, then learns to move smoothly... idk
# @configclass
# class CurriculumCfg:
#     """Start with weak smoothness penalties so the policy first learns to reach,
#     then ramp them up so it learns to move smoothly.
    
#     num_steps=4500 means the weight reaches its final value after 4500 env steps.
#     """

#     action_rate = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
#     )
#     joint_vel = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500},
#     )

##
# Environment configuration
##


@configclass
class HanfordArmIsaaclabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: HanfordArmIsaaclabSceneCfg = HanfordArmIsaaclabSceneCfg(
        num_envs=16, 
        env_spacing=6.0
        )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # add curriculum later

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 2.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation