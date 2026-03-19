# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

""" 
python scripts\skrl\train.py --task=Template-Hanford-Arm-Isaaclab-v0 --num_envs=1 --headless

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

import isaacsim.core.prims as prims

from . import mdp

##
# Global Variables
##

# PROJECT_ROOT = "C:/Users/LICF/projects"
ARM_USD_PATH = "C:/Users/LICF/projects/hanford_wire_manipulator_with_camera_description/usd/robot_pit_end_effector/robot_pit_end_effector_2.usd" # hard coded
TANK_USD_PATH = "C:/Users/LICF/projects/hanford_wire_manipulator_with_camera_description/usd/tank.usd" # hard coded
PTZ_USD_PATH = "C:/Users/LICF/projects/scope89_ptz/usd/scope89_ptz/scope89_ptz.usd"

JOINT_NAMES=[ # list of joint names that the action will be mapped to
                "insert_into_pipe", "rotate_in_pipe", 
                "joint_1", "joint_2", "end_effector_joint",
                "joint_3_pulley_spin",
            ]

PTZ_JOINT_NAMES=["J1", "J2"]

EE_LIGHT_PRIM = "/World/envs/env_0/Robot/pulley_drive/SphereLight"

# ALL_JOINT_NAMES = JOINT_NAMES + [ 
#                    "yaw_mount_to_base", "camera_link_to_zed_x_mini",
#                    "zed_base_to_left", "zed_base_to_right", "zed_left_to_optical", 
#                    "zed_base_to_imu", "zed_right_to_optical", 
#                    "camera_link_to_pulley"
#                    ]
    
# reset arm position randomly
# making these private (adding _) because otherwise it thinks it's an event term 
POSES_W = [  # 3 poses, each [x,y,z,qw,qx,qy,qz] = 0
    [-0.554,  0.01,   2.0, 1.0, 0.0, 0.0, 0.0],
    [ 1.012,  0.414,  2.0, 1.0, 0.0, 0.0, 0.0],
    [ 1.678, -0.976,  2.0, 1.0, 0.0, 0.0, 0.0],
]

CONTACT_BUFFER = 0.3

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
        activate_contact_sensors=False, # add contact sensors and set to true later
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
            intensity=100.0
            ),
    )

    # robot
    robot: ArticulationCfg = ARM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        ) # squiggly but same as example usagef
    
    # ptz
    ptz: ArticulationCfg = PTZ_CFG.replace(
        prim_path="{ENV_REGEX_NS}/PTZ",
    )
    
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
    
    ee_pose = mdp.WorldFrameUniformPoseCommandCfg(
        asset_name="robot",
        body_name="end_effector",
        resampling_time_range=(0.75, 0.75),
        debug_vis=False,
        ranges=mdp.WorldFrameUniformPoseCommandCfg.Ranges(
            pos_x=(-1.782, 2.393),   # world frame tank bounds with 0.3 margin
            pos_y=(-1.387, 1.036),
            pos_z=(0.381, 1.882),    # tank z in world (0.7 base + extent)
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    


@configclass
class ActionsCfg:
    """
    Action specifications for the MDP.
    Use differential IK to solve joint positions.
    
    The action is the pose command.
    
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
        
        arm_collision_flag = ObsTerm(
            func=mdp.collision_observation,
            params={"asset_name": "robot"}
        )
        
        
        def __post_init__(self):
            self.enable_corruption = False # i think this adds some noise during training, set as false for now
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset joints to a random fraction of their limit range.
    
    Also respawn the root from one of 3 locations.
    
    FOR ML: resets in same spot each time
    """
    reset_roots = EventTerm(
        func=mdp.reset_multi_from_3_spots,
        mode="reset",
        params={
            "poses_w": POSES_W,
            "asset_names": ["robot","ptz"],
        },
    )
    # reset_robot_fixed = EventTerm(
    #     func=mdp.reset_robot_fixed,
    #     mode="reset",
    #     params={
    #         "asset_name": "robot",
    #         "pose_w": POSES_W[0, :],
    #     },
    # )
    
    # reset_ptz_fixed = EventTerm(
    #     func=mdp.reset_ptz_fixed,
    #     mode="reset",
    #     params={
    #         "asset_name": "ptz",
    #         "pose_w": POSES_W[2, :],
    #     },
    # )
    
    # reset joints to zero state
    reset_robot_joints = EventTerm( # probably a more direct function than this one exists
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ptz_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ptz"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    

@configclass
class RewardsCfg:
    """Reward terms for the MDP.""" # ignore for now

    # Constant running reward
    alive = RewTerm(
        func=base_mdp.is_alive, 
        weight=1.0
        )
    
    # Failure penalty
    terminating = RewTerm(
        func=base_mdp.is_terminated, 
        weight=-2.0
        )
    
    # Collisions
    arm_collision = RewTerm(
        func=mdp.collision_reward, 
        weight=-1.0,
        params={"asset_name": "robot"}
        )
    
    

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
    
    # (2) Collision
    collided = DoneTerm(func=mdp.check_collision, time_out=True)


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
        self.episode_length_s = 7.0
        # viewer settings
        self.viewer.eye = (3.20865, 4.14945, 9.11065)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation