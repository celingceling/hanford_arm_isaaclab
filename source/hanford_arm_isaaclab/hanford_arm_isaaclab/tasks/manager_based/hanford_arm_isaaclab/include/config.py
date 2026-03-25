from __future__ import annotations

import torch

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
    
# reset root positions
POSES_W = [  # 3 poses, each [x,y,z,qw,qx,qy,qz] = 0
    [-0.55521,  0.01049,   1.10504, 1.0, 0.0, 0.0, 0.0],
    [ 1.0123,  0.41759,  1.10504, 1.0, 0.0, 0.0, 0.0],
    [ 1.671, -0.97459,  1.10504, 1.0, 0.0, 0.0, 0.0],
]

# ptz offset for reset
PTZ_OFFSET_Z = 0.9


CONTACT_BUFFER = 0.3
LIDAR_MAX_DIST = 5.0  # metres — single source of truth for sensor + reward filter


# Tank AABB in world frame (same values as CommandsCfg.ranges but used for
# the collision / in-tank check).  Keep in sync with your ranges.
TANK_LOCAL_MIN = torch.tensor([-1.682, -1.287, 0.381]) # margin = 0.3
TANK_LOCAL_MAX = torch.tensor([ 2.293,  0.936, 1.882])