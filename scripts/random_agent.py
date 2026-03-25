# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import hanford_arm_isaaclab.tasks  # noqa: F401


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    
    # DEBUG: check actual joint names
    robot = env.unwrapped.scene["robot"]
    print("Actuated joints:", robot.joint_names)
    print("Count:", robot.num_joints)
    ee_idx = robot.data.body_names.index("end_effector")

    print("=== AFTER RESET ===")
    print(f"joint_pos:    {robot.data.joint_pos[0].cpu().numpy().round(4)}")
    print(f"joint_target: {robot.data.joint_pos_target[0].cpu().numpy().round(4)}")
    print(f"pos_err:      {(robot.data.joint_pos[0] - robot.data.joint_pos_target[0]).abs().cpu().numpy().round(4)}")
    print(f"ee_pos_w:     {robot.data.body_pos_w[0, ee_idx].cpu().numpy().round(4)}")
    if hasattr(env.unwrapped, 'action_manager'):
        print(f"last_action:  {env.unwrapped.action_manager.action[0].cpu().numpy().round(4)}")
    print("=== END RESET ===")
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1

            # DEBUG: print the exact action being sent
            print(f"action:       {actions[0].cpu().numpy().round(4)}")

            env.step(actions)

        robot = env.unwrapped.scene["robot"]

        joint_pos = robot.data.joint_pos[0].cpu()
        print(f"joint_pos:    {joint_pos.numpy().round(4)}")

        joint_pos_target = robot.data.joint_pos_target[0].cpu()
        print(f"joint_target: {joint_pos_target.numpy().round(4)}")

        diff = (joint_pos - joint_pos_target).abs()
        print(f"pos_err:      {diff.numpy().round(4)}  max={diff.max().item():.4f}")

        # Contact forces
        if hasattr(robot.data, "net_contact_forces_w"):
            forces = robot.data.net_contact_forces_w[0]          # [num_bodies, 3]
            mags = torch.linalg.norm(forces, dim=-1)
            print(f"contact_mags: {mags.cpu().numpy().round(4)}  max={mags.max().item():.4f}")
        else:
            print("contact_mags: net_contact_forces_w not available")

        ee_idx = robot.data.body_names.index("end_effector")
        ee_pos = robot.data.body_pos_w[0, ee_idx].cpu()
        print(f"ee_pos_w:     {ee_pos.numpy().round(4)}")

        print("---")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
