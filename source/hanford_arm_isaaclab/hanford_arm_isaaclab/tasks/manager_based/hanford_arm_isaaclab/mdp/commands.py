from collections.abc import Sequence
import torch
from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique, subtract_frame_transforms
from isaaclab.utils import configclass


class WorldFrameUniformPoseCommand(UniformPoseCommand):
    """Samples pose commands in world frame, then converts to root frame for the policy."""

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)

        # sample position in WORLD frame
        pos_w = torch.zeros(len(env_ids), 3, device=self.device)
        pos_w[:, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        pos_w[:, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        pos_w[:, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        # sample orientation
        euler = torch.zeros(len(env_ids), 3, device=self.device)
        euler[:, 0].uniform_(*self.cfg.ranges.roll)
        euler[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat_w = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        if self.cfg.make_quat_unique:
            quat_w = quat_unique(quat_w)

        # store world frame for metrics/visualization
        self.pose_command_w[env_ids, :3] = pos_w
        self.pose_command_w[env_ids, 3:] = quat_w

        # convert to root frame for policy (what IK action reads)
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            pos_w,
            quat_w,
        )
        self.pose_command_b[env_ids, :3] = pos_b
        self.pose_command_b[env_ids, 3:] = quat_b

    def _update_command(self):
        """Keep pose_command_b in sync with world frame as root moves."""
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
        )
        self.pose_command_b[:, :3] = pos_b
        self.pose_command_b[:, 3:] = quat_b

    def _update_metrics(self):
        # pose_command_w is already set in _resample_command, no conversion needed
        from isaaclab.utils.math import compute_pose_error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)


@configclass
class WorldFrameUniformPoseCommandCfg(UniformPoseCommandCfg):
    class_type: type = WorldFrameUniformPoseCommand