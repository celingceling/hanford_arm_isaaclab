from collections.abc import Sequence
import torch
from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique, subtract_frame_transforms, compute_pose_error
from isaaclab.utils import configclass


# ── tuneable constants ────────────────────────────────────────────────────────
MAX_RESAMPLE_TRIES  = 20    # give up after this many attempts per env
ARM_REACH_MIN       = 0.1  # [m] dead-zone around root (too close = unreachable)
ARM_REACH_MAX       = 1.50  # [m] max reach from root - tune to your arm geometry

# Tank AABB in world frame (same values as CommandsCfg.ranges but used for
# the collision / in-tank check).  Keep in sync with your ranges.
TANK_LOCAL_MIN = torch.tensor([-1.782, -1.387, 0.381])
TANK_LOCAL_MAX = torch.tensor([ 2.393,  1.036, 1.882])
# ─────────────────────────────────────────────────────────────────────────────


class WorldFrameUniformPoseCommand(UniformPoseCommand):
    """Samples pose commands in world frame with reachability + in-tank checks.

    Reachability check:
        The sampled world position must be within [ARM_REACH_MIN, ARM_REACH_MAX]
        of the robot root.  This is a sphere-shell test — cheap and geometry-free.

    In-tank check:
        The sampled world position must lie inside TANK_WORLD_MIN/MAX (AABB).
        This rejects targets that are inside the tank walls or outside the tank
        entirely.  It does NOT check fine collision geometry.

    If a valid sample is not found within MAX_RESAMPLE_TRIES the last valid
    sample (or the default zero pose) is kept so the episode can continue.
    """

    def _sample_world_pos(self, n: int) -> torch.Tensor:
        """Draw n positions uniformly from the configured world-frame ranges."""
        r = torch.empty(n, 3, device=self.device)
        r[:, 0].uniform_(*self.cfg.ranges.pos_x)
        r[:, 1].uniform_(*self.cfg.ranges.pos_y)
        r[:, 2].uniform_(*self.cfg.ranges.pos_z)
        return r

    def _get_anchor_body_idx(self) -> int:
        # IsaacLab 5.1.0: robot typically exposes body names list
        # Try these in order; one of them will exist.
        if hasattr(self.robot, "body_names"):
            names = list(self.robot.body_names)
        else:
            names = list(self.robot.data.body_names)

        return names.index("link_2")

    def _is_reachable(self, pos_w: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Return bool mask [n] — True if pos_w is within the arm's reach from LINK_2."""
        anchor_idx = self._get_anchor_body_idx()
        anchor_pos = self.robot.data.body_pos_w[env_ids, anchor_idx, :]  # [n,3]
        # root_pos = self.robot.data.root_pos_w[env_ids]          # [n, 3]
        dist = torch.norm(pos_w - anchor_pos, dim=-1)             # [n]
        return (dist >= ARM_REACH_MIN) & (dist <= ARM_REACH_MAX)

    def _is_in_tank(self, pos_w: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Return bool mask [n] — True if pos_w is inside the tank AABB."""
        origins = self._env.scene.env_origins[env_ids]              # (n,3)
        mn = TANK_LOCAL_MIN.to(self.device) + origins          # (n,3) via broadcast
        mx = TANK_LOCAL_MAX.to(self.device) + origins          # (n,3)

        return ((pos_w >= mn) & (pos_w <= mx)).all(dim=-1)     # (n,)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.tensor(env_ids, device=self.device) \
            if not isinstance(env_ids, torch.Tensor) else env_ids

        n = len(env_ids_t)

        # Buffers for accepted samples — start from current values so
        # envs that never find a valid sample keep their old command.
        accepted_pos_w  = self.pose_command_w[env_ids_t, :3].clone()
        accepted_quat_w = self.pose_command_w[env_ids_t, 3:].clone()
        pending = torch.ones(n, dtype=torch.bool, device=self.device)  # still need a sample

        for attempt in range(MAX_RESAMPLE_TRIES):
            if not pending.any():
                break

            pending_ids = env_ids_t[pending]        # env indices still pending
            pending_local = torch.where(pending)[0] # local indices into [n]
            m = pending_ids.shape[0]

            # ── sample orientation ────────────────────────────────────────
            euler = torch.zeros(m, 3, device=self.device)
            euler[:, 0].uniform_(*self.cfg.ranges.roll)
            euler[:, 1].uniform_(*self.cfg.ranges.pitch)
            euler[:, 2].uniform_(*self.cfg.ranges.yaw)
            quat_w = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
            if self.cfg.make_quat_unique:
                quat_w = quat_unique(quat_w)

            # ── sample position in world frame ────────────────────────────
            pos_w = self._sample_world_pos(m)
            pos_w = pos_w + self._env.scene.env_origins[pending_ids]  # make it env-local → world

            # ── validity checks ───────────────────────────────────────────
            in_tank = self._is_in_tank(pos_w, pending_ids)  
            reachable  = self._is_reachable(pos_w, pending_ids)
            if attempt == 0:
                print("attempt0 in_tank:", in_tank.float().mean().item(),
                    "reachable:", reachable.float().mean().item())
            valid      = in_tank & reachable

            # accept valid samples
            if valid.any():
                v_local = pending_local[valid]          # local indices of valid ones
                accepted_pos_w[v_local]  = pos_w[valid]
                accepted_quat_w[v_local] = quat_w[valid]
                pending[v_local] = False                # mark as done

        if pending.any():
            n_failed = pending.sum().item()
            print(
                f"[WorldFrameUniformPoseCommand] WARNING: {n_failed} env(s) could not find "
                f"a valid command after {MAX_RESAMPLE_TRIES} attempts. "
                f"Keeping previous command for those envs."
            )

        # ── store world-frame result ──────────────────────────────────────
        self.pose_command_w[env_ids_t, :3] = accepted_pos_w
        self.pose_command_w[env_ids_t, 3:] = accepted_quat_w

        # ── convert to root frame for the IK action ───────────────────────
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids_t],
            self.robot.data.root_quat_w[env_ids_t],
            accepted_pos_w,
            accepted_quat_w,
        )
        self.pose_command_b[env_ids_t, :3] = pos_b
        self.pose_command_b[env_ids_t, 3:] = quat_b

    def _update_command(self):
        """Re-express the fixed world-frame target in the (possibly moving) root frame."""
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
        )
        self.pose_command_b[:, :3] = pos_b
        self.pose_command_b[:, 3:] = quat_b

    def _update_metrics(self):
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"]    = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)


@configclass
class WorldFrameUniformPoseCommandCfg(UniformPoseCommandCfg):
    class_type: type = WorldFrameUniformPoseCommand

# class WorldFrameUniformPoseCommand(UniformPoseCommand):
#     """Samples pose commands in world frame, then converts to root frame for the policy."""

#     def _resample_command(self, env_ids: Sequence[int]):
#         r = torch.empty(len(env_ids), device=self.device)

#         # sample position in WORLD frame
#         pos_w = torch.zeros(len(env_ids), 3, device=self.device)
#         pos_w[:, 0] = r.uniform_(*self.cfg.ranges.pos_x)
#         pos_w[:, 1] = r.uniform_(*self.cfg.ranges.pos_y)
#         pos_w[:, 2] = r.uniform_(*self.cfg.ranges.pos_z)

#         # sample orientation
#         euler = torch.zeros(len(env_ids), 3, device=self.device)
#         euler[:, 0].uniform_(*self.cfg.ranges.roll)
#         euler[:, 1].uniform_(*self.cfg.ranges.pitch)
#         euler[:, 2].uniform_(*self.cfg.ranges.yaw)
#         quat_w = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
#         if self.cfg.make_quat_unique:
#             quat_w = quat_unique(quat_w)

#         # store world frame for metrics/visualization
#         self.pose_command_w[env_ids, :3] = pos_w
#         self.pose_command_w[env_ids, 3:] = quat_w

#         # convert to root frame for policy (what IK action reads)
#         pos_b, quat_b = subtract_frame_transforms(
#             self.robot.data.root_pos_w[env_ids],
#             self.robot.data.root_quat_w[env_ids],
#             pos_w,
#             quat_w,
#         )
#         self.pose_command_b[env_ids, :3] = pos_b
#         self.pose_command_b[env_ids, 3:] = quat_b

#     def _update_command(self):
#         """Keep pose_command_b in sync with world frame as root moves."""
#         pos_b, quat_b = subtract_frame_transforms(
#             self.robot.data.root_pos_w,
#             self.robot.data.root_quat_w,
#             self.pose_command_w[:, :3],
#             self.pose_command_w[:, 3:],
#         )
#         self.pose_command_b[:, :3] = pos_b
#         self.pose_command_b[:, 3:] = quat_b

#     def _update_metrics(self):
#         # pose_command_w is already set in _resample_command, no conversion needed
#         from isaaclab.utils.math import compute_pose_error
#         pos_error, rot_error = compute_pose_error(
#             self.pose_command_w[:, :3],
#             self.pose_command_w[:, 3:],
#             self.robot.data.body_pos_w[:, self.body_idx],
#             self.robot.data.body_quat_w[:, self.body_idx],
#         )
#         self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
#         self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)


# @configclass
# class WorldFrameUniformPoseCommandCfg(UniformPoseCommandCfg):
#     class_type: type = WorldFrameUniformPoseCommand