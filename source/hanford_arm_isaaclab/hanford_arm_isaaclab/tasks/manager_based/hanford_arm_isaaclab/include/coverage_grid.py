import torch

class CoverageGrid:
    """
    Vectorized 3D coverage grid over the arm task space.
    Populated by EE position now; replaced by ZED depth unprojection later.

    ZED handoff:
        Replace mark() calls in rewards.py with mark_from_depth().
        Tensor contract ([num_envs, res^3] float32) is identical.

    Frame convention:
        All positions must be in WORLD frame.
        Grid bounds must match the frame used by ee_pos_w inputs.
        TANK_LOCAL_MIN/MAX from commands.py are numerically world-frame bounds
        when env_origins=(0,0,0) — verify this matches your scene layout.
    """

    def __init__(self, bounds, resolution=10, num_envs=1, device="cuda"):
        self.res        = resolution
        self.device     = device
        self.num_envs   = num_envs
        self.bounds_min = torch.tensor(bounds[0], device=device, dtype=torch.float32)
        self.bounds_max = torch.tensor(bounds[1], device=device, dtype=torch.float32)

        # [num_envs, res, res, res]
        self.grid = torch.zeros(
            (num_envs, resolution, resolution, resolution),
            dtype=torch.bool, device=device
        )
        # Written by reward, read by stagnation penalty and obs
        self._last_new_cells = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor):
        """Clear grid for terminated envs. Called by reset_coverage_buffer event."""
        
        self.grid[env_ids]            = False
        self._last_new_cells[env_ids] = False

    def pos_to_idx(self, pos: torch.Tensor) -> torch.Tensor:
        """[N, 3] world pos → [N, 3] grid indices, clamped to valid range."""
        
        normed = (pos - self.bounds_min) / (self.bounds_max - self.bounds_min)
        
        return (normed * self.res).long().clamp(0, self.res - 1)

    def mark(self, ee_pos_w: torch.Tensor) -> torch.Tensor:
        """
        Mark cells visited by EE world position.
        Called in coverage_gain_placeholder() (reward side — runs before obs).

        Args:
            ee_pos_w: [num_envs, 3] EE positions in world frame
        Returns:
            new_cells: [num_envs] bool — True if a new cell was marked this step
        """
        
        idx     = self.pos_to_idx(ee_pos_w)           # [num_envs, 3]
        env_ids = torch.arange(self.num_envs, device=self.device)
        xi, yi, zi = idx[:, 0], idx[:, 1], idx[:, 2]
        was_visited              = self.grid[env_ids, xi, yi, zi]
        self.grid[env_ids, xi, yi, zi] = True
        self._last_new_cells     = ~was_visited
        
        return self._last_new_cells

    def as_tensor(self) -> torch.Tensor:
        """Returns [num_envs, res^3] float32 — policy observation input."""
        
        return self.grid.view(self.num_envs, -1).float()

    def coverage_pct(self) -> torch.Tensor:
        """Returns [num_envs] float — fraction of grid cells visited. For logging."""
        
        return self.grid.view(self.num_envs, -1).float().mean(dim=-1)



    # ── ZED / Depth Handoff ────────────────────────────────────────────────
    def mark_from_depth(
        self,
        depth:      torch.Tensor,  # [num_envs, H, W] float32, metres
        cam_pose:   torch.Tensor,  # [num_envs, 4, 4] camera-to-world transform
        intrinsics: torch.Tensor,  # [3, 3] pinhole matrix (fx,0,cx / 0,fy,cy / 0,0,1)
    ) -> torch.Tensor:
        """
        STUB — implements this to replace mark().

        Unprojection math (per env, per pixel):
            X_cam = (u - cx) * d / fx
            Y_cam = (v - cy) * d / fy
            Z_cam = d
            [X_w, Y_w, Z_w, 1] = cam_pose @ [X_cam, Y_cam, Z_cam, 1]

        Then call pos_to_idx([X_w, Y_w, Z_w]) and mark grid cells.

        Frame note:
            cam_pose must transform FROM camera frame TO world frame.
            Ensure this matches the convention used in mark().

        Returns:
            new_cells: [num_envs] bool — same contract as mark()
        """
        raise NotImplementedError("Depth unprojection pending — use mark() for now.")