import torch

class CoverageGrid:
    """
    Vectorized 3D coverage grid over the arm task space.

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
        self._last_new_count = torch.zeros(num_envs, dtype=torch.float32, device=device)

    def reset(self, env_ids: torch.Tensor):
        self.grid[env_ids]            = False
        self._last_new_cells[env_ids] = False
        self._last_new_count[env_ids] = 0.0

    def pos_to_idx(self, pos: torch.Tensor) -> torch.Tensor:
        """[N, 3] world pos → [N, 3] grid indices, clamped to valid range."""
        
        # clamp point cloud poses to tank bounds
        normed = (pos - self.bounds_min) / (self.bounds_max - self.bounds_min)
        
        # convert positions to voxel grid indices
        return (normed * self.res).long().clamp(0, self.res - 1)

    def mark(self, ee_pos_w: torch.Tensor) -> torch.Tensor:
        """
        ** NO LONGER USED BC USING LIDAR DATA NOW **
        Mark cells visited by EE world position. 

        Args:
            ee_pos_w: [num_envs, 3] EE positions in world frame
        Returns:
            new_cells: [num_envs] bool — True if a new cell was marked this step
        """
        
        # get grid indicies from pos
        idx     = self.pos_to_idx(ee_pos_w)           # [num_envs, 3]
        env_ids = torch.arange(self.num_envs, device=self.device)
        xi, yi, zi = idx[:, 0], idx[:, 1], idx[:, 2] # extrapolate
        was_visited              = self.grid[env_ids, xi, yi, zi] # reads current bool at each grid cell 
        self.grid[env_ids, xi, yi, zi] = True # mark 
        self._last_new_cells     = ~was_visited # ~ is like ! in c++ (this is dumb), returns if cell was not visited
        
        return self._last_new_cells

    def as_tensor(self) -> torch.Tensor:
        """Returns [num_envs, res^3] float32 — policy observation input."""
        
        return self.grid.view(self.num_envs, -1).float()

    def coverage_pct(self) -> torch.Tensor:
        """Returns [num_envs] float — fraction of grid cells visited. For logging."""
        
        return self.grid.view(self.num_envs, -1).float().mean(dim=-1)

    def mark_from_points(self, pts_w: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Vectorized backend — no Python loop over envs.

        Args:
            pts_w:      [num_envs, N, 3] world-frame points
            valid_mask: [num_envs, N] bool — False entries are skipped entirely.

        Returns:
            new_count: [num_envs] float — newly marked cells normalized by res^3.
        """
        num_envs, N, _ = pts_w.shape
        total_cells    = float(self.res ** 3)

        # ── Flatten ───────────────────────────────────────────────────────
        pts_flat  = pts_w.reshape(-1, 3)                        # [num_envs*N, 3]
        mask_flat = valid_mask.reshape(-1)                      # [num_envs*N] bool

        env_idx = torch.arange(num_envs, device=self.device)\
                       .repeat_interleave(N)                    # [num_envs*N]

        # ── Drop invalid points entirely ──────────────────────────────────
        pts_flat = pts_flat[mask_flat]                          # [M, 3]
        env_idx  = env_idx[mask_flat]                          # [M]

        if pts_flat.shape[0] == 0:
            # No valid points this step
            self._last_new_cells = torch.zeros(num_envs, dtype=torch.bool,    device=self.device)
            self._last_new_count = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
            return self._last_new_count

        # ── Voxelize ──────────────────────────────────────────────────────
        idx_flat        = self.pos_to_idx(pts_flat)             # [M, 3]
        xi, yi, zi      = idx_flat[:, 0], idx_flat[:, 1], idx_flat[:, 2]

        # ── Deduplicate (env, xi, yi, zi) before snapshot ─────────────────
        # Without this, multiple points hitting the same unvisited voxel in
        # one step each count as a new cell — overcounting the reward.
        R      = self.res
        linear = env_idx * R**3 + xi * R**2 + yi * R + zi     # [M] unique key
        linear_u, inv = torch.unique(linear, return_inverse=True)

        env_idx_u = linear_u // R**3
        xi_u      = (linear_u % R**3) // R**2
        yi_u      = (linear_u % R**2) // R
        zi_u      =  linear_u % R
        
        was_visited = self.grid[env_idx_u, xi_u, yi_u, zi_u].clone()
        self.grid[env_idx_u, xi_u, yi_u, zi_u] = True

        # ── Count new cells per env ───────────────────────────────────────
        newly_marked    = (~was_visited).float()                # [M]
        new_count       = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        new_count.scatter_add_(0, env_idx, newly_marked)       # sum per env
        new_count       = new_count / total_cells              # normalize

        self._last_new_cells = new_count > 0
        self._last_new_count = new_count
        return new_count


    def mark_from_lidar(self, pts_w: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """LiDAR entry point — thin wrapper around mark_from_points()."""
        return self.mark_from_points(pts_w, valid_mask)
