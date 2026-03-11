import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import isaacsim.core.prims as prims

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_angle_axis

def make_ptz_debug_visualizers(device: str):
    """Create visualization markers — call ONCE before the sim loop."""

    ee_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/ee_marker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
    )

    ptz_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/ptz_marker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
    )

    # Arrow: thin cylinder pointing along +X (long axis = X via scale)
    # CylinderGeom default long axis is Y, so we rotate 90° around Z in the marker itself
    # by making it very thin in Y/Z and long in X via a mesh offset — easiest is just
    # to use a flat cone pointing along +Z and accept we orient it via the quaternion.
    arrow_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/ptz_arrow",
            markers={
                "arrow": sim_utils.CylinderCfg(
                    radius=0.02,
                    height=0.6,
                    axis="X",  # long axis along X so quaternion rotation is intuitive
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 1.0)),
                ),
            },
        )
    )
    
    # Actual camera pointing direction — orange cylinder
    actual_arrow_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/ptz_actual_arrow",
            markers={
                "arrow": sim_utils.CylinderCfg(
                    radius=0.005,
                    height=0.6,
                    axis="X",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.0)),
                ),
            },
        )
    )

    return ee_marker, ptz_marker, arrow_marker, actual_arrow_marker


def update_ptz_debug_vis(ee_marker, 
                         ptz_marker, 
                         arrow_marker, 
                         actual_arrow_marker,
                         ee_pos_w, 
                         ptz_pos_w, 
                         pan, 
                         tilt, 
                         ptz, 
                         device):
    
    """Update marker positions/orientations each step."""
    
    N = ee_pos_w.shape[0]
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(N, -1)


    # -- EE sphere --
    ee_marker.visualize(translations=ee_pos_w, orientations=identity_quat)

    # -- PTZ sphere --
    ptz_marker.visualize(translations=ptz_pos_w, orientations=identity_quat)

    # -- Arrow: rotate to show desired pointing direction --
    # Arrow asset points along +X by default.
    # We need a quaternion that rotates +X toward the desired direction.
    # Desired direction from pan/tilt (PTZ faces -Y world at pan=0):
    #   forward_x = -sin(pan) * cos(tilt)   (pan rotates in XY around Z)
    #   forward_y = -cos(pan) * cos(tilt)
    #   forward_z = -sin(tilt)              (tilt sign: negative = up)
    cos_tilt = torch.cos(tilt)
    desired_dir = torch.stack([
        -torch.sin(pan) * cos_tilt,
        -torch.cos(pan) * cos_tilt,
        -torch.sin(tilt),
    ], dim=1)  # [N, 3], unit vector in world frame

    # Build quaternion from +X -> desired_dir via axis-angle
    x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(N, -1)
    # cross product gives rotation axis
    rot_axis = torch.linalg.cross(x_axis, desired_dir)  # [N, 3]
    rot_axis_norm = torch.norm(rot_axis, dim=1, keepdim=True).clamp(min=1e-6)
    rot_axis = rot_axis / rot_axis_norm
    # dot product gives cos(angle)
    dot = (x_axis * desired_dir).sum(dim=1).clamp(-1.0, 1.0)  # [N]
    angle = torch.acos(dot)  # [N]
    arrow_quat = quat_from_angle_axis(angle, rot_axis)  # [N, 4] wxyz

    arrow_marker.visualize(translations=ptz_pos_w, orientations=arrow_quat)
    
    # -- Actual camera direction from Tilt_Link body pose --
    cam_body_idx = ptz.find_bodies("Tilt_Link")[0]
    cam_pos_w  = ptz.data.body_pos_w[:, cam_body_idx, :]   # (N,3)
    cam_quat_w = ptz.data.body_quat_w[:, cam_body_idx, :]  # (N,4)
    
    if cam_pos_w.ndim == 3:
        cam_pos_w = cam_pos_w[:, 0, :]
    if cam_quat_w.ndim == 3:
        cam_quat_w = cam_quat_w[:, 0, :]

    # Tilt_Link local forward axis — need to know which axis the camera faces.
    # Start with -Y (common convention); adjust if the orange arrow looks wrong.
    cam_forward_local = torch.tensor([[0.0, -1.0, 0.0]], device=device).expand(N, -1)
    from isaaclab.utils.math import quat_apply
    cam_forward_w = quat_apply(cam_quat_w, cam_forward_local)  # [N, 3]

    # Rotate +X -> cam_forward_w for the cylinder marker
    rot_axis2 = torch.linalg.cross(x_axis, cam_forward_w)
    rot_axis2 = rot_axis2 / torch.norm(rot_axis2, dim=1, keepdim=True).clamp(min=1e-6)
    dot2 = (x_axis * cam_forward_w).sum(dim=1).clamp(-1.0, 1.0)
    angle2 = torch.acos(dot2)
    actual_quat = quat_from_angle_axis(angle2, rot_axis2)

    actual_arrow_marker.visualize(translations=cam_pos_w, orientations=actual_quat)