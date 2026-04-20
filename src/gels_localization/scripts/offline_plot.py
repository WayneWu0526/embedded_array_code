#!/usr/bin/env python3
"""
3D visualization for offline processing.

Contains functions for plotting poses and sensor array in 3D space.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def plot_pose_axes(ax, position, quaternion, axis_length=0.05, label=None, alpha=1.0):
    """
    Plot coordinate axes at a given pose position.

    Args:
        ax: matplotlib 3D axes
        position: (x, y, z) position
        quaternion: (qx, qy, qz, qw) quaternion
        axis_length: length of axis arrows
        label: label for the legend
        alpha: transparency
    """
    x, y, z = position
    qx, qy, qz, qw = quaternion

    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)

    # X axis (red), Y axis (green), Z axis (blue)
    colors = ['red', 'green', 'blue']
    axes = np.eye(3) * axis_length

    for color, axis in zip(colors, axes):
        direction = R @ axis + np.array([x, y, z])
        ax.plot([x, direction[0]], [y, direction[1]], [z, direction[2]],
                color=color, alpha=alpha, linewidth=2)

    # Plot origin point
    if label:
        ax.scatter([x], [y], [z], color='black', s=50, label=label, zorder=5)


def plot_cylinder(ax, center, quaternion, length=0.25, diameter=0.04, color='orange', alpha=0.6):
    """
    Plot a cylinder (electromagnet coil representation) aligned with local z-axis.

    Args:
        ax: matplotlib 3D axes
        center: (x, y, z) center position of cylinder
        quaternion: (qx, qy, qz, qw) orientation
        length: length of cylinder in meters (default 0.25m = 250mm)
        diameter: diameter of cylinder in meters (default 0.04m = 40mm)
        color: cylinder color
        alpha: transparency
    """
    x, y, z = center
    qx, qy, qz, qw = quaternion
    r = diameter / 2
    half_L = length / 2

    # Get rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)

    # Local z-axis direction in world frame (cylinder main axis)
    local_z = np.array([0, 0, 1])
    z_direction = R @ local_z  # World direction of local z-axis

    # Create cylinder along local z-axis
    theta = np.linspace(0, 2 * np.pi, 24)
    theta_grid, t_grid = np.meshgrid(theta, np.linspace(-half_L, half_L, 10))

    # Base cylinder along z-axis
    X_base = r * np.cos(theta_grid)
    Y_base = r * np.sin(theta_grid)
    Z_base = t_grid

    # Rotate to align with local z-axis
    for j in range(X_base.shape[0]):
        for k in range(X_base.shape[1]):
            vec = np.array([X_base[j, k], Y_base[j, k], Z_base[j, k]])
            rotated = R @ vec
            X_base[j, k] = rotated[0] + x
            Y_base[j, k] = rotated[1] + y
            Z_base[j, k] = rotated[2] + z

    ax.plot_surface(X_base, Y_base, Z_base, color=color, alpha=alpha, linewidth=0, shade=True)


def plot_poses_comparison(req, resp=None, json_path=None, model_resp=None):
    """
    Plot all source poses, estimated pose, and ground truth in 3D.

    Args:
        req: MockLocalizeCycleRequest
        resp: MockLocalizationResponse for real data (optional)
        json_path: path to JSON file (optional, for saving plot)
        model_resp: MockLocalizationResponse for model data (optional)
    """
    from localization_service_node import handle_localize_cycle

    if resp is None:
        resp = handle_localize_cycle(req)

    fig = plt.figure(figsize=(14, 10))
    ax = Axes3D(fig)

    # Ground truth pose (black, thicker axes)
    gt = req.ground_truth_pose
    plot_pose_axes(ax, (gt.position.x, gt.position.y, gt.position.z),
                   (gt.orientation.x, gt.orientation.y, gt.orientation.z, gt.orientation.w),
                   axis_length=0.08, label='Ground Truth', alpha=1.0)

    # Source poses (diana7, arm1, arm2 from slot_data slots 0,1,2)
    source_names = ['diana7', 'arm1', 'arm2']
    for i, slot in enumerate(req.slot_data[:3]):  # First 3 slots are sources
        pose = slot.pose
        if pose.position.x != 0 or pose.position.y != 0 or pose.position.z != 0:  # Valid pose
            plot_pose_axes(ax, (pose.position.x, pose.position.y, pose.position.z),
                           (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                           axis_length=0.05, label=source_names[i], alpha=0.7)
            # Draw electromagnet cylinder: L=250mm, D=56mm, center at source position,
            # main axis aligned with local z-axis from quaternion
            plot_cylinder(ax, (pose.position.x, pose.position.y, pose.position.z),
                          (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                          length=0.25, diameter=0.056, color='orange', alpha=0.5)

    # Real estimate (magenta)
    if resp and resp.success:
        p = resp.localization_pose.position
        q = resp.localization_pose.orientation
        plot_pose_axes(ax, (p.x, p.y, p.z),
                       (q.x, q.y, q.z, q.w),
                       axis_length=0.06, label='Real Estimate', alpha=1.0)

    # Model estimate (cyan) - if provided
    if model_resp and model_resp.success:
        p = model_resp.localization_pose.position
        q = model_resp.localization_pose.orientation
        plot_pose_axes(ax, (p.x, p.y, p.z),
                       (q.x, q.y, q.z, q.w),
                       axis_length=0.06, label='Model Estimate', alpha=1.0)

    # Collect all positions for auto-scaling
    positions = [
        [gt.position.x, gt.position.y, gt.position.z],
    ]
    for slot in req.slot_data[:3]:
        pose = slot.pose
        if pose.position.x != 0 or pose.position.y != 0 or pose.position.z != 0:
            positions.append([pose.position.x, pose.position.y, pose.position.z])

    # Add estimates to scaling if available
    if resp and resp.success:
        positions.append([resp.localization_pose.position.x,
                         resp.localization_pose.position.y,
                         resp.localization_pose.position.z])
    if model_resp and model_resp.success:
        positions.append([model_resp.localization_pose.position.x,
                         model_resp.localization_pose.position.y,
                         model_resp.localization_pose.position.z])

    positions = np.array(positions)
    center = positions.mean(axis=0)
    max_dist = np.max(np.linalg.norm(positions - center, axis=1))
    padding = 0.15
    limit_range = max(max_dist * 1.5, 0.1) + padding

    ax.set_xlim([center[0] - limit_range, center[0] + limit_range])
    ax.set_ylim([center[1] - limit_range, center[1] + limit_range])
    ax.set_zlim([center[2] - limit_range, center[2] + limit_range])

    # Set viewing angle (front view: looking at y-z plane)
    ax.view_init(elev=20, azim=0)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Pose Comparison - Cycle {req.cycle_id}')
    ax.legend(loc='upper left', fontsize=8)

    # Save plot to file
    if json_path:
        output_dir = os.path.dirname(os.path.abspath(json_path))
        output_path = os.path.join(output_dir, f'pose_plot_cycle_{req.cycle_id:04d}.png')
    else:
        output_path = f'/tmp/pose_plot_cycle_{req.cycle_id:04d}.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, output_path


def plot_poses_from_json(json_path, estimated_pose=None, model_pose=None, output_path=None):
    """
    Plot source poses, estimated pose, and ground truth from a JSON file in 3D.

    Args:
        json_path: path to cycle JSON file
        estimated_pose: tuple (position, quaternion) for estimated pose.
                        Position: (x, y, z) or None to run localization automatically.
                        Quaternion: (qx, qy, qz, qw).
                        None means run MaPS localization on the JSON data.
        model_pose: tuple (position, quaternion) for model-based estimate (optional).
                    Position: (x, y, z)
                    Quaternion: (qx, qy, qz, qw)
        output_path: path to save plot. If None, saves to same directory as json_path.

    Returns:
        tuple: (fig, output_path)
    """
    import json
    from offline_utils import json_to_request
    from localization_service_node import handle_localize_cycle

    # Load JSON and convert to request
    with open(json_path) as f:
        data = json.load(f)

    req = json_to_request(json_path)

    # Run localization if no estimate provided
    if estimated_pose is None:
        resp = handle_localize_cycle(req)
        if resp.success:
            est_pos = (resp.localization_pose.position.x,
                       resp.localization_pose.position.y,
                       resp.localization_pose.position.z)
            est_quat = (resp.localization_pose.orientation.x,
                        resp.localization_pose.orientation.y,
                        resp.localization_pose.orientation.z,
                        resp.localization_pose.orientation.w)
            estimated_pose = (est_pos, est_quat)
        else:
            estimated_pose = None
    elif estimated_pose is not None:
        # estimated_pose is provided as (position, quaternion) tuple
        # Ensure it's in the right format for plotting
        pass

    fig = plt.figure(figsize=(14, 10))
    ax = Axes3D(fig)

    # Ground truth pose (black, thicker axes)
    gt = req.ground_truth_pose
    plot_pose_axes(ax,
                   (gt.position.x, gt.position.y, gt.position.z),
                   (gt.orientation.x, gt.orientation.y, gt.orientation.z, gt.orientation.w),
                   axis_length=0.08, label='Ground Truth', alpha=1.0)

    # Source poses (diana7, arm1, arm2 from slot_data slots 0,1,2)
    source_names = ['diana7', 'arm1', 'arm2']
    for i, slot in enumerate(req.slot_data[:3]):  # First 3 slots are sources
        pose = slot.pose
        if pose.position.x != 0 or pose.position.y != 0 or pose.position.z != 0:  # Valid pose
            plot_pose_axes(ax,
                           (pose.position.x, pose.position.y, pose.position.z),
                           (pose.orientation.x, pose.orientation.y,
                            pose.orientation.z, pose.orientation.w),
                           axis_length=0.05, label=source_names[i], alpha=0.7)
            # Draw electromagnet cylinder: L=250mm, D=56mm, center at source position,
            # main axis aligned with local z-axis from quaternion
            plot_cylinder(ax,
                          (pose.position.x, pose.position.y, pose.position.z),
                          (pose.orientation.x, pose.orientation.y,
                           pose.orientation.z, pose.orientation.w),
                          length=0.25, diameter=0.056, color='orange', alpha=0.5)

    # Estimated pose (magenta)
    if estimated_pose is not None:
        pos, quat = estimated_pose
        plot_pose_axes(ax, pos, quat,
                       axis_length=0.06, label='Estimated', alpha=1.0)

    # Model estimate (cyan) - if provided
    if model_pose is not None:
        pos, quat = model_pose
        plot_pose_axes(ax, pos, quat,
                       axis_length=0.06, label='Model Estimate', alpha=1.0)

    # Collect all positions for auto-scaling
    positions = [[gt.position.x, gt.position.y, gt.position.z]]
    for slot in req.slot_data[:3]:
        pose = slot.pose
        if pose.position.x != 0 or pose.position.y != 0 or pose.position.z != 0:
            positions.append([pose.position.x, pose.position.y, pose.position.z])

    # Add estimates to scaling if available
    if estimated_pose is not None:
        positions.append(list(estimated_pose[0]))
    if model_pose is not None:
        positions.append(list(model_pose[0]))

    positions = np.array(positions)
    center = positions.mean(axis=0)
    max_dist = np.max(np.linalg.norm(positions - center, axis=1))
    padding = 0.15
    limit_range = max(max_dist * 1.5, 0.1) + padding

    ax.set_xlim([center[0] - limit_range, center[0] + limit_range])
    ax.set_ylim([center[1] - limit_range, center[1] + limit_range])
    ax.set_zlim([center[2] - limit_range, center[2] + limit_range])

    # Set viewing angle (front view: looking at y-z plane)
    ax.view_init(elev=20, azim=0)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Pose Comparison - Cycle {req.cycle_id}')
    ax.legend(loc='upper left', fontsize=8)

    # Save plot to file
    if output_path is None:
        output_dir = os.path.dirname(os.path.abspath(json_path))
        cycle_id = data['header']['cycle_id']
        output_path = os.path.join(output_dir, f'pose_plot_cycle_{cycle_id:04d}.png')

    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, output_path
