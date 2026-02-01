#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import yaml
import os

def quat2mat(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def main():
    path_file = '/home/zhang/embedded_array_ws/src/triple_arm_task/config/paths/current_path.csv'
    config_file = '/home/zhang/embedded_array_ws/src/triple_arm_task/config/start_positions.yaml'
    
    if not os.path.exists(path_file) or not os.path.exists(config_file):
        print(f"Error: path_file or config_file not found.")
        return

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    diana7_start = config['start_positions']['diana7']
    start_pos = np.array(diana7_start['position'])
    start_quat = diana7_start['orientation']
    R_base = quat2mat(start_quat)

    points = []
    quats = []
    with open(path_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append([float(row['x']), float(row['y']), float(row['z'])])
            quats.append([float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])])
    points = np.array(points)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='red', s=100, label='Start Position (Center)')
    
    axis_len = 0.05
    colors = ['r', 'g', 'b'] # X, Y, Z
    labels = ['Base-X', 'Base-Y', 'Base-Z']
    for i in range(3):
        axis_vec = R_base[:, i] * axis_len
        ax.quiver(start_pos[0], start_pos[1], start_pos[2], 
                  axis_vec[0], axis_vec[1], axis_vec[2], 
                  color=colors[i], label=labels[i], arrow_length_ratio=0.1)

    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'k--', alpha=0.5, label='Path Trajectory')
    
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], color='green', s=100, marker='o', label='Trajectory StartPoint')
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color='orange', s=100, marker='*', label='Trajectory EndPoint (Tip)')

    step = max(1, len(points) // 10)
    for i in range(0, len(points), step):
        p = points[i]
        R = quat2mat(quats[i])
        # Tool axes: Red (X), Green (Y), Blue (Z)
        ax.quiver(p[0], p[1], p[2], R[0,0]*0.02, R[1,0]*0.02, R[2,0]*0.02, color='red', alpha=0.5, label='TCP X' if i==0 else "")
        ax.quiver(p[0], p[1], p[2], R[0,1]*0.02, R[1,1]*0.02, R[2,1]*0.02, color='green', alpha=0.5, label='TCP Y' if i==0 else "")
        ax.quiver(p[0], p[1], p[2], R[0,2]*0.03, R[1,2]*0.03, R[2,2]*0.03, color='blue', alpha=0.8, label='TCP Z' if i==0 else "")

    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_zlabel('World Z')
    ax.set_title('Path Visualization (World Frame)')
    ax.legend()

    all_pts = np.vstack([points, start_pos])
    max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), 
                         all_pts[:,1].max()-all_pts[:,1].min(), 
                         all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
    mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
    mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
    mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 强制 3D 箱体长宽高比例一致，防止视觉拉伸
    ax.set_box_aspect([1, 1, 1])

    output_png = '/home/zhang/embedded_array_ws/src/triple_arm_task/config/paths/trajectory_preview.png'
    plt.savefig(output_png)
    print(f"Success: Visualization saved to {output_png}")

if __name__ == '__main__':
    main()
