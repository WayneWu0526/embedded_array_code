#!/usr/bin/env python3

import numpy as np
import os
import csv
import yaml
import rospkg
import sys
import argparse
import rospy

def mat2quat(R):
    """Convert rotation matrix to quaternion (qx, qy, qz, qw)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]

def quat2mat(q):
    """Convert quaternion (qx, qy, qz, qw) to rotation matrix."""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

# Get package path
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('triple_arm_task')

def main():
    # Use rospy.myargv() to filter out ROS-internal arguments (__name, __log, etc.)
    my_argv = rospy.myargv(sys.argv)
    parser = argparse.ArgumentParser(description="Generate scanning paths based on start positions.")
    parser.add_argument('--type', type=str, default='cubic', choices=['cubic', 'screw', 'infinite', 'mesh', 'cone', 'path1', 'path2', 'path3'],
                        help='Type of path to generate (default: cubic)')
    args = parser.parse_args(my_argv[1:])

    # Target directory
    output_dir = os.path.join(pkg_path, 'config', 'paths')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load start positions from YAML
    config_path = os.path.join(pkg_path, 'config', 'start_positions.yaml')
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found. Please run initialize_scan.py first.")
        exit(1)

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Use diana7's pose as the base for path generation
    diana7_start = config_data['start_positions']['diana7']
    base_quat = diana7_start['orientation']
    start_pos = np.array(diana7_start['position'])
    R_base = quat2mat(base_quat)

    print(f"Generating {args.type} path...")
    print(f"Base start_pos: {start_pos}")

    def write_csv(filename, points, quaternions=None):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
            for i, p in enumerate(points):
                q = quaternions[i] if quaternions is not None else base_quat
                writer.writerow(list(p) + list(q))
        print(f"Generated {filepath}")

    if args.type == 'cubic':
        # 1. Cubic Path
        L, W, H = 0.1, 0.1, 0.1 # unit: meters
        waypoints_rel = [
            np.array([L, W/2, 0]), np.array([0, W/2, 0]),
            np.array([0, W/2, H]), np.array([L, W/2, H]),
            np.array([L, -W/2, H]), np.array([0, -W/2, H]),
            np.array([0, -W/2, 0]), np.array([L, W/2, 0])
        ]
        points = [start_pos + p for p in waypoints_rel]
        write_csv('current_path.csv', points)

    elif args.type == 'screw':
        # 2. Screw Path (Helix)
        t = np.linspace(0, 4*np.pi, 200)
        radius = 0.05
        points = []
        for ti in t:
            y = start_pos[1] + 0.02 * ti
            x = start_pos[0] + radius * np.sin(ti)
            z = start_pos[2] + radius * (np.cos(ti) - 1)
            points.append([x, y, z])
        write_csv('current_path.csv', points)

    elif args.type == 'infinite':
        # 3. Infinite Path (Lemniscate)
        t = np.linspace(0, 2*np.pi, 200)
        a = 0.15
        points = []
        for ti in t:
            den = 1 + np.sin(ti)**2
            y_offset = (a * np.cos(ti)) / den
            z_offset = (a * np.sin(ti) * np.cos(ti)) / den
            points.append([start_pos[0], start_pos[1] + y_offset, start_pos[2] + z_offset])
        write_csv('current_path.csv', points)

    elif args.type == 'mesh':
        # 4. Mesh (Voxel) Path
        # Cube 20x20x20cm, start_pos is center of bottom face
        N = 10 # Internal parameter: density (points per edge)
        
        # Define ranges relative to start_pos (center of bottom face)
        x_vals = np.linspace(-0.1, 0.1, N)
        y_vals = np.linspace(-0.1, 0.1, N)
        z_vals = np.linspace(0, 0.2, N)
        
        points = []
        for iz, z in enumerate(z_vals):
            # Flip Y direction every layer to minimize travel (S-pattern in Z)
            # But usually we want to scan each "face" (XY plane) fully
            y_order = y_vals if iz % 2 == 0 else y_vals[::-1]
            for iy, y in enumerate(y_order):
                # Flip X direction every row (S-pattern in XY)
                x_order = x_vals if iy % 2 == 0 else x_vals[::-1]
                for x in x_order:
                    points.append([start_pos[0] + x, start_pos[1] + y, start_pos[2] + z])
        
        print(f"Generated mesh path with {len(points)} points ({N}x{N}x{N} grid)")
        write_csv('current_path.csv', points)

    elif args.type == 'cone':
        # 5. Spiral Cone Path: Axis is Local Z (maps to World X)
        # Apex at Local Z = +10cm (World X +10cm from center)
        # Base at Local Z = -10cm (World X -10cm from center)
        N = 30
        M = 4
        R_max = 0.075 # Max radius (Diameter 15cm)
        L = 0.15       # Total depth along Z
        
        # --- Constant Arc-Length Sampling ---
        t_fine = np.linspace(0, 1, 1000)
        z_f = 0.1 - L * t_fine # From +10cm to -10cm (Axial along Local Z)
        r_f = R_max * t_fine   # From 0 (Apex) to R_max (Base)
        theta_f = 2 * np.pi * M * t_fine
        x_f = r_f * np.cos(theta_f) # Radial
        y_f = r_f * np.sin(theta_f) # Radial
        
        points_f = np.stack([x_f, y_f, z_f], axis=1)
        diffs = np.diff(points_f, axis=0)
        arc_length = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(diffs**2, axis=1)))])
        
        s_uniform = np.linspace(0, arc_length[-1], N)
        t_reordered = np.interp(s_uniform, arc_length, t_fine)
        # -------------------------------------

        points = []
        quaternions = []
        apex_rel = np.array([0, 0, 0.1]) # Vertex point (Local Z = 0.1)

        for t in t_reordered:
            # 1. Position (Spiral starts at Apex and expands towards Base)
            theta = 2 * np.pi * M * t
            # [Radial X, Radial Y, Axial Z]
            p_rel = np.array([R_max * t * np.cos(theta), R_max * t * np.sin(theta), 0.1 - L * t])
            points.append(start_pos + R_base @ p_rel)
            
            # 2. Orientation (Local Z pointing towards the Apex [0, 0, 0.1])
            v_z_local = apex_rel - p_rel
            if np.linalg.norm(v_z_local) < 1e-6:
                v_z_local = np.array([0, 0, 1]) # At the apex
            else:
                v_z_local /= np.linalg.norm(v_z_local)
            
            # Consistent X-Y stabilization:
            # v_x = v_z cross [0, -1, 0] to ensure apex orientation matches start_position (Identity in Local Frame)
            v_x_local = np.cross(v_z_local, np.array([0.0, -1.0, 0.0]))
            if np.linalg.norm(v_x_local) < 1e-6:
                v_x_local = np.cross(v_z_local, np.array([1.0, 0.0, 0.0]))
            v_x_local /= np.linalg.norm(v_x_local)
            v_y_local = np.cross(v_z_local, v_x_local)
            
            Rot_local = np.column_stack([v_x_local, v_y_local, v_z_local])
            quaternions.append(mat2quat(R_base @ Rot_local))
            
        print(f"Fixed Cone Path (Axis along World X): Points: 30, Apex at World-X +10cm")
        write_csv('current_path.csv', points, quaternions)

    elif args.type == 'path1':
        # 6. Path 1: Hemispherical Spiral (Restore to original working version)
        N = 200  # Points
        M = 8    # Turns
        R = 0.1
        
        # --- Constant Arc-Length Sampling ---
        phi_fine = np.linspace(0, np.pi/2, 1000)
        z_f = -R * np.cos(phi_fine)
        r_f = R * np.sin(phi_fine)
        theta_f = 2 * np.pi * M * (phi_fine / (np.pi/2))
        x_f = r_f * np.cos(theta_f)
        y_f = r_f * np.sin(theta_f)
        
        points_f = np.stack([x_f, y_f, z_f], axis=1)
        diffs = np.diff(points_f, axis=0)
        arc_length = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(diffs**2, axis=1)))])
        
        s_uniform = np.linspace(0, arc_length[-1], N)
        phi_reordered = np.interp(s_uniform, arc_length, phi_fine)
        # -------------------------------------

        points = []
        quaternions = []
        
        for phi in phi_reordered:
            # 1. Position
            theta = 2 * np.pi * M * (phi / (np.pi/2))
            p_rel = np.array([R * np.sin(phi) * np.cos(theta), R * np.sin(phi) * np.sin(theta), -R * np.cos(phi)])
            points.append(start_pos + R_base @ p_rel)
            
            # 2. Orientation (Pointing to center of sphere [0,0,0])
            v_z_local = -p_rel / np.linalg.norm(p_rel) if np.linalg.norm(p_rel) > 1e-6 else np.array([0, 0, 1])
            
            v_x_local = np.cross(np.array([1.0, 0.0, 0.0]), v_z_local)
            if np.linalg.norm(v_x_local) < 1e-6:
                v_x_local = np.cross(np.array([0.0, 1.0, 0.0]), v_z_local)
            v_x_local /= np.linalg.norm(v_x_local)
            v_y_local = np.cross(v_z_local, v_x_local)
            
            Rot_local = np.column_stack([v_x_local, v_y_local, v_z_local])
            quaternions.append(mat2quat(R_base @ Rot_local))
            
        print(f"Restored Hemispherical Spiral (Path1) with {N} points.")
        write_csv('current_path.csv', points, quaternions)

    elif args.type == 'path2':
        # Placeholder for Path 2
        points = [start_pos]
        print("Path 2 generating... (Placeholder)")
        write_csv('current_path.csv', points)

    elif args.type == 'path3':
        # Placeholder for Path 3
        points = [start_pos]
        print("Path 3 generating... (Placeholder)")
        write_csv('current_path.csv', points)

if __name__ == '__main__':
    main()
