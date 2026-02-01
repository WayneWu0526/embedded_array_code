#!/usr/bin/env python3

import numpy as np
import os
import csv
import yaml
import rospkg
import sys
import argparse
import rospy

# Get package path
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('triple_arm_task')

def main():
    # Use rospy.myargv() to filter out ROS-internal arguments (__name, __log, etc.)
    my_argv = rospy.myargv(sys.argv)
    parser = argparse.ArgumentParser(description="Generate scanning paths based on start positions.")
    parser.add_argument('--type', type=str, default='cubic', choices=['cubic', 'screw', 'infinite', 'mesh'],
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
    start_pos = diana7_start['position']

    print(f"Generating {args.type} path...")
    print(f"Base start_pos: {start_pos}")

    def write_csv(filename, points):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
            for p in points:
                writer.writerow(list(p) + base_quat)
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

if __name__ == '__main__':
    main()
