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
try:
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('triple_arm_task')
except:
    # Fallback to current dir if rospack fails in some envs
    pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Use rospy.myargv() to filter out ROS-internal arguments (__name, __log, etc.)
    my_argv = rospy.myargv(sys.argv)
    parser = argparse.ArgumentParser(description="Generate scanning paths based on start positions.")
    parser.add_argument('--type', type=str, default='cubic', choices=['cubic', 'screw', 'infinite', 'mesh', 'cone', 'sphere', 'rotation', 'path1', 'path2', 'path3'],
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
        # Fallback for preview generation when file doesn't exist? No, we need start pos.
        sys.exit(1)

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
        # 1. Cubic Path (Depth along Local Z)
        L, W, D = 0.1, 0.1, 0.1 # unit: meters
        waypoints_rel = [
            np.array([L/2, W/2, 0]),   np.array([-L/2, W/2, 0]),
            np.array([-L/2, W/2, D]),  np.array([L/2, W/2, D]),
            np.array([L/2, -W/2, D]),  np.array([-L/2, -W/2, D]),
            np.array([-L/2, -W/2, 0]), np.array([L/2, W/2, 0])
        ]
        points = [start_pos + R_base @ p for p in waypoints_rel]
        write_csv('current_path.csv', points)

    elif args.type == 'screw':
        # 2. Screw Path (Helix along Local Z)
        t = np.linspace(0, 4*np.pi, 200)
        radius = 0.05
        pitch = 0.02
        points = []
        for ti in t:
            p_rel = np.array([radius * np.cos(ti), radius * np.sin(ti), pitch * ti])
            points.append(start_pos + R_base @ p_rel)
        write_csv('current_path.csv', points)

    elif args.type == 'infinite':
        # 3. Infinite Path (Lemniscate in Local XY plane)
        t = np.linspace(0, 2*np.pi, 200)
        a = 0.15
        points = []
        for ti in t:
            den = 1 + np.sin(ti)**2
            x_rel = (a * np.cos(ti)) / den
            y_rel = (a * np.sin(ti) * np.cos(ti)) / den
            p_rel = np.array([x_rel, y_rel, 0])
            points.append(start_pos + R_base @ p_rel)
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
                    # Apply R_base for transformation
                    p_rel = np.array([x, y, z])
                    points.append(start_pos + R_base @ p_rel)
        
        print(f"Generated mesh path (aligned) with {len(points)} points ({N}x{N}x{N} grid)")
        write_csv('current_path.csv', points)

    elif args.type == 'cone':
        # 5. Cone Path: Spiral from Base to Tip
        # Start at Base (R=8cm, Z=-10cm) -> End at Tip (R=0, Z=0)
        # Sequence adjusted: Starts 10cm further back and ends at the robot's initial position.
        N = 200    # Increased points
        M = 10     # Increased turns
        R = 0.08   # Base Radius
        Z_start = -0.1 # Starting depth (-10cm)
        Z_end = 0.0    # Ending depth (Initial position)
        
        points = []
        quaternions = []
        
        for i in range(N):
            t = i / (N - 1)
            curr_r = R * (1 - t) # Shrinks to 0
            curr_z = Z_start + t * (Z_end - Z_start) # Moves from -10cm to 0
            theta = 2.0 * np.pi * M * t
            
            p_rel = np.array([curr_r * np.cos(theta), curr_r * np.sin(theta), curr_z])
            points.append(start_pos + R_base @ p_rel)
            
            # Orientation: Local Z axis points to the vertex at [0, 0, Z_end]
            # Since Z_end = 0, target is [0,0,0]
            v_z_local = np.array([0, 0, Z_end]) - p_rel
            if np.linalg.norm(v_z_local) < 1e-6:
                v_z_local = np.array([0.0, 0.0, 1.0]) # At Tip: Aligns with Start Pose Z
            else:
                v_z_local /= np.linalg.norm(v_z_local)
            
            # X-Axis constraints: "Tip pose should match robot start position"
            # Using [0, -1, 0] as reference to ensure Identity rotation at Tip
            ref_vec = np.array([0.0, -1.0, 0.0]) 
            
            v_x_local = np.cross(v_z_local, ref_vec)
            if np.linalg.norm(v_x_local) < 1e-6:
                v_x_local = np.cross(v_z_local, np.array([1.0, 0.0, 0.0]))
            v_x_local /= np.linalg.norm(v_x_local)
            
            # Y = Z cross X
            v_y_local = np.cross(v_z_local, v_x_local)
            
            Rot_local = np.column_stack([v_x_local, v_y_local, v_z_local])
            quaternions.append(mat2quat(R_base @ Rot_local))
            
        print(f"Generated Spiral Cone Path (Base to Tip). Points: {N}")
        write_csv('current_path.csv', points, quaternions)

    elif args.type == 'sphere':
        # 6. Sphere Path: Hemispherical Spiral
        # Requirements: Start at Equator (high Phi), Spiral to Pole (Phi=0)
        # Pole is at [0,0,0] (Robot Start Pos)
        N = 200  # Points
        M = 8    # Turns
        R = 0.1  # Radius of sphere curvature
        
        # Phi: 0 is Pole (+Z), Pi/2 is Equator
        # We start at phi_start (e.g. 60 deg) and go to 0.
        # But we need to map Sphere Pole to Local Origin [0,0,0]
        # Standard sphere centered at [0,0,0]: pole at [0,0,R]
        # We want Pole at [0,0,0], so Center is at [0,0,-R]
        
        phi_start = np.pi / 3  # 60 degrees latitude
        phi_end = 0.0          # Pole (0 degrees)
        
        points = []
        quaternions = []
        
        for i in range(N):
            t = i / (N - 1)
            phi = phi_start + t * (phi_end - phi_start) # Decreasing phi
            theta = 2.0 * np.pi * M * t
            
            # Parametric equation relative to Center [0,0,-R]
            # Surface Point P_surf = Center + R * [sin(phi)cos(theta), sin(phi)sin(theta), cos(phi)]
            # Z component: -R + R*cos(phi) = R(cos(phi)-1). 
            # Check: when phi=0, z=0 (Correct). when phi=90, z=-R.
            
            x_rel = R * np.sin(phi) * np.cos(theta)
            y_rel = R * np.sin(phi) * np.sin(theta)
            z_rel = R * np.cos(phi) - R
            
            p_rel = np.array([x_rel, y_rel, z_rel])
            points.append(start_pos + R_base @ p_rel)
            
            # Orientation: Keep TCP Z pointing to Tip (Start Pos [0,0,0])
            # This is equivalent to pointing "Normal to surface" for a concave scan?
            # Or "Focus at Point"? 
            # User logic has been "Look at Tip". Tip is at [0,0,0].
            
            v_z_local = np.array([0, 0, 0]) - p_rel
            if np.linalg.norm(v_z_local) < 1e-6:
                 v_z_local = np.array([0.0, 0.0, 1.0])
            else:
                 v_z_local /= np.linalg.norm(v_z_local)
            
            # X-Axis constraint (same as Cone for consistency)
            ref_vec = np.array([0.0, -1.0, 0.0])
            v_x_local = np.cross(v_z_local, ref_vec)
            if np.linalg.norm(v_x_local) < 1e-6:
                v_x_local = np.cross(v_z_local, np.array([1.0, 0.0, 0.0]))
            v_x_local /= np.linalg.norm(v_x_local)
            v_y_local = np.cross(v_z_local, v_x_local)
            
            Rot_local = np.column_stack([v_x_local, v_y_local, v_z_local])
            quaternions.append(mat2quat(R_base @ Rot_local))
            
        print(f"Generated Sphere Path (Spiral to Tip). Points: {N}")
        write_csv('current_path.csv', points, quaternions)

    elif args.type == 'rotation':
        # 7. Rotation Spiral: Fixed Position, Spiraling Orientation
        # Position: Fixed at start_pos
        # Orientation: Z-axis spirals out from Center (0 deg) to Boundary (30 deg)
        N = 50
        M = 4
        Max_Angle = np.radians(30)
        
        points = []
        quaternions = []
        
        for i in range(N):
            t = i / (N - 1)
            phi = t * Max_Angle  # 0 -> 30 deg
            theta = 2.0 * np.pi * M * t
            
            # Position is constant (all at start_pos)
            p_rel = np.array([0.0, 0.0, 0.0])
            points.append(start_pos + R_base @ p_rel)
            
            # Orientation: Z-axis spirals around local +Z
            # In local frame:
            vz_x = np.sin(phi) * np.cos(theta)
            vz_y = np.sin(phi) * np.sin(theta)
            vz_z = np.cos(phi)
            v_z_local = np.array([vz_x, vz_y, vz_z])
            
            # X-Axis constraint for consistency
            ref_vec = np.array([0.0, -1.0, 0.0])
            v_x_local = np.cross(v_z_local, ref_vec)
            if np.linalg.norm(v_x_local) < 1e-6:
                v_x_local = np.cross(v_z_local, np.array([1.0, 0.0, 0.0]))
            v_x_local /= np.linalg.norm(v_x_local)
            v_y_local = np.cross(v_z_local, v_x_local)
            
            Rot_local = np.column_stack([v_x_local, v_y_local, v_z_local])
            quaternions.append(mat2quat(R_base @ Rot_local))
            
        print(f"Generated Rotation Spiral. Points: {N}")
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
