#!/usr/bin/env python3
"""
Extract sensor array measurements transformed to Diana source frame.

This script processes cycle JSON files and transforms:
1. Sensor array position -> in Diana source frame
2. b_hat_locals (magnetic field estimate) -> in Diana source frame

The reference source frame is taken from cycle_0000, slot 0 (Diana).

Usage:
    python extract_source_frame_data.py
"""

import sys
import os
import glob
import numpy as np

# Add scripts directory to path for imports
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

# Import from offline modules
from offline_utils import json_to_request, quaternion_to_rotation_matrix
from localization_service_node import load_configuration, process_hall_data, build_D_cal, run_localization


def transform_to_source_frame(ground_truth_pose, source_pose, b_hat_locals, R_est=None):
    """
    Transform sensor array position and b_hat_locals to source frame.

    Args:
        ground_truth_pose: MockPose - sensor array pose in world frame
        source_pose: MockPose - source (Diana) pose in world frame
        b_hat_locals: np.ndarray (3, M) - magnetic field estimates in sensor frame
        R_est: np.ndarray (3, 3) - rotation from sensor frame to world frame (from MaPS)

    Returns:
        p_in_source: np.ndarray (3,) - sensor array position in source frame
        b_in_source: np.ndarray (3, M) - b_hat_locals in source frame
    """
    # Extract positions
    p_sensor = np.array([
        ground_truth_pose.position.x,
        ground_truth_pose.position.y,
        ground_truth_pose.position.z
    ])
    p_source = np.array([
        source_pose.position.x,
        source_pose.position.y,
        source_pose.position.z
    ])

    # Source rotation matrix (world frame)
    R_source = quaternion_to_rotation_matrix(
        source_pose.orientation.x,
        source_pose.orientation.y,
        source_pose.orientation.z,
        source_pose.orientation.w
    )

    # Sensor array position in source frame
    # p_in_source = R_source^T @ (p_sensor - p_source)
    p_in_source = R_source.T @ (p_sensor - p_source)

    # b_hat_locals from sensor frame to source frame
    # b_hat_locals is in sensor frame, R_est @ b_hat_locals goes to world frame
    # Then R_source^T @ (R_est @ b_hat_locals) goes to source frame
    if R_est is not None:
        b_in_source = R_source.T @ R_est @ b_hat_locals
    else:
        b_in_source = R_source.T @ b_hat_locals

    return p_in_source, b_in_source


def to_cylindrical(p, b):
    """
    Convert from Cartesian to cylindrical coordinates for axisymmetric source.

    Args:
        p: np.ndarray (3,) - position in source frame (x, y, z)
        b: np.ndarray (3,) or (3,1) - magnetic field in source frame

    Returns:
        r: radial distance = sqrt(x^2 + y^2)
        z: axial distance = z
        Br: radial magnetic field component
        Bz: axial magnetic field component
    """
    p = p.flatten()
    b = b.flatten() if b.ndim > 1 else b

    r = np.sqrt(p[0]**2 + p[1]**2)
    z = p[2]

    # Radial direction unit vector (in xy plane, from source axis)
    if r > 1e-10:
        e_r = np.array([p[0], p[1], 0]) / r
    else:
        e_r = np.array([1, 0, 0])  # arbitrary at origin

    # Br = b · e_r (projection onto radial direction)
    Br = np.dot(b, e_r)
    # Bz = b · e_z (z component)
    Bz = b[2]

    return r, z, Br, Bz


def process_cycle(json_path, ref_source_pose):
    """
    Process a single cycle JSON file and transform measurements to reference source frame.

    Args:
        json_path: path to cycle JSON file
        ref_source_pose: reference source pose for the source frame

    Returns:
        dict with cycle_id, slot, p_in_source, b_in_source
    """
    # Load request from JSON
    req = json_to_request(json_path)

    # Build sources and B_meas_cell manually to get R_est
    mode = req.mode
    sensor_ids = list(req.sensor_ids)
    slot_dict = {slot.slot: slot for slot in req.slot_data}

    # Extract B0 for CVT mode (slot 3)
    B0 = None
    if mode == 'CVT' and 3 in slot_dict:
        filtered_B0_data = [r for r in slot_dict[3].sensor_data if r.id in sensor_ids]
        B0 = process_hall_data(filtered_B0_data)

    # Build sources and B_meas_cell (skip B0 slot for CVT)
    sources = []
    B_meas_cell = []

    for slot in req.slot_data:
        if slot.slot == 3:  # Skip B0 slot
            continue

        # Source pose -> m_Ci
        from offline_utils import quaternion_to_rotation_matrix as q_to_r
        p_Ci = np.array([
            slot.pose.position.x,
            slot.pose.position.y,
            slot.pose.position.z
        ])
        quat = np.array([
            slot.pose.orientation.w,
            slot.pose.orientation.x,
            slot.pose.orientation.y,
            slot.pose.orientation.z
        ])
        from offline_utils import quaternion_z_axis
        m_Ci = quaternion_z_axis(quat)
        sources.append({'p_Ci': p_Ci, 'm_Ci': m_Ci})

        # Magnetic field measurement
        filtered_sensor_data = [r for r in slot.sensor_data if r.id in sensor_ids]
        B_meas = process_hall_data(filtered_sensor_data)

        # Background subtraction for CVT
        if mode == 'CVT' and B0 is not None:
            B_meas = B_meas - B0

        B_meas_cell.append(B_meas)

    D_cal = build_D_cal(sensor_ids)

    # Run localization to get R_est
    R_est, p_est, quat_est, details = run_localization(D_cal, sources, B_meas_cell)

    # Only process slot 2 (the effective slot)
    target_slot = 2
    slot_data = req.slot_data[target_slot]

    # Get b_hat_locals for slot 2 (column index = 2)
    b_hat = details['b_hat_locals'][:, 2:3]  # slot 2 -> column 2

    # Transform to reference source frame
    # p_in_source = R_source^T @ (p_sensor - p_source)
    # b_in_source = R_source^T @ R_est @ b_hat_locals
    p_in_source, b_in_source = transform_to_source_frame(
        req.ground_truth_pose,
        ref_source_pose,
        b_hat,
        R_est
    )

    # Store world frame positions for distance verification
    p_sensor_world = np.array([
        req.ground_truth_pose.position.x,
        req.ground_truth_pose.position.y,
        req.ground_truth_pose.position.z
    ])
    p_source_world = np.array([
        slot_data.pose.position.x,
        slot_data.pose.position.y,
        slot_data.pose.position.z
    ])

    return {
        'cycle_id': req.cycle_id,
        'slot': slot_data.slot,
        'p_in_source': p_in_source,
        'b_in_source': b_in_source,
        'source_name': sources[2].get('name', f'slot{target_slot}') if 2 < len(sources) else f'slot{target_slot}',
        'p_in_source_world': p_sensor_world,
        'source_position_world': p_source_world
    }


def main():
    # Default path to result directory
    # SCRIPTS_DIR = /.../gels_localization/scripts
    # os.path.dirname(os.path.dirname(SCRIPTS_DIR)) = /.../gels_localization/src
    # result = /.../src/sensor_data_collection/result
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(SCRIPTS_DIR)),  # src
        'sensor_data_collection', 'result'
    )
    result_dir = os.path.normpath(result_dir)

    # Find cycle JSON files
    pattern = os.path.join(result_dir, 'cycle_*.json')
    files = sorted(glob.glob(pattern))

    if len(files) < 3:
        print(f"Expected 3 cycle files, found {len(files)} in {result_dir}")
        sys.exit(1)

    # Take first 3 files
    files = files[:3]
    print(f"Processing {len(files)} cycle files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # Load configuration
    print("\nLoading localization configuration...")
    load_configuration()

    # Get reference source pose from first file, slot 2 (the effective slot)
    print("\nUsing cycle_0000 slot 2 as reference source frame...")
    ref_req = json_to_request(files[0])
    ref_source_pose = ref_req.slot_data[2].pose  # slot 2 is the effective slot

    print(f"Reference source pose:")
    print(f"  Position: ({ref_source_pose.position.x:.6f}, {ref_source_pose.position.y:.6f}, {ref_source_pose.position.z:.6f})")
    print(f"  Orientation: ({ref_source_pose.orientation.x:.4f}, {ref_source_pose.orientation.y:.4f}, "
          f"{ref_source_pose.orientation.z:.4f}, {ref_source_pose.orientation.w:.4f})")

    # Process all cycles
    all_results = []
    for f in files:
        print(f"\nProcessing {os.path.basename(f)}...")
        result = process_cycle(f, ref_source_pose)
        all_results.append(result)

    # Print results
    print("\n" + "="*80)
    print("RESULTS: 3 measurements (one per cycle) transformed to Diana source frame")
    print("="*80)

    print(f"\n{'Cycle':<8} {'Slot':<6} {'Source':<10} {'p_in_source (m)':<45} {'b_in_source (T)':<45}")
    print("-"*110)

    # Also compute cylindrical coordinates for each result
    cylindrical_results = []
    for r in all_results:
        p = r['p_in_source']
        b = r['b_in_source']
        p_str = f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})"
        b_str = f"({b[0,0]:.6e}, {b[1,0]:.6e}, {b[2,0]:.6e})" if b.shape[1] == 1 else f"({b[0]:.6e}, {b[1]:.6e}, {b[2]:.6e})"
        print(f"{r['cycle_id']:<8} {r['slot']:<6} {r['source_name']:<10} {p_str:<45} {b_str:<45}")

        # Compute cylindrical coordinates
        cyl = to_cylindrical(p, b)
        cylindrical_results.append(cyl)

    # Print cylindrical results
    print("\n" + "="*80)
    print("CYLINDRICAL COORDINATES (r, z, Br, Bz)")
    print("="*80)
    print(f"{'Cycle':<8} {'r (m)':<14} {'z (m)':<14} {'Br (T)':<18} {'Bz (T)':<18}")
    print("-"*70)
    for i, r in enumerate(all_results):
        cyl = cylindrical_results[i]
        print(f"{r['cycle_id']:<8} {cyl[0]:<14.6f} {cyl[1]:<14.6f} {cyl[2]:<18.6e} {cyl[3]:<18.6e}")

    # Verify distance consistency: ||p_in_source|| should equal distance in world frame
    # For cycle 0, source and ref_source are the same, so it should match
    # For cycles 1/2, p_in_source is relative to ref_source (cycle 0), so we compute world dist to ref_source
    print("\n" + "="*80)
    print("DISTANCE VERIFICATION (||p_in_source|| should equal world frame distance)")
    print("="*80)
    print(f"{'Cycle':<8} {'||p_in_source|| (m)':<22} {'world dist to ref (m)':<22} {'diff':<18}")
    print("-"*75)
    for i, r in enumerate(all_results):
        p_src = r['p_in_source']
        p_norm = np.linalg.norm(p_src)

        # World frame distance from sensor to reference source
        p_sensor = np.array([r['p_in_source_world'][0], r['p_in_source_world'][1], r['p_in_source_world'][2]])
        # Use reference source position (cycle_0000 slot 2) for consistency
        ref_source_pos = np.array([all_results[0]['source_position_world'][0],
                                    all_results[0]['source_position_world'][1],
                                    all_results[0]['source_position_world'][2]])
        world_dist = np.linalg.norm(p_sensor - ref_source_pos)

        diff = abs(p_norm - world_dist)
        print(f"{r['cycle_id']:<8} {p_norm:<22.10f} {world_dist:<22.10f} {diff:<18.2e}")

    # Compute statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)

    p_values = np.array([r['p_in_source'] for r in all_results])
    b_values = np.array([r['b_in_source'].flatten() for r in all_results])
    cyl_values = np.array(cylindrical_results)

    print(f"\nPosition in source frame (Cartesian):")
    print(f"  Mean: {np.mean(p_values, axis=0)}")
    print(f"  Std:  {np.std(p_values, axis=0)}")

    print(f"\nMagnetic field in source frame (Cartesian):")
    print(f"  Mean: {np.mean(b_values, axis=0)}")
    print(f"  Std:  {np.std(b_values, axis=0)}")

    print(f"\nCylindrical coordinates:")
    print(f"  r  mean: {np.mean(cyl_values[:, 0]):.6f} m,  std: {np.std(cyl_values[:, 0]):.6f} m")
    print(f"  z  mean: {np.mean(cyl_values[:, 1]):.6f} m,  std: {np.std(cyl_values[:, 1]):.6f} m")
    print(f"  Br mean: {np.mean(cyl_values[:, 2]):.6e} T, std: {np.std(cyl_values[:, 2]):.6e} T")
    print(f"  Bz mean: {np.mean(cyl_values[:, 3]):.6e} T, std: {np.std(cyl_values[:, 3]):.6e} T")

    # Save results to file
    output_path = os.path.join(result_dir, 'source_frame_data.txt')
    with open(output_path, 'w') as f:
        f.write("Sensor Array Measurements in Diana Source Frame\n")
        f.write("="*80 + "\n\n")
        f.write(f"Reference source frame: cycle_0000, slot 2\n\n")

        f.write("Raw data (Cartesian):\n")
        f.write(f"{'Cycle':<8} {'Slot':<6} {'Source':<10} {'p_x':<14} {'p_y':<14} {'p_z':<14} {'b_x':<14} {'b_y':<14} {'b_z':<14}\n")
        for r in all_results:
            p = r['p_in_source']
            b = r['b_in_source'].flatten()
            f.write(f"{r['cycle_id']:<8} {r['slot']:<6} {r['source_name']:<10} "
                    f"{p[0]:<14.6f} {p[1]:<14.6f} {p[2]:<14.6f} "
                    f"{b[0]:<14.6e} {b[1]:<14.6e} {b[2]:<14.6e}\n")

        f.write("\nCylindrical coordinates:\n")
        f.write(f"{'Cycle':<8} {'r (m)':<14} {'z (m)':<14} {'Br (T)':<18} {'Bz (T)':<18}\n")
        for i, r in enumerate(all_results):
            cyl = cylindrical_results[i]
            f.write(f"{r['cycle_id']:<8} {cyl[0]:<14.6f} {cyl[1]:<14.6f} {cyl[2]:<18.6e} {cyl[3]:<18.6e}\n")

        f.write("\nStatistics:\n")
        f.write(f"r  mean: {np.mean(cyl_values[:, 0]):.6f} m,  std: {np.std(cyl_values[:, 0]):.6f} m\n")
        f.write(f"z  mean: {np.mean(cyl_values[:, 1]):.6f} m,  std: {np.std(cyl_values[:, 1]):.6f} m\n")
        f.write(f"Br mean: {np.mean(cyl_values[:, 2]):.6e} T, std: {np.std(cyl_values[:, 2]):.6e} T\n")
        f.write(f"Bz mean: {np.mean(cyl_values[:, 3]):.6e} T, std: {np.std(cyl_values[:, 3]):.6e} T\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
