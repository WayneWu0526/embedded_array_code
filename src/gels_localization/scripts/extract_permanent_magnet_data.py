#!/usr/bin/env python3
"""
Extract permanent magnet measurements in world frame.

This script processes cycle JSON files and extracts:
1. Permanent magnet pose (from slot_data[slot].pose) for each slot
2. b_hat_locals transformed to world frame (R_est @ b_hat_locals)

For permanent magnet data, each cycle has 3 slots corresponding to 3 different
spatial poses of the permanent magnet. We process 5 cycles = 15 results.

Usage:
    python extract_permanent_magnet_data.py
"""

import sys
import os
import glob
import json
import numpy as np

# Add scripts directory to path for imports
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

# Import from offline modules
from offline_utils import json_to_request
from localization_service_node import load_configuration, process_hall_data, build_D_cal, run_localization


def process_cycle(json_path):
    """
    Process a single cycle JSON file and extract permanent magnet data.

    Args:
        json_path: path to cycle JSON file

    Returns:
        list of dicts with cycle_id, slot, permanent_magnet_pose, b_in_world
    """
    req = json_to_request(json_path)

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

        # Permanent magnet pose -> m_Ci (for sources list)
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

    # Run localization to get R_est and b_hat_locals
    R_est, p_est, quat_est, details = run_localization(D_cal, sources, B_meas_cell)

    # Extract results for each slot
    results = []
    for slot_obj in req.slot_data:
        if slot_obj.slot == 3:  # Skip B0 slot
            continue

        # Permanent magnet pose
        pm_pose = {
            'position': {
                'x': slot_obj.pose.position.x,
                'y': slot_obj.pose.position.y,
                'z': slot_obj.pose.position.z
            },
            'orientation': {
                'x': slot_obj.pose.orientation.x,
                'y': slot_obj.pose.orientation.y,
                'z': slot_obj.pose.orientation.z,
                'w': slot_obj.pose.orientation.w
            }
        }

        # b_hat_locals[:, slot_idx] transformed to world frame
        # b_hat_locals is (3, M) where M is number of sources
        # slot 0 -> column 0, slot 1 -> column 1, slot 2 -> column 2
        slot_idx = slot_obj.slot  # slot 0, 1, or 2
        if slot_idx < details['b_hat_locals'].shape[1]:
            b_hat_local = details['b_hat_locals'][:, slot_idx:slot_idx+1]  # shape (3, 1)
            b_in_world = R_est @ b_hat_local  # shape (3, 1)
        else:
            b_in_world = np.zeros((3, 1))

        results.append({
            'cycle_id': req.cycle_id,
            'slot': slot_obj.slot,
            'permanent_magnet_pose': pm_pose,
            'b_in_world': b_in_world.flatten(),
            'sensor_array_pose': {
                'position': {
                    'x': req.ground_truth_pose.position.x,
                    'y': req.ground_truth_pose.position.y,
                    'z': req.ground_truth_pose.position.z
                },
                'orientation': {
                    'x': req.ground_truth_pose.orientation.x,
                    'y': req.ground_truth_pose.orientation.y,
                    'z': req.ground_truth_pose.orientation.z,
                    'w': req.ground_truth_pose.orientation.w
                }
            }
        })

    return results


def main():
    # Default path to result directory
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(SCRIPTS_DIR)),  # src
        'sensor_data_collection', 'result'
    )
    result_dir = os.path.normpath(result_dir)

    # Find all cycle JSON files
    pattern = os.path.join(result_dir, 'cycle_*.json')
    files = sorted(glob.glob(pattern))

    if len(files) < 1:
        print(f"No cycle files found in {result_dir}")
        sys.exit(1)

    print(f"Processing {len(files)} cycle files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # Load configuration
    print("\nLoading localization configuration...")
    load_configuration()

    # Process all cycles
    all_results = []
    for f in files:
        print(f"\nProcessing {os.path.basename(f)}...")
        results = process_cycle(f)
        all_results.extend(results)

    # Print results
    print("\n" + "="*150)
    print("PERMANENT MAGNET DATA: 15 measurements (5 cycles × 3 slots) in world frame")
    print("="*150)

    print(f"\n{'Cycle':<6} {'Slot':<6} {'PM Position (m)':<48} {'PM Orientation (qw,qx,qy,qz)':<50} {'p_sensor (m)':<48} {'b_in_world (T)':<50}")
    print("-"*250)

    for r in all_results:
        pos = r['permanent_magnet_pose']['position']
        ori = r['permanent_magnet_pose']['orientation']
        p_sens = r['sensor_array_pose']['position']
        b = r['b_in_world']
        pos_str = f"({pos['x']:.6f}, {pos['y']:.6f}, {pos['z']:.6f})"
        ori_str = f"({ori['w']:.6f}, {ori['x']:.6f}, {ori['y']:.6f}, {ori['z']:.6f})"
        p_str = f"({p_sens['x']:.6f}, {p_sens['y']:.6f}, {p_sens['z']:.6f})"
        b_str = f"({b[0]:.6e}, {b[1]:.6e}, {b[2]:.6e})"
        print(f"{r['cycle_id']:<6} {r['slot']:<6} {pos_str:<48} {ori_str:<50} {p_str:<48} {b_str:<50}")

    # Save results to JSON file
    output_path = os.path.join(result_dir, 'permanent_magnet_data.json')

    # Build structured output
    output_data = {
        'description': 'Permanent magnet measurements in world frame',
        'note': 'All poses and positions are in world frame (lab_table reference). b is magnetic field in world frame.',
        'measurements': []
    }

    for r in all_results:
        pos = r['permanent_magnet_pose']['position']
        ori = r['permanent_magnet_pose']['orientation']
        p_sens = r['sensor_array_pose']['position']
        ori_sens = r['sensor_array_pose']['orientation']
        b = r['b_in_world']

        measurement = {
            'cycle_id': r['cycle_id'],
            'slot': r['slot'],
            'permanent_magnet_pose': {
                'position': {'x': pos['x'], 'y': pos['y'], 'z': pos['z']},
                'orientation': {'x': ori['x'], 'y': ori['y'], 'z': ori['z'], 'w': ori['w']}
            },
            'sensor_array_pose': {
                'position': {'x': p_sens['x'], 'y': p_sens['y'], 'z': p_sens['z']},
                'orientation': {'x': ori_sens['x'], 'y': ori_sens['y'], 'z': ori_sens['z'], 'w': ori_sens['w']}
            },
            'b_in_world': {'x': float(b[0]), 'y': float(b[1]), 'z': float(b[2])}
        }
        output_data['measurements'].append(measurement)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
