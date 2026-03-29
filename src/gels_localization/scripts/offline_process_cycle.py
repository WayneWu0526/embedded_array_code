#!/usr/bin/env python3
"""
Offline processing script for cycle JSON files.

Reads collected cycle JSON files and runs them through the localization
service to compute pose estimates.

Usage:
    # Process single file
    python offline_process_cycle.py /path/to/cycle_0000.json

    # Process directory
    python offline_process_cycle.py /path/to/result/
"""

import sys
import os
import json
import glob

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import mock classes and localization functions
from test_localization_service import (
    MockLocalizeCycleRequest,
    MockSlotData,
    MockSensorReading,
    MockPose,
)
from localization_service_node import handle_localize_cycle, load_configuration


def json_to_request(json_path):
    """
    Convert a cycle JSON file to a MockLocalizeCycleRequest.

    Args:
        json_path: path to cycle JSON file

    Returns:
        MockLocalizeCycleRequest
    """
    with open(json_path) as f:
        data = json.load(f)

    header = data['header']

    # Build slot_data
    slot_data = []
    for sd in data['slot_data']:
        sensors = [
            MockSensorReading(
                id=r['id'],
                x=r['x'],
                y=r['y'],
                z=r['z']
            )
            for r in sd['sensor_data']
        ]

        pose = MockPose()
        if 'pose' in sd:
            pose = MockPose(
                x=sd['pose']['position']['x'],
                y=sd['pose']['position']['y'],
                z=sd['pose']['position']['z'],
                qx=sd['pose']['rotation']['x'],
                qy=sd['pose']['rotation']['y'],
                qz=sd['pose']['rotation']['z'],
                qw=sd['pose']['rotation']['w']
            )

        slot_data.append(MockSlotData(
            slot=sd['slot'],
            sensor_data=sensors,
            pose=pose
        ))

    # Build ground_truth_pose
    gt = data['ground_truth_pose']
    gt_pose = MockPose(
        x=gt['position']['x'],
        y=gt['position']['y'],
        z=gt['position']['z'],
        qx=gt['rotation']['x'],
        qy=gt['rotation']['y'],
        qz=gt['rotation']['z'],
        qw=gt['rotation']['w']
    )

    return MockLocalizeCycleRequest(
        cycle_id=header['cycle_id'],
        mode=header['mode'],
        sensor_ids=header['sensor_ids'],
        slot_data=slot_data,
        ground_truth_pose=gt_pose
    )


def process_file(json_path):
    """
    Process a single cycle JSON file and print results.

    Args:
        json_path: path to cycle JSON file

    Returns:
        bool: True if localization succeeded
    """
    print(f"\n{'='*60}")
    print(f"Processing: {json_path}")

    req = json_to_request(json_path)
    print(f"Cycle ID: {req.cycle_id}, Mode: {req.mode}")
    print(f"Sensors: {req.sensor_ids}")
    print(f"Slots: {[s.slot for s in req.slot_data]}")

    resp = handle_localize_cycle(req)

    if resp.success:
        p = resp.localization_pose.position
        q = resp.localization_pose.orientation
        print(f"\nResult: success=True")
        print(f"  Position: ({p.x:.6f}, {p.y:.6f}, {p.z:.6f})")
        print(f"  Orientation: ({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})")
        print(f"  Position Error: {resp.position_error:.6f} m")
        print(f"  Orientation Error: {resp.orientation_error:.4f} rad ({resp.orientation_error * 180 / 3.14159:.4f} deg)")
    else:
        print(f"\nResult: success=False")

    return resp.success


def main():
    # Default path
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'sensor_data_collection', 'result', 'cycle_0000.json'
    )
    default_path = os.path.normpath(default_path)

    if len(sys.argv) < 2:
        path = default_path
        print(f"No path provided, using default: {path}")
    else:
        path = sys.argv[1]

    # Load configuration once
    print("Loading localization configuration...")
    load_configuration()
    print()

    if os.path.isfile(path):
        # Single file
        process_file(path)
    elif os.path.isdir(path):
        # Directory - process all cycle_*.json files
        pattern = os.path.join(path, 'cycle_*.json')
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"No cycle_*.json files found in {path}")
            sys.exit(1)

        print(f"Found {len(files)} cycle files")

        success_count = 0
        for f in files:
            if process_file(f):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"Processed {len(files)} files, {success_count} succeeded")
    else:
        print(f"Invalid path: {path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
