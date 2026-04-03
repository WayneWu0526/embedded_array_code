#!/usr/bin/env python3
"""
Offline processing script for cycle JSON files.

Reads collected cycle JSON files and runs them through the localization
service. Processes both real data and model-based synthetic data for comparison.

Usage:
    # Process single file (real + model comparison)
    python offline_process_cycle.py /path/to/cycle_0000.json

    # Process directory
    python offline_process_cycle.py /path/to/result/

    # Skip 3D plot
    python offline_process_cycle.py /path/to/cycle_0000.json --no-plot

3D Plot legend:
    X axis = red, Y axis = green, Z axis = blue
    Ground Truth (black), Sources (diana7/arm1/arm2)
    Real Estimate (magenta), Model Estimate (cyan)
"""

import sys
import os
import glob
import argparse

import numpy as np

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from separate modules
from offline_utils import (
    json_to_request,
    generate_synthetic_request_from_json,
    get_estimator_details,
    compare_model_vs_estimate,
    mock_request_to_json,
    sensitivity_analysis_x_hat_noise,
)
from offline_plot import plot_poses_comparison

# Import localization functions
from localization_service_node import handle_localize_cycle, load_configuration


def process_file(json_path, show_plot=False, run_sensitivity=False, sensor_ids=None):
    """
    Process a single cycle JSON file: run both real data and model data localization.

    Args:
        json_path: path to cycle JSON file
        show_plot: if True, generate 3D pose plot
        run_sensitivity: if True, run x_hat noise sensitivity analysis
        sensor_ids: list of sensor IDs to use (e.g. [1,3,4,6,7,9,10,12]).
                     None means use all sensors from JSON.

    Returns:
        tuple: (real_success: bool, req, resp, model_resp)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {json_path}")

    # Load real data request
    req = json_to_request(json_path)
    print(f"Cycle ID: {req.cycle_id}, Mode: {req.mode}")
    print(f"Sensors: {req.sensor_ids}")
    print(f"Slots: {[s.slot for s in req.slot_data]}")

    # Process real data
    print(f"\n--- Real Data Localization ---")
    resp = handle_localize_cycle(req)

    real_success = False
    if resp.success:
        real_success = True
        p = resp.localization_pose.position
        q = resp.localization_pose.orientation
        print(f"Result: success=True")
        print(f"  Position: ({p.x:.6f}, {p.y:.6f}, {p.z:.6f})")
        print(f"  Orientation: ({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})")
        print(f"  Position Error: {resp.position_error:.6f} m")
        print(f"  Orientation Error: {resp.orientation_error:.4f} rad ({resp.orientation_error * 180 / 3.14159:.4f} deg)")
    else:
        print(f"Result: success=False")

    # Process model data
    print(f"\n--- Model Data Localization ---")
    model_resp = None
    model_success = False
    try:
        model_req, sources, p_array, R_array = generate_synthetic_request_from_json(json_path, cycle_id=req.cycle_id)
        print(f"Model Request: cycle_id={model_req.cycle_id}, mode={model_req.mode}")
        print(f"Sensors: {model_req.sensor_ids}")

        model_resp = handle_localize_cycle(model_req)

        if model_resp.success:
            model_success = True
            p = model_resp.localization_pose.position
            q = model_resp.localization_pose.orientation
            print(f"Result: success=True")
            print(f"  Position: ({p.x:.6f}, {p.y:.6f}, {p.z:.6f})")
            print(f"  Orientation: ({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})")
            print(f"  Position Error: {model_resp.position_error:.6f} m")
            print(f"  Orientation Error: {model_resp.orientation_error:.4f} rad ({model_resp.orientation_error * 180 / 3.14159:.4f} deg)")
        else:
            print(f"Result: success=False")

        # Mock vs Real comparison
        if real_success and model_success:
            # Get B_meas_cell and details for both
            _, _, B_meas_cell, meas_details = get_estimator_details(req, sensor_ids=sensor_ids)
            _, _, mock_B_meas_cell, mock_details = get_estimator_details(model_req, sensor_ids=sensor_ids)

            # Save B_meas comparison plot to same directory as JSON
            output_dir = os.path.dirname(os.path.abspath(json_path))
            bmeas_plot_path = os.path.join(output_dir, f'b_meas_comparison_cycle_{req.cycle_id:04d}.png')

            print("\n=== Mock vs Real Comparison ===")
            compare_model_vs_estimate(B_meas_cell, mock_B_meas_cell, meas_details, mock_details, output_path=bmeas_plot_path)

            # Run sensitivity analysis if requested
            if run_sensitivity:
                from localization_service_node import build_D_cal
                filter_sensor_ids = sensor_ids if sensor_ids else list(range(1, 13))
                D_cal = build_D_cal(filter_sensor_ids)
                sens_output_path = os.path.join(output_dir, f'x_hat_sensitivity_cycle_{req.cycle_id:04d}.png')
                sensitivity_analysis_x_hat_noise(
                    D_cal, sources, mock_B_meas_cell,
                    noise_levels=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                    num_trials=20,
                    output_path=sens_output_path
                )

            pos_diff = np.sqrt(
                (resp.localization_pose.position.x - model_resp.localization_pose.position.x)**2 +
                (resp.localization_pose.position.y - model_resp.localization_pose.position.y)**2 +
                (resp.localization_pose.position.z - model_resp.localization_pose.position.z)**2
            )
            print(f"\n=== Summary ===")
            print(f"Real Data:   Success")
            print(f"Model Data:  Success")
            print(f"Position Diff (Real vs Model): {pos_diff:.6f} m")

    except Exception as e:
        print(f"Model processing failed: {e}")
        import traceback
        traceback.print_exc()

    return (real_success, req, resp, model_resp)


def main():
    parser = argparse.ArgumentParser(
        description='Offline processing for cycle JSON files with real+model comparison.'
    )
    parser.add_argument('path', nargs='?', default=None,
                       help='Path to cycle JSON file or directory containing cycle_*.json files')
    parser.add_argument('--plot', action='store_true',
                       help='Generate 3D pose plot')
    parser.add_argument('--save-mock', action='store_true',
                       help='Save mock data to JSON file')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Run x_hat noise sensitivity analysis')
    parser.add_argument('--sensors', type=str, default=None,
                       help='Comma-separated sensor IDs to use (e.g., "1,3,4,6,7,9,10,12"). Default: all 12 sensors')

    args = parser.parse_args()

    # Default path
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'sensor_data_collection', 'result', 'cycle_0000.json'
    )
    default_path = os.path.normpath(default_path)

    if args.path is None:
        path = default_path
        print(f"No path provided, using default: {path}")
    else:
        path = args.path

    # Load configuration once
    print("Loading localization configuration...")
    load_configuration()
    print()

    if os.path.isfile(path):
        # Parse sensor IDs if provided
        sensor_ids = None
        if args.sensors is not None:
            sensor_ids = [int(x.strip()) for x in args.sensors.split(',')]
            print(f"Using sensors: {sensor_ids}")

        # Single file
        show_plot = args.plot
        success, req, resp, model_resp = process_file(path, show_plot=show_plot, run_sensitivity=args.sensitivity, sensor_ids=sensor_ids)
        if show_plot and success:
            output_path = plot_poses_comparison(req, resp, json_path=path, model_resp=model_resp)
            print(f"Plot saved to: {output_path}")

        # Save mock data if requested
        if args.save_mock:
            model_req, _, _, _ = generate_synthetic_request_from_json(path, cycle_id=req.cycle_id)
            output_dir = os.path.dirname(os.path.abspath(path))
            mock_path = os.path.join(output_dir, f'cycle_{req.cycle_id:04d}_mock.json')
            mock_request_to_json(model_req, mock_path)
            print(f"Mock data saved to: {mock_path}")

        if sys.stdin.isatty():
            input("Press Enter to exit...")
    elif os.path.isdir(path):
        # Directory - process all cycle_*.json files
        pattern = os.path.join(path, 'cycle_*.json')
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"No cycle_*.json files found in {path}")
            sys.exit(1)

        print(f"Found {len(files)} cycle files")

        real_success_count = 0
        model_success_count = 0
        for f in files:
            success, _, _, model_resp = process_file(f, show_plot=False)
            if success:
                real_success_count += 1
            if model_resp is not None:
                model_success_count += 1

        print(f"\n{'='*60}")
        print(f"Processed {len(files)} files, {real_success_count} real succeeded, {model_success_count} model succeeded")
    else:
        print(f"Invalid path: {path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
