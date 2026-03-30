#!/usr/bin/env python3
"""
Offline processing script for cycle JSON files.

Supports two modes:
    1. Localization: Compute pose estimate using MaPS estimator
    2. Calibration: Validate R_CORR, unit conversion, and local field/gradient
                   estimation by comparing p_Ci_est with true p_Ci

Usage:
    # Localization mode (default)
    python sensor_calibration.py /path/to/result/

    # Calibration validation mode
    python sensor_calibration.py --calibration /path/to/result/

    # Single file
    python sensor_calibration.py /path/to/cycle_0000.json
"""

import sys
import os
import json
import glob
import numpy as np

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import mock classes and localization functions
from test_localization_service import (
    MockLocalizeCycleRequest,
    MockSlotData,
    MockSensorReading,
    MockPose,
)
from localization_service_node import (
    handle_localize_cycle,
    load_configuration,
    process_hall_data,
    build_D_cal,
    R_CORR,
    quaternion_to_rotation_matrix,
)


# =============================================================================
# CALIBRATION VALIDATION FUNCTIONS
# Re-implemented from maps_estimator.py for offline calibration analysis
# =============================================================================

def null_space(A, rcond=1e-10):
    """
    Compute null space of matrix A. Equivalent to MATLAB's null(A).

    Args:
        A: m x n matrix
        rcond: Condition number cutoff

    Returns:
        Q_bar: n x k matrix (orthonormal basis for null space)
    """
    _, s, Vh = np.linalg.svd(A, full_matrices=True)
    n = A.shape[1] if A.ndim > 1 else len(A)
    rank = np.sum(s > rcond * s[0]) if s.size > 0 else 0
    k = n - rank
    if k > 0:
        return Vh[rank:, :].T
    else:
        return np.zeros((n, 0))


def estimate_local_field(B_meas, D_cal):
    """
    Estimate local magnetic field at sensor array center.
    Implements Eq. 5 from MaPS paper.

    Args:
        B_meas: 3 x N magnetic field measurements (Tesla)
        D_cal: 3 x N sensor offset matrix

    Returns:
        b_hat: Local field estimate (3,)
    """
    Q_bar = null_space(D_cal)
    g = Q_bar.T @ np.ones(D_cal.shape[1])

    if Q_bar.size == 0:
        b_hat = (B_meas @ np.ones(D_cal.shape[1])) / D_cal.shape[1]
    else:
        b_hat = (B_meas @ Q_bar @ g) / (np.linalg.norm(g) ** 2)

    return b_hat


def estimate_gradient_tensor(B_meas, D_cal):
    """
    Estimate local magnetic gradient tensor.
    Implements Eq. 8-11 from MaPS paper.

    Args:
        B_meas: 3 x N magnetic field measurements (Tesla)
        D_cal: 3 x N sensor offset matrix

    Returns:
        X_hat: 3x3 gradient tensor
        D_delta: 3 x Np pairwise difference matrix (for debugging)
    """
    N = D_cal.shape[1]

    # (Eq. 9-11) Constraint matrix for symmetric traceless gradient tensor X
    # Parametrization: X = [x1 x2 x3; x2 x4 x5; x3 x5 -(x1+x4)]
    S_mat = np.array([
        [1,  0,  0,  0,  0],  # (1,1)
        [0,  1,  0,  0,  0],  # (2,1)
        [0,  0,  1,  0,  0],  # (3,1)
        [0,  1,  0,  0,  0],  # (1,2)
        [0,  0,  0,  1,  0],  # (2,2)
        [0,  0,  0,  0,  1],  # (3,2)
        [0,  0,  1,  0,  0],  # (1,3)
        [0,  0,  0,  0,  1],  # (2,3)
        [-1, 0,  0, -1,  0],  # (3,3)
    ])

    # (Eq. 8) Pairwise differences
    Np = N * (N - 1) // 2
    D_delta = np.zeros((3, Np))
    B_delta = np.zeros((3, Np))
    idx = 0
    for i in range(N):
        for j in range(i):
            D_delta[:, idx] = D_cal[:, i] - D_cal[:, j]
            B_delta[:, idx] = B_meas[:, i] - B_meas[:, j]
            idx += 1

    C_mat = np.kron(D_delta.T, np.eye(3)) @ S_mat
    C_pinv = np.linalg.pinv(C_mat)

    h_vec = B_delta.flatten(order='F')
    x_param = C_pinv @ h_vec
    X_hat = S_mat @ x_param
    X_hat = X_hat.reshape((3, 3), order='F')

    return X_hat, D_delta


def calibration_validate(req):
    """
    Validate calibration by comparing estimated source positions
    with true positions.

    For each slot:
        1. Apply R_CORR and unit conversion to raw sensor data
        2. Estimate local gradient tensor X and local field b
        3. Compute p_Ci_est = p + 3 * R.T @ (X.inv() @ b)
        4. Compare with true p_Ci

    Args:
        req: MockLocalizeCycleRequest

    Returns:
        dict: Validation results per slot
    """
    results = {
        'cycle_id': req.cycle_id,
        'mode': req.mode,
        'slots': []
    }

    sensor_ids = list(req.sensor_ids)
    D_cal = build_D_cal(sensor_ids)

    # Build slot lookup
    slot_dict = {slot.slot: slot for slot in req.slot_data}

    # For CVT mode, get background B0 (slot 3)
    B0 = None
    if req.mode == 'CVT' and 3 in slot_dict:
        B0 = process_hall_data(slot_dict[3].sensor_data)

    # Process each source slot (skip slot 3 which is B0)
    for slot in req.slot_data:
        if slot.slot == 3:  # Skip B0 slot
            continue

        # Get true source position and orientation
        p_true = np.array([
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
        R_true = quaternion_to_rotation_matrix(quat)

        # Process hall data with R_CORR and unit conversion
        B_meas = process_hall_data(slot.sensor_data)

        # Background subtraction for CVT
        if req.mode == 'CVT' and B0 is not None:
            B_meas = B_meas - B0

        # Estimate local field and gradient tensor
        b_hat = estimate_local_field(B_meas, D_cal)
        X_hat, _ = estimate_gradient_tensor(B_meas, D_cal)

        # Compute estimated source position
        # p_Ci_est = p + 3 * R.T @ (X.inv() @ b)
        try:
            X_inv = np.linalg.inv(X_hat)
            p_Ci_est = p_true + 3 * R_true.T @ (X_inv @ b_hat)
        except np.linalg.LinAlgError:
            p_Ci_est = None
            X_inv = None

        # Compute error
        if p_Ci_est is not None:
            position_error = np.linalg.norm(p_Ci_est - p_true)
        else:
            position_error = None

        slot_result = {
            'slot': slot.slot,
            'p_true': p_true,
            'p_est': p_Ci_est,
            'position_error': position_error,
            'b_hat': b_hat,
            'X_hat': X_hat,
            'X_inv': X_inv,
            'B_meas_norm': np.linalg.norm(B_meas, axis=0).mean() if B_meas.size > 0 else None,
        }
        results['slots'].append(slot_result)

    return results


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


def process_file(json_path, calibration_mode=False):
    """
    Process a single cycle JSON file and print results.

    Args:
        json_path: path to cycle JSON file
        calibration_mode: if True, run calibration validation instead of localization

    Returns:
        bool: True if processing succeeded
    """
    print(f"\n{'='*60}")
    print(f"Processing: {json_path}")

    req = json_to_request(json_path)
    print(f"Cycle ID: {req.cycle_id}, Mode: {req.mode}")
    print(f"Sensors: {req.sensor_ids}")
    print(f"Slots: {[s.slot for s in req.slot_data]}")

    if calibration_mode:
        return process_calibration(req)
    else:
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


def process_calibration(req):
    """
    Run calibration validation on a cycle.

    Args:
        req: MockLocalizeCycleRequest

    Returns:
        bool: True if validation completed
    """
    results = calibration_validate(req)

    print(f"\n--- Calibration Validation ---")
    for slot_res in results['slots']:
        print(f"\nSlot {slot_res['slot']}:")
        print(f"  p_true: ({slot_res['p_true'][0]:.4f}, {slot_res['p_true'][1]:.4f}, {slot_res['p_true'][2]:.4f})")

        if slot_res['p_est'] is not None:
            print(f"  p_est:  ({slot_res['p_est'][0]:.4f}, {slot_res['p_est'][1]:.4f}, {slot_res['p_est'][2]:.4f})")
            print(f"  Position Error: {slot_res['position_error']:.6f} m")
        else:
            print(f"  p_est:  None (singular matrix)")

        print(f"  ||b_hat||: {np.linalg.norm(slot_res['b_hat']):.6e} T")
        print(f"  ||X_hat||: {np.linalg.norm(slot_res['X_hat']):.6e}")

        if slot_res['B_meas_norm'] is not None:
            print(f"  Mean ||B_meas||: {slot_res['B_meas_norm']:.6e} T")

    # Summary statistics
    errors = [s['position_error'] for s in results['slots'] if s['position_error'] is not None]
    if errors:
        print(f"\n--- Summary ---")
        print(f"  Mean Position Error: {np.mean(errors):.6f} m")
        print(f"  Max Position Error: {np.max(errors):.6f} m")
        print(f"  Min Position Error: {np.min(errors):.6f} m")

    return True


def main():
    # Default path
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'sensor_data_collection', 'result', 'cycle_0000.json'
    )
    default_path = os.path.normpath(default_path)

    # Parse arguments
    calibration_mode = False
    path = default_path

    for arg in sys.argv[1:]:
        if arg == '--calibration':
            calibration_mode = True
        else:
            path = arg

    if calibration_mode:
        print("Running in CALIBRATION mode")
    else:
        print("Running in LOCALIZATION mode")

    # Load configuration once
    print("Loading localization configuration...")
    load_configuration()
    print()

    if os.path.isfile(path):
        # Single file
        process_file(path, calibration_mode)
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
            if process_file(f, calibration_mode):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"Processed {len(files)} files, {success_count} succeeded")
    else:
        print(f"Invalid path: {path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
