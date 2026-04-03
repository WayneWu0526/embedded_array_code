#!/usr/bin/env python3
"""
Utility functions for offline processing.

Contains quaternion conversions, magnetic dipole model helpers, and model reference computation.
"""

import numpy as np
from mag_dipole_model import mag_dipole_model


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def quaternion_z_axis(q):
    """
    Extract the z-axis direction from a quaternion.

    Args:
        q: quaternion [w, x, y, z]

    Returns:
        np.ndarray: z-axis direction vector (3,)
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    z_axis = np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x**2 + y**2)
    ])
    return z_axis / np.linalg.norm(z_axis)


def generate_hall_data_from_dipole(sensor_ids, p_sensor_array, R_sensor_array, d_list, source_p_Ci, m_Ci, R_CORR, noise_level=1e-10):
    """
    Generate synthetic Hall sensor data using mag_dipole_model.

    Args:
        sensor_ids: list of active sensor IDs (1-12)
        p_sensor_array: 3D position of sensor array center (global frame)
        R_sensor_array: 3x3 rotation matrix of sensor array orientation
        d_list: 12x3 array of sensor positions relative to array center
        source_p_Ci: 3D position of the magnetic dipole source
        m_Ci: 3D magnetic moment vector of the source
        R_CORR: dict of rotation correction matrices per sensor group
        noise_level: standard deviation of Gaussian noise to add

    Returns:
        list of MockSensorReading
    """
    from localization_service_node import get_sensor_group, GS_TO_TESLA
    from offline_mock import MockSensorReading

    data = []
    for sid in sensor_ids:
        sensor_idx = sid - 1  # 0-indexed

        # Sensor position in global frame: p_s = p_array + R @ d_j
        d_j = d_list[sensor_idx]
        p_sensor_global = p_sensor_array + R_sensor_array @ d_j

        # Compute magnetic field at sensor using dipole model (in global frame, Tesla)
        b_global, _ = mag_dipole_model(p_sensor_global, m_Ci, source_p_Ci, order=1)

        # Transform from global to sensor array frame: b_array = R.T @ b_global
        b_array = R_sensor_array.T @ b_global

        # Apply R_CORR.T to get raw sensor measurement (sensor local frame)
        group = get_sensor_group(sid)
        if group is not None and group in R_CORR:
            b_sensor = R_CORR[group].T @ b_array
        else:
            b_sensor = b_array

        # Convert from Tesla to Gs (Hall sensor output is in Gs)
        b_sensor_gs = b_sensor / GS_TO_TESLA

        # Add noise (in Gs)
        x = b_sensor_gs[0] + noise_level * np.random.randn()
        y = b_sensor_gs[1] + noise_level * np.random.randn()
        z = b_sensor_gs[2] + noise_level * np.random.randn()

        data.append(MockSensorReading(id=sid, x=x, y=y, z=z))

    return data


def compute_model_reference(p_sensor_array, R_sensor_array, sources):
    """
    Compute model reference b_model and X_model for each source at the sensor array center.

    Args:
        p_sensor_array: 3D position of sensor array center (global frame)
        R_sensor_array: 3x3 rotation matrix of sensor array orientation
        sources: list of dicts with 'p_Ci' (3,) and 'm_Ci' (3,)

    Returns:
        b_model_list: list of b_model vectors (3,) per source
        X_model_list: list of X_model matrices (3x3) per source
    """
    b_model_list = []
    X_model_list = []
    for src in sources:
        b_global, A_global = mag_dipole_model(p_sensor_array, src['m_Ci'], src['p_Ci'], order=1)
        b_local = R_sensor_array.T @ b_global
        X_local = R_sensor_array.T @ A_global @ R_sensor_array
        b_model_list.append(b_local)
        X_model_list.append(X_local)
    return b_model_list, X_model_list


def json_to_request(json_path):
    """
    Convert a cycle JSON file to a MockLocalizeCycleRequest.

    Args:
        json_path: path to cycle JSON file

    Returns:
        MockLocalizeCycleRequest
    """
    import json
    from offline_mock import MockPose, MockSensorReading, MockSlotData, MockLocalizeCycleRequest

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


def mock_request_to_json(req, json_path):
    """
    Save a MockLocalizeCycleRequest to a JSON file.

    Args:
        req: MockLocalizeCycleRequest to save
        json_path: output path for JSON file
    """
    import json

    def sensor_to_dict(s):
        return {'id': s.id, 'x': s.x, 'y': s.y, 'z': s.z}

    def pose_to_dict(p):
        return {
            'position': {'x': p.position.x, 'y': p.position.y, 'z': p.position.z},
            'rotation': {'x': p.orientation.x, 'y': p.orientation.y,
                         'z': p.orientation.z, 'w': p.orientation.w}
        }

    slot_data = []
    for sd in req.slot_data:
        slot_data.append({
            'slot': sd.slot,
            'sensor_data': [sensor_to_dict(s) for s in sd.sensor_data],
            'pose': pose_to_dict(sd.pose)
        })

    gt = req.ground_truth_pose
    data = {
        'header': {
            'cycle_id': req.cycle_id,
            'mode': req.mode,
            'sensor_ids': req.sensor_ids
        },
        'ground_truth_pose': pose_to_dict(gt),
        'slot_data': slot_data
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def generate_synthetic_request_from_json(json_path, cycle_id=None, mode_override=None):
    """
    Read pose from JSON and generate synthetic sensor data using mag_dipole_model.

    Args:
        json_path: path to cycle JSON file
        cycle_id: override cycle_id (optional)
        mode_override: force 'CVT' or 'CCI' mode (optional, auto-detected from JSON)

    Returns:
        tuple: (MockLocalizeCycleRequest, sources, p_sensor_array, R_sensor_array)
    """
    import json
    from localization_service_node import R_CORR
    from offline_mock import MockPose, MockSlotData, MockLocalizeCycleRequest

    with open(json_path) as f:
        data = json.load(f)

    header = data['header']
    mode = mode_override or header['mode']

    sensor_ids = header['sensor_ids']

    # Ground truth pose of the sensor array
    gt = data['ground_truth_pose']
    p_sensor_array = np.array([gt['position']['x'], gt['position']['y'], gt['position']['z']])
    R_sensor_array = quaternion_to_rotation_matrix(
        gt['rotation']['x'], gt['rotation']['y'],
        gt['rotation']['z'], gt['rotation']['w']
    )

    # Sensor positions relative to array center
    de1, de2, de3 = 1.0e-3, 2.0e-3, 2.1e-3
    d_list = np.array([
        [-de2/2,  de3/2, -de1], [-de2/2,  de3/2,  0   ], [-de2/2,  de3/2,  de1 ],
        [-de2/2, -de3/2, -de1], [-de2/2, -de3/2,  0   ], [-de2/2, -de3/2,  de1 ],
        [ de2/2,  de3/2, -de1], [ de2/2,  de3/2,  0   ], [ de2/2,  de3/2,  de1 ],
        [ de2/2, -de3/2, -de1], [ de2/2, -de3/2,  0   ], [ de2/2, -de3/2,  de1 ],
    ])

    # Build slot_data from JSON
    slot_dict = {sd['slot']: sd for sd in data['slot_data']}
    sources = []
    slot_data_list = []

    # Moment magnitudes
    moment_magnitude = np.array([-450, -500, -1000])

    for slot_idx in [0, 1, 2]:
        if slot_idx not in slot_dict:
            continue
        slot_info = slot_dict[slot_idx]
        pose_data = slot_info.get('pose')
        if not pose_data:
            continue

        p_Ci = np.array([pose_data['position']['x'], pose_data['position']['y'], pose_data['position']['z']])
        q = np.array([pose_data['rotation']['w'], pose_data['rotation']['x'],
                      pose_data['rotation']['y'], pose_data['rotation']['z']])
        z_axis = quaternion_z_axis(q)
        m_Ci = moment_magnitude[slot_idx] * z_axis

        sources.append({'p_Ci': p_Ci, 'm_Ci': m_Ci, 'name': f'slot{slot_idx}'})

        # Generate synthetic sensor data
        sensor_readings = generate_hall_data_from_dipole(
            sensor_ids, p_sensor_array, R_sensor_array, d_list, p_Ci, m_Ci, R_CORR
        )

        pose = MockPose(
            x=pose_data['position']['x'],
            y=pose_data['position']['y'],
            z=pose_data['position']['z'],
            qx=pose_data['rotation']['x'],
            qy=pose_data['rotation']['y'],
            qz=pose_data['rotation']['z'],
            qw=pose_data['rotation']['w']
        )
        slot_data_list.append(MockSlotData(slot=slot_idx, sensor_data=sensor_readings, pose=pose))

    # B0 (slot 3) for CVT - background only
    if mode == 'CVT' and 3 in slot_dict:
        b0_data = generate_hall_data_from_dipole(
            sensor_ids, p_sensor_array, R_sensor_array, d_list,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            R_CORR
        )
        slot_data_list.append(MockSlotData(slot=3, sensor_data=b0_data, pose=MockPose()))

    gt_pose = MockPose(
        x=gt['position']['x'],
        y=gt['position']['y'],
        z=gt['position']['z'],
        qx=gt['rotation']['x'],
        qy=gt['rotation']['y'],
        qz=gt['rotation']['z'],
        qw=gt['rotation']['w']
    )

    cid = cycle_id if cycle_id is not None else header['cycle_id']
    req = MockLocalizeCycleRequest(
        cycle_id=cid,
        mode=mode,
        sensor_ids=sensor_ids,
        slot_data=slot_data_list,
        ground_truth_pose=gt_pose
    )
    return req, sources, p_sensor_array, R_sensor_array


def get_estimator_details(req, sensor_ids=None):
    """
    Get b_hat_locals and X_hat_locals for model comparison.

    Args:
        req: MockLocalizeCycleRequest
        sensor_ids: list of sensor IDs to use (e.g. [1,3,4,6,7,9,10,12]).
                    None means use all sensors from req.sensor_ids.

    Returns (D_cal, sources, B_meas_cell, details).
    """
    from localization_service_node import process_hall_data, build_D_cal, run_localization

    mode = req.mode
    if sensor_ids is None:
        sensor_ids = list(req.sensor_ids)
    slot_data = list(req.slot_data)

    # Build slot lookup dict
    slot_dict = {slot.slot: slot for slot in slot_data}

    # Extract B0 for CVT mode (slot 3)
    B0 = None
    if mode == 'CVT' and 3 in slot_dict:
        # Filter sensor data for B0
        filtered_B0_data = [r for r in slot_dict[3].sensor_data if r.id in sensor_ids]
        B0 = process_hall_data(filtered_B0_data)

    # Build sources and B_meas_cell (skip B0 slot for CVT)
    sources = []
    B_meas_cell = []

    for slot in slot_data:
        if slot.slot == 3:  # Skip B0 slot
            continue

        # Source pose -> m_Ci
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
        m_Ci = quaternion_z_axis(quat)
        sources.append({'p_Ci': p_Ci, 'm_Ci': m_Ci})

        # Magnetic field measurement - filter to selected sensors
        filtered_sensor_data = [r for r in slot.sensor_data if r.id in sensor_ids]
        B_meas = process_hall_data(filtered_sensor_data)

        # Background subtraction for CVT
        if mode == 'CVT' and B0 is not None:
            B_meas = B_meas - B0

        B_meas_cell.append(B_meas)

    D_cal = build_D_cal(sensor_ids)
    _, _, _, details = run_localization(D_cal, sources, B_meas_cell)
    return D_cal, sources, B_meas_cell, details


def x_hat_from_X_hat(X_hat):
    """
    Convert 3x3 symmetric traceless matrix X_hat to 5D parameter x_hat using S_mat.

    X_hat parametrization: X = [x1 x2 x3; x2 x4 x5; x3 x5 -(x1+x4)]
    """
    S_mat = np.array([
        [1,  0,  0,  0,  0],
        [0,  1,  0,  0,  0],
        [0,  0,  1,  0,  0],
        [0,  1,  0,  0,  0],
        [0,  0,  0,  1,  0],
        [0,  0,  0,  0,  1],
        [0,  0,  1,  0,  0],
        [0,  0,  0,  0,  1],
        [-1, 0,  0, -1,  0],
    ])
    X_flat = X_hat.flatten(order='F')  # Column-major (Fortran) order
    x_hat = np.linalg.lstsq(S_mat, X_flat, rcond=None)[0]
    return x_hat


def compare_model_vs_estimate(B_meas_cell, mock_B_meas_cell, meas_details, mock_details, output_path=None):
    """
    Compare real data vs mock data (measurement and estimator details).

    Args:
        B_meas_cell: list of B_meas from real sensor data (3 x N per slot)
        mock_B_meas_cell: list of B_meas from mock/synthetic data (3 x N per slot)
        meas_details: estimator details from real data
        mock_details: estimator details from mock data
        output_path: path to save plot (optional, default /tmp/b_meas_comparison.png)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    slot_names = ['diana7', 'arm1', 'arm2']
    num_slots = min(len(B_meas_cell), len(mock_B_meas_cell))

    # =========================================================================
    # 1. B_meas comparison (sensor-level)
    # =========================================================================
    fig, axes = plt.subplots(2, num_slots, figsize=(5 * num_slots, 8))
    if num_slots == 1:
        axes = axes.reshape(-1, 1)

    for slot_idx in range(num_slots):
        B_real = B_meas_cell[slot_idx]  # 3 x N
        B_mock = mock_B_meas_cell[slot_idx]  # 3 x N
        N = B_real.shape[1]

        diff_norms = np.zeros(N)
        rel_errors = np.zeros(N)
        for i in range(N):
            diff = B_real[:, i] - B_mock[:, i]
            diff_norms[i] = np.linalg.norm(diff)
            b_norm = np.linalg.norm(B_real[:, i])
            rel_errors[i] = np.linalg.norm(diff) / b_norm * 100 if b_norm > 1e-10 else 0

        sensors = np.arange(1, N + 1)

        ax = axes[0, slot_idx]
        ax.bar(sensors, diff_norms * 1e6, color='steelblue', alpha=0.8)
        ax.set_ylabel('|B_real - B_mock| (uT)')
        ax.set_title(f'{slot_names[slot_idx]} (Slot {slot_idx})')
        ax.set_xticks(sensors)
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1, slot_idx]
        ax.bar(sensors, rel_errors, color='coral', alpha=0.8)
        ax.set_xlabel('Sensor ID')
        ax.set_ylabel('Relative Error (%)')
        ax.set_xticks(sensors)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = output_path or '/tmp/b_meas_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"B_meas comparison plot saved to: {save_path}")
    plt.close()

    # =========================================================================
    # 2. b_hat comparison (slot-level, from estimator)
    # =========================================================================
    print("\n=== b_hat Comparison ===")
    b_hat_real = meas_details['b_hat_locals']  # 3 x M
    b_hat_mock = mock_details['b_hat_locals']  # 3 x M

    print(f"{'Slot':<8} {'|b_hat_real|':<15} {'|b_hat_mock|':<15} {'|diff|':<15} {'rel_err(%)':<12}")
    print("-" * 65)
    for i in range(num_slots):
        b_r = b_hat_real[:, i]
        b_m = b_hat_mock[:, i]
        diff_norm = np.linalg.norm(b_r - b_m)
        b_norm = np.linalg.norm(b_r)
        rel_err = diff_norm / b_norm * 100 if b_norm > 1e-10 else 0
        print(f"{slot_names[i]:<8} {np.linalg.norm(b_r)*1e6:>12.4f} uT  {np.linalg.norm(b_m)*1e6:>12.4f} uT  {diff_norm*1e6:>12.4f} uT  {rel_err:>10.4f} %")

    # =========================================================================
    # 3. X_hat comparison (slot-level, from estimator)
    # =========================================================================
    print("\n=== X_hat Comparison ===")
    X_hat_real = meas_details['X_hat_locals']  # list of 3x3
    X_hat_mock = mock_details['X_hat_locals']  # list of 3x3

    print(f"{'Slot':<8} {'||X_hat_real||':<15} {'||X_hat_mock||':<15} {'cond(X_hat_real)':<18} {'cond(X_hat_mock)':<15}")
    print("-" * 75)
    for i in range(num_slots):
        X_r = X_hat_real[i]
        X_m = X_hat_mock[i]
        norm_r = np.linalg.norm(X_r, 'fro')
        norm_m = np.linalg.norm(X_m, 'fro')
        cond_r = np.linalg.cond(X_r)
        cond_m = np.linalg.cond(X_m)
        print(f"{slot_names[i]:<8} {norm_r:>12.6f}  {norm_m:>12.6f}  {cond_r:>15.4f}  {cond_m:>14.4f}")

    # =========================================================================
    # 4. x_hat comparison (5D parametrization)
    # =========================================================================
    print("\n=== x_hat (5D parametrization) Comparison ===")
    print(f"{'Slot':<8} {'||x_hat_real||':<15} {'||x_hat_mock||':<15} {'||diff||':<15} {'rel_err(%)':<12}")
    print("-" * 65)
    for i in range(num_slots):
        x_r = x_hat_from_X_hat(X_hat_real[i])
        x_m = x_hat_from_X_hat(X_hat_mock[i])
        diff_norm = np.linalg.norm(x_r - x_m)
        x_norm_r = np.linalg.norm(x_r)
        x_norm_m = np.linalg.norm(x_m)
        rel_err = diff_norm / x_norm_r * 100 if x_norm_r > 1e-10 else 0
        print(f"{slot_names[i]:<8} {x_norm_r:>12.6f}  {x_norm_m:>12.6f}  {diff_norm:>12.6f}  {rel_err:>10.4f} %")

    # =========================================================================
    # 5. Eigenvalue comparison for X_hat
    # =========================================================================
    print("\n=== X_hat Eigenvalue Comparison ===")
    print(f"{'Slot':<8} {'Eigenvalues (real)':<45} {'Eigenvalues (mock)':<45}")
    print("-" * 100)
    for i in range(num_slots):
        evals_r = np.linalg.eigvalsh(X_hat_real[i])
        evals_m = np.linalg.eigvalsh(X_hat_mock[i])
        evals_r_str = ", ".join([f"{v:.4e}" for v in evals_r])
        evals_m_str = ", ".join([f"{v:.4e}" for v in evals_m])
        print(f"{slot_names[i]:<8} [{evals_r_str}]")
        print(f"{'':.<8} [{evals_m_str}]")

    return save_path


# =========================================================================
# x_hat Noise Sensitivity Analysis
# =========================================================================

def _get_S_mat():
    """Get the S_mat for X_hat parametrization."""
    return np.array([
        [1,  0,  0,  0,  0],
        [0,  1,  0,  0,  0],
        [0,  0,  1,  0,  0],
        [0,  1,  0,  0,  0],
        [0,  0,  0,  1,  0],
        [0,  0,  0,  0,  1],
        [0,  0,  1,  0,  0],
        [0,  0,  0,  0,  1],
        [-1, 0,  0, -1,  0],
    ])


def compute_rho_hat_eigen(X, b_hat):
    """
    Compute rho_hat using eigenvalue decomposition to avoid direct matrix inversion.

    Formula: rho_hat = 3 / (lambda_max * lambda_min) * (X + lambda_med * I) @ b_hat

    Args:
        X: 3x3 symmetric matrix (gradient tensor)
        b_hat: 3-vector (local field estimate)

    Returns:
        3-vector: rho_hat estimate
    """
    # Compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(X)

    # Sort eigenvalues in descending order: lambda_max >= lambda_med >= lambda_min
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    # evecs = evecs[:, idx]  # eigenvectors reordered accordingly

    lambda_max = evals[0]
    lambda_med = evals[1]
    lambda_min = evals[2]

    # Compute rho_hat using eigenvalue-based formula
    # rho_hat = 3 / (lambda_max * lambda_min) * (X + lambda_med * I) @ b_hat
    rho_hat = 3.0 / (lambda_max * lambda_min) * (X + lambda_med * np.eye(3)) @ b_hat

    return rho_hat


def _get_null_basis():
    """Get null basis computation from maps_estimator."""
    from maps_estimator import null, kabsch_solver
    return null, kabsch_solver


def run_localization_with_x_hat_noise(D_cal, sources, B_meas_cell, noise_scale=0.0, noise_seed=None):
    """
    Run MaPS estimator with noise injected into x_hat.

    This is a modified version of run_localization that adds Gaussian noise
    to the x_param (5D parameterization of X_hat) before computing rho_hat.

    Args:
        D_cal: 3 x N sensor offset matrix
        sources: list of source dicts with 'p_Ci' and 'm_Ci'
        B_meas_cell: list of 3 x N magnetic field matrices (one per slot)
        noise_scale: fraction of x_hat std dev to add as noise (0 = no noise)
        noise_seed: random seed for reproducibility

    Returns:
        tuple: (R_est, p_est, details, x_hat_noisy_list)
        x_hat_noisy_list: list of noisy x_hat vectors for each slot
    """
    from maps_estimator import kabsch_solver

    if noise_seed is not None:
        np.random.seed(noise_seed)

    D_cal = np.asarray(D_cal)
    N = D_cal.shape[1]
    M = len(B_meas_cell)

    S_mat = _get_S_mat()

    # Precomputations (same as MaPS_Estimator)
    def null(A, rcond=1e-10):
        """Computes the null space of matrix A."""
        u, s, vh = np.linalg.svd(A)
        rank = np.sum(s > s[0] * rcond)
        return vh[rank:].T

    Q_bar = null(D_cal)
    g = Q_bar.T @ np.ones(N)

    # (Eq. 8) Pairwise differences for local gradient estimation
    Np = N * (N - 1) // 2
    D_delta = np.zeros((3, Np))
    idx = 0
    for i in range(N):
        for j in range(i):
            D_delta[:, idx] = D_cal[:, i] - D_cal[:, j]
            idx += 1

    # (Eq. 9-11) Constraint matrix for symmetric traceless gradient tensor X
    C_mat = np.kron(D_delta.T, np.eye(3)) @ S_mat
    C_pinv = np.linalg.pinv(C_mat)

    # Local Quantity Estimation
    b_hat_locals = np.zeros((3, M))
    X_hat_locals = []
    X_hat_noisy_locals = []
    rho_hats = np.zeros((3, M))
    x_param_orig_list = []
    x_param_noisy_list = []

    for i in range(M):
        B_meas = B_meas_cell[i]

        # (Eq. 5) Local Field Estimator
        if Q_bar.size == 0:
            b_hat_locals[:, i] = (B_meas @ np.ones(N)) / N
        else:
            b_hat_locals[:, i] = (B_meas @ Q_bar @ g) / (np.linalg.norm(g)**2)

        # (Eq. 8-11) Local Gradient Tensor Estimator
        B_delta = np.zeros((3, Np))
        idx = 0
        for ii in range(N):
            for jj in range(ii):
                B_delta[:, idx] = B_meas[:, ii] - B_meas[:, jj]
                idx += 1

        h_vec = B_delta.flatten(order='F')
        x_param_orig = C_pinv @ h_vec
        x_param_noisy = x_param_orig.copy()

        # Add noise to x_param if noise_scale > 0
        if noise_scale > 0:
            noise_std = noise_scale * np.std(x_param_orig)
            x_param_noisy = x_param_orig + noise_std * np.random.randn(5)

        x_param_orig_list.append(x_param_orig)
        x_param_noisy_list.append(x_param_noisy)

        # Original X_hat (not used in final computation when noise is injected)
        X_hat = S_mat @ x_param_orig
        X_hat = X_hat.reshape((3, 3), order='F')

        # Noisy X_hat
        X_hat_noisy = S_mat @ x_param_noisy
        X_hat_noisy = X_hat_noisy.reshape((3, 3), order='F')
        X_hat_noisy_locals.append(X_hat_noisy)

        # (Eq. 13) Local relative displacement estimation using NOISY X_hat
        # Original: rho_hats[:, i] = -3 * solve(X_hat_noisy, b_hat_locals[:, i])
        # New method: use eigenvalue-based computation to avoid matrix inversion
        rho_hats[:, i] = compute_rho_hat_eigen(X_hat_noisy, b_hat_locals[:, i])

    # Global Pose Recovery (using noisy rho_hats)
    U_P = []
    V_P = []
    for i in range(M):
        for j in range(i):
            U_P.append(rho_hats[:, i] - rho_hats[:, j])
            V_P.append(sources[j]['p_Ci'] - sources[i]['p_Ci'])

    U_P = np.column_stack(U_P)
    V_P = np.column_stack(V_P)

    R_est = kabsch_solver(V_P, U_P)

    # (Eq. 18) Position Averaging
    p_ests = np.zeros((3, M))
    for i in range(M):
        p_ests[:, i] = sources[i]['p_Ci'] + R_est @ rho_hats[:, i]
    p_est = np.mean(p_ests, axis=1)

    details = {
        'b_hat_locals': b_hat_locals,
        'X_hat_locals': X_hat_locals,
        'rho_hats': rho_hats,
        'p_ests': p_ests
    }

    return R_est, p_est, details, x_param_noisy_list


def sensitivity_analysis_x_hat_noise(D_cal, sources, B_meas_cell,
                                     noise_levels=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                                     num_trials=10, reference_rho_hats=None, output_path=None):
    """
    Perform sensitivity analysis: inject noise into x_hat at different scales
    and observe the effect on rho_hat estimation.

    Args:
        D_cal: 3 x N sensor offset matrix
        sources: list of source dicts with 'p_Ci' and 'm_Ci'
        B_meas_cell: list of 3 x N magnetic field matrices (one per slot)
        noise_levels: list of noise scales (fraction of x_hat std dev)
        num_trials: number of Monte Carlo trials per noise level
        reference_rho_hats: ground truth rho_hats for error computation (optional)
        output_path: path to save plot (optional)

    Returns:
        dict: results with statistics for each noise level
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    slot_names = ['diana7', 'arm1', 'arm2']
    M = len(B_meas_cell)

    # First run without noise to get baseline
    _, _, baseline_details, _ = run_localization_with_x_hat_noise(D_cal, sources, B_meas_cell, noise_scale=0.0)
    baseline_rho_hats = baseline_details['rho_hats']

    # Reference rho_hats (ground truth) if not provided
    if reference_rho_hats is None:
        reference_rho_hats = baseline_rho_hats

    results = {}

    print("\n" + "=" * 70)
    print("x_hat Noise Sensitivity Analysis")
    print("=" * 70)
    print(f"{'Noise %':<10} {'|d_rho| mean':<15} {'|d_rho| std':<15} {'max |d_rho|':<15}")
    print("-" * 55)

    fig, axes = plt.subplots(1, M, figsize=(5 * M, 4))
    if M == 1:
        axes = [axes]

    for noise_level in noise_levels:
        rho_diffs = []

        for trial in range(num_trials):
            _, _, details, _ = run_localization_with_x_hat_noise(
                D_cal, sources, B_meas_cell,
                noise_scale=noise_level,
                noise_seed=trial if num_trials > 1 else None
            )
            noisy_rho = details['rho_hats']

            for i in range(M):
                diff = noisy_rho[:, i] - baseline_rho_hats[:, i]
                rho_diffs.append(np.linalg.norm(diff))

        rho_diffs = np.array(rho_diffs)
        mean_err = np.mean(rho_diffs) * 1000  # Convert to mm
        std_err = np.std(rho_diffs) * 1000
        max_err = np.max(rho_diffs) * 1000

        results[noise_level] = {
            'mean': mean_err,
            'std': std_err,
            'max': max_err,
            'all_diffs': rho_diffs
        }

        print(f"{noise_level*100:>6.1f}%    {mean_err:>10.4f} mm  {std_err:>10.4f} mm  {max_err:>10.4f} mm")

    print("-" * 55)

    # Plot
    noise_pct = [n * 100 for n in noise_levels]
    means = [results[n]['mean'] for n in noise_levels]
    stds = [results[n]['std'] for n in noise_levels]
    maxes = [results[n]['max'] for n in noise_levels]

    for i in range(M):
        ax = axes[i]
        slot_diffs = [results[n]['all_diffs'][i::M] * 1000 for n in noise_levels]
        # Box plot
        bp = ax.boxplot(slot_diffs, positions=range(len(noise_levels)), widths=0.6)
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f'{n*100:.0f}%' for n in noise_levels])
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('|d_rho| (mm)')
        ax.set_title(f'{slot_names[i]} (Slot {i})')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('x_hat Noise Sensitivity: Effect on rho_hat (log scale)')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    plt.close()

    return results
