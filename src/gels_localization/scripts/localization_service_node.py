#!/usr/bin/env python3
"""
Localization Service Node for GELS algorithm.

=== FRAMEWORK MODE ===
This node provides the interface between ROS cycle data and the MaPS estimator.
Data processing (calibration, coordinate transforms, etc.) should be implemented
as modular components that can be configured/replaced.

Data flow:
    LocalizeCycle.srv (request)
        ↓
    [1] Extract cycle data
        ↓
    [2] Data processing (TODO: implement according to calibration)
        ↓  D_cal, sources, B_meas_cell
    [3] MaPS Estimator
        ↓
    [4] Return LocalizeCycle.srv (response)
"""

import sys
import os
import numpy as np

# Add scripts directory to path for local module imports
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from maps_estimator import MaPS_Estimator

# ROS imports - optional for standalone testing
try:
    import rospy
    from sensor_data_collection.srv import LocalizeCycle, LocalizeCycleResponse
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rospy = None
    LocalizeCycle = None
    LocalizeCycleResponse = None


# =============================================================================
# [1] CONFIGURATION LOADING (load from rosparam)
# =============================================================================

# Global calibration parameters (loaded from yaml)
R_CORR = {}  # Rotation correction matrices per sensor group
D_LIST = None  # Sensor physical offset positions (3 x 12)
MOMENT_LIST = None  # Magnetic moment for each source (slot)
GS_TO_TESLA = 1.0e-4  # Unit conversion factor


def load_configuration():
    """
    Load sensor calibration parameters.
    - From rosparam if ROS is available
    - From defaults if ROS is not available
    """
    global R_CORR, D_LIST, MOMENT_LIST, GS_TO_TESLA

    # Default values
    r_corr_r1 = [1, 0, 0, 0, 0, -1, 0, 1, 0]
    r_corr_r2 = [1, 0, 0, 0, 0, 1, 0, -1, 0]
    r_corr_r3 = [-1, 0, 0, 0, 0, 1, 0, 1, 0]
    r_corr_r4 = [-1, 0, 0, 0, 0, -1, 0, -1, 0]

    de1, de2, de3 = 1.0e-3, 2.0e-3, 2.1e-3
    d_list_default = [
        [-de2/2,  de3/2, -de1], [-de2/2,  de3/2,  0   ], [-de2/2,  de3/2,  de1 ],
        [-de2/2, -de3/2, -de1], [-de2/2, -de3/2,  0   ], [-de2/2, -de3/2,  de1 ],
        [ de2/2,  de3/2, -de1], [ de2/2,  de3/2,  0   ], [ de2/2,  de3/2,  de1 ],
        [ de2/2, -de3/2, -de1], [ de2/2, -de3/2,  0   ], [ de2/2, -de3/2,  de1 ],
    ]

    gs_to_tesla_default = 1.0e-4

    if ROS_AVAILABLE:
        # Load from rosparam (try/except in case roscore is not running)
        try:
            r_corr_r1 = rospy.get_param('~r_corr_r1', r_corr_r1)
            r_corr_r2 = rospy.get_param('~r_corr_r2', r_corr_r2)
            r_corr_r3 = rospy.get_param('~r_corr_r3', r_corr_r3)
            r_corr_r4 = rospy.get_param('~r_corr_r4', r_corr_r4)
            d_list_raw = rospy.get_param('~d_list', None)
            if d_list_raw:
                D_LIST = np.array(d_list_raw)
            else:
                D_LIST = np.array(d_list_default)
            GS_TO_TESLA = rospy.get_param('~gs_to_tesla', gs_to_tesla_default)
            print("[INFO] Configuration loaded from rosparam")
        except (rospy.exceptions.ROSException, ConnectionRefusedError):
            print("[WARN] ROS param server not available, using defaults")
    else:
        # Use defaults
        print("[INFO] Configuration loaded from defaults (ROS not available)")

    # Set R_CORR (shared by both ROS and non-ROS)
    R_CORR = {
        1: np.array(r_corr_r1).reshape(3, 3),
        2: np.array(r_corr_r2).reshape(3, 3),
        3: np.array(r_corr_r3).reshape(3, 3),
        4: np.array(r_corr_r4).reshape(3, 3),
    }
    D_LIST = np.array(d_list_default)
    GS_TO_TESLA = gs_to_tesla_default


def get_sensor_group(sensor_id):
    """Get rotation correction group for sensor id (1-indexed)"""
    if 1 <= sensor_id <= 3:
        return 1
    elif 4 <= sensor_id <= 6:
        return 2
    elif 7 <= sensor_id <= 9:
        return 3
    elif 10 <= sensor_id <= 12:
        return 4
    return None


def quaternion_z_axis(q):
    """
    Extract the z-axis direction from a quaternion.

    Args:
        q: quaternion [w, x, y, z]

    Returns:
        np.ndarray: z-axis direction vector (3,)
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # z-axis of the rotation matrix
    z_axis = np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x**2 + y**2)
    ])

    return z_axis / np.linalg.norm(z_axis)  # Normalize


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to rotation matrix.

    Args:
        q: quaternion [w, x, y, z]

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])
    return R


def compute_rotation_error(R_est, R_true):
    """
    Compute orientation error between two rotation matrices.

    Formula: error = 2 * arcsin(norm(R_est - R_true, 'fro')) / (2*sqrt(2))

    Args:
        R_est: estimated rotation matrix (3x3)
        R_true: ground truth rotation matrix (3x3)

    Returns:
        float: orientation error in radians
    """
    error = np.arccos(np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1.0, 1.0))
    return error


# =============================================================================
# [2] DATA PROCESSING (Implement according to calibration)
# =============================================================================

def process_hall_data(sensor_readings):
    """
    Process raw Hall sensor readings to magnetic field measurements.

    Applies:
    - Rotation correction per sensor group (R_CORR)
    - Unit conversion (Gs -> Tesla)

    Args:
        sensor_readings: list of SensorReading messages from one slot

    Returns:
        np.ndarray: 3 x N matrix of magnetic field (Tesla)
    """
    global GS_TO_TESLA

    N = len(sensor_readings)
    B_meas = np.zeros((3, N))

    for i, reading in enumerate(sensor_readings):
        sensor_id = reading.id
        group = get_sensor_group(sensor_id)

        if group is not None and group in R_CORR:
            # Apply rotation correction
            b_raw = np.array([reading.x, reading.y, reading.z])
            b_corrected = R_CORR[group] @ b_raw
            # Unit conversion: Gs -> Tesla
            B_meas[:, i] = b_corrected * GS_TO_TESLA
        else:
            print(f"[WARN] Unknown sensor id: {sensor_id}")

    return B_meas


def build_D_cal(sensor_ids):
    """
    Build sensor calibration matrix D_cal.

    TODO: D_cal should contain the 3D positions of active sensors
    in the sensor array local frame.

    Args:
        sensor_ids: list of active sensor IDs (1-indexed)

    Returns:
        np.ndarray: 3 x N matrix of sensor positions
    """
    if D_LIST is None:
        print("[ERROR] D_LIST not loaded")
        return np.zeros((3, len(sensor_ids)))

    indices = [sid - 1 for sid in sensor_ids]
    return D_LIST[indices, :].T  # Transpose to 3 x N


# =============================================================================
# [3] MAPS ESTIMATOR CALL
# =============================================================================

def run_localization(D_cal, sources, B_meas_cell):
    """
    Run MaPS estimator.

    Args:
        D_cal: 3 x N sensor offset matrix
        sources: list of source dicts with 'p_Ci' and 'm_Ci'
        B_meas_cell: list of 3 x N magnetic field matrices (one per slot)

    Returns:
        tuple: (R_est, position, quaternion) or (None, None, None) on failure
    """
    try:
        R_est, p_est, _ = MaPS_Estimator(D_cal, sources, B_meas_cell)
        # Convert rotation matrix to quaternion [w, x, y, z]
        trace = np.trace(R_est)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R_est[2, 1] - R_est[1, 2]) * s
            qy = (R_est[0, 2] - R_est[2, 0]) * s
            qz = (R_est[1, 0] - R_est[0, 1]) * s
        else:
            if R_est[0, 0] > R_est[1, 1] and R_est[0, 0] > R_est[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R_est[0, 0] - R_est[1, 1] - R_est[2, 2])
                qw = (R_est[2, 1] - R_est[1, 2]) / s
                qx = 0.25 * s
                qy = (R_est[0, 1] + R_est[1, 0]) / s
                qz = (R_est[0, 2] + R_est[2, 0]) / s
            elif R_est[1, 1] > R_est[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R_est[1, 1] - R_est[0, 0] - R_est[2, 2])
                qw = (R_est[0, 2] - R_est[2, 0]) / s
                qx = (R_est[0, 1] + R_est[1, 0]) / s
                qy = 0.25 * s
                qz = (R_est[1, 2] + R_est[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R_est[2, 2] - R_est[0, 0] - R_est[1, 1])
                qw = (R_est[1, 0] - R_est[0, 1]) / s
                qx = (R_est[0, 2] + R_est[2, 0]) / s
                qy = (R_est[1, 2] + R_est[2, 1]) / s
                qz = 0.25 * s
        quat = np.array([qw, qx, qy, qz])
        quat = quat / np.linalg.norm(quat)  # Normalize
        return R_est, p_est, quat
    except Exception as e:
        print(f"[ERROR] MaPS estimation failed: {e}")
        return None, None, None


# =============================================================================
# [4] SERVICE HANDLER
# =============================================================================

def handle_localize_cycle(req):
    """
    Service handler for localize_cycle.

    CVT mode (4 slots):
        slot 0: arm1 data - B0
        slot 1: arm2 data - B0
        slot 2: diana7 data - B0
        slot 3: B0 (background, no pose)

    CCI mode (3 slots):
        slot 0: arm1 data (no background subtraction)
        slot 1: arm2 data (no background subtraction)
        slot 2: diana7 data (no background subtraction)
    """
    resp = LocalizeCycleResponse()
    resp.success = False

    try:
        # [1] Extract data from request
        cycle_id = req.cycle_id
        mode = req.mode  # 'CVT' or 'CCI'
        sensor_ids = list(req.sensor_ids)
        slot_data = list(req.slot_data)

        # [2] Data processing - build inputs for MaPS

        # Build slot lookup dict
        slot_dict = {slot.slot: slot for slot in slot_data}

        # Extract B0 for CVT mode (slot 3)
        B0 = None
        if mode == 'CVT' and 3 in slot_dict:
            B0 = process_hall_data(slot_dict[3].sensor_data)

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

            # Magnetic field measurement
            B_meas = process_hall_data(slot.sensor_data)

            # Background subtraction for CVT
            if mode == 'CVT' and B0 is not None:
                B_meas = B_meas - B0

            B_meas_cell.append(B_meas)

        D_cal = build_D_cal(sensor_ids)

        # [3] Run localization
        R_est, p_est, quat = run_localization(D_cal, sources, B_meas_cell)

        if p_est is not None:
            # [4] Compute errors
            p_true = np.array([
                req.ground_truth_pose.position.x,
                req.ground_truth_pose.position.y,
                req.ground_truth_pose.position.z
            ])
            quat_true = np.array([
                req.ground_truth_pose.orientation.w,
                req.ground_truth_pose.orientation.x,
                req.ground_truth_pose.orientation.y,
                req.ground_truth_pose.orientation.z
            ])

            position_error = np.linalg.norm(p_est - p_true)
            R_true = quaternion_to_rotation_matrix(quat_true)
            orientation_error = compute_rotation_error(R_est, R_true)

            # [5] Fill response
            resp.success = True
            resp.localization_pose.position.x = p_est[0]
            resp.localization_pose.position.y = p_est[1]
            resp.localization_pose.position.z = p_est[2]
            resp.localization_pose.orientation.x = quat[0]
            resp.localization_pose.orientation.y = quat[1]
            resp.localization_pose.orientation.z = quat[2]
            resp.localization_pose.orientation.w = quat[3]
            resp.position_error = float(position_error)
            resp.orientation_error = float(orientation_error)

            if ROS_AVAILABLE:
                rospy.loginfo(f"Localization succeeded: cycle={cycle_id}, "
                             f"pos=({p_est[0]:.4f}, {p_est[1]:.4f}, {p_est[2]:.4f}), "
                             f"pos_err={position_error:.6f}, ori_err={orientation_error:.6f}")
            else:
                print(f"[INFO] Localization succeeded: cycle={cycle_id}, "
                     f"pos=({p_est[0]:.4f}, {p_est[1]:.4f}, {p_est[2]:.4f}), "
                     f"pos_err={position_error:.6f}, ori_err={orientation_error:.6f}")
        else:
            if ROS_AVAILABLE:
                rospy.logwarn(f"Localization failed: cycle={cycle_id}")
            else:
                print(f"[WARN] Localization failed: cycle={cycle_id}")

    except Exception as e:
        if ROS_AVAILABLE:
            rospy.logerr(f"Service handler error: {e}")
        else:
            print(f"[ERROR] Service handler error: {e}")

    return resp


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not ROS_AVAILABLE:
        print("ROS not available. Cannot start service.")
        print("Use test_localization_service.py for standalone testing.")
        return

    rospy.init_node('gels_localization_service')
    rospy.loginfo("GELS Localization service started (FRAMEWORK MODE)")

    # Load configuration
    load_configuration()

    # Create service
    rospy.Service('localize_cycle', LocalizeCycle, handle_localize_cycle)

    rospy.loginfo("Service 'localize_cycle' is ready")
    rospy.spin()


if __name__ == '__main__':
    main()
