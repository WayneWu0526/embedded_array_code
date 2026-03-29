"""
Nonlinear Least Squares solver using Levenberg-Marquardt for 6-DoF pose estimation.
This serves as a baseline comparison for the MaPS estimator.
"""

import numpy as np
from scipy.optimize import least_squares


def nlls_lm_solver(D_cal: np.ndarray, sources: list, B_meas_cell: list,
                   p0: np.ndarray = None, euler0: np.ndarray = None,
                   max_nfev: int = 100):
    """
    Nonlinear Least Squares pose estimation using Levenberg-Marquardt.

    Args:
        D_cal: 3 x N matrix of sensor offsets in the sensor frame
        sources: List of dicts with fields 'p_Ci' (3,) and 'm_Ci' (3,)
        B_meas_cell: List of M matrices (3 x N) of magnetic measurements
        p0: Initial position estimate (3,), if None uses MaPS result or zeros
        euler0: Initial euler angles [roll, pitch, yaw] in degrees (3,), if None uses MaPS result or zeros
        max_nfev: Maximum number of function evaluations

    Returns:
        R_est: Estimated rotation matrix (3x3)
        p_est: Estimated position (3,)
        result: scipy.optimize.least_squares result object
    """
    from mag_dipole_model import mag_dipole_model, rotm2eul

    M = len(sources)
    N = D_cal.shape[1]

    # If no initial guess provided, use zeros
    if p0 is None:
        p0 = np.zeros(3)
    if euler0 is None:
        euler0 = np.zeros(3)

    # Initial state: [position(3), euler(3)]
    x0 = np.concatenate([p0, np.deg2rad(euler0)])

    # Define residual function
    def residuals(x):
        p = x[:3]
        euler = x[3:6]
        R = eul2rotm_custom(euler)  # ZYX convention

        total_residual = np.zeros(3 * M * N)

        idx = 0
        for i in range(M):
            for j in range(N):
                # Sensor position in global frame
                p_sensor_global = p + R @ D_cal[:, j]

                # Modeled field at sensor
                b_global, _ = mag_dipole_model(p_sensor_global,
                                               sources[i]['m_Ci'],
                                               sources[i]['p_Ci'])

                # Transform to sensor frame
                b_modeled = R.T @ b_global

                # Residual (measured - modeled)
                total_residual[idx:idx+3] = B_meas_cell[i][:, j] - b_modeled
                idx += 3

        return total_residual

    # Run Levenberg-Marquardt optimization
    result = least_squares(
        residuals,
        x0,
        method='lm',
        max_nfev=max_nfev,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12
    )

    # Extract results
    p_est = result.x[:3]
    euler_est = np.rad2deg(result.x[3:6])
    R_est = eul2rotm_custom(result.x[3:6])

    return R_est, p_est, result


def eul2rotm_custom(euler):
    """
    Euler angle to rotation matrix (ZYX convention).
    Custom implementation to avoid import issues.
    """
    rz, ry, rx = euler

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,          0,          1]])

    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [ 0,          1, 0         ],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rx = np.array([[1, 0,          0          ],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])

    return Rz @ Ry @ Rx


def nlls_lm_solver_from_maps(D_cal: np.ndarray, sources: list, B_meas_cell: list,
                              R_init: np.ndarray = None, p_init: np.ndarray = None,
                              max_nfev: int = 100):
    """
    NLLS-LM solver using MaPS estimate as initial guess.
    This provides a fairer comparison by starting from the same initial condition.

    Args:
        D_cal: 3 x N matrix of sensor offsets
        sources: List of source dicts
        B_meas_cell: List of measurement matrices
        R_init: Initial rotation matrix from MaPS
        p_init: Initial position from MaPS
        max_nfev: Maximum function evaluations

    Returns:
        R_est, p_est, result
    """
    from mag_dipole_model import rotm2eul

    # Use MaPS initial guess if provided
    if p_init is None:
        p_init = np.zeros(3)
    if R_init is None:
        R_init = np.eye(3)

    # Convert rotation matrix to euler angles
    euler_init = rotm2eul(R_init, 'ZYX')

    return nlls_lm_solver(D_cal, sources, B_meas_cell,
                         p0=p_init, euler0=euler_init, max_nfev=max_nfev)
