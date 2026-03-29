"""
MaPS Estimator: Implements Algorithm 1 from the paper.
6-DoF Pose Estimation from magnetic measurements.
"""

import numpy as np
from kabsch_solver import kabsch_solver


def MaPS_Estimator(D_cal: np.ndarray, sources: list, B_meas_cell: list):
    """
    MaPS Estimator: Implements Algorithm 1 from the paper.

    Args:
        D_cal: 3 x N matrix of sensor offsets in the sensor frame
        sources: List of dicts with fields 'p_Ci' (3,) and 'm_Ci' (3,)
        B_meas_cell: List of N matrices (3 x N) of magnetic measurements

    Returns:
        R_est: Estimated rotation matrix
        p_est: Estimated position (3,)
        details: Dict with intermediate results for debugging
    """
    D_cal = np.asarray(D_cal)
    N = D_cal.shape[1]
    M = len(B_meas_cell)

    # Precomputations
    # (Eq. 5) Null-space projection for local field estimation
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

    C_mat = np.kron(D_delta.T, np.eye(3)) @ S_mat
    C_pinv = np.linalg.pinv(C_mat)

    # Local Quantity Estimation
    b_hat_locals = np.zeros((3, M))
    X_hat_locals = []
    rho_hats = np.zeros((3, M))

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

        h_vec = B_delta.flatten(order='F')  # Column-major (Fortran) order to match MATLAB
        x_param = C_pinv @ h_vec
        X_hat = S_mat @ x_param
        X_hat = X_hat.reshape((3, 3), order='F')  # Fortran order for column-major
        X_hat_locals.append(X_hat)

        # (Eq. 13) Local relative displacement estimation
        rho_hats[:, i] = -3 * np.linalg.solve(X_hat, b_hat_locals[:, i])

    # Global Pose Recovery
    # (Eq. 14-17) Orientation Recovery via Kabsch Algorithm
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

    # Store details for debugging/analysis
    details = {
        'b_hat_locals': b_hat_locals,
        'X_hat_locals': X_hat_locals,
        'rho_hats': rho_hats,
        'p_ests': p_ests
    }

    return R_est, p_est, details


def null(A: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """
    Computes the null space of matrix A.
    Equivalent to MATLAB's null(A).

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
