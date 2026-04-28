"""
MaPS Estimator: Implements Algorithm 1 from the paper.
6-DoF Pose Estimation from magnetic measurements.
"""

import numpy as np
from kabsch_solver import kabsch_solver


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
    # Compute eigenvalues (sorted in ascending order by default)
    evals = np.linalg.eigvalsh(X)

    # Sort eigenvalues in descending order: lambda_max >= lambda_med >= lambda_min
    evals_sorted = np.sort(evals)[::-1]
    lambda_max = evals_sorted[0]
    lambda_med = evals_sorted[1]
    lambda_min = evals_sorted[2]

    # Compute rho_hat using eigenvalue-based formula
    # rho_hat = 3 / (lambda_max * lambda_min) * (X + lambda_med * I) @ b_hat
    rho_hat = 3.0 / (lambda_max * lambda_min) * (X + lambda_med * np.eye(3)) @ b_hat

    return rho_hat


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
    D_cal = np.asarray(D_cal, dtype=float)
    N = D_cal.shape[1]
    M = len(B_meas_cell)

    # Precomputations
    ones_N = np.ones((N, 1))
    tol = 1e-10

    # Null-space basis of D_cal
    Q = null(D_cal)  # shape: (N, N-r)

    # Construct weight vector w for local field estimation:
    #   D_cal @ w = 0,   1^T w = 1
    if Q.size == 0:
        # Only valid fallback when the array is centered: D_cal @ 1 = 0
        if np.linalg.norm(D_cal @ ones_N) < tol:
            w = ones_N / N
        else:
            raise ValueError(
                "Cannot construct local-field weight vector w: "
                "null(D_cal) is empty and the array is not centered."
            )
    else:
        g = Q.T @ ones_N                      # shape: (N-r, 1)
        g_norm_sq = float(g.T @ g)
        if g_norm_sq < tol:
            raise ValueError(
                "Invalid sensor geometry: 1 has zero projection onto null(D_cal), "
                "so the local field estimator is not identifiable."
            )
        w = (Q @ g) / g_norm_sq              # shape: (N, 1)

    # Generalized centering matrix: Pw = I - w 1^T
    Pw = np.eye(N) - w @ ones_N.T                # shape: (N, N)

    # Constraint matrix for symmetric traceless gradient tensor X
    # Parametrization:
    # X = [[ x1,  x2,  x3],
    #      [ x2,  x4,  x5],
    #      [ x3,  x5, -(x1+x4)]]
    S_mat = np.array([
        [1,  0,  0,  0,  0],   # (1,1)
        [0,  1,  0,  0,  0],   # (2,1)
        [0,  0,  1,  0,  0],   # (3,1)
        [0,  1,  0,  0,  0],   # (1,2)
        [0,  0,  0,  1,  0],   # (2,2)
        [0,  0,  0,  0,  1],   # (3,2)
        [0,  0,  1,  0,  0],   # (1,3)
        [0,  0,  0,  0,  1],   # (2,3)
        [-1, 0,  0, -1,  0],   # (3,3)
    ], dtype=float)

    # C = (D^T \kron I3) S
    C_mat = np.kron(D_cal.T, np.eye(3)) @ S_mat
    if np.linalg.matrix_rank(C_mat, tol=tol) < 5:
        raise ValueError(
            "Gradient tensor is not uniquely identifiable: rank(C_mat) < 5. "
            "At least three magnetometers with non-degenerate geometry are required."
        )

    C_pinv = np.linalg.pinv(C_mat)

    # Local Quantity Estimation
    b_hat_locals = np.zeros((3, M))
    X_hat_locals = []
    rho_hats = np.zeros((3, M))

    for i in range(M):
        B_meas = np.asarray(B_meas_cell[i], dtype=float)  # expected shape: (3, N)
        if B_meas.shape != (3, N):
            raise ValueError(
                f"B_meas_cell[{i}] has shape {B_meas.shape}, but expected (3, {N})."
            )

        # Local field estimator: b_hat = B_meas @ w
        b_hat = B_meas @ w                                # shape: (3, 1)
        b_hat_locals[:, i] = b_hat.ravel()

        # Non-pairwise local gradient estimator:
        #   h = vec(B_meas @ Pw),   x_hat = C^† h,   X_hat = mat(S x_hat)
        B_tilde = B_meas @ Pw                             # shape: (3, N)
        h_vec = B_tilde.reshape(-1, order='F')            # vec(.) in column-major order
        x_hat = C_pinv @ h_vec                            # shape: (5,)
        X_hat = (S_mat @ x_hat).reshape((3, 3), order='F')

        X_hat_locals.append(X_hat)

        # (Eq. 13) Local relative displacement estimation
        # Original: 
        # rho_hats[:, i] = -3 * np.linalg.solve(X_hat, b_hat_locals[:, i])
        # Eigenvalue-based method to avoid matrix inversion:
        rho_hats[:, i] = compute_rho_hat_eigen(X_hat, b_hat_locals[:, i])
        
    # Global Pose Recovery
    # Standard rigid registration / absolute orientation via centered Kabsch

    # Stack global source positions
    p_Ck = np.column_stack([sources[i]['p_Ci'] for i in range(M)])   # shape: (3, M)
    rho_k = rho_hats                                                   # shape: (3, M)

    # Centroids
    p_C_bar = np.mean(p_Ck, axis=1, keepdims=True)                   # shape: (3, 1)
    rho_bar = np.mean(rho_k, axis=1, keepdims=True)                   # shape: (3, 1)

    # Centered correspondences
    p_C_tilde = p_Ck - p_C_bar                                       # shape: (3, M)
    rho_tilde = rho_k - rho_bar                                       # shape: (3, M)

    # Since p_C_k = p - R rho_k, the centered relation is:
    #   P_C_tilde = - R RHO_tilde
    # Therefore solve Kabsch on (P_C_tilde, -RHO_tilde)
    R_est = kabsch_solver(p_C_tilde, -rho_tilde)

    # Position recovery
    p_est = (p_C_bar + R_est @ rho_bar).reshape(3)
    
    # Store details for debugging/analysis
    details = {
        'b_hat_locals': b_hat_locals,
        'X_hat_locals': X_hat_locals,
        'rho_hats': rho_hats,
        'rho_bar': rho_bar,  
        'p_C_bar': p_C_bar
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
