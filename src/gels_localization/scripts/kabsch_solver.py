""" kabsch_solver: Solves the Orthogonal Procrustes problem. Finds R in SO(3) that minimizes ||V - R @ U||_F """
import numpy as np


def kabsch_solver(V: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Solve the Orthogonal Procrustes problem:
        min_{R in SO(3)} ||V - R @ U||_F

    Args:
        V: (3, K) matrix of target/reference vectors
        U: (3, K) matrix of source vectors

    Returns:
        R: (3, 3) rotation matrix in SO(3)
    """
    if V.shape != U.shape:
        raise ValueError(f"Shape mismatch: V{V.shape} vs U{U.shape}")
    if V.shape[0] != 3:
        raise ValueError(f"Expected 3xK inputs, got {V.shape}")

    H = V @ U.T
    L, _, Wh = np.linalg.svd(H)

    d = np.sign(np.linalg.det(L @ Wh))
    if d == 0:
        d = 1.0

    D = np.diag([1.0, 1.0, d])
    R = L @ D @ Wh

    return R