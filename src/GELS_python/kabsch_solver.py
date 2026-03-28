"""
kabsch_solver: Solves the Orthogonal Procrustes problem.
Finds R in SO(3) that minimizes ||V - R @ U||_F
"""

import numpy as np


def kabsch_solver(V: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Kabsch algorithm for rotation estimation.

    Args:
        V: 3 x K matrix (global reference vectors)
        U: 3 x K matrix (local estimated vectors)

    Returns:
        R: 3x3 rotation matrix in SO(3)
    """
    H = V @ U.T

    # numpy.linalg.svd returns (L, S, Wh) where H = L @ S @ Wh
    # Wh is the transposed right singular vectors (Vh in math notation)
    L, _, Wh = np.linalg.svd(H)

    # Reflection check: ensure proper rotation (det = 1)
    d = np.linalg.det(L @ Wh)

    # Force rotation matrix to have det = 1 (SO(3))
    D = np.diag([1, 1, d])

    # Kabsch: R = L @ D @ Wh
    R = L @ D @ Wh

    return R
