"""
mag_dipole_model: Magnetic dipole field and gradient tensor model.
"""

import numpy as np


def mag_dipole_model(p: np.ndarray, m: np.ndarray, pC: np.ndarray):
    """
    Computes the magnetic field and gradient tensor for a dipole.

    Args:
        p: 3x1 observation position (ndarray shape (3,) or (3,1))
        m: 3x1 dipole moment (ndarray shape (3,) or (3,1))
        pC: 3x1 dipole center position (ndarray shape (3,) or (3,1))

    Returns:
        b: Magnetic field vector (3,)
        A: Magnetic gradient tensor (3x3)
    """
    p = p.flatten()
    m = m.flatten()
    pC = pC.flatten()

    mu0 = 4 * np.pi * 1e-7  # Magnetic permeability of free space
    coeff = mu0 / (4 * np.pi)

    r_vec = p - pC
    r = np.linalg.norm(r_vec)
    r_hat = r_vec / r

    # Magnetic field (standard dipole formula)
    b = coeff * (3 * np.dot(m, r_hat) * r_hat - m) / r**3

    # Magnetic gradient tensor A = grad(b)
    I3 = np.eye(3)
    m_dot_r = np.dot(m, r_vec)

    A = coeff * (3 / r**5) * (
        m_dot_r * I3
        + np.outer(r_vec, m)
        + np.outer(m, r_vec)
        - (5 * m_dot_r / r**2) * np.outer(r_vec, r_vec)
    )

    return b, A


def eul2rotm(euler: np.ndarray, order: str = 'ZYX') -> np.ndarray:
    """
    Converts Euler angles to rotation matrix.
    MATLAB's eul2rotm equivalent.

    Args:
        euler: [roll, pitch, yaw] in radians (or [rz, ry, rx] for ZYX)
        order: Rotation order, default 'ZYX' matches MATLAB

    Returns:
        R: 3x3 rotation matrix
    """
    rz, ry, rx = euler

    # Z rotation
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1]
    ])

    # Y rotation
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,          1, 0         ],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # X rotation
    Rx = np.array([
        [1, 0,          0          ],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])

    if order == 'ZYX':
        return Rz @ Ry @ Rx
    else:
        raise NotImplementedError(f"Order {order} not implemented")


def rotm2eul(R: np.ndarray, order: str = 'ZYX') -> np.ndarray:
    """
    Converts rotation matrix to Euler angles.
    MATLAB's rotm2eul equivalent.

    Args:
        R: 3x3 rotation matrix
        order: Rotation order, default 'ZYX' matches MATLAB

    Returns:
        euler: [roll, pitch, yaw] in radians
    """
    if order == 'ZYX':
        # Extract angles from rotation matrix
        # R = Rz @ Ry @ Rx
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0

        return np.array([rx, ry, rz])
    else:
        raise NotImplementedError(f"Order {order} not implemented")
