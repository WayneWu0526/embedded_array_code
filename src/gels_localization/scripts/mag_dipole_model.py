"""
mag_dipole_model: Magnetic dipole field and gradient tensor model.
"""

import numpy as np


def mag_dipole_model(p: np.ndarray, m: np.ndarray, pC: np.ndarray,
                     order: int = 1, L: float = 240.0e-3, D: float = 40.0e-3):
    """
    Computes the magnetic field and gradient tensor for a dipole with optional higher-order corrections.

    Args:
        p: 3x1 observation position (ndarray shape (3,) or (3,1))
        m: 3x1 dipole moment (ndarray shape (3,) or (3,1))
        pC: 3x1 dipole center position (ndarray shape (3,) or (3,1))
        order: Order of the model (1, 3, or 5). Default is 1.
               1: Standard point dipole
               3: Adds octupole correction (b3)
               5: Adds b3 and hexapole correction (b5)
        L: Length of the cylindrical magnet in mm
        D: Diameter of the cylindrical magnet in mm
        
    Returns:
        b: Magnetic field vector (3,)
        A: Magnetic gradient tensor (3x3) - currently unchanged from order=1

    Raises:
        ValueError: If order is not 1, 3, or 5
    """
    if order not in [1, 3, 5]:
        raise ValueError("order must be 1, 3, or 5")

    p = p.flatten()
    m = m.flatten()
    pC = pC.flatten()

    mu0 = 4 * np.pi * 1e-7  # Magnetic permeability of free space
    coeff = mu0 / (4 * np.pi)

    r_vec = p - pC
    r = np.linalg.norm(r_vec)
    r_hat = r_vec / r

    # Magnetic field - order 1 (standard dipole formula)
    b1 = coeff * (3 * np.dot(m, r_hat) * r_hat - m) / r**3

    # Total field starts with order 1
    b = b1.copy()

    # Higher-order corrections for finite-sized cylindrical magnet
    if order >= 3:
        beta = D / L  # aspect ratio
        m_hat = m / (np.linalg.norm(m) + 1e-10)

        m_dot_p_hat = np.dot(m_hat, r_hat)
        m_dot_p_hat_sq = m_dot_p_hat ** 2

        # B3 coefficient: (μ0/4π) * (1/p⁵) * (L/2)² * ((4-3β²)/8)
        coef_b3 = coeff * (1 / r**5) * (L / 2)**2 * ((4 - 3 * beta**2) / 8)

        # B3 matrix: (35*(m̂ᵀ*p̂)² - 15)*p̂*p̂ᵀ - (15*(m̂ᵀ*p̂)² - 3)*I
        term1 = (35 * m_dot_p_hat_sq - 15) * np.outer(r_hat, r_hat)
        term2 = (15 * m_dot_p_hat_sq - 3) * np.eye(3)
        matrix_term = term1 - term2

        # B3 = coef_b3 * matrix_term @ m
        b3 = coef_b3 * (matrix_term @ m)
        b = b1 + b3

    if order >= 5:
        beta = D / L
        m_hat = m / (np.linalg.norm(m) + 1e-10)

        m_dot_p_hat = np.dot(m_hat, r_hat)
        m_dot_p_hat_sq = m_dot_p_hat ** 2
        m_dot_p_hat_4 = m_dot_p_hat_sq ** 2

        # B5 coefficient: (μ0/4π) * (1/p⁷) * (L/2)⁴ * ((15β⁴-60β²+24)/64)
        coef_b5 = coeff * (1 / r**7) * (L / 2)**4 * ((15 * beta**4 - 60 * beta**2 + 24) / 64)

        # B5 matrix terms
        coeff_p = 231 * m_dot_p_hat_4 - (105 / 2) * m_dot_p_hat_sq + 35
        coeff_I = 105 * m_dot_p_hat_4 - 70 * m_dot_p_hat_sq + 5

        term1 = coeff_p * np.outer(r_hat, r_hat)
        term2 = coeff_I * np.eye(3)
        matrix_term = term1 - term2

        # B5 = coef_b5 * matrix_term @ m
        b5 = coef_b5 * (matrix_term @ m)
        b = b1 + b3 + b5

    # Magnetic gradient tensor A = grad(b) - unchanged for now
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
