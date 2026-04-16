#!/usr/bin/env python3
"""
full_ellipsoid_fit.py - Full ellipsoid fitting for QMC6309 sensors

适用于所有传感器：
- 需要完整的 3×3 校正矩阵
- 包括 soft iron 效应和轴不正交

参考论文公式:
    (b_raw - o_i)^T * W * (b_raw - o_i) = 1
其中 W = C_i^T * C_i

校正后:
    b_corr = C_i * (b_raw - o_i)
"""

import numpy as np
from typing import Tuple, Dict
from scipy import optimize


def full_ellipsoid_fit(b_raw: np.ndarray, initial_guess: dict = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Full ellipsoid fitting using least squares

    使用椭球标准方程的代数拟合方法:
    Ax² + By² + Cz² + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1

    通过矩阵形式求解:
    [x² y² z² 2xy 2xz 2yz 2x 2y 2z] · [A B C D E F G H I]^T = 1

    然后从二次型矩阵提取 offset 和 correction matrix

    Args:
        b_raw: N×3 matrix of raw magnetic field measurements
        initial_guess: Optional initial guess for offset

    Returns:
        o_i: Hard iron offset (3,)
        C_i: Correction matrix (3×3)
        info: Fitting information dict
    """
    N = b_raw.shape[0]

    # 构建设计矩阵
    # 每个点对应方程: Ax² + By² + Cz² + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
    x = b_raw[:, 0]
    y = b_raw[:, 1]
    z = b_raw[:, 2]

    # 设计矩阵 (N × 9)
    A_design = np.column_stack([
        x**2,      # A
        y**2,      # B
        z**2,      # C
        2*x*y,     # D
        2*x*z,     # E
        2*y*z,     # F
        2*x,       # G
        2*y,       # H
        2*z        # I
    ])

    # 目标向量
    b_target = np.ones(N)

    # 最小二乘求解
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A_design, b_target, rcond=None)
    except Exception as e:
        raise ValueError(f"Ellipsoid fit failed: {e}")

    A, B, C, D, E, F, G, H, I = coeffs

    # 构建二次型矩阵 W 和线性项向量
    # 椭球方程: [x y z] · W · [x y z]^T + 2·[x y z]·[G H I]^T = 1
    W = np.array([
        [A, D, E],
        [D, B, F],
        [E, F, C]
    ])
    linear = np.array([G, H, I])

    # 计算 offset: o = -W^{-1} · linear
    try:
        W_inv = np.linalg.inv(W)
    except np.linalg.LinAlgError:
        # W 奇异，使用伪逆
        W_inv = np.linalg.pinv(W)

    o_i = -W_inv @ linear

    # 计算中心化后的二次型矩阵
    # (b - o)^T · W · (b - o) = b^T·W·b - 2·o^T·W·b + o^T·W·o = 1
    # 展开后相当于 b^T·W·b 的系数不变，只是平移到中心

    # 提取 correction matrix: C = W^{-1/2}
    # 或者使用 Cholesky 分解: W = L·L^T, C = L^{-1}
    try:
        # 方法1: Cholesky 分解
        L = np.linalg.cholesky(W)
        C_i = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        # W 不是正定矩阵（数值误差），使用特征值分解
        eigenvalues_w, eigenvectors_w = np.linalg.eigh(W)
        # 确保特征值为正
        eigenvalues_w = np.maximum(eigenvalues_w, 1e-10)
        # W = V·Λ·V^T, W^{-1/2} = V·Λ^{-1/2}·V^T
        C_i = eigenvectors_w @ np.diag(1.0 / np.sqrt(eigenvalues_w)) @ eigenvectors_w.T

    # 验证拟合质量
    b_centered = b_raw - o_i
    b_corr = b_centered @ C_i.T  # 等价于 C_i @ b_centered
    radius_corr = np.linalg.norm(b_corr, axis=1)
    radius_raw = np.linalg.norm(b_raw - o_i, axis=1)

    # 计算拟合误差
    predicted = np.sum(b_centered @ W * b_centered, axis=1)
    fit_error = np.std(np.abs(predicted - 1.0))

    # 特征值分析
    eigenvalues_w, _ = np.linalg.eigh(W)
    eigenvalues_w = np.sort(eigenvalues_w)[::-1]

    info = {
        'method': 'full_ellipsoid',
        'W_matrix': W.tolist(),
        'eigenvalues_W': eigenvalues_w.tolist(),
        'eigenvalue_ratio': float(eigenvalues_w[0] / eigenvalues_w[2]) if eigenvalues_w[2] > 0 else float('inf'),
        'offset': o_i.tolist(),
        'C_matrix': C_i.tolist(),
        'radius_raw_std': float(np.std(radius_raw)),
        'radius_corr_std': float(np.std(radius_corr)),
        'radius_corr_mean': float(np.mean(radius_corr)),
        'fit_error': float(fit_error),
        'improvement_ratio': float(np.std(radius_raw) / np.std(radius_corr)) if np.std(radius_corr) > 0 else float('inf'),
        'rank': int(rank),
        'singular_values': s.tolist() if len(s) > 0 else [],
    }

    return o_i, C_i, info


def full_ellipsoid_fit_iterative(b_raw: np.ndarray, initial_offset: np.ndarray = None,
                                  max_iter: int = 100, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    迭代优化版本的 full ellipsoid fitting

    对于噪声较大的数据更鲁棒

    Args:
        b_raw: N×3 matrix of raw magnetic field measurements
        initial_offset: 初始 offset 猜测
        max_iter: 最大迭代次数
        tolerance: 收敛阈值

    Returns:
        o_i, C_i, info
    """
    # 初始猜测
    if initial_offset is None:
        o_i = np.mean(b_raw, axis=0)
    else:
        o_i = np.copy(initial_offset)

    for iteration in range(max_iter):
        b_centered = b_raw - o_i

        # 固定 offset，估计 W
        # 使用椭球拟合
        x = b_centered[:, 0]
        y = b_centered[:, 1]
        z = b_centered[:, 2]

        A_design = np.column_stack([
            x**2, y**2, z**2,
            2*x*y, 2*x*z, 2*y*z,
            2*x, 2*y, 2*z
        ])
        b_target = np.ones(b_raw.shape[0])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_design, b_target, rcond=None)
        except:
            break

        A, B, C, D, E, F, G, H, I = coeffs

        W = np.array([
            [A, D, E],
            [D, B, F],
            [E, F, C]
        ])
        linear = np.array([G, H, I])

        # 更新 offset
        try:
            W_inv = np.linalg.inv(W)
        except:
            W_inv = np.linalg.pinv(W)

        o_new = -W_inv @ linear

        # 检查收敛
        if np.linalg.norm(o_new - o_i) < tolerance:
            o_i = o_new
            break

        o_i = o_new

    # 最终估计 C
    b_centered = b_raw - o_i
    x = b_centered[:, 0]
    y = b_centered[:, 1]
    z = b_centered[:, 2]

    A_design = np.column_stack([
        x**2, y**2, z**2,
        2*x*y, 2*x*z, 2*y*z,
        2*x, 2*y, 2*z
    ])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_design, b_target, rcond=None)
    except:
        coeffs = [1, 1, 1, 0, 0, 0, 0, 0, 0]

    A, B, C, D, E, F, G, H, I = coeffs

    W = np.array([
        [A, D, E],
        [D, B, F],
        [E, F, C]
    ])

    try:
        L = np.linalg.cholesky(W)
        C_i = np.linalg.inv(L)
    except:
        eigenvalues_w, eigenvectors_w = np.linalg.eigh(W)
        eigenvalues_w = np.maximum(eigenvalues_w, 1e-10)
        C_i = eigenvectors_w @ np.diag(1.0 / np.sqrt(eigenvalues_w)) @ eigenvectors_w.T

    # 验证
    b_corr = b_centered @ C_i.T
    radius_corr = np.linalg.norm(b_corr, axis=1)

    info = {
        'method': 'full_ellipsoid_iterative',
        'iterations': iteration + 1,
        'converged': iteration < max_iter - 1,
        'offset': o_i.tolist(),
        'C_matrix': C_i.tolist(),
        'radius_corr_std': float(np.std(radius_corr)),
        'radius_corr_mean': float(np.mean(radius_corr)),
    }

    return o_i, C_i, info
