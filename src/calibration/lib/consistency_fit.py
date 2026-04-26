#!/usr/bin/env python3
"""
consistency_fit.py - Phase 2: 一致性校准算法

功能：
1. 读取 consistency 数据 CSV（已过椭球+旋转校正）
2. 计算 D_i (缩放矩阵) 和 e_i (残余偏置)

使用方法:
    from calibration.lib.consistency_fit import consistency_fit, batch_consistency_fit

参考论文公式:
    b_i^{final} = D_i * b_i^{corr} + e_i           (Phase 2 一致性校正)
"""

import numpy as np
import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from sensor_array_config.base import get_config, SensorArrayConfig, ConsistencyParamsSet, ConsistencyParams, IntrinsicParamsSet

# ============== 常量 ==============
VOLTAGE_ORDER = (5, 4, 3, 2, 1)

# ============== 椭球校正与放大系数 ==============
def apply_ellipsoid_correction_to_data(b_raw: np.ndarray, o_i: np.ndarray, C_i: np.ndarray) -> np.ndarray:
    """
    对原始传感器数据进行椭球校正

    公式: b_corr = C_i * (b_raw - o_i)

    Args:
        b_raw: (N, 3) 原始数据
        o_i: (3,) offset 向量
        C_i: (3, 3) 校正矩阵

    Returns:
        b_corr: (N, 3) 校正后数据
    """
    b_raw = np.asarray(b_raw)
    o_i = np.asarray(o_i)
    C_i = np.asarray(C_i)
    b_centered = b_raw - o_i
    b_corr = b_centered @ C_i.T
    return b_corr


def apply_r_corr_rotation(b_corr: np.ndarray, r_corr: np.ndarray) -> np.ndarray:
    """
    应用 R_CORR 旋转变换

    将传感器局部坐标系转换到参考坐标系
    公式: b_rot = R_CORR @ b_corr

    Args:
        b_corr: (N, 3) 椭球校正后数据
        r_corr: (3, 3) 旋转矩阵

    Returns:
        b_rot: (N, 3) 旋转变换后数据
    """
    b_corr = np.asarray(b_corr)
    r_corr = np.asarray(r_corr)
    return b_corr @ r_corr.T


def build_r_corr_dict(hardware_params) -> Dict[int, np.ndarray]:
    """
    从 hardware params 构建 sensor_id -> R_CORR 矩阵的字典

    Args:
        hardware_params: SensorArrayHardwareParams 对象

    Returns:
        R_CORR dict: {sensor_id: np.ndarray(3, 3)}
    """
    r_corr_dict = {}
    for entry in hardware_params.R_CORR:
        mat = np.array(entry.matrix).reshape(3, 3, order='F')
        for sid in entry.sensor_ids:
            r_corr_dict[sid] = mat
    return r_corr_dict


def compute_amplification_factor(b_raw: np.ndarray, b_corr: np.ndarray) -> Dict[str, float]:
    """
    计算放大系数

    对比原始数据和校正后数据的磁场向量模长

    Args:
        b_raw: (N, 3) 原始数据
        b_corr: (N, 3) 校正后数据

    Returns:
        Dict with raw_norm, corr_norm, and amplification_factor
    """
    b_raw = np.asarray(b_raw)
    b_corr = np.asarray(b_corr)

    # 计算每个样本的向量模长
    raw_norms = np.linalg.norm(b_raw, axis=1)
    corr_norms = np.linalg.norm(b_corr, axis=1)

    raw_norm_mean = float(np.mean(raw_norms))
    corr_norm_mean = float(np.mean(corr_norms))

    # 放大系数 = 校正后模长均值 / 原始数据模长均值
    amp_factor = corr_norm_mean / (raw_norm_mean + 1e-10)

    return {
        'raw_norm_mean': raw_norm_mean,
        'corr_norm_mean': corr_norm_mean,
        'amplification_factor': amp_factor
    }


def apply_ellipsoid_correction_to_csv(
    csv_path: Path,
    intrinsic_params: IntrinsicParamsSet,
    n_sensors: int = 12
) -> np.ndarray:
    """
    对CSV文件中的原始数据进行椭球校正

    Args:
        csv_path: CSV文件路径
        intrinsic_params: 椭球校正参数集合
        n_sensors: 传感器数量

    Returns:
        corrected_data: (N, n_sensors, 3) 校正后的数据
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith('sensor_')]
    raw_data = df[cols].values

    n_samples = raw_data.shape[0]
    corrected_data = np.zeros((n_samples, n_sensors, 3))

    for i in range(n_sensors):
        sensor_id = i + 1
        o_i = np.array(intrinsic_params.params[sensor_id].o_i)
        C_i = np.array(intrinsic_params.params[sensor_id].C_i)

        # 提取该传感器所有样本的原始数据 (N, 3)
        sensor_raw = raw_data[:, i*3:(i+1)*3]
        # 应用椭球校正
        corrected_data[:, i, :] = apply_ellipsoid_correction_to_data(sensor_raw, o_i, C_i)

    return corrected_data


def load_csv_data_raw(csv_path: Path, n_sensors: int = 12) -> np.ndarray:
    """
    加载CSV并提取传感器原始数据（不进行任何校正）

    Args:
        csv_path: CSV 文件路径
        n_sensors: 传感器数量

    Returns:
        data: (N_samples, n_sensors, 3) 原始数据
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith('sensor_')]
    data = df[cols].values
    n = data.shape[0]
    reshaped = np.zeros((n, n_sensors, 3))
    for i in range(n_sensors):
        reshaped[:, i, :] = data[:, i*3:(i+1)*3]
    return reshaped


# ============== 通道-轴 自动检测 ==============
def auto_detect_channel_axis_mapping(
    csv_dir: Path,
    data_files: Dict[str, str] = None,
    n_sensors: int = 12,
) -> Dict[int, str]:
    """
    自动检测通道与磁场轴的对应关系。

    原理：激活某个通道时，哪个方向的磁场绝对值增量最大，
    说明这个通道主要驱动该方向。

    Args:
        csv_dir: CSV 文件目录
        data_files: 数据文件映射
        n_sensors: 传感器数量

    Returns:
        channel_to_axis: {1: 'x', 2: 'y', 3: 'z'} 或类似映射
    """
    if data_files is None:
        data_files = {
            'background': 'consistency_calib_background.csv',
            'ch1_positive': 'consistency_calib_ch1_positive.csv',
            'ch1_negative': 'consistency_calib_ch1_negative.csv',
            'ch2_positive': 'consistency_calib_ch2_positive.csv',
            'ch2_negative': 'consistency_calib_ch2_negative.csv',
            'ch3_positive': 'consistency_calib_ch3_positive.csv',
            'ch3_negative': 'consistency_calib_ch3_negative.csv',
        }

    # 加载背景数据
    bg_path = Path(csv_dir) / data_files['background']
    if not bg_path.exists():
        raise FileNotFoundError(f"Background CSV not found: {bg_path}")
    bg_data = load_csv_data(bg_path, n_sensors)
    bg_mean = compute_stable_mean(bg_data)  # (n_sensors, 3)

    channel_to_axis = {}
    axis_names = ['x', 'y', 'z']

    for ch in [1, 2, 3]:
        pos_path = Path(csv_dir) / data_files[f'ch{ch}_positive']
        neg_path = Path(csv_dir) / data_files[f'ch{ch}_negative']

        if not pos_path.exists() or not neg_path.exists():
            print(f"[WARN] CH{ch} data files not found, using default mapping")
            default_map = {1: 'x', 2: 'y', 3: 'z'}
            channel_to_axis[ch] = default_map[ch]
            continue

        pos_data = load_csv_data(pos_path, n_sensors)
        neg_data = load_csv_data(neg_path, n_sensors)
        pos_mean = compute_stable_mean(pos_data)
        neg_mean = compute_stable_mean(neg_data)

        # 计算相对于背景的增量（取正负极性的平均幅度）
        delta_pos = pos_mean - bg_mean
        delta_neg = neg_mean - bg_mean

        # 检查 delta 是否全为 0 (数据损坏)
        if np.allclose(delta_pos, 0) and np.allclose(delta_neg, 0):
            print(f"[WARN] CH{ch} data appears to be empty/zero, using default X axis")
            channel_to_axis[ch] = 'x'
            continue

        # 计算每个方向的平均绝对增量
        mean_abs_x = (np.abs(delta_pos[:, 0]).mean() + np.abs(delta_neg[:, 0]).mean()) / 2
        mean_abs_y = (np.abs(delta_pos[:, 1]).mean() + np.abs(delta_neg[:, 1]).mean()) / 2
        mean_abs_z = (np.abs(delta_pos[:, 2]).mean() + np.abs(delta_neg[:, 2]).mean()) / 2

        abs_components = [mean_abs_x, mean_abs_y, mean_abs_z]
        max_idx = np.argmax(abs_components)
        detected_axis = axis_names[max_idx]

        print(f"  [Auto-detect] CH{ch} -> {detected_axis.upper()} axis "
              f"(|dx|={mean_abs_x:.4f}, |dy|={mean_abs_y:.4f}, |dz|={mean_abs_z:.4f})")

        channel_to_axis[ch] = detected_axis

    return channel_to_axis


# ============== 场条件枚举 ==============
class FieldCondition:
    ZERO = 0       # 0场 (background)
    POS_X = 1      # +Bx (CH1 positive)
    NEG_X = 2      # -Bx (CH1 negative)
    POS_Y = 3      # +By (CH2 positive)
    NEG_Y = 4      # -By (CH2 negative)
    POS_Z = 5      # +Bz (CH3 positive)
    NEG_Z = 6      # -Bz (CH3 negative)

    @staticmethod
    def all_conditions():
        return [FieldCondition.POS_X, FieldCondition.NEG_X,
                FieldCondition.POS_Y, FieldCondition.NEG_Y,
                FieldCondition.POS_Z, FieldCondition.NEG_Z]

    @staticmethod
    def to_string(cond):
        return {0: '0', 1: '+x', 2: '-x', 3: '+y', 4: '-y', 5: '+z', 6: '-z'}.get(cond, '?')


# ============== 数据类 ==============
@dataclass
class ConsistencyResult:
    """单颗传感器一致性校准结果"""
    sensor_id: int
    csv_file: str
    D_i: List[List[float]]  # 缩放矩阵 (3×3)
    e_i: List[float]        # 残余偏置 (3,)
    fit_info: Dict

    def to_dict(self) -> dict:
        return asdict(self)


# ============== 数据加载 ==============
def load_csv_data(csv_path: Path, n_sensors: int = 12) -> np.ndarray:
    """
    加载 CSV 并提取传感器数据

    Args:
        csv_path: CSV 文件路径
        n_sensors: 传感器数量

    Returns:
        data: (N_samples, n_sensors, 3) 原始数据
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith('sensor_')]
    data = df[cols].values
    n = data.shape[0]
    reshaped = np.zeros((n, n_sensors, 3))
    for i in range(n_sensors):
        reshaped[:, i, :] = data[:, i*3:(i+1)*3]
    return reshaped


def compute_stable_mean(data: np.ndarray, skip: int = 5) -> np.ndarray:
    """
    计算稳定段均值，跳过前后各 skip 个样本

    Args:
        data: (N, 12, 3) 数据
        skip: 跳过的样本数

    Returns:
        mean: (12, 3) 稳定段均值
    """
    n = data.shape[0]
    if n == 0:
        return np.zeros((12, 3))
    
    start = min(skip, max(0, n // 10))
    end = max(start + 1, min(n, n - skip, n * 9 // 10))
    
    # 打印切片范围以调试
    # print(f"[DEBUG] compute_stable_mean: n={n}, slice=[{start}:{end}]")
    
    segment = data[start:end]
    if segment.shape[0] == 0:
        return np.zeros((12, 3))
        
    return segment.mean(axis=0)


# ============== 核心算法 ==============
def fit_D_and_e(
    b_norm_dict: Dict[int, np.ndarray],
    n_sensors: int = 12,
    channel_to_axis: Dict[int, str] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    拟合缩放矩阵 D_i 和残余偏置 e_i

    Phase 2 核心算法：使 n_sensors 颗传感器在各个磁场方向上响应一致。

    原理：
    - 在某一磁场方向上，阵列平均增量为 delta_bar
    - 各传感器增量 delta[sid] 应与 delta_bar 成比例
    - 比例系数即为 D_i 的元素（D_i 为完整 3x3 矩阵，可校正交叉轴耦合）

    Args:
        b_norm_dict: {condition: (n_sensors, 3)} 椭球校正后的数据
        n_sensors: 传感器数量
        channel_to_axis: {ch: axis} 通道到驱动轴的映射（仅用于自动检测，不影响拟合）

    Returns:
        D_list: [D_1, ..., D_n], 每个 D_i 为 3x3 矩阵
        e_list: [e_1, ..., e_n], 每个 e_i ∈ ℝ³
    """
    conditions = FieldCondition.all_conditions()

    # Step 1: 计算每颗传感器在每个条件下的增量
    delta = {}
    for sid in range(1, n_sensors + 1):
        delta[sid] = {}
        b_zero = b_norm_dict[FieldCondition.ZERO][sid - 1]
        for cond in conditions:
            delta[sid][cond] = b_norm_dict[cond][sid - 1] - b_zero

    # Step 2: 计算阵列平均增量
    delta_bar = {}
    for cond in conditions:
        delta_bar[cond] = np.mean([delta[sid][cond] for sid in range(1, n_sensors + 1)], axis=0)

    # 打印 6 个场条件的平均增量
    print("\n  [DEBUG] 阵列平均增量 delta_bar:")
    print(f"  {'Condition':<10} {'X':>12} {'Y':>12} {'Z':>12}")
    print(f"  {'-'*46}")
    for cond in conditions:
        cond_str = FieldCondition.to_string(cond)
        print(f"  {cond_str:<10} {delta_bar[cond][0]:>12.6f} {delta_bar[cond][1]:>12.6f} {delta_bar[cond][2]:>12.6f}")

    # Step 3: 拟合 D_i（完整 3x3 矩阵，使用最小二乘）
    # 对于每个传感器，解 D_i @ delta_c = delta_bar[c] for all c
    # 这是一个线性最小二乘问题: D_i @ M = B, 其中 M 和 B 都是 3x6 矩阵
    D_list = []
    for sid in range(1, n_sensors + 1):
        # 构建矩阵 M = [delta[sid][c1], delta[sid][c2], ..., delta[sid][c6]] (3x6)
        # 构建矩阵 B = [delta_bar[c1], delta_bar[c2], ..., delta_bar[c6]] (3x6)
        M = np.stack([delta[sid][c] for c in conditions], axis=1)  # 3x6
        B = np.stack([delta_bar[c] for c in conditions], axis=1)   # 3x6

        # 解 D_i @ M = B，使用最小二乘
        # D_i = B @ M^T @ (M @ M^T)^-1 或使用 lstsq
        D_i, residuals, rank, s = np.linalg.lstsq(M.T, B.T, rcond=None)  # M.T: 6x3, B.T: 6x3, result: 3x3
        D_i = D_i.T  # 转置得到 3x3
        D_list.append(D_i)

    # Step 4: 计算0场阵列平均向量
    b_bar_zero = np.mean([b_norm_dict[FieldCondition.ZERO][sid - 1] for sid in range(1, n_sensors + 1)], axis=0)

    # Step 5: 拟合 e_i
    e_list = []
    for sid in range(1, n_sensors + 1):
        b_zero = b_norm_dict[FieldCondition.ZERO][sid - 1]
        e_i = b_bar_zero - D_list[sid - 1] @ b_zero
        e_list.append(e_i)

    return D_list, e_list


def apply_consistency_correction(b_ellipsoid: np.ndarray, D_i: np.ndarray, e_i: np.ndarray) -> np.ndarray:
    """
    应用一致性校正公式

    Formula: b_final = D_i * b_ellipsoid + e_i

    Args:
        b_ellipsoid: (N, 3) 或 (3,) 椭球校正后数据
        D_i: (3, 3) 缩放矩阵（对角矩阵）
        e_i: (3,) 残余偏置

    Returns:
        b_final: (N, 3) 或 (3,) 校正后数据
    """
    b_ellipsoid = np.asarray(b_ellipsoid)
    D_i = np.asarray(D_i)
    e_i = np.asarray(e_i)

    return (b_ellipsoid @ D_i.T) + e_i


# ============== 主 API ==============
def consistency_fit(
    csv_dir: Path,
    data_files: Dict[str, str] = None,
    sensor_config: SensorArrayConfig = None,
    auto_detect: bool = True,
    intrinsic_params: IntrinsicParamsSet = None,
    r_corr: Dict[int, np.ndarray] = None,
    logger=None,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict, Optional[float]]:
    """
    执行一致性校准（单次调用完成全部计算）

    注意: 始终尝试自动探测通道-轴映射，因为物理连接或信号源配置可能随实验变化。

    处理顺序（与 TDM 保持一致）:
        raw_data -> ellipsoid correction -> R_CORR rotation -> consistency fit

    Args:
        csv_dir: CSV 文件所在目录
        data_files: 数据文件映射字典
        sensor_config: 传感器配置对象
        auto_detect: 是否自动检测通道-轴映射
        intrinsic_params: 椭球校正参数（如果提供，则对原始数据进行椭球校正）
        r_corr: R_CORR 旋转变换字典 {sensor_id: np.ndarray(3,3)}，
                如果提供，则在椭球校正后应用旋转变换
        logger: 可选的日志函数，用于替代 print() (e.g., rospy.loginfo)

    Returns:
        D_list, e_list, fit_info, amp_factor
        amp_factor: 背景条件的放大系数 (用于方案B逆缩放)，如果没有则返回 None
    """
    if logger is None:
        def logger(*args, **kwargs):
            print(*args, **kwargs)

    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    if data_files is None:
        data_files = {
            'background': 'consistency_calib_background.csv',
            'ch1_positive': 'consistency_calib_ch1_positive.csv',
            'ch1_negative': 'consistency_calib_ch1_negative.csv',
            'ch2_positive': 'consistency_calib_ch2_positive.csv',
            'ch2_negative': 'consistency_calib_ch2_negative.csv',
            'ch3_positive': 'consistency_calib_ch3_positive.csv',
            'ch3_negative': 'consistency_calib_ch3_negative.csv',
        }

    # 始终尝试自动探测，除非明确禁用
    logger("\n" + "=" * 60)
    logger("Auto-detecting channel-to-axis mapping from current data...")
    logger("=" * 60)
    channel_to_axis = auto_detect_channel_axis_mapping(csv_dir, data_files, n_sensors)
    logger("  Mapping found: {}".format(channel_to_axis))
    logger("=" * 60)

    # 根据检测结果构建 condition_map
    axis_to_cond_pos = {'x': FieldCondition.POS_X, 'y': FieldCondition.POS_Y, 'z': FieldCondition.POS_Z}
    axis_to_cond_neg = {'x': FieldCondition.NEG_X, 'y': FieldCondition.NEG_Y, 'z': FieldCondition.NEG_Z}

    condition_map = {'background': FieldCondition.ZERO}
    for ch, axis in channel_to_axis.items():
        condition_map[f'ch{ch}_positive'] = axis_to_cond_pos[axis]
        condition_map[f'ch{ch}_negative'] = axis_to_cond_neg[axis]

    logger(f"\n  Channel-Axis mapping: {channel_to_axis}")
    logger(f"  Condition map: {condition_map}")

    # 加载并处理数据
    # 如果提供了 intrinsic_params，则对原始数据进行椭球校正
    b_norm_means = {}
    amp_factors = {}  # 存储每个条件的放大系数

    for name, filename in data_files.items():
        csv_path = Path(csv_dir) / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        sensors_raw = load_csv_data_raw(csv_path, n_sensors)  # 加载原始数据

        if intrinsic_params is not None:
            # Step 1: 对每个传感器应用椭球校正
            sensors_ellipsoid = np.zeros_like(sensors_raw)
            for sid in range(1, n_sensors + 1):
                o_i = np.array(intrinsic_params.params[sid].o_i)
                C_i = np.array(intrinsic_params.params[sid].C_i)
                sensors_ellipsoid[:, sid-1, :] = apply_ellipsoid_correction_to_data(
                    sensors_raw[:, sid-1, :], o_i, C_i
                )

            # Step 2: 应用 R_CORR 旋转变换（如果提供）
            if r_corr is not None:
                sensors_rot = np.zeros_like(sensors_ellipsoid)
                for sid in range(1, n_sensors + 1):
                    if sid in r_corr:
                        sensors_rot[:, sid-1, :] = apply_r_corr_rotation(
                            sensors_ellipsoid[:, sid-1, :], r_corr[sid]
                        )
                    else:
                        sensors_rot[:, sid-1, :] = sensors_ellipsoid[:, sid-1, :]
                sensors_corr = sensors_rot
            else:
                sensors_corr = sensors_ellipsoid

            # 计算放大系数（使用稳定段数据）
            raw_mean = compute_stable_mean(sensors_raw)
            corr_mean = compute_stable_mean(sensors_corr)
            amp_factor = compute_amplification_factor(raw_mean, corr_mean)
            amp_factors[condition_map[name]] = amp_factor

            b_norm_means[condition_map[name]] = corr_mean
        else:
            # 不进行椭球校正，直接使用
            b_norm_means[condition_map[name]] = compute_stable_mean(sensors_raw)

    # 打印放大系数
    if intrinsic_params is not None:
        logger("\n" + "=" * 60)
        logger("Amplification Factor (after ellipsoid correction)")
        logger("=" * 60)
        logger(f"  {'Condition':<12} {'Raw Norm':>12} {'Corr Norm':>12} {'Amp Factor':>12}")
        logger("  " + "-" * 50)
        for cond, amp in amp_factors.items():
            cond_str = FieldCondition.to_string(cond)
            logger(f"  {cond_str:<12} {amp['raw_norm_mean']:>12.6f} {amp['corr_norm_mean']:>12.6f} {amp['amplification_factor']:>12.4f}")
        logger("=" * 60)

    # 拟合 D_i 和 e_i
    D_list, e_list = fit_D_and_e(b_norm_means, n_sensors, channel_to_axis)

    # 仅保留算法元数据，不再存储具体的通道映射
    fit_info = {
        'method': 'relative_consistency_fit',
        'n_sensors': n_sensors,
    }

    # 获取背景条件的放大系数 (用于方案B)
    amp_factor_background = amp_factors.get(FieldCondition.ZERO, {}).get('amplification_factor')

    return D_list, e_list, fit_info, amp_factor_background


def batch_consistency_fit(
    csv_dir: Path,
    output_path: Path = None,
    sensor_config: SensorArrayConfig = None,
    auto_detect: bool = True,
    intrinsic_params: IntrinsicParamsSet = None,
    r_corr: Dict[int, np.ndarray] = None,
    logger=None,
) -> Tuple[List[ConsistencyResult], Optional[float]]:
    """
    批量处理一致性校准，返回完整结果

    Args:
        csv_dir: CSV 文件所在目录
        output_path: 可选，输出 JSON 路径
        sensor_config: 可选，传感器配置对象
        auto_detect: 是否自动检测通道-轴映射（默认True）
        intrinsic_params: 可选，椭球校正参数（如果提供，则对原始数据进行椭球校正）
        r_corr: 可选，R_CORR 旋转变换字典 {sensor_id: np.ndarray(3,3)}
        logger: 可选的日志函数 (e.g., rospy.loginfo)

    Returns:
        (results, amp_factor_background)
        results: List of ConsistencyResult for all n_sensors sensors
        amp_factor_background: 背景条件放大系数 (用于方案B逆缩放)
    """
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    D_list, e_list, fit_info, amp_factor_background = consistency_fit(
        csv_dir, data_files=None, sensor_config=sensor_config,
        auto_detect=auto_detect, intrinsic_params=intrinsic_params,
        r_corr=r_corr, logger=logger
    )

    results = []
    for i in range(n_sensors):
        result = ConsistencyResult(
            sensor_id=i + 1,
            csv_file=str(csv_dir),
            D_i=D_list[i].tolist(),
            e_i=e_list[i].tolist(),
            fit_info=fit_info
        )
        results.append(result)

    # 保存参数 (包含 amp_factor)
    if output_path is not None:
        save_consistency_params(results, output_path, sensor_config, amp_factor_background)

    return results, amp_factor_background


def validate_consistency(
    csv_dir: Path,
    D_list: List[np.ndarray],
    e_list: List[np.ndarray],
    sensor_config: SensorArrayConfig = None,
    channel_to_axis: Dict[int, str] = None,
    intrinsic_params: IntrinsicParamsSet = None,
    r_corr: Dict[int, np.ndarray] = None,
) -> Dict:
    """
    验证一致性校正效果

    处理顺序（与 TDM 保持一致）:
        raw_data -> ellipsoid correction -> R_CORR rotation -> consistency correction

    Args:
        csv_dir: CSV 文件所在目录
        D_list: D_i 矩阵列表
        e_list: e_i 向量列表
        sensor_config: 可选，传感器配置对象
        channel_to_axis: 可选，通道-轴映射，如 {1: 'x', 2: 'y', 3: 'z'}，
                         如果为 None，则自动检测
        intrinsic_params: 可选，椭球校正参数（如果提供，则对原始数据进行椭球校正）
        r_corr: 可选，R_CORR 旋转变换字典 {sensor_id: np.ndarray(3,3)}

    Returns:
        包含校正前后标准差对比的字典
    """
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    data_files = {
        'background': 'consistency_calib_background.csv',
        'ch1_positive': 'consistency_calib_ch1_positive.csv',
        'ch1_negative': 'consistency_calib_ch1_negative.csv',
        'ch2_positive': 'consistency_calib_ch2_positive.csv',
        'ch2_negative': 'consistency_calib_ch2_negative.csv',
        'ch3_positive': 'consistency_calib_ch3_positive.csv',
        'ch3_negative': 'consistency_calib_ch3_negative.csv',
    }

    # 自动检测或使用提供的映射
    if channel_to_axis is None:
        print("\n  [validate] Auto-detecting channel-to-axis mapping...")
        channel_to_axis = auto_detect_channel_axis_mapping(csv_dir, data_files, n_sensors)

    axis_to_cond_pos = {'x': FieldCondition.POS_X, 'y': FieldCondition.POS_Y, 'z': FieldCondition.POS_Z}
    axis_to_cond_neg = {'x': FieldCondition.NEG_X, 'y': FieldCondition.NEG_Y, 'z': FieldCondition.NEG_Z}

    condition_map = {'background': FieldCondition.ZERO}
    for ch, axis in channel_to_axis.items():
        condition_map[f'ch{ch}_positive'] = axis_to_cond_pos[axis]
        condition_map[f'ch{ch}_negative'] = axis_to_cond_neg[axis]

    # 加载并处理数据（与 consistency_fit 保持一致的处理流程）
    b_norm_means = {}
    for name, filename in data_files.items():
        csv_path = Path(csv_dir) / filename
        if not csv_path.exists():
            continue

        sensors_raw = load_csv_data_raw(csv_path, n_sensors)  # 加载原始数据

        if intrinsic_params is not None:
            # Step 1: 椭球校正
            sensors_ellipsoid = np.zeros_like(sensors_raw)
            for sid in range(1, n_sensors + 1):
                o_i = np.array(intrinsic_params.params[sid].o_i)
                C_i = np.array(intrinsic_params.params[sid].C_i)
                sensors_ellipsoid[:, sid-1, :] = apply_ellipsoid_correction_to_data(
                    sensors_raw[:, sid-1, :], o_i, C_i
                )

            # Step 2: R_CORR 旋转变换
            if r_corr is not None:
                sensors_rot = np.zeros_like(sensors_ellipsoid)
                for sid in range(1, n_sensors + 1):
                    if sid in r_corr:
                        sensors_rot[:, sid-1, :] = apply_r_corr_rotation(
                            sensors_ellipsoid[:, sid-1, :], r_corr[sid]
                        )
                    else:
                        sensors_rot[:, sid-1, :] = sensors_ellipsoid[:, sid-1, :]
                sensors_corr = sensors_rot
            else:
                sensors_corr = sensors_ellipsoid

            b_norm_means[condition_map[name]] = compute_stable_mean(sensors_corr)
        else:
            # 不进行校正，直接使用
            b_norm_means[condition_map[name]] = compute_stable_mean(sensors_raw)

    conditions = [FieldCondition.ZERO] + FieldCondition.all_conditions()
    validation = {'conditions': [], 'axes': [], 'before': [], 'after': [], 'improvement_pct': []}

    for cond in conditions:
        before = b_norm_means[cond]
        after = np.stack([apply_consistency_correction(before[s-1], D_list[s-1], e_list[s-1])
                         for s in range(1, n_sensors + 1)])

        for axis, axis_name in enumerate(['X', 'Y', 'Z']):
            std_b = float(np.std(before[:, axis]))
            std_a = float(np.std(after[:, axis]))
            imp = float((std_b - std_a) / (std_b + 1e-10) * 100)
            validation['conditions'].append(FieldCondition.to_string(cond))
            validation['axes'].append(axis_name)
            validation['before'].append(std_b)
            validation['after'].append(std_a)
            validation['improvement_pct'].append(imp)

    return validation


def save_consistency_params(
    results: List[ConsistencyResult],
    output_path: Path,
    sensor_config: SensorArrayConfig = None,
    amp_factor: float = None,
):
    """保存一致性校准参数到 JSON 文件"""
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    params_set = ConsistencyParamsSet(
        params={
            r.sensor_id: ConsistencyParams(D_i=r.D_i, e_i=r.e_i)
            for r in results
        },
        amp_factor=amp_factor
    )
    params_set.to_json(str(output_path))
    print(f"Consistency params saved to: {output_path}")


def load_consistency_params(
    params_path: Path,
    sensor_config: SensorArrayConfig = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Optional[float]]:
    """
    从 JSON 文件加载一致性校准参数

    Args:
        params_path: 参数文件路径
        sensor_config: 可选，传感器配置对象

    Returns:
        (D_dict, e_dict, amp_factor): sensor_id -> 矩阵/向量, amp_factor or None
    """
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    params_set = ConsistencyParamsSet.from_json(str(params_path))
    D_dict = {sid: np.array(p.D_i) for sid, p in params_set.params.items()}
    e_dict = {sid: np.array(p.e_i) for sid, p in params_set.params.items()}
    return D_dict, e_dict, params_set.amp_factor


def parse_magnitude_txt(magnitude_path: Path) -> Dict[str, Dict[int, float]]:
    """
    解析 magnitude.txt 文件，提取各 channel 各 voltage 下的参考磁场强度

    Args:
        magnitude_path: magnitude.txt 文件路径

    Returns:
        {channel: {voltage: magnitude}}
        例如: {'z': {5: 25.0, 4: 20.0, 3: 14.8, 2: 9.4, 1: 4.3}, 'y': {5: 25.0, 4: 20.0, 3: 14.8, 2: 10.0, 1: 4.8}, 'x': {5: 25.0, 4: 19.8, 3: 15.0, 2: 9.7, 1: 4.8}}
    """
    result = {}

    content = magnitude_path.read_text()
    # Match channel blocks
    channel_blocks = re.split(r'channel,\s*(\w+)', content)
    # channel_blocks[0] is empty or before first match, [1] is channel name, [2] is content, etc.
    for i in range(1, len(channel_blocks), 2):
        channel = channel_blocks[i]
        block = channel_blocks[i + 1] if i + 1 < len(channel_blocks) else ''

        # Extract magnitude line
        mag_match = re.search(r'magnitude,\s*([\d.,\s]+)', block)
        if mag_match:
            mag_values = [float(v.strip()) for v in mag_match.group(1).split(',')]
            if len(mag_values) != len(VOLTAGE_ORDER):
                raise ValueError(
                    f"Channel '{channel}': expected {len(VOLTAGE_ORDER)} magnitude values "
                    f"(for voltages {VOLTAGE_ORDER}), but got {len(mag_values)} values: {mag_values}"
                )
            result[channel] = {VOLTAGE_ORDER[j]: mag_values[j] for j in range(len(VOLTAGE_ORDER))}
        else:
            print(f"[WARNING] Channel '{channel}' has no magnitude line, skipping")

    if not result:
        print("[WARNING] parse_magnitude_txt: no valid channel data found, returning empty dict")

    return result


def load_manual_calibration_data(
    data_dir: Path,
    channel: str,
    voltage: int,
    n_sensors: int = 12
) -> np.ndarray:
    """
    加载指定 channel 和 voltage 的手动标定数据

    Args:
        data_dir: 父目录 (e.g., .../sensor_data_collection/data)
        channel: 'x', 'y', or 'z'
        voltage: 1, 2, 3, 4, 5

    Returns:
        data: (N_samples, n_sensors, 3) 原始传感器数据
    """
    if n_sensors < 1:
        raise ValueError(f"n_sensors must be >= 1, got {n_sensors}")

    if channel not in ('x', 'y', 'z'):
        raise ValueError(f"channel must be 'x', 'y', or 'z', got '{channel}'")
    if voltage not in (1, 2, 3, 4, 5):
        raise ValueError(f"voltage must be 1, 2, 3, 4, or 5, got {voltage}")

    csv_path = data_dir / f"manual_{channel}" / f"manual_record_{voltage}V.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Manual calibration CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith('sensor_')]
    if not cols:
        raise ValueError(f"No sensor_* columns found in {csv_path}")
    data = df[cols].values
    n_samples = data.shape[0]
    if n_samples == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")
    reshaped = np.zeros((n_samples, n_sensors, 3))
    for i in range(n_sensors):
        reshaped[:, i, :] = data[:, i*3:(i+1)*3]
    return reshaped


def consistency_check_by_magnitude(
    data_dir: Path,
    magnitude_path: Path,
    sensor_config: SensorArrayConfig = None,
    intrinsic_params: IntrinsicParamsSet = None,
    channel: str = 'x',
    voltage: int = 5,
    logger=None,
) -> Dict:
    """
    基于 magnitude.txt 参考值进行一致性检验

    算法：
    1. 解析 magnitude.txt 获取指定 channel/voltage 的参考磁场强度
    2. 加载 manual_{channel}/manual_record_{voltage}V.csv 原始数据
    3. 对每颗传感器应用椭球校准: b_corr = C_i @ (b_raw - o_i)
    4. 计算每行校准后数据的模值: |b_corr|
    5. 计算比例系数: ratio = reference_magnitude / |b_corr|
    6. 统计各传感器的 mean_ratio 和 std_ratio

    Args:
        data_dir: .../sensor_data_collection/data 目录
        magnitude_path: magnitude.txt 文件路径
        sensor_config: 传感器配置 (默认 QMC6309)
        intrinsic_params: 椭球校准内参 (o_i, C_i)
        channel: 'x', 'y', or 'z'
        voltage: 1, 2, 3, 4, 5
        logger: 日志函数 (默认 print)

    Returns:
        {
            'channel': channel,
            'voltage': voltage,
            'reference_magnitude': float,
            'sensor_results': {
                sensor_id: {
                    'mean_ratio': float,
                    'std_ratio': float,
                    'calibrated_magnitudes': [float],  # list of |b_corr| per sample
                }
            }
        }
    """
    if logger is None:
        def logger(*args, **kwargs):
            print(*args, **kwargs)

    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    # Step 1: 解析 magnitude.txt 获取参考磁场强度
    magnitude_data = parse_magnitude_txt(magnitude_path)
    if channel not in magnitude_data:
        raise ValueError(f"Channel '{channel}' not found in magnitude.txt. Available: {list(magnitude_data.keys())}")
    if voltage not in magnitude_data[channel]:
        raise ValueError(f"Voltage {voltage} not found for channel '{channel}'. Available: {list(magnitude_data[channel].keys())}")

    reference_magnitude = magnitude_data[channel][voltage]

    # Step 2: 加载手动标定原始数据
    raw_data = load_manual_calibration_data(data_dir, channel, voltage, n_sensors)
    n_samples = raw_data.shape[0]

    # Step 3-6: 对每颗传感器应用椭球校正，计算模值和比例系数
    sensor_results = {}

    for i in range(n_sensors):
        sensor_id = i + 1
        b_raw = raw_data[:, i, :]  # (N, 3)

        # 应用椭球校准（如果提供了 intrinsic_params）
        if intrinsic_params is not None:
            o_i = np.array(intrinsic_params.params[sensor_id].o_i)
            C_i = np.array(intrinsic_params.params[sensor_id].C_i)
            b_corr = apply_ellipsoid_correction_to_data(b_raw, o_i, C_i)
        else:
            b_corr = b_raw

        # Step 4: 计算校准后数据的模值
        magnitudes = np.linalg.norm(b_corr, axis=1)  # (N,)

        # Step 5: 计算比例系数
        ratios = reference_magnitude / magnitudes

        # Step 6: 统计 mean_ratio 和 std_ratio
        mean_ratio = float(np.mean(ratios))
        std_ratio = float(np.std(ratios))

        sensor_results[sensor_id] = {
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'calibrated_magnitudes': magnitudes.tolist(),
        }

    # 打印统计信息
    logger("\n" + "=" * 50)
    logger("Consistency Check by Magnitude")
    logger("=" * 50)
    logger(f"  Channel: {channel}, Voltage: {voltage}V")
    logger(f"  Reference magnitude: {reference_magnitude:.4f}")
    logger(f"  Number of samples: {n_samples}")
    logger(f"  Number of sensors: {n_sensors}")
    logger("-" * 50)
    logger(f"  {'Sensor':<10} {'Mean Ratio':>12} {'Std Ratio':>12}")
    logger(f"  {'-'*36}")
    for sid in range(1, n_sensors + 1):
        sr = sensor_results[sid]
        logger(f"  {sid:<10} {sr['mean_ratio']:>12.4f} {sr['std_ratio']:>12.4f}")
    logger(f"  {'-'*36}")

    # Overall statistics
    all_mean_ratios = [sensor_results[sid]['mean_ratio'] for sid in range(1, n_sensors + 1)]
    all_std_ratios = [sensor_results[sid]['std_ratio'] for sid in range(1, n_sensors + 1)]
    logger(f"  {'Overall Mean:':<10} {np.mean(all_mean_ratios):>12.4f} {np.std(all_mean_ratios):>12.4f}")
    logger(f"  {'Overall Std:':<10} {np.std(all_mean_ratios):>12.4f} {np.std(all_std_ratios):>12.4f}")
    logger("=" * 50)

    return {
        'channel': channel,
        'voltage': voltage,
        'reference_magnitude': reference_magnitude,
        'sensor_results': sensor_results,
    }


def batch_consistency_check_by_magnitude(
    data_dir: Path,
    magnitude_path: Path,
    sensor_config: SensorArrayConfig = None,
    intrinsic_params: IntrinsicParamsSet = None,
    verbose: bool = True,
    logger=None,
) -> List[Dict]:
    """
    批量执行所有 channel 和 voltage 的 magnitude-based 一致性检验

    遍历所有 channel 和 voltage 组合，
    对每种组合调用 consistency_check_by_magnitude。

    Args:
        data_dir: .../sensor_data_collection/data 目录
        magnitude_path: magnitude.txt 文件路径
        sensor_config: 传感器配置 (默认 QMC6309)
        intrinsic_params: 椭球校准内参
        verbose: 是否输出详细信息 (默认 True)
        logger: 日志函数 (默认 print)

    Returns:
        List of results for each (channel, voltage) combination.
        Each result is the dict returned by consistency_check_by_magnitude.
    """
    if logger is None:
        if verbose:
            def logger(*args, **kwargs):
                print(*args, **kwargs)
        else:
            def logger(*args, **kwargs):
                pass

    results = []

    # Get channels from magnitude.txt dynamically
    magnitude_data = parse_magnitude_txt(magnitude_path)
    channels = list(magnitude_data.keys())

    for channel in channels:
        for voltage in VOLTAGE_ORDER:
            try:
                result = consistency_check_by_magnitude(
                    data_dir=data_dir,
                    magnitude_path=magnitude_path,
                    sensor_config=sensor_config,
                    intrinsic_params=intrinsic_params,
                    channel=channel,
                    voltage=voltage,
                    logger=logger,
                )
                results.append(result)
            except Exception as e:
                logger(f"[ERROR] Failed for channel={channel}, voltage={voltage}: {e}")
                continue

    return results


def compute_sensor_gains(
    data_dir: Path,
    magnitude_path: Path,
    sensor_config: SensorArrayConfig = None,
    intrinsic_params: IntrinsicParamsSet = None,
    logger=None,
) -> Dict[int, Dict]:
    """
    计算每个传感器的模值增益 s_i

    算法：
    1. 加载 manual_x/y/z 目录下所有电压(1-5V)的数据
    2. 对每个 sensor_i:
       - 拼接所有 channel/voltage 的校准后模值 (N, 1)
       - 拼接对应的 reference magnitude 列向量 (N, 1) = magnitude * ones(N, 1)
       - 用最小二乘求解: ref = s_i * cal => s_i = (cal^T * ref) / (cal^T * cal)

    Args:
        data_dir: .../sensor_data_collection/data 目录
        magnitude_path: magnitude.txt 文件路径
        sensor_config: 传感器配置 (默认 QMC6309)
        intrinsic_params: 椭球校准内参 (o_i, C_i)
        logger: 日志函数 (默认 print)

    Returns:
        {sensor_id: {
            's_i': float,           # 增益系数
            'r_squared': float,     # R² 决定系数
            'rmse': float,          # RMSE
            'n_samples': int        # 样本数
        }}
    """
    if logger is None:
        def logger(*args, **kwargs):
            print(*args, **kwargs)

    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    # Step 1: 解析 magnitude.txt
    magnitude_data = parse_magnitude_txt(magnitude_path)

    # Step 2: 收集所有 channel/voltage 组合
    # 构建 (cal_magnitudes, ref_magnitudes) 数据对
    all_data = []  # list of (cal_mag, ref_mag) tuples per sensor

    channels = list(magnitude_data.keys())

    for channel in channels:
        for voltage in VOLTAGE_ORDER:
            try:
                raw_data = load_manual_calibration_data(data_dir, channel, voltage, n_sensors)
            except FileNotFoundError as e:
                logger(f"[WARN] {e}")
                continue

            ref_mag = magnitude_data[channel][voltage]

            # 对每颗传感器应用椭球校正并计算模值
            for i in range(n_sensors):
                sensor_id = i + 1
                b_raw = raw_data[:, i, :]  # (N, 3)

                if intrinsic_params is not None:
                    o_i = np.array(intrinsic_params.params[sensor_id].o_i)
                    C_i = np.array(intrinsic_params.params[sensor_id].C_i)
                    b_corr = apply_ellipsoid_correction_to_data(b_raw, o_i, C_i)
                else:
                    b_corr = b_raw

                # 计算模值
                cal_mags = np.linalg.norm(b_corr, axis=1, keepdims=True)  # (N, 1)
                ref_vec = np.full((b_corr.shape[0], 1), ref_mag)  # (N, 1)

                all_data.append((cal_mags, ref_vec))

    # Step 3: 对每颗传感器拼接所有数据并求解最小二乘
    results = {}

    for i in range(n_sensors):
        sensor_id = i + 1

        # 拼接该传感器所有 channel/voltage 的数据
        cal_list = [all_data[j][0][:, 0] for j in range(i, len(all_data), n_sensors)]
        ref_list = [all_data[j][1][:, 0] for j in range(i, len(all_data), n_sensors)]

        if not cal_list:
            logger(f"[WARN] No data for sensor {sensor_id}")
            continue

        cal_all = np.concatenate(cal_list)  # (N_total,)
        ref_all = np.concatenate(ref_list)  # (N_total,)

        n_samples = len(cal_all)

        # 最小二乘求解: ref = s_i * cal
        # s_i = (cal^T * ref) / (cal^T * cal)
        cal_col = cal_all.reshape(-1, 1)  # (N, 1)
        ref_col = ref_all.reshape(-1, 1)  # (N, 1)

        s_i = float((cal_col.T @ ref_col) / (cal_col.T @ cal_col))

        # 计算拟合值和残差
        ref_pred = s_i * cal_col
        residuals = ref_col - ref_pred

        # RMSE
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        # R² 决定系数
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((ref_col - np.mean(ref_col)) ** 2))
        r_squared = 1.0 - (ss_res / (ss_tot + 1e-10))

        results[sensor_id] = {
            's_i': s_i,
            'r_squared': r_squared,
            'rmse': rmse,
            'n_samples': n_samples,
        }

    # 打印摘要
    logger("\n" + "=" * 50)
    logger("Compute Sensor Gains Summary")
    logger("=" * 50)
    logger(f"  {'Sensor':<10} {'s_i':>12} {'R²':>12} {'RMSE':>12} {'N':>8}")
    logger(f"  {'-' * 50}")
    for sid in range(1, n_sensors + 1):
        if sid in results:
            r = results[sid]
            logger(f"  {sid:<10} {r['s_i']:>12.6f} {r['r_squared']:>12.6f} {r['rmse']:>12.6f} {r['n_samples']:>8}")
    logger(f"  {'-' * 50}")

    s_values = [results[sid]['s_i'] for sid in range(1, n_sensors + 1) if sid in results]
    if s_values:
        logger(f"  {'Mean s_i:':<10} {np.mean(s_values):>12.6f}")
        logger(f"  {'Std s_i:':<10} {np.std(s_values):>12.6f}")
    logger("=" * 50)

    return results


def compute_rowwise_sensor_consistency_metric(
    data: np.ndarray,
    sensor_ids: List[int] = None,
) -> Dict:
    """
    计算逐行传感器间分量标准差（within-row sensor std）

    对数据形状 (N, 12, 3)，对每一行的 12 个传感器计算 x/y/z 三个分量的 std。

    Args:
        data: (N, 12, 3) 数据
        sensor_ids: 可选，传感器 ID 列表（不用于计算，仅保留在返回中）

    Returns:
        包含聚合指标的字典
    """
    data = np.asarray(data)
    assert data.ndim == 3 and data.shape[1] == 12, \
        f"Expected (N, 12, 3), got {data.shape}"
    n_rows = data.shape[0]

    std_x_per_row = np.std(data[:, :, 0], axis=1)
    std_y_per_row = np.std(data[:, :, 1], axis=1)
    std_z_per_row = np.std(data[:, :, 2], axis=1)

    all_stds = np.concatenate([std_x_per_row, std_y_per_row, std_z_per_row])

    return {
        'grand_mean_std': float(np.mean(all_stds)),
        'grand_max_std': float(np.max(all_stds)),
        'percentile_95_std': float(np.percentile(all_stds, 95)),
        'std_x_per_row': std_x_per_row,
        'std_y_per_row': std_y_per_row,
        'std_z_per_row': std_z_per_row,
        'n_rows': n_rows,
    }


def compare_calibration_methods(
    data_dir: Path,
    magnitude_path: Path,
    intrinsic_params: IntrinsicParamsSet,
    r_corr_dict: Dict[int, np.ndarray],
    sensor_gains: Dict[int, Dict],
    sensor_config: SensorArrayConfig = None,
    logger=None,
) -> Dict:
    """
    对比两个校准路径的逐行传感器一致性指标

    Path A (baseline): R_CORR × b_raw
    Path B (full):     R_CORR × s_i × (C_i × (b_raw - o_i))

    Args:
        data_dir: .../sensor_data_collection/data 目录
        magnitude_path: magnitude.txt 路径
        intrinsic_params: 椭球参数 (o_i, C_i)
        r_corr_dict: {sensor_id: (3,3)} R_CORR 矩阵字典
        sensor_gains: {sensor_id: {'s_i': float}} 模值增益字典
        sensor_config: 传感器配置
        logger: 日志函数

    Returns:
        包含两个路径指标的完整对比结果字典
    """
    if logger is None:
        def logger(*args, **kwargs):
            print(*args, **kwargs)

    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    magnitude_data = parse_magnitude_txt(magnitude_path)
    channels = list(magnitude_data.keys())

    # 收集所有数据路径
    all_raw_data = []  # list of (channel, voltage, data) tuples
    for channel in channels:
        for voltage in VOLTAGE_ORDER:
            try:
                raw_data = load_manual_calibration_data(
                    data_dir, channel, voltage, n_sensors
                )
                all_raw_data.append((channel, voltage, raw_data))
            except FileNotFoundError:
                logger(f"[WARN] Missing: manual_{channel}/manual_record_{voltage}V.csv")
                continue

    # ---------- Path A: R_CORR × b_raw ----------
    all_path_a_concat = []
    # ---------- Path B: R_CORR × s_i × (C_i × (b_raw - o_i)) ----------
    all_path_b_concat = []
    # ---------- 按 channel 分组 ----------
    per_channel_a = {ch: [] for ch in channels}
    per_channel_b = {ch: [] for ch in channels}
    # ---------- 按 voltage 分组 ----------
    per_voltage_a = {v: [] for v in VOLTAGE_ORDER}
    per_voltage_b = {v: [] for v in VOLTAGE_ORDER}

    for channel, voltage, b_raw in all_raw_data:
        # Path A
        b_a = np.zeros_like(b_raw)
        for sid in range(1, n_sensors + 1):
            b_a[:, sid-1, :] = apply_r_corr_rotation(b_raw[:, sid-1, :], r_corr_dict[sid])

        # Path B
        b_b = np.zeros_like(b_raw)
        for sid in range(1, n_sensors + 1):
            o_i = np.array(intrinsic_params.params[sid].o_i)
            C_i = np.array(intrinsic_params.params[sid].C_i)
            s_i = sensor_gains[sid]['s_i']
            b_ellipsoid = apply_ellipsoid_correction_to_data(b_raw[:, sid-1, :], o_i, C_i)
            b_with_gain = s_i * b_ellipsoid
            b_b[:, sid-1, :] = apply_r_corr_rotation(b_with_gain, r_corr_dict[sid])

        # 收集用于全局指标
        all_path_a_concat.append(b_a)
        all_path_b_concat.append(b_b)
        per_channel_a[channel].append(b_a)
        per_channel_b[channel].append(b_b)
        per_voltage_a[voltage].append(b_a)
        per_voltage_b[voltage].append(b_b)

    # 合并所有数据
    all_path_a = np.concatenate(all_path_a_concat, axis=0)  # (N_total, 12, 3)
    all_path_b = np.concatenate(all_path_b_concat, axis=0)

    # 计算全局指标
    metric_a = compute_rowwise_sensor_consistency_metric(all_path_a)
    metric_b = compute_rowwise_sensor_consistency_metric(all_path_b)

    # 计算 per-channel 指标
    per_channel_results = {}
    for ch in channels:
        if per_channel_a[ch]:
            ca = np.concatenate(per_channel_a[ch], axis=0)
            cb = np.concatenate(per_channel_b[ch], axis=0)
            per_channel_results[ch] = {
                'path_a': compute_rowwise_sensor_consistency_metric(ca),
                'path_b': compute_rowwise_sensor_consistency_metric(cb),
            }

    # 计算 per-voltage 指标
    per_voltage_results = {}
    for v in VOLTAGE_ORDER:
        if per_voltage_a[v]:
            va = np.concatenate(per_voltage_a[v], axis=0)
            vb = np.concatenate(per_voltage_b[v], axis=0)
            per_voltage_results[v] = {
                'path_a': compute_rowwise_sensor_consistency_metric(va),
                'path_b': compute_rowwise_sensor_consistency_metric(vb),
            }

    # ---------- 新增：逐 sensor 逐 channel/voltage 模值重复性指标 ----------
    # 对每颗传感器、每个 channel/voltage，计算跨行模值 std（衡量单传感器重复性）
    per_sensor_mag_std = {}
    for sid in range(1, n_sensors + 1):
        sid_key = f"sensor_{sid}"
        per_sensor_mag_std[sid_key] = {}
        for ch in channels:
            per_sensor_mag_std[sid_key][ch] = {}
            for v in VOLTAGE_ORDER:
                # 找到对应 channel/voltage 的原始索引
                for idx, (c, volt, _) in enumerate(all_raw_data):
                    if c == ch and volt == v:
                        # Path A: (N, 3) 取第 sid-1 号传感器 → 计算每行模值 → std across rows
                        mag_a = np.linalg.norm(all_path_a_concat[idx][:, sid-1, :], axis=1)
                        mag_b = np.linalg.norm(all_path_b_concat[idx][:, sid-1, :], axis=1)
                        per_sensor_mag_std[sid_key][ch][v] = {
                            'path_a': float(np.std(mag_a)),
                            'path_b': float(np.std(mag_b)),
                        }
                        break

    # 聚合到 per-sensor 级别（跨所有 channel/voltage 的平均）
    per_sensor_agg = {}
    for sid in range(1, n_sensors + 1):
        sid_key = f"sensor_{sid}"
        all_a = []
        all_b = []
        for ch in channels:
            for v in VOLTAGE_ORDER:
                if sid_key in per_sensor_mag_std and ch in per_sensor_mag_std[sid_key] and v in per_sensor_mag_std[sid_key][ch]:
                    all_a.append(per_sensor_mag_std[sid_key][ch][v]['path_a'])
                    all_b.append(per_sensor_mag_std[sid_key][ch][v]['path_b'])
        if all_a:
            per_sensor_agg[sid_key] = {
                'path_a': float(np.mean(all_a)),
                'path_b': float(np.mean(all_b)),
            }

    # 聚合到总体（跨所有 sensor/channel/voltage）
    all_mag_a = []
    all_mag_b = []
    for sid_key in per_sensor_agg:
        for ch in channels:
            for v in VOLTAGE_ORDER:
                if ch in per_sensor_mag_std[sid_key] and v in per_sensor_mag_std[sid_key][ch]:
                    all_mag_a.append(per_sensor_mag_std[sid_key][ch][v]['path_a'])
                    all_mag_b.append(per_sensor_mag_std[sid_key][ch][v]['path_b'])
    grand_mag_std = {
        'path_a': float(np.mean(all_mag_a)),
        'path_b': float(np.mean(all_mag_b)),
    }

    result = {
        'path_a': {
            'method': 'R_CORR × b_raw',
            'metric': metric_a,
        },
        'path_b': {
            'method': 'R_CORR × s_i × (C_i × (b_raw - o_i))',
            'metric': metric_b,
        },
        'per_channel_results': per_channel_results,
        'per_voltage_results': per_voltage_results,
        'total_samples': all_path_a.shape[0],
        'per_sensor_mag_std': per_sensor_mag_std,
        'per_sensor_agg': per_sensor_agg,
        'grand_mag_std': grand_mag_std,
    }

    # 打印摘要
    logger("\n" + "=" * 60)
    logger("Calibration Comparison Summary")
    logger("=" * 60)
    logger(f"  Total samples: {all_path_a.shape[0]}")
    logger("")
    logger(f"  {'Metric':<25} {'Path A (R_CORR)':>18} {'Path B (Full)':>18} {'Improvement':>12}")
    logger(f"  {'-' * 75}")

    for key in ['grand_mean_std', 'grand_max_std', 'percentile_95_std']:
        va = metric_a[key]
        vb = metric_b[key]
        imp = (va - vb) / (va + 1e-10) * 100
        logger(f"  {key:<25} {va:>18.6f} {vb:>18.6f} {imp:>+11.1f}%")

    logger(f"  {'-' * 75}")
    logger("  Per-channel:")
    for ch in channels:
        if ch in per_channel_results:
            ma = per_channel_results[ch]['path_a']['grand_mean_std']
            mb = per_channel_results[ch]['path_b']['grand_mean_std']
            imp = (ma - mb) / (ma + 1e-10) * 100
            logger(f"    {ch.upper()}: mean_std A={ma:.4f}, B={mb:.4f}, imp={imp:+.1f}%")

    logger("  Per-voltage:")
    for v in VOLTAGE_ORDER:
        if v in per_voltage_results:
            ma = per_voltage_results[v]['path_a']['grand_mean_std']
            mb = per_voltage_results[v]['path_b']['grand_mean_std']
            imp = (ma - mb) / (ma + 1e-10) * 100
            logger(f"    {v}V: mean_std A={ma:.4f}, B={mb:.4f}, imp={imp:+.1f}%")

    # ---------- 打印新指标：逐 sensor 模值重复性 ----------
    logger("")
    logger("Within-Sensor Magnitude Repeatability (std of |b| across rows, per sensor)")
    logger("=" * 80)
    logger(f"  {'Sensor':<12} {'Path A (Raw)':>15} {'Path B (Full)':>15} {'Improvement':>12}")
    logger(f"  {'-' * 60}")
    improvements = []
    for sid in range(1, n_sensors + 1):
        sid_key = f"sensor_{sid}"
        if sid_key in per_sensor_agg:
            ma = per_sensor_agg[sid_key]['path_a']
            mb = per_sensor_agg[sid_key]['path_b']
            imp = (ma - mb) / (ma + 1e-10) * 100
            improvements.append(imp)
            logger(f"  {sid_key:<12} {ma:>15.6f} {mb:>15.6f} {imp:>+11.1f}%")
    logger(f"  {'-' * 60}")
    if improvements:
        logger(f"  {'Mean improvement:':<12} {np.mean(improvements):>+11.1f}%  "
               f"(Path A mean={grand_mag_std['path_a']:.4f}, Path B mean={grand_mag_std['path_b']:.4f})")
    logger("=" * 80)
    logger("")
    logger("  Per-sensor, per-channel, per-voltage magnitude std detail:")
    for sid in range(1, n_sensors + 1):
        sid_key = f"sensor_{sid}"
        logger(f"    --- {sid_key} ---")
        for ch in channels:
            if ch in per_sensor_mag_std.get(sid_key, {}):
                for v in VOLTAGE_ORDER:
                    if v in per_sensor_mag_std[sid_key][ch]:
                        ma = per_sensor_mag_std[sid_key][ch][v]['path_a']
                        mb = per_sensor_mag_std[sid_key][ch][v]['path_b']
                        imp = (ma - mb) / (ma + 1e-10) * 100
                        logger(f"      {ch.upper()} @ {v}V: raw={ma:.4f}, calib={mb:.4f}, imp={imp:+.1f}%")
    logger("=" * 80)

    logger("")
    logger("=" * 60)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Phase 2 Consistency Calibration - Fit D_i and e_i',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default data directory
  python -m calibration.lib.consistency_fit /home/zhang/embedded_array_ws/src/calibration/data/consistency

  # Process with custom output
  python -m calibration.lib.consistency_fit ./data/consistency -o ./consistency_params.json

  # Validate existing params
  python -m calibration.lib.consistency_fit ./data/consistency --validate
        """
    )
    parser.add_argument('csv_dir', nargs='?', default=None,
                        help='Directory containing consistency CSV files')
    parser.add_argument('-o', '--output', dest='output_path',
                        help='Output JSON path (default: serial_processor/config/consistency_params.json)')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation after fitting')
    parser.add_argument('--no-save', action='store_true',
                        help='Only compute, do not save')
    parser.add_argument('--no-auto-detect', action='store_true',
                        help='Use hardcoded channel-axis mapping instead of auto-detection')

    args = parser.parse_args()

    # Default path
    if args.csv_dir is None:
        csv_dir = Path(__file__).parent.parent.parent / 'data' / 'consistency'
    else:
        csv_dir = Path(args.csv_dir)

    # Default output (only used when --no-save is NOT set)
    if args.output_path is None:
        output_path = Path(__file__).parent.parent.parent.parent / 'sensor_array_config' / 'config' / 'QMC6309' / 'consistency_params.json'
    else:
        output_path = Path(args.output_path)

    print("=" * 70)
    print("Phase 2: Consistency Calibration (CLI)")
    print("=" * 70)
    print(f"  CSV dir:   {csv_dir}")
    if args.no_save:
        print(f"  Output:    (disabled by --no-save)")
    else:
        print(f"  Output:    {output_path}")
    print()

    if not csv_dir.exists():
        print(f"[ERROR] CSV directory not found: {csv_dir}")
        sys.exit(1)

    # Determine output path (None if --no-save)
    save_path = None if args.no_save else output_path

    # Run consistency fit
    auto_detect = not args.no_auto_detect
    results, amp_factor = batch_consistency_fit(csv_dir, save_path, sensor_config=get_config("QMC6309"), auto_detect=auto_detect)

    # Print amp_factor if available
    if amp_factor is not None:
        print(f"\n  Amplification factor (background): {amp_factor:.4f}")

    # Print fitted parameters
    print("\n  Fitted parameters:")
    print("  " + "-" * 65)
    print(f"  {'Sensor':<8} {'D_ix':<10} {'D_iy':<10} {'D_iz':<10} {'e_ix':<9} {'e_iy':<9} {'e_iz':<9}")
    print("  " + "-" * 65)
    for i, r in enumerate(results):
        D = np.array(r.D_i)
        e = np.array(r.e_i)
        print(f"  {r.sensor_id:<8} {D[0,0]:<10.4f} {D[1,1]:<10.4f} {D[2,2]:<10.4f} "
              f"{e[0]:<+9.4f} {e[1]:<+9.4f} {e[2]:<+9.4f}")

    # Always run validation (unless --no-save but still show report)
    print("\n" + "=" * 70)
    print("Validation Report")
    print("=" * 70)
    D_list = [np.array(r.D_i) for r in results]
    e_list = [np.array(r.e_i) for r in results]
    validation = validate_consistency(csv_dir, D_list, e_list)

    print(f"\n  Dispersion (std of 12 sensors):")
    print("  " + "-" * 55)
    print(f"  {'Condition':<8} {'Axis':<6} {'校正前':<12} {'校正后':<12} {'改善'}")
    print("  " + "-" * 55)

    improvements = []
    for i in range(len(validation['conditions'])):
        cond = validation['conditions'][i]
        axis = validation['axes'][i]
        std_b = validation['before'][i]
        std_a = validation['after'][i]
        imp = validation['improvement_pct'][i]
        improvements.append(imp)
        print(f"  {cond:<8} {axis:<6} {std_b:<12.6f} {std_a:<12.6f} {imp:>+6.1f}%")

    # Summary statistics (exclude severe degradation outliers for mean calc)
    improvements_filtered = [x for x in improvements if x > -50]
    print("\n  Summary:")
    print("  " + "-" * 40)
    print(f"  {'Mean improvement:':<22} {np.mean(improvements_filtered):>+.1f}%")
    print(f"  {'Median improvement:':<22} {np.median(improvements_filtered):>+.1f}%")
    print(f"  {'Max improvement:':<22} {np.max(improvements_filtered):>+.1f}%")
    print(f"  {'Min improvement:':<22} {np.min(improvements_filtered):>+.1f}%")

    # Per-condition summary
    print("\n  Per-condition mean improvement:")
    print("  " + "-" * 40)
    conditions_list = ['0', '+x', '-x', '+y', '-y', '+z', '-z']
    for cond in conditions_list:
        indices = [i for i, c in enumerate(validation['conditions']) if c == cond]
        if indices:
            mean_imp = np.mean([improvements[i] for i in indices])
            bar = '=' * int(max(0, mean_imp) / 5)
            print(f"  {cond:<8} {mean_imp:>+6.1f}%  {bar}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
