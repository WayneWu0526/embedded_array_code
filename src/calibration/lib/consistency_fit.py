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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from sensor_array_config.base import get_config, SensorArrayConfig, ConsistencyParamsSet, ConsistencyParams, IntrinsicParamsSet


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
    D_i: List[List[float]]  # 缩放矩阵 (3×3) - 对角矩阵
    e_i: List[float]        # 残余偏置 (3,)
    d_i: Dict[str, float]   # 对角元素 {'x': d_ix, 'y': d_iy, 'z': d_iz}
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
def fit_D_and_e(b_norm_dict: Dict[int, np.ndarray], n_sensors: int = 12) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
    logger=None,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict, Optional[float]]:
    """
    执行一致性校准（单次调用完成全部计算）

    注意: 始终尝试自动探测通道-轴映射，因为物理连接或信号源配置可能随实验变化。

    Args:
        csv_dir: CSV 文件所在目录
        data_files: 数据文件映射字典
        sensor_config: 传感器配置对象
        auto_detect: 是否自动检测通道-轴映射
        intrinsic_params: 椭球校正参数（如果提供，则对原始数据进行椭球校正）
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
            # 对每个传感器应用椭球校正
            sensors_corr = np.zeros_like(sensors_raw)
            for sid in range(1, n_sensors + 1):
                o_i = np.array(intrinsic_params.params[sid].o_i)
                C_i = np.array(intrinsic_params.params[sid].C_i)
                sensors_corr[:, sid-1, :] = apply_ellipsoid_correction_to_data(
                    sensors_raw[:, sid-1, :], o_i, C_i
                )

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
    D_list, e_list = fit_D_and_e(b_norm_means, n_sensors)

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
        logger=logger
    )

    results = []
    for i in range(n_sensors):
        result = ConsistencyResult(
            sensor_id=i + 1,
            csv_file=str(csv_dir),
            D_i=D_list[i].tolist(),
            e_i=e_list[i].tolist(),
            d_i={'x': float(D_list[i][0,0]), 'y': float(D_list[i][1,1]), 'z': float(D_list[i][2,2])},
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
) -> Dict:
    """
    验证一致性校正效果

    Args:
        csv_dir: CSV 文件所在目录
        D_list: D_i 矩阵列表
        e_list: e_i 向量列表
        sensor_config: 可选，传感器配置对象
        channel_to_axis: 可选，通道-轴映射，如 {1: 'x', 2: 'y', 3: 'z'}，
                         如果为 None，则自动检测

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

    # 数据已经是 ellipsoid + rotation 校正后的，直接使用
    b_norm_means = {}
    for name, filename in data_files.items():
        csv_path = Path(csv_dir) / filename
        if not csv_path.exists():
            continue
        sensors_raw = load_csv_data(csv_path, n_sensors)
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
    results = batch_consistency_fit(csv_dir, save_path, sensor_config=get_config("QMC6309"), auto_detect=auto_detect)

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
