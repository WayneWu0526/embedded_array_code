#!/usr/bin/env python3
"""
apply_ellipsoid_and_rccorr.py - 对consistency_calib.csv应用椭球校正和R_CORR旋转变换

处理流程:
    raw_data -> ellipsoid correction (C_i, o_i) -> R_CORR rotation -> output

输入: src/calibration/data/consistency_calib.csv
输出: src/calibration/data/rotated_ellip_calibrated_consistency_data.csv
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# 路径配置
SCRIPT_DIR = Path(__file__).parent.resolve()  # .../src/calibration/scripts
CALIB_DIR = SCRIPT_DIR.parent / 'data'       # .../src/calibration/data
CONSISTENCY_DIR = CALIB_DIR / 'consistency'  # .../src/calibration/data/consistency
INPUT_CSV = CALIB_DIR / 'consistency_calib.csv'
OUTPUT_CSV = CALIB_DIR / 'rotated_ellip_calibrated_consistency_data.csv'
RAW_OUTPUT_CSV = CALIB_DIR / 'raw_consistency_data.csv'  # 原始数据（未校正）
INTRINSIC_PARAMS_PATH = Path('/home/zhang/embedded_array_ws/src/sensor_array_config/sensor_array_config/config/qmc6309/intrinsic_params.json')
HARDWARE_PARAMS_PATH = Path('/home/zhang/embedded_array_ws/src/sensor_array_config/sensor_array_config/config/qmc6309/sensor_array_params.json')


def load_intrinsic_params(path: Path) -> dict:
    """加载椭球校正参数"""
    with open(path, 'r') as f:
        data = json.load(f)
    # 转换为 sensor_id -> {o_i, C_i}
    result = {}
    for sid, params in data.items():
        result[int(sid)] = {
            'o_i': np.array(params['o_i']),
            'C_i': np.array(params['C_i'])
        }
    return result


def load_hardware_params(path: Path) -> dict:
    """加载硬件参数，构建 sensor_id -> R_CORR 矩阵的字典"""
    with open(path, 'r') as f:
        data = json.load(f)

    r_corr_dict = {}
    for entry in data['R_CORR']:
        mat = np.array(entry['matrix']).reshape(3, 3, order='F')
        for sid in entry['sensor_ids']:
            r_corr_dict[sid] = mat
    return r_corr_dict


def apply_ellipsoid_correction(b_raw: np.ndarray, o_i: np.ndarray, C_i: np.ndarray) -> np.ndarray:
    """
    对原始传感器数据进行椭球校正

    公式: b_corr = C_i @ (b_raw - o_i)

    Args:
        b_raw: (N, 3) 原始数据
        o_i: (3,) offset 向量
        C_i: (3, 3) 校正矩阵

    Returns:
        b_corr: (N, 3) 校正后数据
    """
    b_centered = b_raw - o_i
    b_corr = b_centered @ C_i.T
    return b_corr


def apply_r_corr_rotation(b_corr: np.ndarray, r_corr: np.ndarray) -> np.ndarray:
    """
    应用 R_CORR 旋转变换

    公式: b_rot = R_CORR @ b_corr

    Args:
        b_corr: (N, 3) 椭球校正后数据
        r_corr: (3, 3) 旋转矩阵

    Returns:
        b_rot: (N, 3) 旋转变换后数据
    """
    return b_corr @ r_corr.T


def process_csv(input_path: Path, output_path: Path,
               intrinsic_params: dict, r_corr_dict: dict,
               n_sensors: int = 12):
    """
    处理CSV文件，应用椭球校正和R_CORR旋转变换

    Args:
        input_path: 输入CSV路径
        output_path: 输出CSV路径
        intrinsic_params: {sensor_id: {o_i, C_i}}
        r_corr_dict: {sensor_id: R_CORR matrix}
        n_sensors: 传感器数量
    """
    # 读取CSV
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Columns: {list(df.columns[:5])}... ({len(df.columns)} total)")

    # 准备输出DataFrame
    out_df = df.copy()

    # 对每个传感器应用校正
    for sid in range(1, n_sensors + 1):
        col_x = f'sensor_{sid}_x'
        col_y = f'sensor_{sid}_y'
        col_z = f'sensor_{sid}_z'

        # 提取原始数据 (N, 3)
        b_raw = df[[col_x, col_y, col_z]].values

        # 椭球校正
        o_i = intrinsic_params[sid]['o_i']
        C_i = intrinsic_params[sid]['C_i']
        b_ellipsoid = apply_ellipsoid_correction(b_raw, o_i, C_i)

        # R_CORR旋转
        r_corr = r_corr_dict[sid]
        b_rot = apply_r_corr_rotation(b_ellipsoid, r_corr)

        # 写回DataFrame
        out_df[col_x] = b_rot[:, 0]
        out_df[col_y] = b_rot[:, 1]
        out_df[col_z] = b_rot[:, 2]

        # 打印统计信息
        raw_norm = np.linalg.norm(b_raw, axis=1).mean()
        corr_norm = np.linalg.norm(b_rot, axis=1).mean()
        print(f"  Sensor {sid:2d}: raw_norm={raw_norm:.4f}, corr_norm={corr_norm:.4f}")

    # 保存结果，传感器数据保留6位小数
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    for col in sensor_cols:
        out_df[col] = out_df[col].round(6)
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved corrected data to: {output_path}")


def compute_background_from_channel_pairs(
    calibrated_csv_path: Path,
    n_sensors: int = 12,
    output_path: Path = None
):
    """
    从已校正的CSV文件读取数据，计算每个channel的 (positive + negative) / 2 背景场。

    Args:
        calibrated_csv_path: rotated_ellip_calibrated_consistency_data.csv 路径
        n_sensors: 传感器数量
        output_path: 输出CSV路径（可选）

    Returns:
        DataFrame: 3行 (channel 1/2/3) x 13列 (channel, sensor_1 ~ sensor_12, 模长)
    """
    df = pd.read_csv(calibrated_csv_path)
    print(f"Loaded {len(df)} rows from {calibrated_csv_path}")

    results = []

    for ch in [1, 2, 3]:
        # 获取 positive 和 negative 行
        pos_row = df[(df['channel'] == f'ch{ch}') & (df['polarity'] == 'positive')]
        neg_row = df[(df['channel'] == f'ch{ch}') & (df['polarity'] == 'negative')]

        if pos_row.empty or neg_row.empty:
            print(f"[WARN] Channel {ch}: data not found")
            continue

        # 计算每个传感器的 (positive + negative) / 2 均值，再取模长
        sensor_magnitudes = []
        for sid in range(1, n_sensors + 1):
            col_x = f'sensor_{sid}_x'
            col_y = f'sensor_{sid}_y'
            col_z = f'sensor_{sid}_z'

            # 取平均值 (positive + negative) / 2
            b_avg = (
                pos_row[[col_x, col_y, col_z]].values +
                neg_row[[col_x, col_y, col_z]].values
            ) / 2.0

            # 计算模长
            magnitude = float(np.linalg.norm(b_avg))
            sensor_magnitudes.append(magnitude)

        row = [ch] + sensor_magnitudes
        results.append(row)
        print(f"  Channel {ch}: computed background from pos/neg pair")

    # 构建输出DataFrame
    cols = ['channel'] + [f'sensor_{sid}' for sid in range(1, n_sensors + 1)]
    out_df = pd.DataFrame(results, columns=cols)

    if output_path is not None:
        out_df.to_csv(output_path, index=False)
        print(f"\nSaved background field to: {output_path}")

    return out_df


def compute_b_ref_from_calibrated(
    calibrated_csv_path: Path,
    n_sensors: int = 12,
    output_path: Path = None
):
    """
    计算每个channel的 b_ref = mean((positive - negative) / 2) across all sensors.
    先对12个传感器的 (positive - negative) / 2 向量取平均，再计算模长，结果取6位小数。

    Args:
        calibrated_csv_path: rotated_ellip_calibrated_consistency_data.csv 路径
        n_sensors: 传感器数量
        output_path: 输出CSV路径（可选）

    Returns:
        DataFrame: 3行 (channel 1/2/3) x 4列 (channel, b_ref_x, b_ref_y, b_ref_z, magnitude)
    """
    df = pd.read_csv(calibrated_csv_path)
    print(f"Loaded {len(df)} rows from {calibrated_csv_path}")

    results = []

    for ch in [1, 2, 3]:
        # 获取 positive 和 negative 行
        pos_row = df[(df['channel'] == f'ch{ch}') & (df['polarity'] == 'positive')]
        neg_row = df[(df['channel'] == f'ch{ch}') & (df['polarity'] == 'negative')]

        if pos_row.empty or neg_row.empty:
            print(f"[WARN] Channel {ch}: data not found")
            continue

        # 计算每个传感器的 (positive - negative) / 2
        sensor_vectors = []
        for sid in range(1, n_sensors + 1):
            col_x = f'sensor_{sid}_x'
            col_y = f'sensor_{sid}_y'
            col_z = f'sensor_{sid}_z'

            # (positive - negative) / 2
            b_diff = (
                pos_row[[col_x, col_y, col_z]].values -
                neg_row[[col_x, col_y, col_z]].values
            ) / 2.0
            sensor_vectors.append(b_diff[0])

        # 对12个传感器的向量取平均
        b_avg = np.mean(sensor_vectors, axis=0)
        magnitude = float(np.linalg.norm(b_avg))

        row = [ch, round(b_avg[0], 6), round(b_avg[1], 6), round(b_avg[2], 6), round(magnitude, 6)]
        results.append(row)
        print(f"  Channel {ch}: b_ref = ({b_avg[0]:.6f}, {b_avg[1]:.6f}, {b_avg[2]:.6f}), |b_ref| = {magnitude:.6f}")

    # 构建输出DataFrame
    cols = ['channel', 'b_ref_x', 'b_ref_y', 'b_ref_z', 'magnitude']
    out_df = pd.DataFrame(results, columns=cols)

    if output_path is not None:
        out_df.to_csv(output_path, index=False)
        print(f"\nSaved b_ref to: {output_path}")

    return out_df


def main():
    print("=" * 70)
    print("Ellipsoid Correction + R_CORR Rotation for Consistency Data")
    print("=" * 70)

    # 检查文件存在
    if not INPUT_CSV.exists():
        print(f"ERROR: Input CSV not found: {INPUT_CSV}")
        return

    # 加载校准参数
    print(f"\nLoading intrinsic params from: {INTRINSIC_PARAMS_PATH}")
    intrinsic_params = load_intrinsic_params(INTRINSIC_PARAMS_PATH)
    print(f"  Loaded params for {len(intrinsic_params)} sensors")

    print(f"\nLoading hardware params from: {HARDWARE_PARAMS_PATH}")
    r_corr_dict = load_hardware_params(HARDWARE_PARAMS_PATH)
    print(f"  Loaded R_CORR for {len(r_corr_dict)} sensors")

    # Step 1: 保存原始数据（未校正）
    print("\n" + "-" * 70)
    print("Step 1: Saving raw data (no correction)")
    print("-" * 70)
    df_raw = pd.read_csv(INPUT_CSV)
    df_raw.to_csv(RAW_OUTPUT_CSV, index=False)
    print(f"Saved raw data to: {RAW_OUTPUT_CSV}")

    # Step 2: 从 raw data 计算背景场
    print("\n" + "=" * 70)
    print("Step 2: Computing background field from RAW data")
    print("=" * 70)
    bg_raw_output_path = CALIB_DIR / 'channel_background_field_raw.csv'
    bg_raw_df = compute_background_from_channel_pairs(
        RAW_OUTPUT_CSV, n_sensors=12, output_path=bg_raw_output_path
    )
    print(f"\nBackground field (RAW) CSV:\n{bg_raw_df.to_string(index=False)}")

    # Step 3: 处理CSV - 椭球校正 + R_CORR旋转
    print("\n" + "=" * 70)
    print("Step 3: Ellipsoid Correction + R_CORR Rotation")
    print("=" * 70)
    process_csv(INPUT_CSV, OUTPUT_CSV, intrinsic_params, r_corr_dict)

    # Step 4: 从校正后数据计算背景场
    print("\n" + "=" * 70)
    print("Step 4: Computing background field from CALIBRATED data")
    print("=" * 70)
    bg_cal_output_path = CALIB_DIR / 'channel_background_field_calibrated.csv'
    bg_cal_df = compute_background_from_channel_pairs(
        OUTPUT_CSV, n_sensors=12, output_path=bg_cal_output_path
    )
    print(f"\nBackground field (CALIBRATED) CSV:\n{bg_cal_df.to_string(index=False)}")

    # Step 5: 计算 b_ref = (positive - negative) / 2
    print("\n" + "=" * 70)
    print("Step 5: Computing b_ref = (positive - negative) / 2 from CALIBRATED data")
    print("=" * 70)
    b_ref_output_path = CALIB_DIR / 'b_ref_calibrated.csv'
    b_ref_df = compute_b_ref_from_calibrated(
        OUTPUT_CSV, n_sensors=12, output_path=b_ref_output_path
    )
    print(f"\nb_ref CSV:\n{b_ref_df.to_string(index=False)}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()