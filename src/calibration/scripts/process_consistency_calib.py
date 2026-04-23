#!/usr/bin/env python3
"""
process_consistency_calib.py - 后处理consistency_calib.csv

处理流程:
    consistency_calib.csv -> 椭球校正 -> (positive - negative) / 2 -> 输出

输入: src/calibration/data/consistency_calib.csv
      src/sensor_array_config/.../intrinsic_params.json
输出: src/calibration/data/channel_background_field_calibrated.csv

输出格式 (3行 x 40列):
    channel, b_ref_x, b_ref_y, b_ref_z,
    sensor_1_x, sensor_1_y, sensor_1_z,
    sensor_2_x, sensor_2_y, sensor_2_z, ...,
    sensor_12_x, sensor_12_y, sensor_12_z

第二阶段: 计算每个传感器的旋转矩阵R
    - 对b_ref沿每个channel归一化
    - 对每个sensor在三个channel下的校正值归一化
    - 用Kabsch算法计算旋转矩阵R

用法:
    rosrun calibration process_consistency_calib.py

或指定b_ref参数:
    rosrun calibration process_consistency_calib.py \
        _b_ref_ch1_x:=0.194062 _b_ref_ch1_y:=13.270273 _b_ref_ch1_z:=-0.34768 \
        _b_ref_ch2_x:=-14.045894 _b_ref_ch2_y:=-0.115834 _b_ref_ch2_z:=0.484175 \
        _b_ref_ch3_x:=-0.006901 _b_ref_ch3_y:=0.380408 _b_ref_ch3_z:=14.927272
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
import rospy

# 路径配置
SCRIPT_DIR = Path(__file__).parent.resolve()
CALIB_DIR = SCRIPT_DIR.parent / 'data'
INPUT_CSV = CALIB_DIR / 'consistency_calib.csv'
OUTPUT_CSV = CALIB_DIR / 'channel_background_field_calibrated.csv'
B_REF_CSV = CALIB_DIR / 'b_ref_calibrated.csv'
R_MATRIX_CSV = CALIB_DIR / 'sensor_R_matrices.csv'
INTRINSIC_PARAMS_PATH = Path('/home/zhang/embedded_array_ws/src/sensor_array_config/sensor_array_config/config/qmc6309/intrinsic_params.json')


def kabsch_solver(V: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Solve the Orthogonal Procrustes problem:
        min_{R in SO(3)} ||V - R @ U||_F

    Args:
        V: (3, 3) matrix of target/reference vectors (columns)
        U: (3, 3) matrix of source vectors (columns)

    Returns:
        R: (3, 3) rotation matrix in SO(3)
    """
    if V.shape != U.shape:
        raise ValueError(f"Shape mismatch: V{V.shape} vs U{U.shape}")
    if V.shape != (3, 3):
        raise ValueError(f"Expected 3x3 inputs, got {V.shape}")

    H = V @ U.T
    L, _, Wh = np.linalg.svd(H)

    d = np.sign(np.linalg.det(L @ Wh))
    if d == 0:
        d = 1.0

    D = np.diag([1.0, 1.0, d])
    R = L @ D @ Wh

    return R


def load_intrinsic_params(path: Path) -> dict:
    """加载椭球校正参数"""
    with open(path, 'r') as f:
        data = json.load(f)
    result = {}
    for sid, params in data.items():
        result[int(sid)] = {
            'o_i': np.array(params['o_i']),
            'C_i': np.array(params['C_i'])
        }
    return result


def load_b_ref_from_csv(csv_path: Path) -> dict:
    """从b_ref_calibrated.csv加载b_ref参考值"""
    df = pd.read_csv(csv_path)
    b_ref = {}
    for _, row in df.iterrows():
        ch = int(row['channel'])
        b_ref[ch] = {
            'x': row['b_ref_x'],
            'y': row['b_ref_y'],
            'z': row['b_ref_z']
        }
    return b_ref


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


def process_consistency_calib(
    input_path: Path,
    output_path: Path,
    intrinsic_params: dict,
    b_ref: dict,
    n_sensors: int = 12
):
    """
    处理consistency_calib.csv，应用椭球校正并计算(positive+negative)/2

    Args:
        input_path: 输入CSV路径 (consistency_calib.csv)
        output_path: 输出CSV路径
        intrinsic_params: {sensor_id: {o_i, C_i}}
        b_ref: {channel: {x, y, z}}
        n_sensors: 传感器数量
    """
    df = pd.read_csv(input_path)
    rospy.loginfo("Loaded %d rows from %s", len(df), input_path)

    results = []

    for ch in [1, 2, 3]:
        # 获取 positive 和 negative 行 (channel可能是"ch1"或1)
        pos_row = df[(df['channel'] == f'ch{ch}') & (df['polarity'] == 'positive')]
        neg_row = df[(df['channel'] == f'ch{ch}') & (df['polarity'] == 'negative')]

        if pos_row.empty or neg_row.empty:
            rospy.logwarn("Channel %d: data not found", ch)
            continue

        # 对每个传感器进行椭球校正，然后计算 (positive + negative) / 2
        sensor_vectors = []
        for sid in range(1, n_sensors + 1):
            col_x = f'sensor_{sid}_x'
            col_y = f'sensor_{sid}_y'
            col_z = f'sensor_{sid}_z'

            # 获取原始数据
            pos_data = pos_row[[col_x, col_y, col_z]].values[0]
            neg_data = neg_row[[col_x, col_y, col_z]].values[0]

            # 椭球校正
            o_i = intrinsic_params[sid]['o_i']
            C_i = intrinsic_params[sid]['C_i']
            pos_corr = apply_ellipsoid_correction(pos_data.reshape(1, 3), o_i, C_i)[0]
            neg_corr = apply_ellipsoid_correction(neg_data.reshape(1, 3), o_i, C_i)[0]

            # (positive - negative) / 2  消除背景场
            avg_corr = (pos_corr - neg_corr) / 2.0
            sensor_vectors.append(avg_corr)

        # 获取b_ref
        b_ref_vec = b_ref.get(ch, {'x': 0.0, 'y': 0.0, 'z': 0.0})

        # 构建输出行: channel, b_ref_x, b_ref_y, b_ref_z, sensor_1_x, ...
        row = [ch, b_ref_vec['x'], b_ref_vec['y'], b_ref_vec['z']]
        for sv in sensor_vectors:
            row.extend([sv[0], sv[1], sv[2]])

        results.append(row)
        rospy.loginfo("Channel %d: processed %d sensors, b_ref=(%.4f, %.4f, %.4f)",
                      ch, n_sensors, b_ref_vec['x'], b_ref_vec['y'], b_ref_vec['z'])

    # 构建输出DataFrame
    cols = ['channel', 'b_ref_x', 'b_ref_y', 'b_ref_z']
    for sid in range(1, n_sensors + 1):
        cols.extend([f'sensor_{sid}_x', f'sensor_{sid}_y', f'sensor_{sid}_z'])

    out_df = pd.DataFrame(results, columns=cols)

    # 保留6位小数
    sensor_cols = [c for c in out_df.columns if c.startswith('sensor_')]
    for col in sensor_cols:
        out_df[col] = out_df[col].round(6)

    out_df.to_csv(output_path, index=False)
    rospy.loginfo("Saved to: %s", output_path)

    return out_df


def compute_R_matrices(calibrated_csv_path: Path, output_path: Path, n_sensors: int = 12):
    """
    计算每个传感器的旋转矩阵R

    算法:
    1. 读取calibrated_csv中每个sensor在3个channel下的校正值
    2. 对b_ref沿每个channel归一化: b_ref_norm_i = b_ref_i / ||b_ref_i||
    3. 对sensor值归一化: sensor_norm_i = sensor_i / ||b_ref_i|| (用对应的b_ref magnitude归一化)
    4. 用Kabsch算法求解: R @ sensor_norm ≈ b_ref_norm

    Args:
        calibrated_csv_path: channel_background_field_calibrated.csv 路径
        output_path: 输出CSV路径
        n_sensors: 传感器数量
    """
    df = pd.read_csv(calibrated_csv_path)
    rospy.loginfo("Loaded %d rows from %s", len(df), calibrated_csv_path)

    # 提取b_ref向量并归一化 (3 channels)
    b_ref_vectors = np.zeros((3, 3))  # shape: (3, 3), each row is a channel
    b_ref_magnitudes = np.zeros(3)
    for i, ch in enumerate([1, 2, 3]):
        row = df[df['channel'] == ch].iloc[0]
        b_ref_vectors[i] = [row['b_ref_x'], row['b_ref_y'], row['b_ref_z']]
        b_ref_magnitudes[i] = np.linalg.norm(b_ref_vectors[i])

    rospy.loginfo("b_ref magnitudes: %s", b_ref_magnitudes)

    # 归一化b_ref: 每个channel的b_ref除以其模长
    b_ref_normalized = b_ref_vectors / b_ref_magnitudes[:, np.newaxis]
    rospy.loginfo("Normalized b_ref (rows=channels):\n%s", b_ref_normalized)

    results = []

    for sid in range(1, n_sensors + 1):
        # 提取sensor在3个channel下的校正值 (shape: 3x3)
        sensor_vectors = np.zeros((3, 3))  # rows: channel, cols: x,y,z
        for i, ch in enumerate([1, 2, 3]):
            row = df[df['channel'] == ch].iloc[0]
            col_x = f'sensor_{sid}_x'
            col_y = f'sensor_{sid}_y'
            col_z = f'sensor_{sid}_z'
            sensor_vectors[i] = [row[col_x], row[col_y], row[col_z]]

        rospy.loginfo("Sensor %d raw vectors (rows=channels):\n%s", sid, sensor_vectors)

        # 用对应的b_ref magnitude归一化sensor向量
        sensor_normalized = sensor_vectors / b_ref_magnitudes[:, np.newaxis]
        rospy.loginfo("Sensor %d normalized:\n%s", sid, sensor_normalized)

        # Kabsch: R @ sensor_normalized.T ≈ b_ref_normalized.T
        # 即 R @ U ≈ V, 其中 U=sensor_normalized.T, V=b_ref_normalized.T
        # 或者说 R @ sensor_normalized (as rows)  ≈ b_ref_normalized (as rows)
        #
        # 实际上: 我们希望 R @ s_i ≈ b_i_norm for each channel i
        # 所以 V = b_ref_normalized.T (3x3), U = sensor_normalized.T (3x3)
        # R @ sensor_normalized.T ≈ b_ref_normalized.T
        #
        # kabsch(V, U) returns R such that R @ U ≈ V
        # 所以应该 kabsch(b_ref_normalized.T, sensor_normalized.T)

        V = b_ref_normalized.T  # target: (3, 3), columns are normalized b_ref vectors
        U = sensor_normalized.T  # source: (3, 3), columns are normalized sensor vectors

        R = kabsch_solver(V, U)

        # 验证: R @ sensor_normalized.T should ≈ b_ref_normalized.T
        residual = np.linalg.norm(R @ sensor_normalized.T - b_ref_normalized.T)
        rospy.loginfo("Sensor %d: R =\n%s", sid, R)
        rospy.loginfo("Sensor %d: residual = %.6f", sid, residual)

        # 保存R矩阵 (flattened row-major)
        row_data = [sid] + R.flatten().tolist()
        results.append(row_data)

    # 构建输出DataFrame
    cols = ['sensor_id']
    for i in range(3):
        for j in range(3):
            cols.append(f'R_{i}{j}')
    out_df = pd.DataFrame(results, columns=cols)

    # 保留6位小数
    for col in cols[1:]:
        out_df[col] = out_df[col].round(6)

    out_df.to_csv(output_path, index=False)
    rospy.loginfo("Saved R matrices to: %s", output_path)

    return out_df


def apply_R_correction(
    calibrated_csv_path: Path,
    r_matrix_csv_path: Path,
    output_path: Path,
    n_sensors: int = 12
):
    """
    对椭球校正后的数据应用R矩阵校正

    公式: sensor_corrected = R @ sensor_ellipsoid

    Args:
        calibrated_csv_path: channel_background_field_calibrated.csv 路径
        r_matrix_csv_path: sensor_R_matrices.csv 路径
        output_path: 输出CSV路径
        n_sensors: 传感器数量
    """
    df_cal = pd.read_csv(calibrated_csv_path)
    df_r = pd.read_csv(r_matrix_csv_path)

    rospy.loginfo("Loaded %d rows from %s", len(df_cal), calibrated_csv_path)
    rospy.loginfo("Loaded %d R matrices from %s", len(df_r), r_matrix_csv_path)

    results = []

    for ch in [1, 2, 3]:
        row_cal = df_cal[df_cal['channel'] == ch].iloc[0]
        # b_ref保持不变
        b_ref_x = row_cal['b_ref_x']
        b_ref_y = row_cal['b_ref_y']
        b_ref_z = row_cal['b_ref_z']

        row_result = [ch, b_ref_x, b_ref_y, b_ref_z]

        for sid in range(1, n_sensors + 1):
            # 获取R矩阵
            r_row = df_r[df_r['sensor_id'] == sid].iloc[0]
            R = np.array([
                [r_row['R_00'], r_row['R_01'], r_row['R_02']],
                [r_row['R_10'], r_row['R_11'], r_row['R_12']],
                [r_row['R_20'], r_row['R_21'], r_row['R_22']]
            ])

            # 获取椭球校正后的传感器值
            col_x = f'sensor_{sid}_x'
            col_y = f'sensor_{sid}_y'
            col_z = f'sensor_{sid}_z'
            sensor_vec = np.array([
                row_cal[col_x],
                row_cal[col_y],
                row_cal[col_z]
            ])

            # 应用R校正: R @ sensor
            sensor_corrected = R @ sensor_vec

            row_result.extend([sensor_corrected[0], sensor_corrected[1], sensor_corrected[2]])

        results.append(row_result)

    # 构建输出DataFrame
    cols = ['channel', 'b_ref_x', 'b_ref_y', 'b_ref_z']
    for sid in range(1, n_sensors + 1):
        cols.extend([f'sensor_{sid}_x', f'sensor_{sid}_y', f'sensor_{sid}_z'])

    out_df = pd.DataFrame(results, columns=cols)

    # 保留6位小数
    sensor_cols = [c for c in out_df.columns if c.startswith('sensor_')]
    for col in sensor_cols:
        out_df[col] = out_df[col].round(6)

    out_df.to_csv(output_path, index=False)
    rospy.loginfo("Saved R-corrected data to: %s", output_path)

    return out_df


def main():
    rospy.init_node('process_consistency_calib', anonymous=True)

    # 尝试从ROS参数获取b_ref，如果没有则从CSV加载
    b_ref = {}
    for ch in [1, 2, 3]:
        try:
            x = rospy.get_param(f'~b_ref_ch{ch}_x', None)
            y = rospy.get_param(f'~b_ref_ch{ch}_y', None)
            z = rospy.get_param(f'~b_ref_ch{ch}_z', None)
            if x is not None and y is not None and z is not None:
                b_ref[ch] = {'x': x, 'y': y, 'z': z}
                rospy.loginfo("Using ROS param b_ref for channel %d: (%.4f, %.4f, %.4f)", ch, x, y, z)
        except:
            pass

    # 如果没有从ROS参数获取到所有b_ref，则从CSV加载
    if len(b_ref) < 3:
        rospy.loginfo("Loading b_ref from: %s", B_REF_CSV)
        b_ref_csv = load_b_ref_from_csv(B_REF_CSV)
        for ch, vals in b_ref_csv.items():
            if ch not in b_ref:
                b_ref[ch] = vals
                rospy.loginfo("Using CSV b_ref for channel %d: (%.4f, %.4f, %.4f)",
                              ch, vals['x'], vals['y'], vals['z'])

    # 检查输入文件
    if not INPUT_CSV.exists():
        rospy.logerr("Input CSV not found: %s", INPUT_CSV)
        sys.exit(1)

    # 加载校准参数
    rospy.loginfo("Loading intrinsic params from: %s", INTRINSIC_PARAMS_PATH)
    intrinsic_params = load_intrinsic_params(INTRINSIC_PARAMS_PATH)
    rospy.loginfo("Loaded params for %d sensors", len(intrinsic_params))

    # 处理数据
    rospy.loginfo("=" * 60)
    rospy.loginfo("Processing consistency calibration data")
    rospy.loginfo("=" * 60)

    out_df = process_consistency_calib(
        INPUT_CSV, OUTPUT_CSV, intrinsic_params, b_ref, n_sensors=12
    )

    rospy.loginfo("")
    rospy.loginfo("=" * 60)
    rospy.loginfo("Output CSV (%d rows x %d cols):", len(out_df), len(out_df.columns))
    rospy.loginfo("=" * 60)
    print(out_df.to_string(index=False))

    # 第二阶段: 计算每个传感器的旋转矩阵R
    rospy.loginfo("")
    rospy.loginfo("=" * 60)
    rospy.loginfo("Computing rotation matrices R for each sensor")
    rospy.loginfo("=" * 60)

    r_df = compute_R_matrices(OUTPUT_CSV, R_MATRIX_CSV, n_sensors=12)

    rospy.loginfo("")
    rospy.loginfo("=" * 60)
    rospy.loginfo("R matrices (%d rows):", len(r_df))
    rospy.loginfo("=" * 60)
    print(r_df.to_string(index=False))

    # 第三阶段: 应用R矩阵校正
    rospy.loginfo("")
    rospy.loginfo("=" * 60)
    rospy.loginfo("Applying R matrix correction")
    rospy.loginfo("=" * 60)

    r_corrected_csv = CALIB_DIR / 'channel_background_field_R_corrected.csv'
    r_corr_df = apply_R_correction(OUTPUT_CSV, R_MATRIX_CSV, r_corrected_csv, n_sensors=12)

    rospy.loginfo("")
    rospy.loginfo("=" * 60)
    rospy.loginfo("R-corrected data (%d rows x %d cols):", len(r_corr_df), len(r_corr_df.columns))
    rospy.loginfo("=" * 60)
    print(r_corr_df.to_string(index=False))

    rospy.loginfo("")
    rospy.loginfo("Done!")


if __name__ == '__main__':
    main()
