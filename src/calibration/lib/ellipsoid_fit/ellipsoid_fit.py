#!/usr/bin/env python3
"""
ellipsoid_fit.py - QMC6309 传感器椭球拟合

参考论文公式:
    b_i^{raw} = A_i * b^{true} + o_i
    b_i^{corr} = C_i * (b_i^{raw} - o_i)

其中 C_i 是 3×3 校正矩阵，o_i 是 hard iron offset。
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict

from .full_ellipsoid_fit import full_ellipsoid_fit
from sensor_array_config.base import get_config, SensorArrayConfig, IntrinsicParamsSet, IntrinsicParams


@dataclass
class CalibrationResult:
    """单颗传感器校准结果"""
    sensor_id: int
    csv_file: str
    o_i: List[float]      # Hard iron offset (3,)
    C_i: List[List[float]]  # Correction matrix (3×3)
    eigenvalue_ratio: float
    eigenvalues: List[float]
    radius_raw_std: float
    radius_corr_std: float
    improvement_ratio: float
    fit_info: Dict

    def to_dict(self) -> dict:
        return asdict(self)


def ellipsoid_fit(b_raw: np.ndarray, sensor_id: int,
                  csv_file: str = "",
                  use_iterative: bool = False) -> CalibrationResult:
    """
    单颗传感器椭球拟合

    Args:
        b_raw: N×3 原始磁场采样数据
        sensor_id: 传感器编号 (1-12)
        csv_file: 原始数据文件名
        use_iterative: 是否使用迭代方法

    Returns:
        CalibrationResult: 校准结果
    """
    # 计算 eigenvalue ratio
    cov = np.cov(b_raw.T)
    eigenvalues_all = np.linalg.eigvalsh(cov)
    eigenvalues_all = np.sort(eigenvalues_all)
    eigenvalue_ratio = eigenvalues_all[-1] / eigenvalues_all[0]

    # 执行椭球拟合
    if use_iterative:
        from .full_ellipsoid_fit import full_ellipsoid_fit_iterative
        o_i, C_i, fit_info = full_ellipsoid_fit_iterative(b_raw)
    else:
        o_i, C_i, fit_info = full_ellipsoid_fit(b_raw)

    # 计算校正后的统计
    b_centered = b_raw - o_i
    b_corr = b_centered @ C_i.T
    radius_corr = np.linalg.norm(b_corr, axis=1)
    radius_raw = np.linalg.norm(b_raw - np.mean(b_raw, axis=0), axis=1)

    # 构建结果
    result = CalibrationResult(
        sensor_id=sensor_id,
        csv_file=csv_file,
        o_i=o_i.tolist(),
        C_i=C_i.tolist(),
        eigenvalue_ratio=float(eigenvalue_ratio),
        eigenvalues=eigenvalues_all.tolist(),
        radius_raw_std=float(np.std(radius_raw)),
        radius_corr_std=float(np.std(radius_corr)),
        improvement_ratio=float(np.std(radius_raw) / np.std(radius_corr)) if np.std(radius_corr) > 1e-10 else float('inf'),
        fit_info=fit_info
    )

    return result


def apply_calibration(b_raw: np.ndarray, o_i: np.ndarray, C_i: np.ndarray) -> np.ndarray:
    """
    将校准参数应用到原始数据

    公式: b_corr = C_i * (b_raw - o_i)

    Args:
        b_raw: N×3 原始数据
        o_i: (3,) offset 向量
        C_i: (3×3) 校正矩阵

    Returns:
        b_corr: N×3 校正后数据
    """
    b_raw = np.asarray(b_raw)
    o_i = np.asarray(o_i)
    C_i = np.asarray(C_i)

    b_centered = b_raw - o_i
    b_corr = b_centered @ C_i.T

    return b_corr


def batch_ellipsoid_fit(csv_path: Path, output_dir: Path = None,
                        use_iterative: bool = False,
                        sensor_config: SensorArrayConfig = None) -> List[CalibrationResult]:
    """
    批量处理单个 CSV 文件中所有传感器的校准

    Args:
        csv_path: CSV 文件路径
        output_dir: 输出目录（可选，用于保存参数）
        use_iterative: 是否使用迭代方法
        sensor_config: 传感器配置（可选，默认使用 QMC6309）

    Returns:
        List of CalibrationResult for all sensors
    """
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    import pandas as pd

    # 读取数据
    df = pd.read_csv(csv_path)

    results = []

    for sensor_id in sensor_config.get_sensor_ids():
        col_x = f'sensor_{sensor_id}_x'
        col_y = f'sensor_{sensor_id}_y'
        col_z = f'sensor_{sensor_id}_z'

        if col_x not in df.columns:
            print(f"[WARN] Sensor {sensor_id} columns not found in {csv_path.name}")
            continue

        b_raw = df[[col_x, col_y, col_z]].values.astype(float)

        # 过滤无效数据
        valid_mask = ~(np.isnan(b_raw).any(axis=1) | np.isinf(b_raw).any(axis=1))
        b_raw = b_raw[valid_mask]

        if len(b_raw) < 100:
            print(f"[WARN] Sensor {sensor_id}: insufficient valid data ({len(b_raw)} points)")
            continue

        # 执行校准
        result = ellipsoid_fit(
            b_raw=b_raw,
            sensor_id=sensor_id,
            csv_file=csv_path.name,
            use_iterative=use_iterative
        )
        results.append(result)

        # 打印结果
        print(f"  Sensor {sensor_id:2d}: ratio={result.eigenvalue_ratio:6.2f}, "
              f"improvement={result.improvement_ratio:8.2f}x, "
              f"radius: {result.radius_raw_std:.4f} → {result.radius_corr_std:.4f}")

    # 保存参数
    if output_dir is not None:
        output_path = Path(output_dir) / f"intrinsic_params_{Path(csv_path).stem}.json"
        save_calibration_params(results, output_path, sensor_config=sensor_config)

    return results


def save_calibration_params(results: List[CalibrationResult], output_path: Path,
                             sensor_config: SensorArrayConfig = None):
    """
    保存校准参数到 JSON 文件

    Args:
        results: 校准结果列表
        output_path: 输出文件路径
        sensor_config: 传感器配置（可选，默认使用 QMC6309）
    """
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    params_set = IntrinsicParamsSet(params={
        r.sensor_id: IntrinsicParams(o_i=r.o_i, C_i=r.C_i)
        for r in results
    })
    params_set.to_json(str(output_path))
    print(f"\nCalibration params saved to: {output_path}")


def load_calibration_params(params_path: Path) -> Dict:
    """
    从 JSON 文件加载校准参数

    Args:
        params_path: 参数文件路径

    Returns:
        参数字典
    """
    with open(params_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    print("Ellipsoid Fit Module for QMC6309 Sensors")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - ellipsoid_fit(): Fit single sensor")
    print("  - batch_ellipsoid_fit(): Fit all 12 sensors from CSV")
    print("  - apply_calibration(): Apply calibration to raw data")
    print("  - save/load_calibration_params(): Save/load params")
