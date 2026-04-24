#!/usr/bin/env python3
"""
ellipsoid_fit - QMC6309 传感器椭球校准模块

使用方法:
    from calibration.lib.ellipsoid_fit import ellipsoid_fit, batch_ellipsoid_fit

    # 单颗传感器校准（拟合 + 评价）
    result = ellipsoid_fit(b_raw, sensor_id=1)

    # 批量校准
    results = batch_ellipsoid_fit('data.csv')

评估（使用预计算参数，无需拟合）:
    # 单颗传感器评估
    result = ellipsoid_fit(b_raw, sensor_id=1, o_i=o_i, C_i=C_i)

    # 批量评估（自动从配置加载参数）
    results = batch_ellipsoid_fit('data.csv', sensor_type='QMC6309', evaluate_only=True)
"""

from .ellipsoid_fit import (
    ellipsoid_fit,
    batch_ellipsoid_fit,
    apply_calibration,
    save_calibration_params,
    load_calibration_params,
    CalibrationResult,
)

from .full_ellipsoid_fit import full_ellipsoid_fit, full_ellipsoid_fit_iterative

__all__ = [
    'ellipsoid_fit',
    'batch_ellipsoid_fit',
    'apply_calibration',
    'save_calibration_params',
    'load_calibration_params',
    'CalibrationResult',
    'full_ellipsoid_fit',
    'full_ellipsoid_fit_iterative',
]
