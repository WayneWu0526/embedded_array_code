#!/usr/bin/env python3
"""
ellipsoid_fit - QMC6309 传感器椭球校准模块

使用方法:
    from calibration.ellipsoid_fit import ellipsoid_fit, batch_ellipsoid_fit

    # 单颗传感器校准
    result = ellipsoid_fit(b_raw, sensor_id=1)

    # 批量校准
    results = batch_ellipsoid_fit('data.csv')
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
    # Main fitting
    'ellipsoid_fit',
    'batch_ellipsoid_fit',
    'apply_calibration',
    'save_calibration_params',
    'load_calibration_params',
    'CalibrationResult',

    # Full ellipsoid (底层实现)
    'full_ellipsoid_fit',
    'full_ellipsoid_fit_iterative',
]
