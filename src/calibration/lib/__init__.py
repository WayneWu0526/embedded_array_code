#!/usr/bin/env python3
"""
consistency_fit - Phase 2: 一致性校准模块

使用方法:
    from calibration.lib.consistency_fit import consistency_fit, batch_consistency_fit

    # 执行一致性校准
    D_list, e_list, fit_info = consistency_fit(csv_dir='data/consistency/')

    # 批量处理并保存
    results = batch_consistency_fit(csv_dir, output_path='report/consistency_params.json')
"""

from .consistency_fit import (
    consistency_fit,
    batch_consistency_fit,
    apply_consistency_correction,
    validate_consistency,
    save_consistency_params,
    load_consistency_params,
    load_ellipsoid_params,
    load_csv_data,
    compute_stable_mean,
    fit_D_and_e,
    FieldCondition,
    ConsistencyResult,
)

__all__ = [
    # Main fitting API
    'consistency_fit',
    'batch_consistency_fit',
    'validate_consistency',

    # Apply calibration
    'apply_consistency_correction',

    # Save/load
    'save_consistency_params',
    'load_consistency_params',
    'load_ellipsoid_params',

    # Utilities
    'load_csv_data',
    'compute_stable_mean',
    'fit_D_and_e',

    # Data classes
    'FieldCondition',
    'ConsistencyResult',
]
