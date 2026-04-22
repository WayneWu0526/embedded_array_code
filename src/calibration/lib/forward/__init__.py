"""
calibration.lib.forward - 正向校正模块

提供 Strategy 模式的可插拔校正步骤，可独立使用或组合。
"""
from .base import CorrectionStep
from .processor import CalibrationProcessor
from .ellipsoid import EllipsoidCorrection
from .r_corr import RCorrRotation
from .consistency import ConsistencyCorrection

__all__ = [
    "CorrectionStep",
    "CalibrationProcessor",
    "EllipsoidCorrection",
    "RCorrRotation",
    "ConsistencyCorrection",
]