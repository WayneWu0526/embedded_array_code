"""R_CORR 旋转变换步骤"""
import numpy as np
from typing import Tuple

from .base import CorrectionStep


class RCorrRotation(CorrectionStep):
    """
    R_CORR 旋转变换：将传感器局部坐标系转换到参考坐标系。

    公式: b_rot = R_CORR @ b

    参数来源: sensor_config.hardware.R_CORR
    """

    def __init__(self, sensor_config: "SensorArrayConfig"):
        self._r_corr = {}

        try:
            hw = sensor_config.hardware
            for entry in hw.R_CORR:
                mat = np.array(entry.matrix).reshape(3, 3, order='F')
                for sid in entry.sensor_ids:
                    self._r_corr[sid] = mat
        except Exception:
            pass

        # fallback: 无参数时直通
        if not self._r_corr:
            n = sensor_config.manifest.n_sensors
            for sid in range(1, n + 1):
                self._r_corr[sid] = np.eye(3)

    @property
    def name(self) -> str:
        return "r_corr"

    def apply(
        self, x: float, y: float, z: float, sensor_id: int
    ) -> Tuple[float, float, float]:
        R = self._r_corr.get(sensor_id)
        if R is None:
            return x, y, z
        b = np.array([x, y, z])
        return tuple((R @ b).tolist())
