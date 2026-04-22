"""椭球校正步骤"""
import numpy as np
from typing import Tuple

from .base import CorrectionStep


class EllipsoidCorrection(CorrectionStep):
    """
    Phase 1 椭球校正。

    公式: b_corr = (b_raw - o_i) @ C_i.T

    参数来源: sensor_config.intrinsic
    """

    def __init__(self, sensor_config: "SensorArrayConfig"):
        self._offset = {}
        self._correction = {}

        try:
            intrinsic = sensor_config.intrinsic
            if intrinsic and intrinsic.params:
                for sid, params in intrinsic.params.items():
                    self._offset[sid] = np.array(params.o_i)
                    self._correction[sid] = np.array(params.C_i)
        except Exception:
            pass

        # fallback: 无参数时用单位阵
        if not self._offset:
            n = sensor_config.manifest.n_sensors
            for sid in range(1, n + 1):
                self._offset[sid] = np.zeros(3)
                self._correction[sid] = np.eye(3)

    @property
    def name(self) -> str:
        return "ellipsoid"

    def apply(
        self, x: float, y: float, z: float, sensor_id: int
    ) -> Tuple[float, float, float]:
        o_i = self._offset.get(sensor_id, np.zeros(3))
        C_i = self._correction.get(sensor_id, np.eye(3))
        b = np.array([x, y, z])
        b_corr = (b - o_i) @ C_i.T
        return b_corr[0], b_corr[1], b_corr[2]
