"""一致性校正步骤"""
import numpy as np
from typing import Tuple

from .base import CorrectionStep


class ConsistencyCorrection(CorrectionStep):
    """
    Phase 2 一致性校正。

    公式: b_final = D_i @ b + e_i, 再除以 amp_factor (方案B)

    参数来源: sensor_config.consistency
    """

    def __init__(self, sensor_config: "SensorArrayConfig"):
        self._D = {}
        self._e = {}

        try:
            consistency = sensor_config.consistency
            if consistency and consistency.params:
                for sid, params in consistency.params.items():
                    self._D[sid] = np.array(params.D_i)
                    self._e[sid] = np.array(params.e_i)
                self._amp_factor = consistency.amp_factor
            else:
                self._amp_factor = None
        except Exception:
            self._amp_factor = None

        # fallback: 无参数时用单位阵
        if not self._D:
            n = sensor_config.manifest.n_sensors
            for sid in range(1, n + 1):
                self._D[sid] = np.eye(3)
                self._e[sid] = np.zeros(3)
            self._amp_factor = 1.0

        if self._amp_factor is None:
            self._amp_factor = 1.0

    @property
    def name(self) -> str:
        return "consistency"

    def apply(
        self, x: float, y: float, z: float, sensor_id: int
    ) -> Tuple[float, float, float]:
        D = self._D.get(sensor_id, np.eye(3))
        e = self._e.get(sensor_id, np.zeros(3))
        b = np.array([x, y, z])
        b_cons = D @ b + e
        if self._amp_factor is not None and self._amp_factor != 0:
            b_cons = b_cons / self._amp_factor
        return tuple(b_cons.tolist())