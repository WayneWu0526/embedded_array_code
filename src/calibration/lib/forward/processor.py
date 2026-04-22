"""校正处理门面类"""
from typing import List, Optional, Tuple

from .base import CorrectionStep


class CalibrationProcessor:
    """
    门面类：持有校正步骤列表，按序调用。

    Args:
        sensor_config: 传感器配置对象（提供 intrinsic/hardware/consistency 参数）
        steps: 校正步骤列表，默认为 [Ellipsoid, RCorr, Consistency]。
               传入空列表则不做任何校正（直通）。
    """

    def __init__(
        self,
        sensor_config: "SensorArrayConfig",
        steps: Optional[List[CorrectionStep]] = None,
    ):
        self._config = sensor_config
        self._steps = steps or self._default_steps(sensor_config)

    def _default_steps(self, config):
        from .ellipsoid import EllipsoidCorrection
        from .r_corr import RCorrRotation
        from .consistency import ConsistencyCorrection

        return [
            EllipsoidCorrection(config),
            RCorrRotation(config),
            ConsistencyCorrection(config),
        ]

    def apply(
        self, sensor_id: int, x: float, y: float, z: float
    ) -> Tuple[float, float, float]:
        """
        依次通过所有 step，返回最终校正结果。

        Args:
            sensor_id: 传感器编号
            x, y, z: 原始磁场分量

        Returns:
            (cx, cy, cz): 最终校正后的磁场分量
        """
        cx, cy, cz = x, y, z
        for step in self._steps:
            cx, cy, cz = step.apply(cx, cy, cz, sensor_id)
        return cx, cy, cz

    @property
    def steps(self) -> List[CorrectionStep]:
        """返回当前步骤列表（用于 introspection）"""
        return list(self._steps)
