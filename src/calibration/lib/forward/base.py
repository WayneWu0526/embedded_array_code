"""校正步骤抽象基类"""
from abc import ABC, abstractmethod
from typing import Tuple


class CorrectionStep(ABC):
    """Strategy 模式抽象基类：单一校正步骤"""

    @property
    @abstractmethod
    def name(self) -> str:
        """步骤名称，用于日志和 introspection"""
        pass

    @abstractmethod
    def apply(
        self, x: float, y: float, z: float, sensor_id: int
    ) -> Tuple[float, float, float]:
        """
        对单颗传感器的单次采样进行校正。

        Args:
            x, y, z: 校正前的磁场分量 (Gauss)
            sensor_id: 传感器编号 (1-12)

        Returns:
            (cx, cy, cz): 校正后的磁场分量
        """
        pass

    def reset(self):
        """可选：重置内部状态（供 processor 调用）"""
        pass
