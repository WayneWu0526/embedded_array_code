from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SensorChipConfig:
    name: str
    manufacturer: str
    bit_width: int
    gs_to_tesla: float
    adu_to_gs: float
    range_gs: Optional[float] = None
    notes: str = ""


@dataclass(frozen=True)
class ArrayManifest:
    name: str
    sensor_type: str
    n_sensors: int
    n_groups: int
    sensors_per_group: int
    calibration_status: str
    default_startup_sensors: str
    frame_id: str
    notes: str = ""


@dataclass(frozen=True)
class AffineModelParams:
    D_i: List[List[float]]
    e_i: List[float]


@dataclass(frozen=True)
class AffineModelParamsSet:
    params: Dict[int, AffineModelParams]


@dataclass(frozen=True)
class RCorrEntry:
    sensor_ids: List[int]
    matrix: List[float]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.matrix, dtype=float).reshape(3, 3, order="F")


@dataclass(frozen=True)
class HardwareParams:
    d_list: List[List[float]]
    R_CORR: List[RCorrEntry]
    description: str = ""


@dataclass(frozen=True)
class ArrayConfig:
    name: str
    manifest: ArrayManifest
    sensor_type: SensorChipConfig
    hardware: HardwareParams
    affine_model: AffineModelParamsSet

    @property
    def adu_to_gs(self) -> float:
        return self.sensor_type.adu_to_gs

    @property
    def gs_to_si(self) -> float:
        return self.sensor_type.gs_to_tesla

    def get_sensor_ids(self) -> List[int]:
        return list(range(1, self.manifest.n_sensors + 1))

    def get_group_for_sensor(self, sensor_id: int) -> int:
        return (sensor_id - 1) // self.manifest.sensors_per_group

    def get_sensors_in_group(self, group: int) -> List[int]:
        start = group * self.manifest.sensors_per_group + 1
        return list(range(start, start + self.manifest.sensors_per_group))


@dataclass(frozen=True)
class ImuConfig:
    name: str
    imu_type: str
    imu_frame_id: str
    publish_scaled_imu: bool
    accel_range_g: float
    gyro_range_dps: float
    accel_odr_hz: float
    gyro_odr_hz: float
    accel_lsb_per_g: float
    gyro_lsb_per_dps: float
    temp_lsb_per_c: float
    temp_offset_c: float
    position_m: List[float]
    rotation_deg: float
    axis_transform_matrix: List[List[float]]

    def axis_transform_numpy(self) -> np.ndarray:
        return np.array(self.axis_transform_matrix, dtype=float).reshape(3, 3)


@dataclass(frozen=True)
class BoardProfile:
    name: str
    firmware_protocol: str
    default_array_config: str
    default_imu_config: str
    array_configs: Dict[str, str]
    startup_sensors: Dict[str, str]
