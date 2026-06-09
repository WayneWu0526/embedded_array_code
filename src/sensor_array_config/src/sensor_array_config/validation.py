import numpy as np

from .schemas import ArrayConfig, BoardProfile, ImuConfig, SensorChipConfig


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def validate_sensor_chip_config(config: SensorChipConfig):
    _require(bool(config.name), "sensor type name is required")
    _require(config.bit_width > 0, f"{config.name}: bit_width must be positive")
    _require(config.adu_to_gs > 0.0, f"{config.name}: adu_to_gs must be positive")
    _require(config.gs_to_tesla > 0.0, f"{config.name}: gs_to_tesla must be positive")


def validate_array_config(config: ArrayConfig):
    n_sensors = config.manifest.n_sensors
    sensor_ids = set(range(1, n_sensors + 1))
    _require(config.manifest.name == config.name, f"{config.name}: manifest name mismatch")
    _require(config.manifest.sensor_type == config.sensor_type.name, f"{config.name}: sensor type mismatch")
    _require(len(config.hardware.d_list) == n_sensors, f"{config.name}: n_sensors != len(d_list)")

    r_corr_ids = set()
    for entry in config.hardware.R_CORR:
        _require(len(entry.matrix) == 9, f"{config.name}: R_CORR matrix must have 9 values")
        r_corr_ids.update(int(sid) for sid in entry.sensor_ids)
    _require(r_corr_ids == sensor_ids, f"{config.name}: R_CORR must cover all sensors")

    affine_ids = set(config.affine_model.params)
    _require(affine_ids == sensor_ids, f"{config.name}: affine params must cover all sensors")
    for sid, params in config.affine_model.params.items():
        _require(np.array(params.D_i).shape == (3, 3), f"{config.name}: sensor {sid} D_i must be 3x3")
        _require(np.array(params.e_i).reshape(-1).shape == (3,), f"{config.name}: sensor {sid} e_i must have length 3")


def validate_imu_config(config: ImuConfig):
    _require(bool(config.name), "IMU config name is required")
    _require(np.array(config.axis_transform_matrix).shape == (3, 3), f"{config.name}: axis_transform_matrix must be 3x3")
    _require(len(config.position_m) == 3, f"{config.name}: position_m must have length 3")
    _require(config.accel_lsb_per_g > 0.0, f"{config.name}: accel_lsb_per_g must be positive")
    _require(config.gyro_lsb_per_dps > 0.0, f"{config.name}: gyro_lsb_per_dps must be positive")


def validate_profile(profile: BoardProfile):
    _require(bool(profile.name), "profile name is required")
    _require(bool(profile.default_array_config), f"{profile.name}: default_array_config is required")
    _require(bool(profile.default_imu_config), f"{profile.name}: default_imu_config is required")
