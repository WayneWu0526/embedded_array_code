# Sensor Array Config

This directory contains hardware configuration bundles loaded by `sensor_array_config`.

- `sensor_types/`: chip-level magnetic sensor parameters.
- `arrays/`: concrete sensor-array geometry, R_CORR, and affine calibration.
- `imu_types/`: IMU scale, frame, pose, and axis transform parameters.
- `profiles/`: board-level combinations of arrays, IMU, firmware protocol, and startup sensor commands.

Use `get_array_config(name)`, `get_imu_config(name)`, and `get_profile(name)`.
