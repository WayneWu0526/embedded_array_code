# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Repository Overview

This is a ROS Noetic catkin workspace for the Mi-Gels magnetic sensor array
system. The current main path is continuous MagGrad collection from STM32 sensor
streams, with optional FY8300 signal-generator control and robot/TF context.

## Current Package Map

| Package | Role |
| --- | --- |
| `sensor_data_collection` | Main MagGrad collection launch/nodes, plus legacy TDM collection under `config/legacy` and `launch/legacy` |
| `serial_processor` | STM32 serial bridges: MagGrad stream, legacy TDM stream, manual-record tools |
| `sensor_array_config` | Sensor-array, IMU, and profile configs for QMC6309/AK09973D/TMAG3001 arrays |
| `calibration` | Calibration utilities and center-field estimator tools |
| `triple_arm_task` | Triple-arm scan/exploration experiments |
| `triple_arm_visual_servo` | MoveIt and visual-servo trajectory experiments |
| `gels_localization` | Legacy/reference GELS localization service and offline analysis |

## Build Commands

Build the related `zlab_robots` workspace first when `signal_generator` or robot
bringup dependencies are needed:

```bash
cd ~/zlab_robots
catkin build
source devel/setup.bash

cd ~/embedded_array_ws_Mi-Gels
catkin build
source devel/setup.bash
```

On this Mac workspace, the repository path is usually:

```bash
cd ~/Developer/embedded_array_ws_Mi-Gels
```

## Main Run Commands

Current MagGrad collection:

```bash
roslaunch sensor_data_collection maggrad_continuous_collection.launch \
  profile:=maggrad_dual_v1 \
  array_config:=qmc6309_12ch_v1 \
  imu_config:=icm42670 \
  output_dir:=$(pwd)/data
```

Start and stop recording:

```bash
rostopic pub /maggrad_continuous_collection/record_trigger std_msgs/Bool "data: true" -1
rostopic pub /maggrad_continuous_collection/record_trigger std_msgs/Bool "data: false" -1
```

Manual STM32 recording helpers:

```bash
./tools/run_stm32_manual_migels.sh
./tools/manual_record_enter.sh
```

Legacy TDM collection is kept for reference:

```bash
roslaunch sensor_data_collection data_collection.launch
roslaunch sensor_data_collection test_stm32.launch
```

These launch files resolve through `src/sensor_data_collection/launch/legacy/`.

## Data Policy

Do not add new runtime outputs under `src/`. Use one of:

- `data/` for normal local experiments and staging inside this workspace.
- NAS or another external archive for large historical datasets.

Tracked config files under `src/**/config/` are allowed. Generated CSV, JSONL,
HTML, PDF, plot outputs, and archived experiment files should stay ignored or
outside the repository.

## Important Paths

- Current workflow: `docs/mi-gels-workflow.md`
- Cleanup notes: `docs/repository-cleanup.md`
- Historical plans/specs: `docs/design-history/`
- Sensor config guide: `src/sensor_array_config/config/README.md`
- Legacy protocol docs: `src/sensor_data_collection/README.md`
- TDM communication test: `src/sensor_data_collection/docs/communication_test.md`

## Development Notes

- For quick Python checks, use `/Users/lawkaho/.venvs/codex/bin/python` unless a
  project-local virtual environment is clearly required.
- `sensor_array_config` uses a standard Python src-layout. The Python package is
  under `src/sensor_array_config/src/sensor_array_config`.
- Keep `src/CMakeLists.txt` as the catkin top-level symlink. It can appear broken
  on machines without `/opt/ros/noetic`, but is valid in the ROS Noetic runtime.
- Prefer fixing actual structure, paths, build metadata, and runtime output
  locations over adding per-directory README files.
