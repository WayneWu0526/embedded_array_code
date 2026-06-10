# embedded_array_ws_Mi-Gels

ROS Noetic catkin workspace for Mi-Gels magnetic sensor array data collection.

## Current Main Path

The current Mi-Gels main path is continuous MagGrad collection with STM32
MagGrad streaming and optional FY8300 signal control:

```bash
cd ~/zlab_robots
catkin build
source devel/setup.bash

cd ~/Developer/embedded_array_ws_Mi-Gels
catkin build
source devel/setup.bash

roslaunch sensor_data_collection maggrad_continuous_collection.launch \
  hardware_config:=tmag3001 \
  output_dir:=$(pwd)/data
```

Start and stop recording:

```bash
rostopic pub /maggrad_continuous_collection/record_trigger std_msgs/Bool "data: true" -1
rostopic pub /maggrad_continuous_collection/record_trigger std_msgs/Bool "data: false" -1
```

## Workspace Map

| Path | Role |
| --- | --- |
| `src/sensor_data_collection/` | Main collection package: MagGrad collection, legacy TDM collection, launch/config files |
| `src/serial_processor/` | STM32 serial bridge and sensor stream publishing |
| `src/sensor_array_config/` | Board-level magnetic sensor and IMU hardware configs |
| `src/calibration/` | Calibration and center-field estimation utilities |
| `src/triple_arm_task/` | Triple-arm scan/exploration experiments |
| `src/triple_arm_visual_servo/` | MoveIt/visual-servo experiments |
| `src/gels_localization/` | Legacy/reference GELS localization code |
| `docs/` | Workflow notes, cleanup notes, design plans |
| `tools/` | Small wrapper scripts for repeatable launch/control flows |
| `data/` | Local-only data staging; large experiment outputs should stay outside git |

## Manual Calibration Helper

On zlab, when multiple worktrees contain a package named `sensor_data_collection`,
prefer the wrapper script so the Mi-Gels workspace is selected explicitly:

```bash
cd ~/embedded_array_ws_Mi-Gels
./tools/run_stm32_manual_migels.sh
```

For manual calibration recording, use an Enter-only controller in another
terminal after launching `stm32_manual.launch`:

```bash
cd ~/embedded_array_ws_Mi-Gels
./tools/manual_record_enter.sh
```

## Data Policy

Do not write new experiment outputs under `src/`. Use the workspace-local
`data/` directory by default, or NAS/external storage for large archives.
Source/config JSON files remain trackable, but runtime CSV/JSONL/HTML/PDF
outputs are ignored.

## More Details

- [Mi-Gels workflow](docs/mi-gels-workflow.md)
- [Documentation index](docs/README.md)
- [Backlog](docs/backlog.md)
- [Design history](docs/design-history/README.md)
- [Tooling notes](tools/README.md)
- [Repository cleanup notes](docs/repository-cleanup.md)
- [Legacy sensor data collection protocol](src/sensor_data_collection/README.md)
