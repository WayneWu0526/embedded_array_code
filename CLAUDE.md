# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a ROS Noetic catkin workspace (`embedded_array_ws`) for a **sensor array data collection system**. The system collects magnetic field sensor data synchronized with robot arm poses for EKF-based pose estimation.

### Key Packages

| Package | Purpose |
|---------|---------|
| `sensor_data_collection` | Main data collection node - collects Hall sensor data, TF poses, and signal generator status |
| `serial_processor` | Serial communication bridge with STM32 development board; publishes Hall sensor data via topic and service |
| `triple_arm_task` | Triple arm exploration/calibration task |
| `triple_arm_visual_servo` | Visual servo control for robotic arms |

### Hardware Setup

- **ZED2i camera** → USB → PC (visual localization via AprilTag/TagSLAM)
- **Sensor array** → STM32 → USB → PC (Hall sensor data at 921600 baud)
- **FY8300 signal generator** → USB → PC (control) + TTL → STM32 (trigger)

## Build Commands

```bash
# Build zlab_robots first (contains signal_generator dependency)
cd ~/zlab_robots && catkin build

# Then build embedded_array_ws
cd /home/zhang/embedded_array_ws
catkin build
source devel/setup.bash
```

**Note**: `embedded_array_ws` depends on `signal_generator` from `zlab_robots`. Ensure `zlab_robots/devel/setup.bash` is sourced before building.

## Run Commands

```bash
# Data collection (main system)
roslaunch sensor_data_collection data_collection.launch

# For calibration/localization (separate zlab_robots workspace at ~/zlab_robots)
roslaunch zlab_robots_calibration localization_tagslam.launch
```

## Architecture

### Data Collection Flow (sensor_data_collection)

1. **Main thread**: ROS callbacks for Hall (~75fps via topic), visual/~10fps TF timer
2. **Background save thread**: Daemon thread periodically flushes thread-safe `DataBuffer` to temp JSON files every `flush_interval` seconds
3. **On shutdown**: Merges temp JSON files into final `sensor_data_{timestamp}.json`

Key classes in `data_collection_node.py`:
- `DataBuffer` - thread-safe deque with 3 channels (hall_data, visual_data, signal_data)
- `BackgroundSaver` - daemon thread for non-blocking file I/O
- `DataCollector` - main ROS node

### TF Frame Hierarchy (reference: lab_table)

| Frame | Description |
|-------|-------------|
| `diana7_em_tcp_filt` | Diana7 robot end-effector pose (filtered) |
| `arm1_em_tcp_filt` | Arm1 end-effector pose (filtered) |
| `arm2_em_tcp_filt` | Arm2 end-effector pose (filtered) |
| `sensor_array_filt` | Sensor array pose (filtered, ground truth) |
| `cam0` | ZED2i camera frame |

TF lookup is done at 10fps via `rospy.Timer(rospy.Duration(0.1), ...)`.

### PC-STM32 Binary Protocol

Downlink command (8 bytes): Header(0xAA55, 2B) + Version(1B) + Mode(1B) + Bitmap(2B) + SettlingTime(2B)
- Mode: 0x01=CVT (4 slots), 0x02=CCI (3 slots)
- SettlingTime: 0.01ms units, range 0~655.35ms

Uplink data (variable): Header + Version + cycle_id + slot + Bitmap + Timestamp(8B) + sensor_data(N×13B) + cycle_end

### Slot-Pose Mapping (TDM - Time Division Multiplexing)

Each cycle has multiple slots, where each slot corresponds to a different EM coil activation:

| slot | CVT mode (4 slots) | CCI mode (3 slots) |
|------|-------------------|-------------------|
| 0 | diana7_em_tcp_filt pose | diana7_em_tcp_filt pose |
| 1 | arm1_em_tcp_filt pose | arm1_em_tcp_filt pose |
| 2 | arm2_em_tcp_filt pose | arm2_em_tcp_filt pose |
| 3 | sensor_data only | — |
| cycle_end | sensor_array_filt pose (ground truth) | sensor_array_filt pose (ground truth) |

PC stores each completed cycle as JSON: `output_dir/cycle_{cycle_id:04d}.json`

### Related Workspace

**zlab_robots** (`~/zlab_robots`) is a separate workspace containing:
- `zlab_robots_calibration` - TagSLAM + AprilTag localization, filter nodes, frame reprojector
- `signal_generator` - FY8300 signal generator driver (formerly in `instrument_ws`)
- Runs independently; publishes TF for `diana7_em_tcp`, `arm1_em_tcp`, `arm2_em_tcp`, `sensor_array` (unfiltered) which `sensor_data_collection` subscribes to

## Key Files

- `src/sensor_data_collection/README.md` - Full protocol specification and system documentation
- `src/sensor_data_collection/scripts/data_collection_node.py` - Main collection node
- `src/serial_processor/scripts/serial_node.py` - STM32 serial bridge
- `src/sensor_data_collection/config/task_params.yaml` - Collection parameters (period, cycles, flush_interval)
- `src/sensor_data_collection/launch/data_collection.launch` - Launch file

## Configuration Parameters (task_params.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| period | 1.0s | Duration of each cycle |
| num_cycles | 10 | Number of cycles to collect |
| output_dir | ~/sensor_data | JSON output directory |
| flush_interval | 2.0s | Background save interval |
| start_with_signal | true | Wait for FY8300 trigger before collecting |
