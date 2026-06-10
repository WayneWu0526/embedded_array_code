# Mi-Gels 工作流程

本文档描述当前仓库中更适合 Mi-Gels 的主流程。旧 GELS/TDM 的 `launch/legacy/data_collection.launch`、`serial_node_tdm.py` 和 `gels_localization` 仍可作为历史算法和离线处理参考，但不作为 Mi-Gels 连续采集的默认入口。

## 1. 推荐使用的 ROS 包

| 包 | Mi-Gels 中的定位 |
| --- | --- |
| `serial_processor` | STM32 二进制流解析；`serial_node_maggrad.py` 发布磁传感器、IMU、原始/标定后 topic |
| `sensor_data_collection` | 连续采集；`maggrad_continuous_collection_node.py` 将磁场、IMU、线圈状态、TF 同步写入 CSV/JSONL |
| `sensor_array_config` | 板级 hardware config；当前包含 QMC6309、AK09973D、TMAG3001 三种磁传感器板 |
| `calibration` | Mi-Gels 相关标定与中心场估计脚本 |
| `triple_arm_visual_servo` | 需要机械臂轨迹/视觉伺服时使用 |
| `gels_localization` | 旧 GELS 定位服务和离线算法参考，非默认在线入口 |

## 2. 推荐启动流程

### 环境准备

```bash
cd ~/zlab_robots
catkin build
source devel/setup.bash

cd ~/Developer/embedded_array_ws_Mi-Gels
catkin build
source devel/setup.bash
```

`signal_generator` 和 `zlab_robots_calibration` 来自 `zlab_robots`，所以要先 source 该 workspace。

### 启动连续采集

```bash
roslaunch sensor_data_collection maggrad_continuous_collection.launch
```

常用参数：

```bash
roslaunch sensor_data_collection maggrad_continuous_collection.launch \
  hardware_config:=tmag3001 \
  port:=auto \
  baudrate:=115200 \
  output_dir:=$(pwd)/data \
  record_tf:=true
```

如果需要看相机图像：

```bash
roslaunch sensor_data_collection maggrad_continuous_collection.launch show_image:=true
```

开始/停止记录：

```bash
rostopic pub /maggrad_continuous_collection/record_trigger std_msgs/Bool "data: true" -1
rostopic pub /maggrad_continuous_collection/record_trigger std_msgs/Bool "data: false" -1
```

输出文件默认为：

```text
data/maggrad_continuous_*.csv
```

默认写到工作区根目录 `data/`，避免把实验结果混进源码目录。

## 3. Topic 和数据含义

`serial_node_maggrad.py` 发布：

| Topic | 含义 |
| --- | --- |
| `stm_uplink_raw` | 原始磁传感器数据，已应用 `R_CORR` 方向统一，未应用 affine 标定 |
| `stm_uplink` | 应用 affine 标定后的磁传感器数据 |
| `stm_magnitude_raw` | 原始磁场模长 |
| `stm_magnitude` | 标定后磁场模长 |
| `maggrad/imu_raw` | ICM42670 原始 IMU |
| `maggrad/imu` | 单位换算后的 ROS `sensor_msgs/Imu` |

`maggrad_continuous_collection_node.py` 默认订阅 `stm_uplink_raw`。如果要记录 affine 标定后的数据，在 `src/sensor_data_collection/config/maggrad_continuous_collection.yaml` 中设置：

```yaml
use_calibrated: true
```

连续采集 CSV 包含：

- PC 时间、STM32 序号和 tick；
- 当前线圈状态标签；
- 12 个传感器三轴磁场；
- 原始/换算后 IMU；
- `sensor_array_filt`、`diana7_em_tcp_filt`、`arm1_em_tcp_filt`、`arm2_em_tcp_filt` 相对 `lab_table` 的 TF。

## 4. 线圈时序

Mi-Gels 当前使用 FY8300 三通道连续输出，采集节点只按时间给每一帧磁场打标签，不在线切换 FY8300。

配置文件：

- `src/sensor_data_collection/config/signal_params_maggrad_continuous.yaml`
- `src/sensor_data_collection/config/maggrad_continuous_collection.yaml`

默认一周期 1 s：

| 状态 | 相位 | 占空比 |
| --- | --- | --- |
| `coil_1_on` | 0 deg | 25% |
| `coil_2_on` | 90 deg | 25% |
| `coil_3_on` | 180 deg | 25% |
| `all_off` | 剩余 25% | 背景 |

后处理时应按 `coil_state`/`coil_channel` 分组，先做背景段和通道段质量检查，再进入定位或模型拟合。

## 5. 建议的实验数据流程

1. 采集前确认 `zlab_robots_calibration` 能发布 `lab_table` 和需要的 `_filt` TF。
2. 启动 `maggrad_continuous_collection.launch`，确认 `serial_node_maggrad` 已识别正确 `hardware_config`。
3. 先短录 5-10 s，检查 CSV 中 `coil_state` 是否周期正确，TF 列是否非空。
4. 正式录制时保持 `output_dir` 指向工作区根目录 `data/`，大型归档再手动转移到 NAS 或外部数据仓库。
5. 分析时保留原始 CSV，只把分析脚本、配置和小型摘要报告提交到 Git。
6. 对外共享大文件时使用归档目录、NAS、DVC、Git LFS 或发布压缩包，不再直接提交到源码仓库。

## 6. 旧 GELS/TDM 流程的保留边界

旧流程入口：

- `src/sensor_data_collection/launch/legacy/data_collection.launch`
- `src/sensor_data_collection/scripts/data_collection_node.py`
- `src/serial_processor/scripts/serial_node_tdm.py`
- `src/gels_localization/launch/localization_service.launch`

这些文件适合保留为：

- 已有 cycle JSON 数据的复现入口；
- GELS 定位服务的算法参考；
- TDM slot-pose 映射的历史实现。

如果后续只维护 Mi-Gels，可以将旧流程移动到 `legacy/` 或独立分支，但不建议在没有备份历史结果的情况下直接删除。
