# 仓库整理建议

当前仓库同时包含源码、配置、历史采集结果、分析报告和旧 GELS/TDM 流程。为了让 Mi-Gels 后续维护更清楚，建议按下面方式分层。

## 1. 建议保留为主线的内容

| 路径 | 处理建议 | 原因 |
| --- | --- | --- |
| `src/serial_processor/scripts/serial_node_maggrad.py` | 主线保留 | Mi-Gels STM32 磁传感器/IMU 数据入口 |
| `src/sensor_data_collection/scripts/maggrad_continuous_collection_node.py` | 主线保留 | Mi-Gels 连续采集入口 |
| `src/sensor_data_collection/launch/maggrad_continuous_collection.launch` | 主线保留 | 推荐 launch |
| `src/sensor_data_collection/config/*maggrad*` | 主线保留 | 连续采集和 FY8300 时序配置 |
| `src/sensor_array_config/` | 主线保留 | QMC6309/AK09973D 阵列参数和 affine 标定 |
| `src/calibration/` | 主线保留 | Mi-Gels 标定和中心场估计 |
| `src/triple_arm_visual_servo/` | 按需保留 | 机械臂/视觉伺服实验仍可能需要 |

## 2. 建议标记为 legacy 的内容

| 路径 | 处理建议 | 说明 |
| --- | --- | --- |
| `src/gels_localization/` | 保留但标记 legacy | 旧 GELS 定位服务和离线算法 |
| `src/sensor_data_collection/scripts/data_collection_node.py` | 保留但标记 legacy | 旧 TDM/cycle JSON 采集节点 |
| `src/sensor_data_collection/launch/legacy/data_collection.launch` | legacy | 旧在线 GELS 流程入口 |
| `src/serial_processor/scripts/serial_node_tdm.py` | 保留但标记 legacy | 旧 PC-STM32 TDM 协议 |
| `src/sensor_data_collection/config/legacy/params_cvt.yaml` | legacy | 旧 CVT 参数 |
| `src/sensor_data_collection/config/legacy/params_cci.yaml` | legacy | 旧 CCI 参数 |
| `src/sensor_data_collection/config/legacy/signal_params_cvt.yaml` | legacy | 旧 CVT 信号配置 |
| `src/sensor_data_collection/config/legacy/signal_params_cci.yaml` | legacy | 旧 CCI 信号配置 |
| `src/sensor_data_collection/config/legacy/task_params.yaml` | legacy | 旧 cycle 采集参数 |

可选整理方式：

```text
src/legacy_gels/
  gels_localization/
  sensor_data_collection_legacy/
  serial_node_tdm.py
```

但这会影响 ROS 包名、launch 中的 `$(find ...)` 和 package 依赖，需要单独做一次可编译迁移。短期更稳妥的做法是保留路径，只在文档和 README 中明确默认流程。

## 3. 历史结果和输出

以下路径曾经是历史采集/分析结果，不应继续作为源码树的一部分：

| 路径 | 当前作用 | 建议 |
| --- | --- | --- |
| 路径 | 原作用 | 当前处理 |
| --- | --- | --- |
| `src/sensor_data_collection/data/` | MagGrad CSV、manual CSV、分析报告 | 从源码树移出；新结果写到 `data/` 或外部归档 |
| `src/sensor_data_collection/data_backup/` | 手动标定备份数据 | 从源码树移出 |
| `src/sensor_data_collection/result_2/` | 旧 cycle JSON 结果 | 从源码树移出 |
| `src/sensor_data_collection/result_5/` | 旧 cycle JSON 结果 | 从源码树移出 |
| `src/sensor_data_collection/result_6_old_config/` | 旧配置结果 | 从源码树移出 |
| `src/sensor_data_collection/result_merged_all/` | 合并后的旧结果 | 从源码树移出 |
| `src/sensor_data_collection/plot/` | 绘图脚本和输出图 | 保留脚本；生成图迁到 `data/archive/sensor_data_collection/plot/` |

仓库已经更新 `.gitignore`，后续新产生的 CSV、JSONL、HTML、PDF、`data/` 和 `result*/` 默认不会再被 Git 跟踪。注意：已经被 Git 跟踪的历史文件不会因为 `.gitignore` 自动消失。

## 4. 推荐清理步骤

### 安全清理

```bash
find src -path '*/__pycache__/*' -delete
find src -type d -name '__pycache__' -empty -delete
```

这只删除 Python 缓存，不影响源码和实验数据。

### 将新实验输出放到工作区 data/

```bash
mkdir -p data
roslaunch sensor_data_collection maggrad_continuous_collection.launch output_dir:=$(pwd)/data
```

### 如果需要在其他机器复现旧数据清理

先归档仍存在的历史目录：

```bash
mkdir -p ~/mi_gels_archive
rsync -a src/sensor_data_collection/data/ ~/mi_gels_archive/data/
rsync -a src/sensor_data_collection/data_backup/ ~/mi_gels_archive/data_backup/
rsync -a src/sensor_data_collection/result_*/ ~/mi_gels_archive/results/
```

再从 Git 索引移除但保留本地文件：

```bash
git rm -r --cached src/sensor_data_collection/data \
  src/sensor_data_collection/data_backup \
  src/sensor_data_collection/result_2 \
  src/sensor_data_collection/result_5 \
  src/sensor_data_collection/result_6_old_config \
  src/sensor_data_collection/result_merged_all
```

这一步会改变仓库历史后的工作区呈现，建议单独开分支并确认归档完整后再做。

## 5. 建议的目标结构

```text
embedded_array_ws_Mi-Gels/
  README.md
  docs/
    README.md
    backlog.md
    design-history/
    mi-gels-workflow.md
    repository-cleanup.md
  src/
    serial_processor/
    sensor_data_collection/
    sensor_array_config/
    calibration/
    triple_arm_visual_servo/
    gels_localization/        # legacy/reference
```

实验输出不再放在 `src/` 下，默认放到工作区根目录 `data/`，大型归档再放到 NAS 或外部数据仓库。
