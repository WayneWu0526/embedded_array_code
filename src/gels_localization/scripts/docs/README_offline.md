# Offline Processing

## offline_process_cycle.py

离线处理 cycle JSON 文件，运行 MaPS 定位算法。

## 使用方法

```bash
# 进入 scripts 目录
cd src/gels_localization/scripts/

# 处理目录下的所有 cycle JSON 文件
python offline_process_cycle.py /path/to/result/

# 直接指定单个文件
python offline_process_cycle.py /path/to/cycle_0000.json
```

## 输出说明

每个 cycle 输出：
- **Position**: 估计位置 (x, y, z) 单位: m
- **Orientation**: 四元数 (x, y, z, w)
- **Position Error**: 位置误差，单位: m
- **Orientation Error**: 姿态误差，单位: rad