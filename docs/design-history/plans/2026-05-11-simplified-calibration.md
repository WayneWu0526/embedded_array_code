# Simplified Calibration Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify calibration in `serial_node_tdm.py` — remove ellipsoid correction and一致性（D_i, e_i）方法，改为仅使用 R_CORR 和归一化方法的 D_i, e_i（通过 SensorArrayConfig 加载）。

**Architecture:** 新pipeline：`b_raw → R_CORR @ b_raw → D_i @ (...) + e_i → b_corrected`。其中 D_i, e_i 来自归一化校准，通过 `SensorArrayConfig.normalized` 属性读取（数据来源为 `normalized_params.json`）。

**Tech Stack:** ROS Noetic, Python numpy, serial_processor package, sensor_array_config package

---

## 两种校准方法的本质区别

| | 一致性方法 (`calibration_fit.py`) | 归一化方法 (`calibration_fit_normalized.py`) |
|---|---|---|
| 参考目标 | 传感器间均值 `o_bar` | 归一化参考场 `b_ref_norm` |
| 训练信号 | 最小化传感器间偏差 | 最小化与归一化 b_ref 的偏差 |
| 对幅度敏感度 | 敏感（原始 b_ref 直接参与） | 钝化（b_ref 幅度归一化） |
| 产出文件 | `consistency_params.json` | `normalized_params.json` |

---

## File Structure

- **Modify:** `src/serial_processor/scripts/serial_node_tdm.py`
  - 移除 ellipsoid correction 和 amp_factor
  - D_i, e_i 改为从 `sensor_config.normalized` 加载
  - 移除 `_apply_ellipsoid_correction` 方法
  - 移除 ellipsoid-related publisher (`pub_ellip`, `pub_ellip_magnitude`)

- **Modify:** `src/sensor_array_config/sensor_array_config/base.py`
  - 新增 `NormalizedParamsSet` 数据类

- **Modify:** `src/sensor_array_config/sensor_array_config/qmc6309.py`
  - 新增 `normalized` 属性，从 `normalized_params.json` 读取

- **Modify:** `src/calibration/scripts/calibration_fit_normalized.py`
  - 输出路径改为 `normalized_params.json` 到 sensor_array_config 目录

---

## Task 1: Add NormalizedParamsSet to base.py

**Files:**
- Modify: `src/sensor_array_config/sensor_array_config/base.py`

- [ ] **Step 1: Add NormalizedParamsSet dataclass**

在 `ConsistencyParamsSet` 之后添加：

```python
# ---------- normalized params (Phase 2 variant: trained on b_ref_norm) ----------
@dataclass
class NormalizedParamsSet:
    params: Dict[int, ConsistencyParams]  # reuses D_i, e_i from ConsistencyParams

    @classmethod
    def from_json(cls, path: str) -> "NormalizedParamsSet":
        with open(path) as f:
            raw = json.load(f)
        params = {}
        if "sensors" in raw:
            for entry in raw["sensors"]:
                params[entry["sensor_id"]] = ConsistencyParams(
                    D_i=entry["D_i"],
                    e_i=entry["e_i"]
                )
        else:
            for sid, entry in raw.items():
                if sid == "amp_factor":
                    continue
                params[int(sid)] = ConsistencyParams(
                    D_i=entry["D_i"],
                    e_i=entry["e_i"]
                )
        return cls(params=params)

    def to_json(self, path: str):
        raw = {str(k): {"D_i": v.D_i, "e_i": v.e_i} for k, v in self.params.items()}
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)
```

- [ ] **Step 2: Add `normalized` abstract property to SensorArrayConfig**

在 `SensorArrayConfig` 类的 `consistency` 属性之后添加：

```python
    @property
    @abstractmethod
    def normalized(self) -> NormalizedParamsSet:
        ...
```

- [ ] **Step 3: Commit**

```bash
git add src/sensor_array_config/sensor_array_config/base.py
git commit -m "feat(sensor_array_config): add NormalizedParamsSet and normalized property"
```

---

## Task 2: Implement normalized property in QMC6309Config

**Files:**
- Modify: `src/sensor_array_config/sensor_array_config/qmc6309.py`

- [ ] **Step 1: Update import**

在 import 语句中，将 `NormalizedParamsSet` 添加到 from 列表：
```python
from .base import (
    SensorArrayConfig, SensorArrayManifest,
    IntrinsicParamsSet, ConsistencyParamsSet, NormalizedParamsSet, SensorArrayHardwareParams
)
```

- [ ] **Step 2: Add normalized property**

在 `QMC6309Config` 类 `consistency` 属性之后添加：

```python
    @property
    def normalized(self) -> NormalizedParamsSet:
        return NormalizedParamsSet.from_json(
            os.path.join(self._config_root, "normalized_params.json")
        )
```

- [ ] **Step 3: Commit**

```bash
git add src/sensor_array_config/sensor_array_config/qmc6309.py
git commit -m "feat(sensor_array_config): implement normalized property in QMC6309Config"
```

---

## Task 3: Update calibration_fit_normalized.py output path

**Files:**
- Modify: `src/calibration/scripts/calibration_fit_normalized.py`

- [ ] **Step 1: Change output path**

修改文件末尾的保存路径。输出到 sensor_array_config 目录：

```python
    cal_out = Path('/home/zhang/embedded_array_ws/src/sensor_array_config/sensor_array_config/config/qmc6309/normalized_params.json')
```

注意：JSON 的 key 需改为 `D_i` 和 `e_i`（与 `ConsistencyParams` 一致），以便 `NormalizedParamsSet.from_json` 能正确解析。当前输出是 `"D"` 和 `"e"`，需改为 `"D_i"` 和 `"e_i"`。

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile src/calibration/scripts/calibration_fit_normalized.py`

- [ ] **Step 3: Commit**

```bash
git add src/calibration/scripts/calibration_fit_normalized.py
git commit -m "feat(calibration): output normalized_params.json to sensor_array_config"
```

---

## Task 4: Update serial_node_tdm.py to use sensor_config.normalized

**Files:**
- Modify: `src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 1: Remove _load_normalized_calibration method**

Delete the `_load_normalized_calibration` method entirely (was added in the previous worktree session).

- [ ] **Step 2: Replace with config-based loading in __init__**

在 `_load_sensor_array_params()` 调用之后，将：

```python
        # Load normalized calibration D_i, e_i
        self._load_normalized_calibration()
```

替换为：

```python
        # Load normalized calibration D_i, e_i from sensor config
        norm = self._sensor_config.normalized
        self.D_matrix = {}
        self.e_bias = {}
        for sid, params in norm.params.items():
            self.D_matrix[sid] = np.array(params.D_i)
            self.e_bias[sid] = np.array(params.e_i)
        rospy.loginfo(f"Loaded normalized calibration for {len(self.D_matrix)} sensors from config")
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -m py_compile src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 4: Commit**

```bash
git add src/serial_processor/scripts/serial_node_tdm.py
git commit -m "refactor: load normalized D_i,e_i from sensor_config.normalized"
```

---

## Task 5: Verify Final Calibration Flow

**Files:**
- Verify: `src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 1: Verify calibration pipeline**

确认校准循环为：
```
b_raw → R_CORR → D_i,e_i (from sensor_config.normalized) → b_corrected
```

代码应为：
```python
                            corrected_sensors = []
                            for s in msg.sensor_data:
                                cx, cy, cz = s.x, s.y, s.z
                                if s.id in self.R_CORR:
                                    b_rot = self.R_CORR[s.id] @ np.array([cx, cy, cz])
                                    cx, cy, cz = b_rot[0], b_rot[1], b_rot[2]
                                if s.id in self.D_matrix and s.id in self.e_bias:
                                    b_cons = self.D_matrix[s.id] @ np.array([cx, cy, cz]) + self.e_bias[s.id]
                                    cx, cy, cz = b_cons[0], b_cons[1], b_cons[2]
                                corrected_sensors.append(SensorData(id=s.id, x=cx, y=cy, z=cz))
```

- [ ] **Step 2: Verify no ellipsoid/amp_factor code**

确认不存在：
- `_apply_ellipsoid_correction`
- `pub_ellip`, `pub_ellip_magnitude`
- `_amp_factor`

- [ ] **Step 3: Syntax check all files**

Run: `python3 -m py_compile src/serial_processor/scripts/serial_node_tdm.py`
Run: `python3 -m py_compile src/sensor_array_config/sensor_array_config/base.py`
Run: `python3 -m py_compile src/sensor_array_config/sensor_array_config/qmc6309.py`
Run: `python3 -m py_compile src/calibration/scripts/calibration_fit_normalized.py`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: complete simplified calibration via SensorArrayConfig.normalized"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `base.py` | 新增 `NormalizedParamsSet` 数据类和 `normalized` 抽象属性 |
| `qmc6309.py` | 实现 `normalized` 属性，从 `normalized_params.json` 读取 |
| `calibration_fit_normalized.py` | 输出改为 `normalized_params.json` 到 sensor_array_config 目录；key 改为 `D_i`/`e_i` |
| `serial_node_tdm.py` | D_i, e_i 从 `sensor_config.normalized` 加载；移除 ellipsoid/amp_factor 代码 |

**New pipeline:**
```
b_raw → R_CORR[s] @ b_raw → D_i @ (R_CORR @ b_raw) + e_i → b_corrected
```
