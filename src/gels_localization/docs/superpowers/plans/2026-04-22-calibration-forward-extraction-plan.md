# Calibration Forward 模块抽取实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `serial_node_tdm.py` 中的内联校正逻辑抽取为 Strategy 模式的可插拔模块，置于 `src/calibration/lib/forward/`，TDM 代码零改动即可替换/禁用任意校正步骤。

**Architecture:** Strategy 模式。`CalibrationProcessor` 作为 facade 持有 `CorrectionStep` 列表，按序调用。三个 step：ellipsoid、R_CORR、consistency，各自独立，可按需替换/禁用。

**Tech Stack:** Python, numpy, ROS (catkin), sensor_array_config

---

## 文件结构

```
src/calibration/lib/forward/
    __init__.py          # public exports: CalibrationProcessor, EllipsoidCorrection,
                         # RCorrRotation, ConsistencyCorrection, CorrectionStep
    base.py              # CorrectionStep ABC
    processor.py         # CalibrationProcessor
    ellipsoid.py         # EllipsoidCorrection step
    r_corr.py            # RCorrRotation step
    consistency.py       # ConsistencyCorrection step

src/serial_processor/scripts/serial_node_tdm.py
    # 删除内联 _load_calibration_params, _load_sensor_array_params,
    # _apply_ellipsoid_correction 及相关属性，替换为 CalibrationProcessor
```

---

## Task 1: 创建 forward 模块目录和 __init__.py

**Files:**
- Create: `src/calibration/lib/forward/__init__.py`
- Verify: `src/calibration/lib/__init__.py` 已存在（不需要改）

- [ ] **Step 1: 创建目录**

```bash
mkdir -p /home/zhang/embedded_array_ws/src/calibration/lib/forward
```

- [ ] **Step 2: 创建 __init__.py**

```python
"""
calibration.lib.forward - 正向校正模块

提供 Strategy 模式的可插拔校正步骤，可独立使用或组合。
"""
from .base import CorrectionStep
from .processor import CalibrationProcessor
from .ellipsoid import EllipsoidCorrection
from .r_corr import RCorrRotation
from .consistency import ConsistencyCorrection

__all__ = [
    "CorrectionStep",
    "CalibrationProcessor",
    "EllipsoidCorrection",
    "RCorrRotation",
    "ConsistencyCorrection",
]
```

- [ ] **Step 3: 提交**

```bash
git add src/calibration/lib/forward/
git commit -m "feat(calibration): create forward module directory and __init__.py"
```

---

## Task 2: 创建 base.py — CorrectionStep 抽象基类

**Files:**
- Create: `src/calibration/lib/forward/base.py`

- [ ] **Step 1: 编写 base.py**

```python
"""校正步骤抽象基类"""
from abc import ABC, abstractmethod
from typing import Tuple


class CorrectionStep(ABC):
    """Strategy 模式抽象基类：单一校正步骤"""

    @property
    @abstractmethod
    def name(self) -> str:
        """步骤名称，用于日志和 introspection"""
        pass

    @abstractmethod
    def apply(
        self, x: float, y: float, z: float, sensor_id: int
    ) -> Tuple[float, float, float]:
        """
        对单颗传感器的单次采样进行校正。

        Args:
            x, y, z: 校正前的磁场分量 (Gauss)
            sensor_id: 传感器编号 (1-12)

        Returns:
            (cx, cy, cz): 校正后的磁场分量
        """
        pass

    def reset(self):
        """可选：重置内部状态（供 processor 调用）"""
        pass
```

- [ ] **Step 2: 提交**

```bash
git add src/calibration/lib/forward/base.py
git commit -m "feat(calibration): add CorrectionStep abstract base class"
```

---

## Task 3: 创建 processor.py — CalibrationProcessor facade

**Files:**
- Create: `src/calibration/lib/forward/processor.py`

- [ ] **Step 1: 编写 processor.py**

```python
"""校正处理门面类"""
from typing import List, Optional, Tuple

from .base import CorrectionStep


class CalibrationProcessor:
    """
    门面类：持有校正步骤列表，按序调用。

    Args:
        sensor_config: 传感器配置对象（提供 intrinsic/hardware/consistency 参数）
        steps: 校正步骤列表，默认为 [Ellipsoid, RCorr, Consistency]。
               传入空列表则不做任何校正（直通）。
    """

    def __init__(
        self,
        sensor_config: "SensorArrayConfig",
        steps: Optional[List[CorrectionStep]] = None,
    ):
        self._config = sensor_config
        self._steps = steps or self._default_steps(sensor_config)

    def _default_steps(self, config):
        from .ellipsoid import EllipsoidCorrection
        from .r_corr import RCorrRotation
        from .consistency import ConsistencyCorrection

        return [
            EllipsoidCorrection(config),
            RCorrRotation(config),
            ConsistencyCorrection(config),
        ]

    def apply(
        self, sensor_id: int, x: float, y: float, z: float
    ) -> Tuple[float, float, float]:
        """
        依次通过所有 step，返回最终校正结果。

        Args:
            sensor_id: 传感器编号
            x, y, z: 原始磁场分量

        Returns:
            (cx, cy, cz): 最终校正后的磁场分量
        """
        cx, cy, cz = x, y, z
        for step in self._steps:
            cx, cy, cz = step.apply(cx, cy, cz, sensor_id)
        return cx, cy, cz

    @property
    def steps(self) -> List[CorrectionStep]:
        """返回当前步骤列表（用于 introspection）"""
        return list(self._steps)
```

- [ ] **Step 2: 提交**

```bash
git add src/calibration/lib/forward/processor.py
git commit -m "feat(calibration): add CalibrationProcessor facade"
```

---

## Task 4: 创建 ellipsoid.py — EllipsoidCorrection step

**Files:**
- Create: `src/calibration/lib/forward/ellipsoid.py`

- [ ] **Step 1: 编写 ellipsoid.py**

```python
"""椭球校正步骤"""
import numpy as np
from typing import Tuple

from .base import CorrectionStep


class EllipsoidCorrection(CorrectionStep):
    """
    Phase 1 椭球校正。

    公式: b_corr = (b_raw - o_i) @ C_i.T

    参数来源: sensor_config.intrinsic
    """

    def __init__(self, sensor_config: "SensorArrayConfig"):
        self._offset = {}
        self._correction = {}

        try:
            intrinsic = sensor_config.intrinsic
            if intrinsic and intrinsic.params:
                for sid, params in intrinsic.params.items():
                    self._offset[sid] = np.array(params.o_i)
                    self._correction[sid] = np.array(params.C_i)
        except Exception:
            pass

        # fallback: 无参数时用单位阵
        if not self._offset:
            n = sensor_config.manifest.n_sensors
            for sid in range(1, n + 1):
                self._offset[sid] = np.zeros(3)
                self._correction[sid] = np.eye(3)

    @property
    def name(self) -> str:
        return "ellipsoid"

    def apply(
        self, x: float, y: float, z: float, sensor_id: int
    ) -> Tuple[float, float, float]:
        o_i = self._offset.get(sensor_id, np.zeros(3))
        C_i = self._correction.get(sensor_id, np.eye(3))
        b = np.array([x, y, z])
        b_corr = (b - o_i) @ C_i.T
        return b_corr[0], b_corr[1], b_corr[2]
```

- [ ] **Step 2: 提交**

```bash
git add src/calibration/lib/forward/ellipsoid.py
git commit -m "feat(calibration): add EllipsoidCorrection step"
```

---

## Task 5: 创建 r_corr.py — RCorrRotation step

**Files:**
- Create: `src/calibration/lib/forward/r_corr.py`

- [ ] **Step 1: 编写 r_corr.py**

```python
"""R_CORR 旋转变换步骤"""
import numpy as np
from typing import Tuple

from .base import CorrectionStep


class RCorrRotation(CorrectionStep):
    """
    R_CORR 旋转变换：将传感器局部坐标系转换到参考坐标系。

    公式: b_rot = R_CORR @ b

    参数来源: sensor_config.hardware.R_CORR
    """

    def __init__(self, sensor_config: "SensorArrayConfig"):
        self._r_corr = {}

        try:
            hw = sensor_config.hardware
            for entry in hw.R_CORR:
                mat = np.array(entry.matrix).reshape(3, 3, order='F')
                for sid in entry.sensor_ids:
                    self._r_corr[sid] = mat
        except Exception:
            pass

        # fallback: 无参数时直通
        if not self._r_corr:
            n = sensor_config.manifest.n_sensors
            for sid in range(1, n + 1):
                self._r_corr[sid] = np.eye(3)

    @property
    def name(self) -> str:
        return "r_corr"

    def apply(
        self, x: float, y: float, z: float, sensor_id: int
    ) -> Tuple[float, float, float]:
        R = self._r_corr.get(sensor_id)
        if R is None:
            return x, y, z
        b = np.array([x, y, z])
        return tuple((R @ b).tolist())
```

- [ ] **Step 2: 提交**

```bash
git add src/calibration/lib/forward/r_corr.py
git commit -m "feat(calibration): add RCorrRotation step"
```

---

## Task 6: 创建 consistency.py — ConsistencyCorrection step

**Files:**
- Create: `src/calibration/lib/forward/consistency.py`

- [ ] **Step 1: 编写 consistency.py**

```python
"""一致性校正步骤"""
import numpy as np
from typing import Tuple

from .base import CorrectionStep


class ConsistencyCorrection(CorrectionStep):
    """
    Phase 2 一致性校正。

    公式: b_final = D_i @ b + e_i, 再除以 amp_factor (方案B)

    参数来源: sensor_config.consistency
    """

    def __init__(self, sensor_config: "SensorArrayConfig"):
        self._D = {}
        self._e = {}

        try:
            consistency = sensor_config.consistency
            if consistency and consistency.params:
                for sid, params in consistency.params.items():
                    self._D[sid] = np.array(params.D_i)
                    self._e[sid] = np.array(params.e_i)
                self._amp_factor = consistency.amp_factor
            else:
                self._amp_factor = None
        except Exception:
            self._amp_factor = None

        # fallback: 无参数时用单位阵
        if not self._D:
            n = sensor_config.manifest.n_sensors
            for sid in range(1, n + 1):
                self._D[sid] = np.eye(3)
                self._e[sid] = np.zeros(3)
            self._amp_factor = 1.0

        if self._amp_factor is None:
            self._amp_factor = 1.0

    @property
    def name(self) -> str:
        return "consistency"

    def apply(
        self, x: float, y: float, z: float, sensor_id: int
    ) -> Tuple[float, float, float]:
        D = self._D.get(sensor_id, np.eye(3))
        e = self._e.get(sensor_id, np.zeros(3))
        b = np.array([x, y, z])
        b_cons = D @ b + e
        if self._amp_factor is not None and self._amp_factor != 0:
            b_cons = b_cons / self._amp_factor
        return tuple(b_cons.tolist())
```

- [ ] **Step 2: 提交**

```bash
git add src/calibration/lib/forward/consistency.py
git commit -m "feat(calibration): add ConsistencyCorrection step"
```

---

## Task 7: 修改 serial_node_tdm.py — 替换为 CalibrationProcessor

**Files:**
- Modify: `src/serial_processor/scripts/serial_node_tdm.py`

需要修改的具体位置：

1. **删除的方法**（约 line 100-181）：
   - `_load_calibration_params` 整个方法
   - `_load_sensor_array_params` 整个方法
   - `_apply_ellipsoid_correction` 整个方法

2. **删除的属性**（在 `__init__` 中）：
   - `self.offset`, `self.correction`, `self.D_matrix`, `self.e_bias`, `self._amp_factor`
   - `self.R_CORR`, `self._d_list`, `self._n_sensors`, `self._n_groups`, `self._sensors_per_group`, `self._sensor_to_group`
   - 删除 `self._load_calibration_params()` 和 `self._load_sensor_array_params()` 调用

3. **新增的代码**（在 `__init__` 中）：
   - 添加 `from calibration.lib.forward import CalibrationProcessor`
   - 添加 `self._calib = CalibrationProcessor(self._sensor_config)`

4. **修改 run() 方法**（约 line 379-395）：
   - 删除内联 ellipsoid 校正循环
   - 替换为 `cx, cy, cz = self._calib.apply(s.id, s.x, s.y, s.z)`

- [ ] **Step 1: 读取 serial_node_tdm.py 确认要删除的具体行**

```bash
# 确认 _load_calibration_params 起始行
grep -n "_load_calibration_params\|_load_sensor_array_params\|_apply_ellipsoid_correction" \
  /home/zhang/embedded_array_ws/src/serial_processor/scripts/serial_node_tdm.py
```

- [ ] **Step 2: 确认删除后，添加 import 和 self._calib**

在 `from sensor_array_config import get_config, SensorArrayConfig` 下方添加：
```python
from calibration.lib.forward import CalibrationProcessor
```

在 `self._sensor_config: SensorArrayConfig = get_config(self._sensor_type)` 之后添加：
```python
self._calib = CalibrationProcessor(self._sensor_config)
```

- [ ] **Step 3: 修改 run() 循环中校正逻辑**

约 line 379-395，将：
```python
cx, cy, cz = self._apply_ellipsoid_correction(s.x, s.y, s.z, s.id)
if s.id in self.R_CORR:
    b_rot = self.R_CORR[s.id] @ np.array([cx, cy, cz])
    cx, cy, cz = b_rot[0], b_rot[1], b_rot[2]
if s.id in self.D_matrix and s.id in self.e_bias:
    b_cons = self.D_matrix[s.id] @ np.array([cx, cy, cz]) + self.e_bias[s.id]
    cx, cy, cz = b_cons[0], b_cons[1], b_cons[2]
if self._amp_factor is not None and self._amp_factor != 0:
    cx, cy, cz = cx / self._amp_factor, cy / self._amp_factor, cz / self._amp_factor
```

替换为：
```python
cx, cy, cz = self._calib.apply(s.id, s.x, s.y, s.z)
```

- [ ] **Step 4: 删除 _load_calibration_params, _load_sensor_array_params, _apply_ellipsoid_correction 方法及删除相关的 self.* 属性初始化**

- [ ] **Step 5: 验证 Python 语法**

```bash
python3 -m py_compile /home/zhang/embedded_array_ws/src/serial_processor/scripts/serial_node_tdm.py
```

- [ ] **Step 6: 提交**

```bash
git add src/serial_processor/scripts/serial_node_tdm.py
git commit -m "refactor(serial_node_tdm): replace inline calibration with CalibrationProcessor"
```

---

## Task 8: 数值一致性验证

**Files:**
- Test: 临时验证脚本（不需要提交）

- [ ] **Step 1: 编写验证脚本**

使用 `sensor_array_config` 中已有的参数，对比旧逻辑（内联）和新逻辑（CalibrationProcessor）的输出是否一致：

```python
#!/usr/bin/env python3
"""验证 CalibrationProcessor 与原内联逻辑数值一致"""
import numpy as np
from sensor_array_config import get_config
from calibration.lib.forward import CalibrationProcessor

config = get_config("QMC6309")

# 模拟原始内联逻辑
intrinsic = config.intrinsic
hardware = config.hardware
consistency = config.consistency

offset = {}
correction = {}
for sid, params in intrinsic.params.items():
    offset[sid] = np.array(params.o_i)
    correction[sid] = np.array(params.C_i)

r_corr = {}
for entry in hardware.R_CORR:
    mat = np.array(entry.matrix).reshape(3, 3, order='F')
    for sid in entry.sensor_ids:
        r_corr[sid] = mat

D_matrix = {}
e_bias = {}
for sid, params in consistency.params.items():
    D_matrix[sid] = np.array(params.D_i)
    e_bias[sid] = np.array(params.e_i)
amp_factor = consistency.amp_factor or 1.0

# 随机测试数据
np.random.seed(42)
test_data = [
    (sid, float(np.random.randn()),
     float(np.random.randn()),
     float(np.random.randn()))
    for _ in range(100)
    for sid in range(1, 13)
]

calib = CalibrationProcessor(config)

errors = []
for sid, x, y, z in test_data:
    # 旧逻辑
    o_i = offset.get(sid, np.zeros(3))
    C_i = correction.get(sid, np.eye(3))
    cx, cy, cz = ((np.array([x, y, z]) - o_i) @ C_i.T).tolist()
    if sid in r_corr:
        b_rot = r_corr[sid] @ np.array([cx, cy, cz])
        cx, cy, cz = b_rot.tolist()
    if sid in D_matrix and sid in e_bias:
        b_cons = D_matrix[sid] @ np.array([cx, cy, cz]) + e_bias[sid]
        cx, cy, cz = b_cons.tolist()
    if amp_factor != 0:
        cx, cy, cz = cx / amp_factor, cy / amp_factor, cz / amp_factor
    old_result = (cx, cy, cz)

    # 新逻辑
    new_result = calib.apply(sid, x, y, z)

    diff = tuple(np.array(new_result) - np.array(old_result))
    if not np.allclose(diff, (0, 0, 0), atol=1e-9):
        errors.append((sid, x, y, z, old_result, new_result, diff))

if errors:
    print(f"FAIL: {len(errors)} / {len(test_data)} mismatches")
    for e in errors[:5]:
        print(f"  sid={e[0]}, input=({e[1]:.4f},{e[2]:.4f},{e[3]:.4f})")
        print(f"  old={e[4]}, new={e[5]}, diff={e[6]}")
else:
    print(f"PASS: All {len(test_data)} tests match")
```

- [ ] **Step 2: 运行验证脚本**

```bash
cd /home/zhang/embedded_array_ws
python3 verify_calibration_forward.py
```

- [ ] **Step 3: 清理验证脚本**（不提交）

---

## Task 9: 确认 TDM 无残留引用

- [ ] **Step 1: 确认已删除方法不再被引用**

```bash
grep -n "offset\|correction\|D_matrix\|e_bias\|_amp_factor\|R_CORR\|_d_list\|_load_calibration_params\|_load_sensor_array_params\|_apply_ellipsoid_correction" \
  /home/zhang/embedded_array_ws/src/serial_processor/scripts/serial_node_tdm.py | \
  grep -v "^.*self\._\(calib\|sensor_config\)"
```

预期输出为空（或仅有新模块引用）。如有残留，手动清理。

---

## Task 10: 最终构建验证

- [ ] **Step 1: catkin build**

```bash
cd /home/zhang/embedded_array_ws
catkin build
```

预期：无错误

---

## 依赖关系图

```
Task 1 (目录/__init__.py)
    ↓
Task 2 (base.py) ← 无依赖
    ↓
Task 3 (processor.py) ← 依赖 Task 2
    ↓
Task 4 (ellipsoid.py) ← 依赖 Task 2
Task 5 (r_corr.py) ← 依赖 Task 2
Task 6 (consistency.py) ← 依赖 Task 2
    ↓
Task 7 (serial_node_tdm.py) ← 依赖 Task 1-6
    ↓
Task 8 (验证) ← 依赖 Task 1-7
    ↓
Task 9 (确认无残留)
    ↓
Task 10 (构建)
```

**可并行执行的任务：** Task 2-6 可并行（相互无依赖）
**串行执行的任务：** Task 3 需等 Task 2；Task 7 需等 Task 1-6；Task 8 需等 Task 7
