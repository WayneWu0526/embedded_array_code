# Calibration Forward 模块抽取设计

## 1. 背景与目标

**动机：** `serial_node_tdm.py` 内嵌了校准逻辑（ellipsoid、R_CORR、consistency），与 TDM 强耦合。后续可能需要按场景启用/禁用某个校正步骤（如丢弃 consistency），或替换校正算法，而不希望改动 `serial_node_tdm.py`。

**目标：** 将校正逻辑抽取为独立模块，通过统一接口调用，TDM 代码保持不变。

**约束：** 校正参数一旦标定即固定，不需要运行时动态开关。模块可替换性在代码级别实现。

---

## 2. 架构：Strategy 模式

```
serial_node_tdm.py
    └── CalibrationProcessor (facade, 统一入口)
             ├── EllipsoidCorrection (step 1)
             ├── RCorrRotation (step 2)
             └── ConsistencyCorrection (step 3, optional)
```

### 核心接口

```python
# calibration/lib/forward/base.py
class CorrectionStep(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, x: float, y: float, z: float, sensor_id: int) -> Tuple[float, float, float]:
        """对单颗传感器的单次采样进行校正"""
        pass

    def reset(self):
        """可选：重置内部状态"""
        pass
```

```python
# calibration/lib/forward/processor.py
class CalibrationProcessor:
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

    def apply(self, sensor_id: int, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """依次通过所有 step，返回最终校正结果"""
        cx, cy, cz = x, y, z
        for step in self._steps:
            cx, cy, cz = step.apply(cx, cy, cz, sensor_id)
        return cx, cy, cz
```

---

## 3. 目录结构

```
src/calibration/lib/forward/
    __init__.py          # public exports
    base.py              # CorrectionStep ABC
    processor.py          # CalibrationProcessor facade
    ellipsoid.py          # EllipsoidCorrection step
    r_corr.py             # RCorrRotation step
    consistency.py        # ConsistencyCorrection step
```

与现有模块的关系：
- `ellipsoid_fit/` — Phase 1 标定算法（拟合参数）
- `consistency_fit.py` — Phase 2 标定算法（拟合参数）
- `forward/` — 正向校正（消费参数，应用校正）

---

## 4. 各 Step 详细设计

### 4.1 EllipsoidCorrection

**参数来源：** `sensor_config.intrinsic`

```python
# 加载逻辑（__init__ 时）
self._offset = {}
self._correction = {}  # C_i
for sid, params in sensor_config.intrinsic.params.items():
    self._offset[sid] = np.array(params.o_i)
    self._correction[sid] = np.array(params.C_i)

# fallback
if not self._offset:
    for sid in range(1, sensor_config.manifest.n_sensors + 1):
        self._offset[sid] = np.zeros(3)
        self._correction[sid] = np.eye(3)
```

**apply 逻辑：**
```python
def apply(self, x, y, z, sensor_id):
    o_i = self._offset.get(sensor_id, np.zeros(3))
    C_i = self._correction.get(sensor_id, np.eye(3))
    b = np.array([x, y, z])
    b_corr = (b - o_i) @ C_i.T
    return b_corr[0], b_corr[1], b_corr[2]
```

### 4.2 RCorrRotation

**参数来源：** `sensor_config.hardware.R_CORR`

```python
# 加载逻辑
self._r_corr = {}
for entry in sensor_config.hardware.R_CORR:
    mat = np.array(entry.matrix).reshape(3, 3, order='F')
    for sid in entry.sensor_ids:
        self._r_corr[sid] = mat
```

**apply 逻辑：**
```python
def apply(self, x, y, z, sensor_id):
    R = self._r_corr.get(sensor_id)
    if R is None:
        return x, y, z
    b = np.array([x, y, z])
    return tuple((R @ b).tolist())
```

### 4.3 ConsistencyCorrection

**参数来源：** `sensor_config.consistency`

```python
# 加载逻辑
self._D = {}
self._e = {}
for sid, params in sensor_config.consistency.params.items():
    self._D[sid] = np.array(params.D_i)
    self._e[sid] = np.array(params.e_i)
self._amp_factor = sensor_config.consistency.amp_factor  # may be None

# fallback
if not self._D:
    n = sensor_config.manifest.n_sensors
    for sid in range(1, n + 1):
        self._D[sid] = np.eye(3)
        self._e[sid] = np.zeros(3)
    self._amp_factor = 1.0
```

**apply 逻辑：**
```python
def apply(self, x, y, z, sensor_id):
    D = self._D.get(sensor_id, np.eye(3))
    e = self._e.get(sensor_id, np.zeros(3))
    b = np.array([x, y, z])
    b_cons = D @ b + e
    if self._amp_factor is not None and self._amp_factor != 0:
        b_cons = b_cons / self._amp_factor
    return tuple(b_cons.tolist())
```

---

## 5. serial_node_tdm.py 改动

**删除的内容：**
- `_load_calibration_params()` 方法
- `_load_sensor_array_params()` 方法
- `_apply_ellipsoid_correction()` 方法
- `self.offset`, `self.correction`, `self.D_matrix`, `self.e_bias`, `self._amp_factor` 属性
- `self.R_CORR`, `self._d_list`, `self._n_sensors`, `self._n_groups`, `self._sensors_per_group`, `self._sensor_to_group` 属性

**替换为：**

```python
from calibration.lib.forward import CalibrationProcessor

class SerialNodeTDM:
    def __init__(self):
        ...
        self._calib = CalibrationProcessor(self._sensor_config)
        ...

    # 在 run() 循环中（约第 380 行附近）：
    for s in msg.sensor_data:
        cx, cy, cz = self._calib.apply(s.id, s.x, s.y, s.z)
        corrected_sensors.append(SensorData(id=s.id, x=cx, y=cy, z=cz))
```

---

## 6. 可替换性示例

**场景：丢弃 consistency 校正**

```python
from calibration.lib.forward import (
    CalibrationProcessor,
    EllipsoidCorrection,
    RCorrRotation,
)

steps = [
    EllipsoidCorrection(sensor_config),
    RCorrRotation(sensor_config),
    # 不加 ConsistencyCorrection
]
calib = CalibrationProcessor(sensor_config, steps=steps)
```

TDM 代码完全不变，只要创建 `CalibrationProcessor` 时指定 `steps` 即可。

---

## 7. 错误处理

每个 Step 在找不到对应 sensor_id 参数时，均 fallback 到**直通**（原样返回输入），不抛异常。

---

## 8. 验证计划

1. 抽取后的模块输出与原 `serial_node_tdm.py` 内联逻辑**数值一致**
2. 按场景测试：full pipeline / 仅 ellipsoid / ellipsoid+R_CORR
3. 确认 `serial_node_tdm.py` 中已删除的代码不再被引用
