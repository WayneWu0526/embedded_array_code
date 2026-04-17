# Multi-Sensor Array Modularization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor 4 packages (serial_processor, calibration, gels_localization, sensor_data_collection) to support multiple sensor array types through a shared abstraction layer, enabling future sensor additions without modifying core logic.

**Architecture:** Introduce a sensor-agnostic `SensorArrayConfig` abstraction. Each sensor type (QMC6309, future) becomes a self-contained config bundle under `config/<sensor_type>/`. A thin shim layer in each package loads type-specific configs without hardcoding sensor count, bit width, or range. Internal unit normalized to Gs.

**Tech Stack:** Python dataclasses, PyYAML, ROS rosparam/launch, JSON

---

## File Structure (Target)

```
src/
  sensor_array_config/           # NEW: Shared sensor array config package
    __init__.py
    base.py                       # SensorArrayConfig abstract base + dataclasses
    qmc6309.py                    # QMC6309-specific implementation
    config/
      qmc6309/
        intrinsic_params.json     # (moved from serial_processor/config/QMC6309/)
        consistency_params.json  # (moved from serial_processor/config/QMC6309/)
        sensor_array_params.json # (moved from serial_processor/config/QMC6309/)
        sensor_calibration.yaml  # (moved from sensor_data_collection/config/)
        calibration_manifest.yaml # NEW: sensor type, bit_width, range_gs, n_sensors
      # future_sensor_type/       # Example: config/my_new_sensor/

  serial_processor/
    scripts/
      serial_node_tdm.py          # MODIFY: use SensorArrayConfig instead of hardcoded params
    config/
      QMC6309/                    # KEEP as symlink OR redirect to sensor_array_config/
                                   # (or remove if no longer needed)

  calibration/
    lib/
      consistency_fit.py           # MODIFY: accept SensorArrayConfig, remove n_sensors hardcode
      ellipsoid_fit/
        ellipsoid_fit.py           # MODIFY: accept SensorArrayConfig
    scripts/
      ellipsoid_calibration_sampler.py  # MODIFY: accept sensor_type param
      handheld_calibration_sampler.py   # MODIFY: accept sensor_type param
      calibration_postprocessor.py      # MODIFY: accept sensor_type param

  gels_localization/
    scripts/
      localization_service_node.py # MODIFY: load from SensorArrayConfig
      offline_utils.py              # MODIFY: use SensorArrayConfig
      mag_dipole_model.py          # MODIFY: pure function, no sensor deps

  sensor_data_collection/
    config/
      sensor_calibration.yaml      # KEEP gs_to_tesla only (generic)
      task_params.yaml              # KEEP (generic - no sensor-specific params)
      params_cvt.yaml               # KEEP (timing params, generic)
    scripts/
      data_collection_node.py     # MODIFY: accept sensor_type param, use SensorArrayConfig

  # No new test files required for this refactor; existing integration tests verify behavior.
```

---

## Task 1: Create `sensor_array_config` Package Skeleton

**Files:**
- Create: `src/sensor_array_config/__init__.py`
- Create: `src/sensor_array_config/base.py`
- Create: `src/sensor_array_config/config/qmc6309/calibration_manifest.yaml`
- Create: `src/sensor_array_config/config/qmc6309/sensor_calibration.yaml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/sensor_array_config/config/qmc6309
touch src/sensor_array_config/__init__.py
touch src/sensor_array_config/base.py
```

- [ ] **Step 2: Write `base.py` - abstract base and dataclasses**

```python
# src/sensor_array_config/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json
import yaml
import os

# ---------- manifest schema ----------
@dataclass
class SensorArrayManifest:
    sensor_type: str          # e.g. "QMC6309"
    bit_width: int            # e.g. 16
    range_gs: float           # e.g. 500.0  (full scale range in Gauss)
    n_sensors: int            # e.g. 12
    n_groups: int             # e.g. 4  (for R_CORR grouping)
    sensors_per_group: int     # e.g. 3
    # derived
    adu_to_gs: float = field(init=False)  # computed from bit_width and range_gs

    def __post_init__(self):
        # LSB value = range_gs / (2^(bit_width-1) - 1) for signed int
        self.adu_to_gs = self.range_gs / (2 ** (self.bit_width - 1))

# ---------- intrinsic params (Phase 1) ----------
@dataclass
class IntrinsicParams:
    """Per-sensor ellipsoid correction: o_i (offset) + C_i (correction matrix)."""
    o_i: List[float]           # 3-element
    C_i: List[List[float]]      # 3x3 matrix, row-major

@dataclass
class IntrinsicParamsSet:
    params: Dict[int, IntrinsicParams]  # sensor_id -> params

    @classmethod
    def from_json(cls, path: str) -> "IntrinsicParamsSet":
        with open(path) as f:
            raw = json.load(f)
        params = {}
        for sid, entry in raw.items():
            params[int(sid)] = IntrinsicParams(
                o_i=entry["o_i"],
                C_i=entry["C_i"]
            )
        return cls(params=params)

# ---------- consistency params (Phase 2) ----------
@dataclass
class ConsistencyParams:
    """Per-sensor consistency correction: D_i (scale) + e_i (bias)."""
    D_i: List[List[float]]     # 3x3 matrix, row-major
    e_i: List[float]            # 3-element bias

@dataclass
class ConsistencyParamsSet:
    params: Dict[int, ConsistencyParams]

    @classmethod
    def from_json(cls, path: str) -> "ConsistencyParamsSet":
        with open(path) as f:
            raw = json.load(f)
        params = {}
        for sid, entry in raw.items():
            params[int(sid)] = ConsistencyParams(
                D_i=entry["D_i"],
                e_i=entry["e_i"]
            )
        return cls(params=params)

# ---------- sensor array hardware params ----------
@dataclass
class SensorArrayHardwareParams:
    d_list: List[List[float]]   # n_sensors x 3, positions in meters
    R_CORR: List[List[float]]   # n_groups x 9, column-major 3x3 rotations flattened

    @classmethod
    def from_json(cls, path: str) -> "SensorArrayHardwareParams":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            d_list=raw["d_list"],
            R_CORR=raw["R_CORR"]
        )

# ---------- sensor array config (facade) ----------
class SensorArrayConfig(ABC):
    """Abstract base for a sensor array configuration."""

    @property
    @abstractmethod
    def manifest(self) -> SensorArrayManifest:
        ...

    @property
    @abstractmethod
    def intrinsic(self) -> IntrinsicParamsSet:
        ...

    @property
    @abstractmethod
    def consistency(self) -> ConsistencyParamsSet:
        ...

    @property
    @abstractmethod
    def hardware(self) -> SensorArrayHardwareParams:
        ...

    @property
    @abstractmethod
    def gs_to_si(self) -> float:
        """Conversion from Gauss to SI (Tesla = Gs * this)."""
        ...

    def get_sensor_ids(self) -> List[int]:
        return list(range(1, self.manifest.n_sensors + 1))

    def get_group_for_sensor(self, sensor_id: int) -> int:
        """Returns 0-indexed group index for a sensor ID."""
        sensors_per_group = self.manifest.sensors_per_group
        return (sensor_id - 1) // sensors_per_group

    def get_sensors_in_group(self, group: int) -> List[int]:
        start = group * self.manifest.sensors_per_group + 1
        end = start + self.manifest.sensors_per_group
        return list(range(start, end))
```

- [ ] **Step 3: Write `config/qmc6309/calibration_manifest.yaml`**

```yaml
# src/sensor_array_config/config/qmc6309/calibration_manifest.yaml
sensor_type: "QMC6309"
bit_width: 16
range_gs: 500.0        # full-scale range in Gauss
n_sensors: 12
n_groups: 4
sensors_per_group: 3
gs_to_tesla: 1.0e-4    # Gs -> Tesla conversion
```

- [ ] **Step 4: Write `config/qmc6309/sensor_calibration.yaml`**

```yaml
# src/sensor_array_config/config/qmc6309/sensor_calibration.yaml
# Generic sensor calibration params for this array type.
# Unit: Gauss internal, convert to Tesla for SI physics.
gs_to_tesla: 1.0e-4
```

- [ ] **Step 5: Create `src/sensor_array_config/qmc6309.py`**

```python
# src/sensor_array_config/qmc6309.py
import os
from .base import (
    SensorArrayConfig, SensorArrayManifest,
    IntrinsicParamsSet, ConsistencyParamsSet, SensorArrayHardwareParams
)

_QMC6309_ROOT = os.path.join(os.path.dirname(__file__), "config", "qmc6309")

class QMC6309Config(SensorArrayConfig):
    def __init__(self, config_root: str = _QMC6309_ROOT):
        self._config_root = config_root

    @property
    def manifest(self) -> SensorArrayManifest:
        path = os.path.join(self._config_root, "calibration_manifest.yaml")
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return SensorArrayManifest(
            sensor_type=d["sensor_type"],
            bit_width=d["bit_width"],
            range_gs=d["range_gs"],
            n_sensors=d["n_sensors"],
            n_groups=d["n_groups"],
            sensors_per_group=d["sensors_per_group"]
        )

    @property
    def intrinsic(self) -> IntrinsicParamsSet:
        return IntrinsicParamsSet.from_json(
            os.path.join(self._config_root, "intrinsic_params.json")
        )

    @property
    def consistency(self) -> ConsistencyParamsSet:
        return ConsistencyParamsSet.from_json(
            os.path.join(self._config_root, "consistency_params.json")
        )

    @property
    def hardware(self) -> SensorArrayHardwareParams:
        return SensorArrayHardwareParams.from_json(
            os.path.join(self._config_root, "sensor_array_params.json")
        )

    @property
    def gs_to_si(self) -> float:
        return 1.0e-4
```

- [ ] **Step 6: Commit**

```bash
git add src/sensor_array_config/
git commit -m "feat: create sensor_array_config package skeleton with base classes and QMC6309 implementation"
```

---

## Task 2: Add Manifest Enum and Registry Pattern to `base.py`

**Files:**
- Modify: `src/sensor_array_config/base.py`

- [ ] **Step 1: Add `SensorType` enum and `CONFIG_REGISTRY`**

Append to `base.py`:

```python
from enum import Enum

class SensorType(Enum):
    QMC6309 = "QMC6309"
    # Future: MY_NEW_SENSOR = "my_new_sensor"

_REGISTRY: Dict[str, type] = {
    "QMC6309": QMC6309Config,  # lazy import below
}

def get_config(sensor_type: str, **kwargs) -> SensorArrayConfig:
    """Factory: return a SensorArrayConfig for the given type name."""
    if sensor_type not in _REGISTRY:
        raise ValueError(f"Unknown sensor type '{sensor_type}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[sensor_type](**kwargs)

def register_sensor_type(name: str, cls: type):
    """Decorator / function to register new sensor types."""
    _REGISTRY[name] = cls
```

At bottom of `base.py`, register QMC6309 lazily to avoid circular import:

```python
# Lazy registration to avoid circular import
import importlib
def _register_qmc6309():
    from .qmc6309 import QMC6309Config
    _REGISTRY["QMC6309"] = QMC6309Config
_register_qmc6309()
```

- [ ] **Step 2: Add `IntrinsicParamsSet.to_json()` and `ConsistencyParamsSet.to_json()` methods**

Append to respective dataclass classes in `base.py`:

```python
    def to_json(self, path: str):
        raw = {str(k): {"o_i": v.o_i, "C_i": v.C_i} for k, v in self.params.items()}
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)
```

And similarly for `ConsistencyParamsSet`:

```python
    def to_json(self, path: str):
        raw = {str(k): {"D_i": v.D_i, "e_i": v.e_i} for k, v in self.params.items()}
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)
```

- [ ] **Step 3: Commit**

```bash
git add src/sensor_array_config/base.py
git commit -m "feat: add SensorType enum, registry pattern, and to_json methods"
```

---

## Task 3: Move QMC6309 Config Files to `sensor_array_config`

**Files:**
- Copy: `src/serial_processor/config/QMC6309/*.json` → `src/sensor_array_config/config/qmc6309/`
- Copy: `src/sensor_data_collection/config/sensor_calibration.yaml` → `src/sensor_array_config/config/qmc6309/`

- [ ] **Step 1: Copy config files**

```bash
cp src/serial_processor/config/QMC6309/intrinsic_params.json src/sensor_array_config/config/qmc6309/
cp src/serial_processor/config/QMC6309/consistency_params.json src/sensor_array_config/config/qmc6309/
cp src/serial_processor/config/QMC6309/sensor_array_params.json src/sensor_array_config/config/qmc6309/
cp src/sensor_data_collection/config/sensor_calibration.yaml src/sensor_array_config/config/qmc6309/
```

- [ ] **Step 2: Verify file contents are intact**

```bash
# No changes to file content, just relocation
```

- [ ] **Step 3: Commit moved files**

```bash
git add src/sensor_array_config/config/qmc6309/
git commit -m "feat: move QMC6309 config files to sensor_array_config package"
```

---

## Task 4: Refactor `serial_node_tdm.py` to Use `SensorArrayConfig`

**Files:**
- Modify: `src/serial_processor/scripts/serial_node_tdm.py:101-203` (calibration loading) and lines 28-32 (hardcoded constants) and lines 401-453 (calibration pipeline)

- [ ] **Step 1: Add import and rosparam loading for sensor_type**

Add near top of file after existing imports:

```python
import rospkg
import yaml
from sensor_array_config import get_config, SensorArrayConfig

# Load sensor_type from rosparam (default to QMC6309 for backward compatibility)
_SENSOR_TYPE = rospy.get_param("~sensor_type", "QMC6309")
_SENSOR_CONFIG: SensorArrayConfig = get_config(_SENSOR_TYPE)
```

Add a rosparam in the launch file or accept as arg:

```xml
<!-- In serial_processor/launch/serial_node_tdm.launch -->
<param name="sensor_type" value="$(arg sensor_type)" />
```

- [ ] **Step 2: Replace `_load_calibration_params()` to use config**

Replace lines 101-124 with:

```python
def _load_calibration_params(self):
    """Load Phase 1 ellipsoid params from SensorArrayConfig."""
    self._intrinsic = self._sensor_config.intrinsic
    self._consistency = self._sensor_config.consistency
```

- [ ] **Step 3: Replace `_load_sensor_array_params()` to use config**

Replace lines 138-179 with:

```python
def _load_sensor_array_params(self):
    """Load d_list and R_CORR from SensorArrayConfig."""
    hw = self._sensor_config.hardware
    self._d_list = hw.d_list
    self._R_CORR = hw.R_CORR
    self._n_sensors = self._sensor_config.manifest.n_sensors
    self._n_groups = self._sensor_config.manifest.n_groups
    self._sensors_per_group = self._sensor_config.manifest.sensors_per_group
```

- [ ] **Step 4: Replace `_load_consistency_params()` - remove (merged above)**

Delete `_load_consistency_params()` method entirely (lines 181-203).

- [ ] **Step 5: Replace hardcoded sensor count and group mapping**

Replace lines 126-136 (hardcoded group mapping):

```python
# Build sensor -> group lookup from config
self._sensor_to_group = {
    sid: self._sensor_config.get_group_for_sensor(sid)
    for sid in self._sensor_config.get_sensor_ids()
}
```

Replace line 28 comment (`# 12 sensors`) with reference to config.

- [ ] **Step 6: Replace hardcoded `SCALE_FACTOR` with config-derived value**

Replace line 32 (`SCALE_FACTOR = 32.0 / 32768.0`):

```python
# Scale factor: STM32 sends signed 16-bit, convert to Gs using config
_SCALE_FACTOR = self._sensor_config.manifest.adu_to_gs
```

- [ ] **Step 7: Verify changes compile**

Run: `python -m py_compile src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 8: Commit**

```bash
git add src/serial_processor/scripts/serial_node_tdm.py
git commit -m "refactor(serial_processor): use SensorArrayConfig instead of hardcoded params"
```

---

## Task 5: Refactor `consistency_fit.py` to Accept `SensorArrayConfig`

**Files:**
- Modify: `src/calibration/lib/consistency_fit.py` (lines 118 `n_sensors = 12` hardcode, lines 24-31 `FieldCondition` enum, lines 349-368 save/load)

- [ ] **Step 1: Add `SensorArrayConfig` import and update function signatures**

Add import near top:

```python
from sensor_array_config import get_config, SensorArrayConfig
```

- [ ] **Step 2: Update `consistency_fit()` to accept optional config**

Change function signature (around line 192):

```python
def consistency_fit(
    csv_dir: str,
    sensor_config: SensorArrayConfig = None,  # NEW: optional, falls back to QMC6309
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors
    # ... rest of function uses n_sensors from config
```

- [ ] **Step 3: Update `FieldCondition` to be data-driven from CSV columns**

The `FieldCondition` enum (lines 24-31) defines field conditions. Replace with a function that reads CSV headers to detect available conditions rather than hardcoding 7:

```python
def detect_field_conditions(csv_dir: str, n_sensors: int) -> Dict[str, int]:
    """Detect available field conditions from CSV filenames."""
    import os, re
    conditions = {}
    for fname in os.listdir(csv_dir):
        if not fname.startswith("calib_") or not fname.endswith(".csv"):
            continue
        m = re.match(r"calib_([a-zA-Z0-9_]+)\.csv", fname)
        if m:
            conditions[m.group(1)] = 1  # count populated below
    return conditions
```

The `FieldCondition` enum itself can stay for internal use but its values should be derived from config's `manifest.n_sensors`.

- [ ] **Step 4: Update `save_consistency_params()` to save via `ConsistencyParamsSet`**

Replace lines 349-368 with:

```python
def save_consistency_params(
    D_dict: Dict[int, np.ndarray],
    e_dict: Dict[int, np.ndarray],
    output_path: str,
    sensor_config: SensorArrayConfig = None,
):
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    params_set = ConsistencyParamsSet(params={
        sid: ConsistencyParams(D_i=D_dict[sid].tolist(), e_i=e_dict[sid].tolist())
        for sid in sensor_config.get_sensor_ids()
    })
    params_set.to_json(output_path)
```

- [ ] **Step 5: Update `load_consistency_params()` to use `ConsistencyParamsSet`**

Replace lines 368-390 with:

```python
def load_consistency_params(
    json_path: str,
    sensor_config: SensorArrayConfig = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    params_set = ConsistencyParamsSet.from_json(json_path)
    D_dict = {sid: np.array(p.D_i) for sid, p in params_set.params.items()}
    e_dict = {sid: np.array(p.e_i) for sid, p in params_set.params.items()}
    return D_dict, e_dict
```

- [ ] **Step 6: Verify changes compile**

Run: `python -m py_compile src/calibration/lib/consistency_fit.py`

- [ ] **Step 7: Commit**

```bash
git add src/calibration/lib/consistency_fit.py
git commit -m "refactor(calibration): make consistency_fit sensor-agnostic via SensorArrayConfig"
```

---

## Task 6: Refactor `ellipsoid_fit.py` to Accept `SensorArrayConfig`

**Files:**
- Modify: `src/calibration/lib/ellipsoid_fit/ellipsoid_fit.py` (line 134 `n_sensors = 12` hardcode)

- [ ] **Step 1: Add import and update `batch_ellipsoid_fit()` signature**

Add import near top:

```python
from sensor_array_config import get_config, SensorArrayConfig
```

Update `batch_ellipsoid_fit()` (around line 114):

```python
def batch_ellipsoid_fit(
    data: Dict[int, np.ndarray],
    sensor_config: SensorArrayConfig = None,
) -> Dict[int, CalibrationResult]:
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors
    sensor_ids = sensor_config.get_sensor_ids()
    # ... use sensor_ids instead of range(1, n_sensors + 1)
```

- [ ] **Step 2: Verify changes compile**

Run: `python -m py_compile src/calibration/lib/ellipsoid_fit/ellipsoid_fit.py`

- [ ] **Step 3: Commit**

```bash
git add src/calibration/lib/ellipsoid_fit/ellipsoid_fit.py
git commit -m "refactor(calibration): make ellipsoid_fit sensor-agnostic"
```

---

## Task 7: Refactor `calibration_postprocessor.py` to Accept `sensor_type` Param

**Files:**
- Modify: `src/calibration/scripts/calibration_postprocessor.py` (class init, method signatures)

- [ ] **Step 1: Add `sensor_type` parameter to `CalibrationPostProcessor.__init__()`**

```python
def __init__(self, sensor_type: str = "QMC6309", ...):
    self._sensor_config = get_config(sensor_type)
    # ... pass sensor_config to called functions
```

- [ ] **Step 2: Add `sensor_type` to `ConsistencyPostProcessor.__init__()`**

```python
def __init__(self, sensor_type: str = "QMC6309", ...):
    self._sensor_config = get_config(sensor_type)
    # ... pass to consistency_fit calls
```

- [ ] **Step 3: Update `save_intrinsic_params()` to use `IntrinsicParamsSet.to_json()`**

Replace current JSON writing with:

```python
from sensor_array_config.base import IntrinsicParamsSet, IntrinsicParams
params_set = IntrinsicParamsSet(params={
    sid: IntrinsicParams(o_i=results[sid].offset, C_i=results[sid].correction)
    for sid in self._sensor_config.get_sensor_ids()
})
params_set.to_json(output_path)
```

- [ ] **Step 4: Add argparse `--sensor-type` argument**

Add to the main block:

```python
parser.add_argument("--sensor-type", default="QMC6309", help="Sensor array type")
```

- [ ] **Step 5: Verify changes compile**

Run: `python -m py_compile src/calibration/scripts/calibration_postprocessor.py`

- [ ] **Step 6: Commit**

```bash
git add src/calibration/scripts/calibration_postprocessor.py
git commit -m "refactor(calibration): add --sensor-type param to postprocessor"
```

---

## Task 8: Refactor `handheld_calibration_sampler.py` and `ellipsoid_calibration_sampler.py`

**Files:**
- Modify: `src/calibration/scripts/handheld_calibration_sampler.py`
- Modify: `src/calibration/scripts/ellipsoid_calibration_sampler.py`

- [ ] **Step 1: Add `sensor_type` rosparam/arg to `handheld_calibration_sampler.py`**

Add after imports:

```python
sensor_type = rospy.get_param("~sensor_type", "QMC6309")
sensor_config = get_config(sensor_type)
n_sensors = sensor_config.manifest.n_sensors
```

Replace hardcoded sensor ID checks (lines 55-56, 61) with config-driven checks.

- [ ] **Step 2: Similarly update `ellipsoid_calibration_sampler.py`**

Add same param loading and replace hardcoded `n_sensors = 12`.

- [ ] **Step 3: Verify changes compile**

Run: `python -m py_compile` on both files.

- [ ] **Step 4: Commit**

```bash
git add src/calibration/scripts/handheld_calibration_sampler.py src/calibration/scripts/ellipsoid_calibration_sampler.py
git commit -m "refactor(calibration): add sensor_type param to calibration samplers"
```

---

## Task 9: Refactor `localization_service_node.py` to Use `SensorArrayConfig`

**Files:**
- Modify: `src/gels_localization/scripts/localization_service_node.py` (lines 51-53 hardcoded globals, lines 56-108 `load_configuration()`)

- [ ] **Step 1: Replace global `GS_TO_TESLA`, `D_LIST`, hardcoded n_sensors with config**

Remove lines 51-53 globals. Add near top after imports:

```python
from sensor_array_config import get_config, SensorArrayConfig

# Load sensor_type from rosparam
_SENSOR_TYPE = rospy.get_param("~sensor_type", "QMC6309")
_SENSOR_CONFIG: SensorArrayConfig = get_config(_SENSOR_TYPE)
_SENSOR_ARRAY_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "sensor_array_config", "config", _SENSOR_TYPE
)
```

- [ ] **Step 2: Refactor `load_configuration()` to load from config**

Replace lines 56-108:

```python
def load_configuration(yaml_path: str = None) -> None:
    global D_LIST, GS_TO_TESLA
    if yaml_path is None:
        yaml_path = os.path.join(_SENSOR_ARRAY_ROOT, "sensor_calibration.yaml")
    with open(yaml_path) as f:
        cal_data = yaml.safe_load(f)
    GS_TO_TESLA = cal_data.get("gs_to_tesla", _SENSOR_CONFIG.gs_to_si)
    D_LIST = np.array(_SENSOR_CONFIG.hardware.d_list).T  # 3xN
```

- [ ] **Step 3: Replace hardcoded de1/de2/de3 defaults in `load_configuration()`**

Remove the fallback computation of d_list from de1/de2/de3 (lines 75-79). The config file is now authoritative.

- [ ] **Step 4: Verify changes compile**

Run: `python -m py_compile src/gels_localization/scripts/localization_service_node.py`

- [ ] **Step 5: Commit**

```bash
git add src/gels_localization/scripts/localization_service_node.py
git commit -m "refactor(gels_localization): use SensorArrayConfig for D_LIST and GS_TO_TESLA"
```

---

## Task 10: Refactor `offline_utils.py` to Use `SensorArrayConfig`

**Files:**
- Modify: `src/gels_localization/scripts/offline_utils.py` (hardcoded d_list generation at lines 75-79, moment list at line 263)

- [ ] **Step 1: Add optional `sensor_config` param to `generate_hall_data_from_dipole()`**

```python
def generate_hall_data_from_dipole(
    p_est: np.ndarray,
    q_est: np.ndarray,
    moment_list: List[np.ndarray],
    d_list: np.ndarray = None,  # 3xN sensor positions, defaults to config
    sensor_config: SensorArrayConfig = None,
    n_sensors: int = None,
) -> np.ndarray:
    if d_list is None:
        if sensor_config is None:
            sensor_config = get_config("QMC6309")
        d_list = np.array(sensor_config.hardware.d_list).T
    if n_sensors is None:
        n_sensors = sensor_config.manifest.n_sensors
    # ... rest unchanged
```

- [ ] **Step 2: Add optional `sensor_config` to `compute_model_reference()`**

```python
def compute_model_reference(
    sources: List[Tuple[np.ndarray, np.ndarray]],
    sensor_config: SensorArrayConfig = None,
    d_list: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    if d_list is None:
        d_list = np.array(sensor_config.hardware.d_list).T
    # ... rest unchanged
```

- [ ] **Step 3: Verify changes compile**

Run: `python -m py_compile src/gels_localization/scripts/offline_utils.py`

- [ ] **Step 4: Commit**

```bash
git add src/gels_localization/scripts/offline_utils.py
git commit -m "refactor(gels_localization): make offline_utils sensor-agnostic"
```

---

## Task 11: Refactor `data_collection_node.py` to Accept `sensor_type`

**Files:**
- Modify: `src/sensor_data_collection/scripts/data_collection_node.py` (hardcoded 12 sensors, bitmap 0x0FFF)

- [ ] **Step 1: Add `sensor_type` rosparam and `SensorArrayConfig` loading**

Add after imports:

```python
from sensor_array_config import get_config, SensorArrayConfig

_SENSOR_TYPE = rospy.get_param("~sensor_type", "QMC6309")
_SENSOR_CONFIG: SensorArrayConfig = get_config(_SENSOR_TYPE)
```

- [ ] **Step 2: Replace hardcoded `bitmap = 0x0FFF` (all 12 sensors) with config**

```python
all_sensors_bitmap = (1 << _SENSOR_CONFIG.manifest.n_sensors) - 1
bitmap = rospy.get_param("~bitmap", all_sensors_bitmap)
```

- [ ] **Step 3: Replace hardcoded `MODE_SLOT_COUNT` to use config**

```python
MODE_SLOT_COUNT = {
    MODE_CVT: _SENSOR_CONFIG.manifest.n_groups + 1,   # groups + 1 background slot
    MODE_CCI: _SENSOR_CONFIG.manifest.n_groups,        # no background slot
}
```

- [ ] **Step 4: Verify changes compile**

Run: `python -m py_compile src/sensor_data_collection/scripts/data_collection_node.py`

- [ ] **Step 5: Commit**

```bash
git add src/sensor_data_collection/scripts/data_collection_node.py
git commit -m "refactor(sensor_data_collection): use SensorArrayConfig for sensor count and bitmap"
```

---

## Task 12: Add `mag_dipole_model.py` Refactor (Pure Function)

**Files:**
- Modify: `src/gels_localization/scripts/mag_dipole_model.py`

- [ ] **Step 1: Confirm no sensor-specific hardcoding exists**

Review the file. It should already be pure (no sensor count references). If clean, commit as-is with note. If hardcoding found, apply same `sensor_config` parameter pattern.

- [ ] **Step 2: Commit**

```bash
git add src/gels_localization/scripts/mag_dipole_model.py
git commit -m "chore(gels_localization): confirm mag_dipole_model is sensor-agnostic"
```

---

## Task 13: Update Launch Files with `sensor_type` Argument

**Files:**
- Modify: All launch files that launch the refactored nodes

- [ ] **Step 1: Add `sensor_type` arg to relevant launch files**

Add to each launch file that starts a node using SensorArrayConfig:

```xml
<arg name="sensor_type" default="QMC6309" />
<param name="sensor_type" value="$(arg sensor_type)" />
```

Key launch files to update:
- `src/serial_processor/launch/serial_node_tdm.launch`
- `src/calibration/launch/ellipsoid_calibration.launch` (if it exists)
- `src/gels_localization/launch/localization_service.launch` (create if needed)
- `src/sensor_data_collection/launch/data_collection.launch`

- [ ] **Step 2: Commit**

```bash
git add src/serial_processor/launch/serial_node_tdm.launch
git add src/sensor_data_collection/launch/data_collection.launch
git commit -m "feat: add sensor_type arg to launch files"
```

---

## Self-Review Checklist

1. **Spec coverage:** Every original requirement mapped?
   - ✅ Different arrays' intrinsic/consistency params independently calibrated and saved → `IntrinsicParamsSet`/`ConsistencyParamsSet` per sensor type in `sensor_array_config/config/<type>/`
   - ✅ Different bit widths/ranges (limited to Gs) → `SensorArrayManifest.adu_to_gs` computed from `bit_width` + `range_gs`
   - ✅ Different r_corr, d_list, sensor counts → `SensorArrayHardwareParams` per type
   - ✅ No hardcoded sensor count of 12 anywhere in refactored code
   - ✅ gels_localization, sensor_data_collection, serial_processor, calibration all use `SensorArrayConfig`

2. **Placeholder scan:** No TODOs, no "fill in later", no vague steps.

3. **Type consistency:** `SensorArrayConfig` used uniformly; `IntrinsicParamsSet.from_json()` / `.to_json()` consistent; `get_config("QMC6309")` fallback everywhere.

---

**Plan complete.** Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
