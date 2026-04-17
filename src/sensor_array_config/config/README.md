# Sensor Array Configuration Guide

This directory contains configuration bundles for each supported sensor array type. Each subdirectory (e.g., `qmc6309/`) is a self-contained configuration package for one sensor array.

---

## Adding a New Sensor Array Type

Suppose you want to add support for a new sensor array called `MY_SENSOR`.

### Step 1: Create Configuration Directory

```bash
mkdir -p src/sensor_array_config/config/my_sensor
```

### Step 2: Create `calibration_manifest.yaml`

This file contains the sensor array's hardware metadata:

```yaml
sensor_type: "MY_SENSOR"
bit_width: 16              # ADC bit width (e.g. 16 for 16-bit ADC)
range_gs: 500.0           # Full-scale range in Gauss
n_sensors: 16              # Number of sensors in the array
n_groups: 4                 # Number of R_CORR rotation groups
sensors_per_group: 4       # Sensors per group
gs_to_tesla: 1.0e-4       # Gauss to Tesla conversion
```

| Field | Description |
|-------|-------------|
| `bit_width` | ADC bit width (affects `adu_to_gs` computation) |
| `range_gs` | Full-scale range in Gauss |
| `n_sensors` | Total sensor count |
| `n_groups` | Number of R_CORR groups (typically 1 per EM coil channel) |
| `sensors_per_group` | Sensors per group (n_groups × sensors_per_group should = n_sensors) |
| `gs_to_tesla` | Unit conversion: `B_Tesla = B_Gauss × gs_to_tesla` |

The derived value `adu_to_gs` is computed automatically as:
```
adu_to_gs = range_gs / (2^(bit_width - 1))
```

### Step 3: Create `sensor_array_params.json`

Sensor positions and rotation correction matrices:

```json
{
  "d_list": [
    [-0.001, 0.00105, -0.001],
    ...
    // 16 entries, one per sensor, positions in meters relative to array center
  ],
  "R_CORR": [
    [1, 0, 0, 0, 0, -1, 0, 1, 0],
    // ... 4 groups × 9 elements (3x3 column-major rotation matrices)
  ]
}
```

### Step 4: Create `intrinsic_params.json`

Ellipsoid calibration parameters (Phase 1):

```json
{
  "1": {"o_i": [...], "C_i": [[...], [...], [...]]},
  "2": {"o_i": [...], "C_i": [[...], [...], [...]]},
  ...
  "16": {"o_i": [...], "C_i": [[...], [...], [...]]}
}
```

Each sensor entry:
- `o_i`: 3-element offset vector [ox, oy, oz]
- `C_i`: 3×3 correction matrix (row-major)

### Step 5: Create `consistency_params.json`

Consistency calibration parameters (Phase 2):

```json
{
  "1": {"D_i": [[...], [...], [...]], "e_i": [...]},
  "2": {"D_i": [[...], [...], [...]], "e_i": [...]},
  ...
  "16": {"D_i": [[...], [...], [...]], "e_i": [...]}
}
```

Each sensor entry:
- `D_i`: 3×3 scale matrix (row-major)
- `e_i`: 3-element bias vector

### Step 6: Create `sensor_calibration.yaml`

Unit conversion for this sensor type:

```yaml
gs_to_tesla: 1.0e-4
```

### Step 7: Implement `MY_SENSOR` Class

Create `src/sensor_array_config/my_sensor.py`:

```python
import os
from .base import (
    SensorArrayConfig, SensorArrayManifest,
    IntrinsicParamsSet, ConsistencyParamsSet, SensorArrayHardwareParams
)

_MY_SENSOR_ROOT = os.path.join(os.path.dirname(__file__), "config", "my_sensor")

class MySensorConfig(SensorArrayConfig):
    def __init__(self, config_root: str = _MY_SENSOR_ROOT):
        self._config_root = config_root

    @property
    def manifest(self) -> SensorArrayManifest:
        import yaml
        path = os.path.join(self._config_root, "calibration_manifest.yaml")
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
        import yaml
        path = os.path.join(self._config_root, "sensor_calibration.yaml")
        with open(path) as f:
            return yaml.safe_load(f)["gs_to_tesla"]
```

### Step 8: Register the New Type

In `src/sensor_array_config/base.py`, add to `_REGISTRY`:

```python
def _lazy_register():
    from .qmc6309 import QMC6309Config
    from .my_sensor import MySensorConfig  # <-- add this
    _REGISTRY["QMC6309"] = QMC6309Config
    _REGISTRY["MY_SENSOR"] = MySensorConfig  # <-- add this
_register_qmc6309()
```

### Step 9: Use the New Sensor Type

```bash
# Via roslaunch
roslaunch sensor_data_collection data_collection.launch sensor_type:=MY_SENSOR

# Via rosparam
rosparam set /data_collection_node/sensor_type MY_SENSOR
```

---

## Directory Structure

```
src/sensor_array_config/config/
├── README.md                    # This file
├── qmc6309/                    # QMC6309 sensor array bundle
│   ├── calibration_manifest.yaml
│   ├── intrinsic_params.json
│   ├── consistency_params.json
│   ├── sensor_array_params.json
│   └── sensor_calibration.yaml
└── my_sensor/                  # Example: MY_SENSOR bundle
    ├── calibration_manifest.yaml
    ├── intrinsic_params.json
    ├── consistency_params.json
    ├── sensor_array_params.json
    └── sensor_calibration.yaml
```

---

## Key Interfaces

### `SensorArrayManifest` (computed fields)

| Field | Description |
|-------|-------------|
| `adu_to_gs` | LSB value: `range_gs / (2^(bit_width - 1))` |

### `SensorArrayConfig` (abstract base)

| Property | Returns | Description |
|----------|---------|-------------|
| `manifest` | `SensorArrayManifest` | Metadata (n_sensors, bit_width, etc.) |
| `intrinsic` | `IntrinsicParamsSet` | Phase 1 ellipsoid params |
| `consistency` | `ConsistencyParamsSet` | Phase 2 consistency params |
| `hardware` | `SensorArrayHardwareParams` | d_list and R_CORR matrices |
| `gs_to_si` | `float` | Gs → Tesla conversion factor |

### Helper Methods

```python
config.get_sensor_ids()              # [1, 2, ..., n_sensors]
config.get_group_for_sensor(sid)     # 0-indexed group for sensor ID
config.get_sensors_in_group(group)  # list of sensor IDs in a group
```

---

## Backward Compatibility

All nodes default to `sensor_type:=QMC6309` if not specified, so existing launch files continue to work without modification.
