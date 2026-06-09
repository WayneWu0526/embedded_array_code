# Remove Legacy Calibration Pipeline — Keep Only Normalized Method

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ellipsoid and consistency calibration code and data, keeping only the normalized calibration pipeline (R_CORR + normalized D_i, e_i).

**Architecture:** The sensor array calibration pipeline is simplified to a single stage: `b_raw → R_CORR → D_i,e_i (normalized) → b_corrected`. All Phase 1 (ellipsoid) and Phase 2 (consistency) classes, configs, and scripts are removed.

**Tech Stack:** ROS Noetic, Python numpy, sensor_array_config package

---

## File Structure Changes

### Files to DELETE (no longer needed):

| File/Directory | Reason |
|----------------|--------|
| `src/sensor_array_config/sensor_array_config/config/qmc6309/consistency_params.json` | Replaced by normalized_params.json |
| `src/sensor_array_config/sensor_array_config/config/qmc6309/intrinsic_params.json` | Ellipsoid params, no longer used |
| `src/calibration/scripts/calibration_fit.py` | Old consistency calibration script |
| `src/calibration/scripts/consistency_calibration.py` | Old consistency calibration script |
| `src/calibration/src/calibration/consistency_fit.py` | Old consistency fit library |

### Files to MODIFY:

| File | Changes |
|------|---------|
| `src/sensor_array_config/sensor_array_config/base.py` | Remove `ConsistencyParams`, `ConsistencyParamsSet`, `IntrinsicParams`, `IntrinsicParamsSet`. Keep `NormalizedParamsSet`, `SensorArrayManifest`, `SensorArrayHardwareParams`, `RCorrEntry`, `SensorArrayConfig`. |
| `src/sensor_array_config/sensor_array_config/qmc6309.py` | Remove `intrinsic` and `consistency` properties. Keep `manifest`, `hardware`, `normalized`, `gs_to_si`. |
| `src/serial_processor/scripts/serial_node_tdm.py` | Already uses normalized pipeline. Verify no references to `consistency` or `intrinsic`. |
| `src/calibration/scripts/calibration_fit_normalized.py` | Already produces normalized_params.json. Update to remove old comments. |

---

## Task 1: Clean up base.py — Remove unused classes

**Files:**
- Modify: `src/sensor_array_config/sensor_array_config/base.py`

- [ ] **Step 1: Identify all classes in base.py**

Read the file. It should have these classes:
- `SensorArrayManifest` — KEEP
- `IntrinsicParams` — REMOVE
- `IntrinsicParamsSet` — REMOVE
- `ConsistencyParams` — REMOVE
- `ConsistencyParamsSet` — REMOVE
- `NormalizedParamsSet` — KEEP
- `RCorrEntry` — KEEP
- `SensorArrayHardwareParams` — KEEP
- `SensorArrayConfig` — KEEP (update abstract properties)

- [ ] **Step 2: Remove unused classes**

Delete these class definitions entirely:
- `IntrinsicParams` (the @dataclass with `o_i`, `C_i`)
- `IntrinsicParamsSet` (the @dataclass with `params: Dict[int, IntrinsicParams]`)
- `ConsistencyParams` (the @dataclass with `D_i`, `e_i`)
- `ConsistencyParamsSet` (the @dataclass with `params: Dict[int, ConsistencyParams]`, `amp_factor`)

Keep:
- `NormalizedParamsSet`
- `RCorrEntry`
- `SensorArrayHardwareParams`
- `SensorArrayManifest`
- `SensorArrayConfig`

- [ ] **Step 3: Update SensorArrayConfig abstract class**

The abstract class currently has:
```python
@property
@abstractmethod
def intrinsic(self) -> IntrinsicParamsSet: ...

@property
@abstractmethod
def consistency(self) -> ConsistencyParamsSet: ...
```

Remove these two abstract properties. The class should only have:
```python
@property
@abstractmethod
def manifest(self) -> SensorArrayManifest: ...

@property
@abstractmethod
def hardware(self) -> SensorArrayHardwareParams: ...

@property
@abstractmethod
def normalized(self) -> NormalizedParamsSet: ...

@property
@abstractmethod
def gs_to_si(self) -> float: ...
```

- [ ] **Step 4: Verify syntax**

Run: `python3 -m py_compile src/sensor_array_config/sensor_array_config/base.py`

- [ ] **Step 5: Commit**

```bash
git add src/sensor_array_config/sensor_array_config/base.py
git commit -m "refactor(sensor_array_config): remove IntrinsicParams, ConsistencyParams, and unused abstract properties"
```

---

## Task 2: Clean up qmc6309.py — Remove unused properties

**Files:**
- Modify: `src/sensor_array_config/sensor_array_config/qmc6309.py`

- [ ] **Step 1: Review current qmc6309.py**

Read the file. It currently has:
- `manifest` property — KEEP
- `intrinsic` property — REMOVE
- `consistency` property — REMOVE
- `hardware` property — KEEP
- `normalized` property — KEEP
- `gs_to_si` property — KEEP

- [ ] **Step 2: Remove `intrinsic` and `consistency` properties**

Delete these from `QMC6309Config`:
```python
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
```

- [ ] **Step 3: Update import**

The import from `.base` currently includes `IntrinsicParamsSet` and `ConsistencyParamsSet`. Update to:
```python
from .base import (
    SensorArrayConfig, SensorArrayManifest,
    NormalizedParamsSet, SensorArrayHardwareParams
)
```

- [ ] **Step 4: Verify syntax**

Run: `python3 -m py_compile src/sensor_array_config/sensor_array_config/qmc6309.py`

- [ ] **Step 5: Commit**

```bash
git add src/sensor_array_config/sensor_array_config/qmc6309.py
git commit -m "refactor(sensor_array_config): remove intrinsic and consistency properties from QMC6309Config"
```

---

## Task 3: Delete legacy calibration files

**Files to DELETE:**

- [ ] **Step 1: Delete consistency_params.json**

```bash
rm src/sensor_array_config/sensor_array_config/config/qmc6309/consistency_params.json
git add src/sensor_array_config/sensor_array_config/config/qmc6309/consistency_params.json
git commit -m "chore: remove consistency_params.json (replaced by normalized_params.json)"
```

- [ ] **Step 2: Delete intrinsic_params.json**

```bash
rm src/sensor_array_config/sensor_array_config/config/qmc6309/intrinsic_params.json
git add src/sensor_array_config/sensor_array_config/config/qmc6309/intrinsic_params.json
git commit -m "chore: remove intrinsic_params.json (ellipsoid calibration no longer used)"
```

- [ ] **Step 3: Delete calibration_fit.py**

```bash
rm src/calibration/scripts/calibration_fit.py
git add src/calibration/scripts/calibration_fit.py
git commit -m "chore: remove calibration_fit.py (old consistency calibration script)"
```

- [ ] **Step 4: Delete consistency_calibration.py**

```bash
rm src/calibration/scripts/consistency_calibration.py
git add src/calibration/scripts/consistency_calibration.py
git commit -m "chore: remove consistency_calibration.py (old consistency calibration script)"
```

- [ ] **Step 5: Delete consistency_fit.py**

```bash
rm src/calibration/src/calibration/consistency_fit.py
git add src/calibration/src/calibration/consistency_fit.py
git commit -m "chore: remove consistency_fit.py (old consistency fit library)"
```

---

## Task 4: Verify serial_node_tdm.py has no legacy references

**Files:**
- Verify: `src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 1: Check for any remaining references**

Search for these terms — all should return 0 results:
```bash
grep -n "intrinsic\|consistency\|amp_factor\|_load_calibration_params\|_apply_ellipsoid\|ellipsoid" src/serial_processor/scripts/serial_node_tdm.py
```

If any remain, remove them.

- [ ] **Step 2: Verify R_CORR loading**

The file should only load R_CORR (from hardware) and normalized D_i, e_i (from normalized). Verify:
- `_load_sensor_array_params()` loads `hardware.R_CORR` ✅
- D_matrix/e_bias come from `sensor_config.normalized` ✅

- [ ] **Step 3: Verify syntax**

Run: `python3 -m py_compile src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 4: Commit if changed**

```bash
git add src/serial_processor/scripts/serial_node_tdm.py
git commit -m "refactor(serial_node): verify clean of legacy calibration references"
```

---

## Task 5: Update calibration_fit_normalized.py comments

**Files:**
- Modify: `src/calibration/scripts/calibration_fit_normalized.py`

- [ ] **Step 1: Clean up docstring**

Update the module docstring (lines 1-11) to remove references to "Comparison output: residual_std_normalized.csv vs residual_std_table.csv" since we're removing the old method. Replace with a clean description of the normalized calibration pipeline.

- [ ] **Step 2: Verify output path is correct**

The output path at line 191 should point to `normalized_params.json` in the sensor_array_config directory.

- [ ] **Step 3: Verify syntax**

Run: `python3 -m py_compile src/calibration/scripts/calibration_fit_normalized.py`

- [ ] **Step 4: Commit**

```bash
git add src/calibration/scripts/calibration_fit_normalized.py
git commit -m "chore(calibration): clean up docstring in calibration_fit_normalized.py"
```

---

## Task 6: Final verification — Build and syntax check

- [ ] **Step 1: Syntax check all sensor_array_config files**

Run:
```bash
python3 -m py_compile src/sensor_array_config/sensor_array_config/base.py
python3 -m py_compile src/sensor_array_config/sensor_array_config/qmc6309.py
python3 -m py_compile src/serial_processor/scripts/serial_node_tdm.py
python3 -m py_compile src/calibration/scripts/calibration_fit_normalized.py
```

Expected: All pass with no output.

- [ ] **Step 2: Verify no import errors**

Run:
```bash
cd /home/zhang/embedded_array_ws && python3 -c "from sensor_array_config import get_config; cfg = get_config('QMC6309'); print('manifest:', cfg.manifest.n_sensors); print('hardware R_CORR groups:', len(cfg.hardware.R_CORR)); print('normalized params:', len(cfg.normalized.params))"
```

Expected: Should print numbers without errors.

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "refactor: remove legacy ellipsoid/consistency calibration, keep normalized only"
```

---

## Summary: What Remains

| Component | Status |
|-----------|--------|
| R_CORR | ✅ Kept (from hardware) |
| NormalizedParamsSet | ✅ Kept |
| normalized D_i, e_i | ✅ Kept (from normalized_params.json) |
| Ellipsoid params (intrinsic) | ❌ Removed |
| Consistency params | ❌ Removed |
| calibration_fit.py | ❌ Removed |
| consistency_calibration.py | ❌ Removed |
| consistency_fit.py | ❌ Removed |

**New pipeline:**
```
b_raw → R_CORR[s] @ b_raw → D_i @ (...) + e_i → b_corrected
```
