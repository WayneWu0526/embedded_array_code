# Center Field Estimator — Subset Sensor Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `CenterFieldEstimator` to accept an optional `sensor_ids` parameter at init, enabling center field estimation using only a specified subset of sensors.

**Architecture:** Modify `__init__` to accept `sensor_ids` list, validate it, filter `d_list` and `R_CORR` to the selected subset, and recompute `w` from the subset's `D_cal`. No changes to `estimate_from_row` or `estimate_batch`.

**Tech Stack:** Python, numpy, sensor_array_config

---

## File Structure

| File | Purpose |
|------|---------|
| `src/calibration/lib/center_field_estimator.py` | **Modify** — add `sensor_ids` param to `__init__`, filter d_list/R_CORR, recompute w |
| `src/calibration/scripts/test_center_field_estimator.py` | **Modify** — add subset validation and usage tests |

---

## Task 1: Add `sensor_ids` Parameter and Validation to `__init__`

**Files:**
- Modify: `src/calibration/lib/center_field_estimator.py` (`__init__` and `_load_r_corr`)
- Modify: `src/calibration/scripts/test_center_field_estimator.py`

- [ ] **Step 1: Write the failing test**

Add to `test_center_field_estimator.py`:

```python
def test_init_with_sensor_ids():
    """Test init with valid sensor subset."""
    estimator = CenterFieldEstimator(sensor_ids=[1, 2, 3])
    assert estimator.w is not None
    assert estimator.w.shape[0] == 3  # 3 sensors
    print("test_init_with_sensor_ids PASSED")

def test_init_default_all_sensors():
    """Test init without sensor_ids uses all 12."""
    estimator = CenterFieldEstimator()
    assert estimator.w.shape[0] == 12
    print("test_init_default_all_sensors PASSED")

def test_invalid_sensor_id():
    """Test ValueError for out-of-range sensor ID."""
    try:
        CenterFieldEstimator(sensor_ids=[1, 2, 99])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Sensor ID 99 not found" in str(e)
        print("test_invalid_sensor_id PASSED")

def test_too_few_sensors():
    """Test ValueError when fewer than 3 sensors."""
    try:
        CenterFieldEstimator(sensor_ids=[1, 2])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "at least 3 sensors" in str(e)
        print("test_too_few_sensors PASSED")

def test_duplicate_sensor_ids():
    """Test ValueError for duplicate IDs."""
    try:
        CenterFieldEstimator(sensor_ids=[1, 1, 2, 3])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Duplicate" in str(e)
        print("test_duplicate_sensor_ids PASSED")
```

Update `if __name__ == "__main__":` to call these tests.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/zhang/embedded_array_ws && python src/calibration/scripts/test_center_field_estimator.py`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'sensor_ids'`

- [ ] **Step 3: Implement validation and subset support in `__init__`**

Replace the existing `__init__` and `_load_r_corr` methods in `center_field_estimator.py` with:

```python
def __init__(self, sensor_config=None, sensor_ids=None):
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    self.sensor_config = sensor_config
    self.full_d_list = np.array(sensor_config.hardware.d_list)  # (12, 3)

    # Default: all 12 sensors
    if sensor_ids is None:
        sensor_ids = list(range(1, 13))

    # Validate sensor_ids
    self._validate_sensor_ids(sensor_ids)

    # Filter d_list to selected sensors
    self.sensor_ids = sensor_ids
    indices = [sid - 1 for sid in sensor_ids]
    self.d_list = self.full_d_list[indices, :]  # (N_selected, 3)

    self._load_r_corr()
    self._precompute_weight_vector()

def _validate_sensor_ids(self, sensor_ids):
    """Validate sensor_ids list."""
    if not isinstance(sensor_ids, (list, tuple)):
        raise ValueError(f"sensor_ids must be a list or tuple of integers, got {type(sensor_ids).__name__}")

    if len(sensor_ids) == 0:
        raise ValueError(f"sensor_ids must have at least 3 sensors, got 0")

    if len(sensor_ids) < 3:
        raise ValueError(f"sensor_ids must have at least 3 sensors, got {len(sensor_ids)}")

    seen = set()
    for sid in sensor_ids:
        if not isinstance(sid, int):
            raise ValueError(f"sensor_ids must be integers in 1-12, got {type(sid).__name__}")
        if sid < 1 or sid > 12:
            raise ValueError(f"Sensor ID {sid} not found. Available: 1-12")
        if sid in seen:
            raise ValueError(f"Duplicate sensor IDs: {sensor_ids}")
        seen.add(sid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/zhang/embedded_array_ws && python src/calibration/scripts/test_center_field_estimator.py`
Expected: PASS (all new tests + existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/calibration/lib/center_field_estimator.py src/calibration/scripts/test_center_field_estimator.py
git commit -m "feat(center_field): add sensor_ids parameter to CenterFieldEstimator

- Validate sensor_ids in range 1-12, no duplicates, at least 3
- Filter d_list to selected sensors
- Default: all 12 sensors (backward compatible)
- Filter R_CORR to selected sensors in _load_r_corr"
```

---

## Task 2: Filter R_CORR to Selected Sensors

**Files:**
- Modify: `src/calibration/lib/center_field_estimator.py` (`_load_r_corr`)

- [ ] **Step 1: Verify _load_r_corr uses self.sensor_ids**

Read current `_load_r_corr` at line ~34-40 in `center_field_estimator.py`. The method currently iterates `sensor_config.hardware.R_CORR` and stores in `self.R_CORR`. We need to filter to only the selected sensor IDs.

Current implementation:
```python
def _load_r_corr(self):
    self.R_CORR = {}
    for entry in self.sensor_config.hardware.R_CORR:
        mat = np.array(entry.matrix).reshape(3, 3, order='F')
        for sid in entry.sensor_ids:
            self.R_CORR[sid] = mat
```

Replace with:
```python
def _load_r_corr(self):
    """Load R_CORR matrices for selected sensors only."""
    self.R_CORR = {}
    for entry in self.sensor_config.hardware.R_CORR:
        mat = np.array(entry.matrix).reshape(3, 3, order='F')
        for sid in entry.sensor_ids:
            if sid in self.sensor_ids:
                self.R_CORR[sid] = mat
```

- [ ] **Step 2: Run tests to verify they still pass**

Run: `cd /home/zhang/embedded_array_ws && python src/calibration/scripts/test_center_field_estimator.py`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/calibration/lib/center_field_estimator.py
git commit -m "feat(center_field): filter R_CORR to selected sensor_ids in _load_r_corr"
```

---

## Task 3: Verify Backward Compatibility

**Files:**
- Modify: `src/calibration/scripts/test_center_field_estimator.py`

- [ ] **Step 1: Add backward compatibility test**

Add to test file:
```python
def test_estimate_with_subset():
    """Test estimate_from_row with subset sensors."""
    import pandas as pd

    estimator = CenterFieldEstimator(sensor_ids=[1, 2, 3])
    csv_path = Path(__file__).resolve().parent.parent / '../../../sensor_data_collection/data/manual_x/manual_record_5V.csv'
    df = pd.read_csv(csv_path)
    b_raw = df.values[:10]  # 10 rows

    b_hats = estimator.estimate_batch(b_raw)
    assert b_hats.shape == (10, 3)
    assert np.all(np.isfinite(b_hats))
    mags = np.linalg.norm(b_hats, axis=1)
    assert np.all(mags > 1.0)
    print(f"test_estimate_with_subset PASSED: shape={b_hats.shape}, mean_mag={np.mean(mags):.2f}")
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd /home/zhang/embedded_array_ws && python src/calibration/scripts/test_center_field_estimator.py::test_estimate_with_subset -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/calibration/scripts/test_center_field_estimator.py
git commit -m "test(center_field): verify estimate_batch works with sensor subset"
```

---

## Self-Review Checklist

- [ ] Spec coverage: All requirements from design doc implemented?
  - sensor_ids parameter at init ✓
  - Validation: range 1-12 ✓
  - Validation: at least 3 sensors ✓
  - Validation: no duplicates ✓
  - d_list filtered to selected sensors ✓
  - R_CORR filtered to selected sensors ✓
  - w recomputed from subset D_cal ✓
  - Default all 12 sensors (backward compatible) ✓
- [ ] Placeholder scan: No TBD/TODO remaining ✓
- [ ] Type consistency: sensor_ids is list of int, w.shape[0] == len(sensor_ids) ✓
- [ ] All tests pass ✓
- [ ] All tasks committed ✓
