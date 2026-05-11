# Center Field Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable module that applies R_CORR correction and estimates local center magnetic field `b_hat` per row using the gels_localization null-space method.

**Architecture:** Single class `CenterFieldEstimator` in `calibration/lib/center_field_estimator.py`. Precomputes weight vector `w` from fixed `d_list` on init. R_CORR loaded from `SensorArrayConfig`. Uses null-space projection to estimate center field.

**Tech Stack:** Python, numpy, sensor_array_config (existing)

---

## File Structure

| File | Purpose |
|------|---------|
| `src/calibration/lib/center_field_estimator.py` | **New** — main module with `CenterFieldEstimator` class |
| `src/calibration/lib/__init__.py` | **Modify** — export `CenterFieldEstimator` |
| `src/calibration/scripts/test_center_field_estimator.py` | **New** — verification script using manual_record_5V.csv |

---

## Task 1: Create `CenterFieldEstimator` Class Skeleton

**Files:**
- Create: `src/calibration/lib/center_field_estimator.py`
- Test: `src/calibration/scripts/test_center_field_estimator.py`

- [ ] **Step 1: Write the failing test**

```python
# src/calibration/scripts/test_center_field_estimator.py
import numpy as np
from calibration.lib.center_field_estimator import CenterFieldEstimator

def test_estimator_init():
    """Test that estimator initializes and precomputes w."""
    estimator = CenterFieldEstimator()
    assert estimator.w is not None
    assert estimator.w.shape == (12, 1)
    print("test_estimator_init PASSED")

def test_estimator_basic():
    """Test that estimate_from_row returns a (3,) vector."""
    estimator = CenterFieldEstimator()
    b_raw = np.random.randn(36)  # one row
    b_hat = estimator.estimate_from_row(b_raw)
    assert b_hat.shape == (3,)
    print("test_estimator_basic PASSED")

if __name__ == "__main__":
    test_estimator_init()
    test_estimator_basic()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python src/calibration/scripts/test_center_field_estimator.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'calibration'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/calibration/lib/center_field_estimator.py
import numpy as np
import sys
from pathlib import Path

# Add calibration lib path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from sensor_array_config.base import get_config


def null(A: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Computes null space of matrix A. Same as gels_localization."""
    _, s, Vh = np.linalg.svd(A, full_matrices=True)
    n = A.shape[1]
    rank = np.sum(s > rcond * s[0]) if s.size > 0 else 0
    k = n - rank
    if k > 0:
        return Vh[rank:, :].T
    else:
        return np.zeros((n, 0))


class CenterFieldEstimator:
    """Estimates local center magnetic field from sensor array measurements."""

    def __init__(self, sensor_config=None):
        if sensor_config is None:
            sensor_config = get_config("QMC6309")
        self.sensor_config = sensor_config
        self.d_list = np.array(sensor_config.hardware.d_list)  # (12, 3)
        self._load_r_corr()
        self._precompute_weight_vector()

    def _load_r_corr(self):
        """Load R_CORR matrices from sensor config into dict sensor_id -> np.array(3,3)."""
        self.R_CORR = {}
        for entry in self.sensor_config.hardware.R_CORR:
            mat = np.array(entry.matrix).reshape(3, 3, order='F')
            for sid in entry.sensor_ids:
                self.R_CORR[sid] = mat

    def _precompute_weight_vector(self):
        """Precompute w from d_list null space. Called once on init."""
        D_cal = self.d_list.T  # (3, 12)
        Q = null(D_cal)
        ones_N = np.ones((12, 1))
        g = Q.T @ ones_N
        g_norm_sq = float(g.T @ g)
        if g_norm_sq < 1e-10:
            raise ValueError("Cannot construct w: 1 has zero projection onto null(D_cal)")
        self.w = (Q @ g) / g_norm_sq  # (12, 1)

    def estimate_from_row(self, b_raw_row):
        """Estimate center field for a single row."""
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python src/calibration/scripts/test_center_field_estimator.py`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 5: Commit**

```bash
git add src/calibration/lib/center_field_estimator.py src/calibration/scripts/test_center_field_estimator.py
git commit -m "feat(center_field): add CenterFieldEstimator class skeleton

- Add null() helper matching gels_localization
- Add CenterFieldEstimator with d_list and R_CORR loading
- Add _precompute_weight_vector() using null(D_cal)"
```

---

## Task 2: Implement `apply_r_corr` and `estimate_from_row`

**Files:**
- Modify: `src/calibration/lib/center_field_estimator.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to test_center_field_estimator.py
def test_r_corr_shape():
    """Test apply_r_corr preserves shape."""
    estimator = CenterFieldEstimator()
    b_raw = np.random.randn(12, 3)
    b_rcorr = estimator.apply_r_corr(b_raw)
    assert b_rcorr.shape == (12, 3)
    print("test_r_corr_shape PASSED")

def test_estimate_known_row():
    """Test estimate_from_row returns finite values."""
    estimator = CenterFieldEstimator()
    # Use first row of actual data (approximate values from manual_record_5V.csv)
    # Just check shape and finiteness
    b_raw = np.random.randn(36)
    b_hat = estimator.estimate_from_row(b_raw)
    assert b_hat.shape == (3,)
    assert np.all(np.isfinite(b_hat))
    print("test_estimate_known_row PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python src/calibration/scripts/test_center_field_estimator.py`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement `apply_r_corr`**

Add to `CenterFieldEstimator` class:

```python
def apply_r_corr(self, b_raw):
    """Apply R_CORR to raw sensor data.

    Args:
        b_raw: (N, 3) or (N*3,) raw sensor readings

    Returns:
        b_rcorr: (N, 3) with R_CORR applied per sensor
    """
    if b_raw.ndim == 1:
        b_raw = b_raw.reshape(12, 3)
    b_rcorr = np.zeros_like(b_raw)
    for sid in range(1, 13):
        sensor_idx = sid - 1
        if sid in self.R_CORR:
            b_rcorr[sensor_idx] = self.R_CORR[sid] @ b_raw[sensor_idx]
        else:
            b_rcorr[sensor_idx] = b_raw[sensor_idx]
    return b_rcorr
```

- [ ] **Step 4: Implement `estimate_from_row`**

Add to `CenterFieldEstimator` class:

```python
def estimate_from_row(self, b_raw_row):
    """Estimate center field for a single row.

    Args:
        b_raw_row: (36,) or (12, 3) raw sensor data

    Returns:
        b_hat: (3,) center field estimate
    """
    b_rcorr = self.apply_r_corr(b_raw_row)
    if b_rcorr.ndim == 2 and b_rcorr.shape[0] == 12:
        B_meas = b_rcorr.T  # (3, 12)
    else:
        raise ValueError(f"Unexpected shape after R_CORR: {b_rcorr.shape}")
    b_hat = B_meas @ self.w  # (3, 1) -> (3,)
    return b_hat.ravel()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python src/calibration/scripts/test_center_field_estimator.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/calibration/lib/center_field_estimator.py
git commit -m "feat(center_field): implement apply_r_corr and estimate_from_row

- apply_r_corr: per-sensor R_CORR rotation using sensor config
- estimate_from_row: B_meas @ w -> b_hat (3,)"
```

---

## Task 3: Implement `estimate_batch`

**Files:**
- Modify: `src/calibration/lib/center_field_estimator.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to test_center_field_estimator.py
def test_batch():
    """Test estimate_batch processes multiple rows."""
    estimator = CenterFieldEstimator()
    # 5 rows of data
    b_raw = np.random.randn(5, 36)
    b_hats = estimator.estimate_batch(b_raw)
    assert b_hats.shape == (5, 3)
    assert np.all(np.isfinite(b_hats))
    print(f"test_batch PASSED: {b_hats.shape}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python src/calibration/scripts/test_center_field_estimator.py::test_batch -v`
Expected: FAIL — `AttributeError: 'CenterFieldEstimator' object has no attribute 'estimate_batch'`

- [ ] **Step 3: Implement `estimate_batch`**

Add to `CenterFieldEstimator` class:

```python
def estimate_batch(self, b_raw):
    """Estimate center field for multiple rows.

    Args:
        b_raw: (N, 36) or (N, 12, 3) raw sensor data

    Returns:
        b_hats: (N, 3) center field estimates
    """
    if b_raw.ndim == 2 and b_raw.shape[1] == 36:
        b_raw = b_raw.reshape(b_raw.shape[0], 12, 3)

    N = b_raw.shape[0]
    b_hats = np.zeros((N, 3))
    for i in range(N):
        b_hats[i] = self.estimate_from_row(b_raw[i])
    return b_hats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python src/calibration/scripts/test_center_field_estimator.py::test_batch -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/calibration/lib/center_field_estimator.py
git commit -m "feat(center_field): add estimate_batch for N-row processing"
```

---

## Task 4: Verify with Real Data (manual_record_5V.csv)

**Files:**
- Modify: `src/calibration/scripts/test_center_field_estimator.py`

- [ ] **Step 1: Write the verification test**

```python
# Add to test_center_field_estimator.py
def test_real_data():
    """Verify with actual manual_record_5V.csv data."""
    import pandas as pd
    from calibration.lib.center_field_estimator import CenterFieldEstimator

    csv_path = Path(__file__).parent.parent / '../../sensor_data_collection/data/manual_x/manual_record_5V.csv'
    df = pd.read_csv(csv_path)
    b_raw = df.values[:100]  # first 100 rows

    estimator = CenterFieldEstimator()
    b_hats = estimator.estimate_batch(b_raw)

    print(f"Input shape: {b_raw.shape}")
    print(f"Output shape: {b_hats.shape}")
    print(f"b_hat stats: mean={np.mean(b_hats, axis=0)}, std={np.std(b_hats, axis=0)}")
    print(f"b_hat magnitude: mean={np.mean(np.linalg.norm(b_hats, axis=1)):.2f}")

    # Sanity: magnitude should be in reasonable range (10-100 µT)
    mags = np.linalg.norm(b_hats, axis=1)
    assert np.all(mags > 1.0), f"Magnitude too small: {mags.min()}"
    assert np.all(mags < 200.0), f"Magnitude too large: {mags.max()}"
    print("test_real_data PASSED")
```

- [ ] **Step 2: Run verification**

Run: `python src/calibration/scripts/test_center_field_estimator.py::test_real_data -v`
Expected: PASS — prints b_hat statistics

- [ ] **Step 3: Commit**

```bash
git add src/calibration/scripts/test_center_field_estimator.py
git commit -m "test(center_field): add real data verification with manual_record_5V.csv"
```

---

## Task 5: Update `__init__.py` Export

**Files:**
- Modify: `src/calibration/lib/__init__.py`

- [ ] **Step 1: Check current exports**

Run: `head -20 src/calibration/lib/__init__.py`

- [ ] **Step 2: Add export**

Add line:
```python
from .center_field_estimator import CenterFieldEstimator
```

- [ ] **Step 3: Commit**

```bash
git add src/calibration/lib/__init__.py
git commit -m "feat: export CenterFieldEstimator from calibration.lib"
```

---

## Self-Review Checklist

- [ ] Spec coverage: All requirements from design doc implemented?
  - R_CORR loading from sensor config ✓
  - apply_r_corr per-sensor ✓
  - estimate_from_row returns (3,) ✓
  - estimate_batch returns (N, 3) ✓
  - Precomputed w ✓
- [ ] Placeholder scan: No TBD/TODO/NotImplementedError remaining ✓
- [ ] Type consistency: `w.shape == (12, 1)`, `b_hat.shape == (3,)`, `b_hats.shape == (N, 3)` ✓
- [ ] All tests pass ✓
- [ ] All tasks committed ✓
