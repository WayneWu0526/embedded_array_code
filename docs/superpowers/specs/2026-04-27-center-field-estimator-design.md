# Center Field Estimator Module Design

## Overview

Build a reusable module that, given one row of raw sensor data (12 sensors × 3 axes = 36 values), applies R_CORR correction to align all sensors to a common frame, then estimates the local center magnetic field `b_hat` using the gels_localization null-space method.

## Data Flow

```
b_raw_row (36,) or (12, 3)
    │
    ▼
R_CORR Correction (per-sensor 3×3 rotation)
    │ Each sensor group (1-3, 4-6, 7-9, 10-12) gets its own R_CORR
    ▼
b_RCORR (12, 3) — all sensors now in unified reference frame
    │
    ▼
Transpose → B_meas (3, 12) — gels_localization expected format
    │
    ▼
b_hat = B_meas @ w  (weight vector w from null(D_cal))
    │
    ▼
b_hat (3,) — center field estimate for this row
```

## Key Components

### 1. Precomputed Weight Vector `w`

Since `d_list` (sensor positions) is fixed, `w` can be computed once:

```python
D_cal = d_list.T           # (3, 12)
Q = null(D_cal)             # (12, N-r)
w = (Q @ (Q.T @ ones_12)) / ||Q.T @ ones_12||²   # (12, 1)
```

`w` satisfies: `D_cal @ w = 0` (null-space) and `1^T w = 1` (sums to 1).

### 2. R_CORR Correction

Loaded from `sensor_array_config` hardware params, structured as:

| Group | Sensor IDs | R_CORR Matrix |
|-------|-----------|---------------|
| 0 | 1, 2, 3 | `[[1,0,0],[0,0,-1],[0,1,0]]` |
| 1 | 4, 5, 6 | `[[1,0,0],[0,0,1],[0,-1,0]]` |
| 2 | 7, 8, 9 | `[[-1,0,0],[0,0,1],[0,1,0]]` |
| 3 | 10,11,12| `[[-1,0,0],[0,0,-1],[0,-1,0]]` |

Matrices are stored as 9-element row-major (Fortran order in JSON).

### 3. Module Interface

```python
# src/calibration/lib/center_field_estimator.py

class CenterFieldEstimator:
    """Estimates local center magnetic field from sensor array measurements."""

    def __init__(self, sensor_config: SensorArrayConfig = None):
        """Initialize with sensor config (loads d_list and R_CORR)."""
        if sensor_config is None:
            sensor_config = get_config("QMC6309")

        self.d_list = np.array(sensor_config.hardware.d_list)  # (12, 3)
        self._precompute_weight_vector()

    def _precompute_weight_vector(self):
        """Precompute w once from d_list."""
        D_cal = self.d_list.T  # (3, 12)
        Q = null(D_cal)
        ones_N = np.ones((12, 1))
        g = Q.T @ ones_N
        g_norm_sq = float(g.T @ g)
        self.w = (Q @ g) / g_norm_sq  # (12, 1)

    def apply_r_corr(self, b_raw: np.ndarray) -> np.ndarray:
        """Apply R_CORR to raw sensor data.

        Args:
            b_raw: (N, 3) or (36,) raw sensor readings

        Returns:
            b_rcorr: (N, 3) with R_CORR applied
        """
        ...

    def estimate_from_row(self, b_raw_row: np.ndarray) -> np.ndarray:
        """Estimate center field for a single row.

        Args:
            b_raw_row: (36,) or (12, 3) raw sensor data

        Returns:
            b_hat: (3,) center field estimate
        """
        ...

    def estimate_batch(self, b_raw: np.ndarray) -> np.ndarray:
        """Estimate center field for multiple rows.

        Args:
            b_raw: (N, 36) or (N, 12, 3) raw sensor data

        Returns:
            b_hats: (N, 3) center field estimates
        """
        ...
```

## File Location

`src/calibration/lib/center_field_estimator.py`

New file, not modifying existing ellipsoid_fit or consistency_fit.

## Dependencies

- `numpy`
- `sensor_array_config.base` (SensorArrayConfig, get_config)
- Reference: `gels_localization/scripts/maps_estimator.py` null-space logic (lines 62-83)

## Verification

1. Load `manual_record_5V.csv` (N rows × 36 cols)
2. Run each row through `CenterFieldEstimator.estimate_from_row()`
3. Verify output shape: N × 3
4. Sanity check: `b_hat` magnitude should be close to expected field magnitude (~25-50 µT for typical lab environment)

## Next Steps

After this module is built and verified:
- Use the N × 3 `b_hat` array as reference for full matrix C and offset o calibration
- Compare with existing ellipsoid fitting approach
