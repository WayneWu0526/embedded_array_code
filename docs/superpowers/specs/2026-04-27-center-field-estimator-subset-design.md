# Center Field Estimator — Subset Sensor Selection

## Overview

Extend `CenterFieldEstimator` to allow restricting center field estimation to a subset of sensors, specified by sensor IDs at initialization time.

## Motivation

When validating calibration quality or comparing sensor groups, it is useful to estimate the center field using only a specific subset of sensors (e.g., sensors 1, 2, 3) rather than all 12.

## Design

### API Change

```python
class CenterFieldEstimator:
    def __init__(
        self,
        sensor_config=None,
        sensor_ids: List[int] = None,  # NEW parameter
    ):
        """Initialize the estimator.

        Args:
            sensor_config: SensorArrayConfig (default: QMC6309)
            sensor_ids: List of sensor IDs (1-12) to use for estimation.
                        Default [1,2,...,12] (all sensors).
                        Must have at least 3 sensors.
        """
```

### Behavior

1. **Default** (`sensor_ids=None`): Uses all 12 sensors, identical to current behavior.

2. **Validation**:
   - All IDs must be integers in range 1-12
   - No duplicate IDs
   - At least 3 sensors (required for a valid null-space solution)
   - Invalid IDs raise `ValueError` with message listing valid range

3. **Weight vector computation**: After selecting the subset of `d_list` rows, `D_cal_subset = d_list[sensor_ids-1].T` is used to compute `w_subset`. The computation is identical to the full case.

4. **R_CORR filtering**: Only R_CORR matrices for the selected sensor IDs are kept in `self.R_CORR`.

### Backward Compatibility

`CenterFieldEstimator()` with no arguments behaves exactly as before (all 12 sensors). No existing code breaks.

### Files Modified

- `src/calibration/lib/center_field_estimator.py`

### Error Messages

```
ValueError: sensor_ids must be a list of integers in range 1-12, got [0, 2, 3]
ValueError: sensor_ids must have at least 3 sensors, got 2
ValueError: Duplicate sensor IDs: [1, 1, 2]
ValueError: Sensor ID 99 not found. Available: 1-12
```

### Testing

- `test_init_with_sensor_ids()` — verify subset initialization
- `test_invalid_sensor_ids()` — verify validation errors
- `test_too_few_sensors()` — verify minimum 3-sensor requirement
- `test_duplicate_sensor_ids()` — verify duplicate detection
- `test_estimate_batch_with_subset()` — verify batch works with subset
- Existing tests continue to pass (backward compatibility)
