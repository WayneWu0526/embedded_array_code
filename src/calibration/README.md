# calibration

Offline calibration utilities for Mi-Gels sensor arrays.

This package generates calibration parameters. It does not apply calibration
online. Runtime application happens in `serial_processor`:

```text
STM32 raw ADU
-> adu_to_gs
-> R_CORR
-> stm_uplink_raw
-> affine D_i @ b + e_i
-> stm_uplink
```

## Manual Affine Calibration

1. Record manual calibration samples:

```bash
roslaunch sensor_data_collection stm32_manual.launch \
  array_config:=qmc6309_12ch_v1
```

Start and stop recording:

```bash
rostopic pub /maggrad_manual_record/record_trigger std_msgs/Bool "data: true" -1
rostopic pub /maggrad_manual_record/record_trigger std_msgs/Bool "data: false" -1
```

The default input directory is:

```text
data/manual_calibration/
```

2. Fit affine calibration:

```bash
rosrun calibration calibration_affine_model.py
```

If `--array-config` is omitted, the script lists available array configs and
prompts for a selection.

For non-interactive runs:

```bash
rosrun calibration calibration_affine_model.py \
  --array-config ak09973d_12ch_v1 \
  --data-dir $(pwd)/data/manual_calibration
```

3. Output:

```text
src/sensor_array_config/config/arrays/<array_config>/affine_model_params.json
```

The output schema is:

```json
{
  "sensors": [
    {
      "sensor_id": 1,
      "D_i": [[...], [...], [...]],
      "e_i": [...]
    }
  ]
}
```

## Boundary

- `calibration` computes and writes `affine_model_params.json`.
- `sensor_array_config` stores array geometry, R_CORR, and affine parameters.
- `serial_processor` loads those parameters and publishes calibrated
  `stm_uplink`.
- `sensor_data_collection` records input CSVs and runtime experiment data.
