# calibration

当前标定流程说明在 calibration 包内：

```text
src/calibration/README.md
```

边界：

- `sensor_data_collection` 负责采集 `data/manual_calibration/manual_record_*.csv`。
- `calibration` 负责生成 `affine_model_params.json`。
- `serial_processor` 负责在线应用 R_CORR 和 affine 参数。

