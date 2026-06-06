# MagGrad continuous analysis: maggrad_continuous_20260605_223648.csv

- File: `/Users/lawkaho/Developer/embedded_array_ws_Mi-Gels/src/sensor_data_collection/data/maggrad_continuous_20260605_223648.csv`
- Rows parsed: 72744
- Duration: 151.547 s
- Mean AK snapshot rate: 480.011 Hz
- AK seq gaps: 0
- dt mean/p50/p95/max: 2.083 / 2.390 / 2.636 / 17.335 ms

## Coil state distribution
- coil_1_on: 18237 rows, mean |B|=1.595 G
- coil_2_on: 18244 rows, mean |B|=6.786 G
- coil_3_on: 18156 rows, mean |B|=5.146 G
- all_off: 18107 rows, mean |B|=10.987 G

## Response vs all_off
- coil_1_on: mean per-sensor vector delta=10.216 G, max sensor delta=10.507 G
- coil_2_on: mean per-sensor vector delta=14.966 G, max sensor delta=15.455 G
- coil_3_on: mean per-sensor vector delta=12.273 G, max sensor delta=12.730 G

## Phase settling check
- coil_1_on: bins=50, first 50ms mean=4.732 G, last 100ms mean=0.833 G, delta=-3.899 G
- coil_2_on: bins=50, first 50ms mean=5.171 G, last 100ms mean=7.205 G, delta=2.034 G
- coil_3_on: bins=50, first 50ms mean=5.041 G, last 100ms mean=5.194 G, delta=0.153 G
- all_off: bins=50, first 50ms mean=8.289 G, last 100ms mean=11.691 G, delta=3.402 G

## TF completeness
- sensor_array_filt: 72744/72744 (100.00%) complete, pos_mean=[0.23244678507049976, 0.14853070909033878, 0.1859379693304743], pos_span=[0.0019289137089318342, 0.0010581279585486214, 0.0013260859570708161]
- diana7_em_tcp_filt: 72744/72744 (100.00%) complete, pos_mean=[0.19863249453156978, -0.08050235645876846, 0.23359574959138216], pos_span=[0.001917572031368997, 0.0013391482571880886, 0.0003129427005150731]
- arm1_em_tcp_filt: 72744/72744 (100.00%) complete, pos_mean=[0.01112917823693735, 0.2806235695208843, 0.2583619516251051], pos_span=[0.0032694083868736856, 0.0011329609808067298, 0.0010239419352953694]
- arm2_em_tcp_filt: 72744/72744 (100.00%) complete, pos_mean=[0.3177015271572755, 0.26705826731285554, 0.3332326273016997], pos_span=[0.0012937883957566676, 0.001266064994586813, 0.0010523122283755404]

## IMU raw
- accel raw norm mean/p50/p95: 8217.07 / 8217.04 / 8227.90

## Generated visualization
- `/Users/lawkaho/Developer/embedded_array_ws_Mi-Gels/src/sensor_data_collection/data/maggrad_continuous_20260605_223648_visualization.html`

## Phase label alignment check

The CSV coil labels are not aligned with the actual FY8300 phase. With the labels as recorded, `all_off` has the highest mean magnitude, which is physically unlikely for a clean off window.

A coarse host-time phase sweep suggests that adding about `0.73 s` to the label phase makes the off window lowest:

- offset ~= 0.735 s: coil_1_on=7.158 G, coil_2_on=5.058 G, coil_3_on=11.398 G, all_off=0.930 G
- offset ~= 0.730 s: coil_1_on=7.143 G, coil_2_on=5.072 G, coil_3_on=11.402 G, all_off=0.932 G

Conclusion: sensor stream and TF recording are healthy, but coil-state labels need hardware/host phase alignment before using this file for model fitting.
