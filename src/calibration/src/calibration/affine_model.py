"""Affine calibration helpers for sensor arrays."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


AFFINE_PARAM_COUNT = 12  # 9 matrix terms + 3 bias terms


@dataclass
class ManualRecordSet:
    b_raw_rs: np.ndarray
    b_ref: np.ndarray
    b_ref_norm: np.ndarray
    b_corr: np.ndarray
    mean_mag: float
    n_rows: int


def solve_per_sensor(b_corr_all, b_ref_all):
    n_total, n_sensor, _ = b_corr_all.shape
    results = {}
    eye3 = np.eye(3)
    for sid_idx in range(n_sensor):
        b_corr_i = b_corr_all[:, sid_idx, :]
        A = np.zeros((n_total * 3, AFFINE_PARAM_COUNT))
        b_vec = np.zeros(n_total * 3)
        for row_idx in range(n_total):
            b_c = b_corr_i[row_idx]
            b_r = b_ref_all[row_idx]
            for axis in range(3):
                row = row_idx * 3 + axis
                A[row, 0:3] = b_c[0] * eye3[axis]
                A[row, 3:6] = b_c[1] * eye3[axis]
                A[row, 6:9] = b_c[2] * eye3[axis]
                A[row, 9:12] = eye3[axis]
                b_vec[row] = b_r[axis]
        x, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        D = x[0:9].reshape(3, 3, order="F")
        e = x[9:12].reshape(3, 1)
        residuals = A @ x - b_vec
        rmse = np.sqrt(np.mean(residuals ** 2))
        results[sid_idx + 1] = {"D": D.tolist(), "e": e.tolist(), "rmse": float(rmse)}
    return results


def delta_o_per_row(b_raw_rs, estimator, D_arr, e_arr, b_ref_eval):
    n_rows = b_raw_rs.shape[0]
    n_sensors = estimator.n_sensors
    delta_o = np.zeros(n_rows)
    for row_idx in range(n_rows):
        b_corr_n = estimator._filter_to_selected_sensors(b_raw_rs[row_idx])
        o_n = np.zeros((n_sensors, 3))
        for sensor_idx in range(n_sensors):
            o_n[sensor_idx] = (
                D_arr[sensor_idx] @ b_corr_n[sensor_idx, :]
                + e_arr[sensor_idx]
                - b_ref_eval[row_idx]
            )
        o_bar_n = np.mean(o_n, axis=0)
        delta_o[row_idx] = np.sqrt(np.mean(np.sum((o_n - o_bar_n) ** 2, axis=1)))
    return delta_o


def delta_o_pre(b_raw_rs, estimator):
    n_rows = b_raw_rs.shape[0]
    delta_o = np.zeros(n_rows)
    for row_idx in range(n_rows):
        b_ref_n = estimator.estimate_from_row(b_raw_rs[row_idx])
        b_corr_n = estimator._filter_to_selected_sensors(b_raw_rs[row_idx])
        o_n = b_corr_n - b_ref_n
        o_bar_n = np.mean(o_n, axis=0)
        delta_o[row_idx] = np.sqrt(np.mean(np.sum((o_n - o_bar_n) ** 2, axis=1)))
    return delta_o


def load_manual_record_sets(data_dir: Path, estimator, array_config_name: str) -> Dict[str, ManualRecordSet]:
    csv_files = sorted(data_dir.glob("manual_record_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No manual_record_*.csv files found in {data_dir}")

    n_sensors = estimator.n_sensors
    expected_cols = n_sensors * 3
    records = {}
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        b_raw = df.values.astype(np.float64)
        if b_raw.shape[1] != expected_cols:
            raise ValueError(
                f"{csv_path} has {b_raw.shape[1]} columns, expected {expected_cols} "
                f"for array_config={array_config_name}"
            )
        b_raw_rs = b_raw.reshape(-1, n_sensors, 3)
        b_ref, b_corr = estimator.estimate_batch(b_raw)
        mag = np.linalg.norm(b_ref, axis=1)
        mean_mag = mag.mean()
        b_ref_norm = b_ref * (mean_mag / mag[:, None])
        records[csv_path.name] = ManualRecordSet(
            b_raw_rs=b_raw_rs,
            b_ref=b_ref,
            b_ref_norm=b_ref_norm,
            b_corr=b_corr,
            mean_mag=mean_mag,
            n_rows=b_raw.shape[0],
        )
    return records


def stack_record_sets(records: Dict[str, ManualRecordSet]):
    return {
        "b_raw_rs": np.concatenate([record.b_raw_rs for record in records.values()], axis=0),
        "b_ref": np.concatenate([record.b_ref for record in records.values()], axis=0),
        "b_ref_norm": np.concatenate([record.b_ref_norm for record in records.values()], axis=0),
        "b_corr": np.concatenate([record.b_corr for record in records.values()], axis=0),
    }


def result_arrays(results, n_sensors):
    D_arr = np.array([results[sid]["D"] for sid in range(1, n_sensors + 1)])
    e_arr = np.array([results[sid]["e"] for sid in range(1, n_sensors + 1)]).squeeze()
    return D_arr, e_arr


def affine_results_payload(results):
    return {
        "sensors": [
            {
                "sensor_id": int(sensor_id),
                "D_i": params["D"],
                "e_i": list(np.array(params["e"]).flatten()),
            }
            for sensor_id, params in sorted(results.items())
        ]
    }


def write_affine_model_params(path: Path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(affine_results_payload(results), f, indent=2)
