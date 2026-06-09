#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import (
    CenterFieldEstimator,
    affine_results_payload,
    load_manual_record_sets,
    solve_per_sensor,
    write_affine_model_params,
)
from sensor_array_config import get_array_config


class AffineModelTest(unittest.TestCase):
    def test_center_field_estimator_loads_all_array_configs(self):
        for name in ("qmc6309_12ch_v1", "ak09973d_12ch_v1", "tmag3001_12ch_v1"):
            estimator = CenterFieldEstimator(sensor_config=get_array_config(name))
            self.assertEqual(estimator.n_sensors, 12)
            self.assertEqual(estimator.w.shape, (12, 1))

    def test_manual_record_column_mismatch_fails(self):
        estimator = CenterFieldEstimator(sensor_config=get_array_config("qmc6309_12ch_v1"))
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "manual_record_bad.csv"
            pd.DataFrame([[1.0, 2.0]]).to_csv(csv_path, index=False)
            with self.assertRaisesRegex(ValueError, "expected 36"):
                load_manual_record_sets(Path(tmpdir), estimator, "qmc6309_12ch_v1")

    def test_affine_payload_schema(self):
        results = {
            1: {
                "D": np.eye(3).tolist(),
                "e": [[0.1], [0.2], [0.3]],
                "rmse": 0.0,
            }
        }
        payload = affine_results_payload(results)
        self.assertEqual(list(payload), ["sensors"])
        self.assertEqual(payload["sensors"][0]["sensor_id"], 1)
        self.assertEqual(np.array(payload["sensors"][0]["D_i"]).shape, (3, 3))
        self.assertEqual(len(payload["sensors"][0]["e_i"]), 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "affine_model_params.json"
            write_affine_model_params(out_path, results)
            saved = json.loads(out_path.read_text())
            self.assertEqual(saved, payload)

    def test_solve_per_sensor_result_shapes(self):
        b_corr = np.array([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ])
        b_ref = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        results = solve_per_sensor(b_corr, b_ref)
        self.assertEqual(set(results), {1, 2})
        for params in results.values():
            self.assertEqual(np.array(params["D"]).shape, (3, 3))
            self.assertEqual(np.array(params["e"]).reshape(-1).shape, (3,))

    def test_cli_array_config_argument_does_not_prompt(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "calibration_affine_model.py"
        spec = importlib.util.spec_from_file_location("calibration_affine_model", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.select_array_config("ak09973d_12ch_v1"),
            "ak09973d_12ch_v1",
        )


if __name__ == "__main__":
    unittest.main()
