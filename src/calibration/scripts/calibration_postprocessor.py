#!/usr/bin/env python3
"""
Calibration Post-Processor - Runs after data sampling completes

Supports both:
- Phase 1: Ellipsoid calibration (handheld/ellipsoid data)
- Phase 2: Consistency calibration (CH1/CH2/CH3 multi-channel data)

Usage (Phase 1 - ellipsoid):
    from calibration_postprocessor import CalibrationPostProcessor
    processor = CalibrationPostProcessor(
        csv_path='/path/to/ellipsoid_calib_20260415_120000.csv',
        calibration_type='ellipsoid'
    )
    processor.run()

Usage (Phase 2 - consistency):
    from calibration_postprocessor import ConsistencyPostProcessor
    processor = ConsistencyPostProcessor(
        csv_dir='/path/to/consistency/',
        calibration_type='consistency'
    )
    processor.run()
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Add workspace src root to path
script_dir = Path(__file__).parent
src_root = script_dir.parent.parent
sys.path.insert(0, str(src_root))

from sensor_array_config import get_config, SensorArrayConfig
from calibration.lib.ellipsoid_fit import ellipsoid_fit
from calibration.lib.consistency_fit import (
    consistency_fit,
    batch_consistency_fit,
    validate_consistency,
    save_consistency_params,
    ConsistencyResult,
)


# ============== Phase 1: Ellipsoid Post-Processor ==============

def apply_calibration_ellipsoid(b_raw: np.ndarray, o_i: np.ndarray, C_i: np.ndarray) -> np.ndarray:
    """
    Apply ellipsoid calibration parameters to raw data.
    Formula: b_corr = C_i * (b_raw - o_i)
    """
    b_raw = np.asarray(b_raw)
    o_i = np.asarray(o_i)
    C_i = np.asarray(C_i)
    b_centered = b_raw - o_i
    b_corr = b_centered @ C_i.T
    return b_corr


class CalibrationPostProcessor:
    """
    Post-processor for Phase 1 ellipsoid calibration.

    Reads the collected CSV, performs ellipsoid calibration,
    and outputs results to sensor_array_config/config/{sensor_type}/intrinsic_params.json.
    """

    def __init__(self, csv_path: str, calibration_type: str = 'ellipsoid', sensor_type: str = 'QMC6309'):
        """
        Initialize the post-processor.

        Args:
            csv_path: Path to the CSV file created by sampling
            calibration_type: 'ellipsoid' or 'handheld'
            sensor_type: Sensor type name, used as subdirectory under sensor_array_config/config/
        """
        self.csv_path = Path(csv_path)
        self.calibration_type = calibration_type
        self.sensor_type = sensor_type
        self._sensor_config = get_config(sensor_type)

        # Setup paths
        self.project_dir = script_dir.parent
        self.report_dir = self.project_dir / 'report'
        self.config_dir = src_root / 'sensor_array_config' / 'config' / self.sensor_type
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.csv_name = self.csv_path.stem

    def run(self):
        """Execute the full post-processing pipeline."""
        print("\n" + "=" * 70)
        print("Phase 1: 单颗传感器椭球校准 (Ellipsoid Fitting)")
        print("=" * 70)
        print(f"\n数据来源: {self.csv_name}.csv")
        print(f"校准类型: {self.calibration_type}")
        print()

        if not self.csv_path.exists():
            print(f"[ERROR] CSV file not found: {self.csv_path}")
            return

        df = pd.read_csv(self.csv_path)
        print(f"读取数据点数: {len(df)}")

        print("\n正在校准 12 颗传感器...")
        print("-" * 70)
        print(f"{'传感器':^6} {'ratio':^8} {'改善倍数':^10} {'radius_std: 校正前 → 校正后'}")
        print("-" * 70)

        for sensor_id in self._sensor_config.get_sensor_ids():
            col_x = f'sensor_{sensor_id}_x'
            col_y = f'sensor_{sensor_id}_y'
            col_z = f'sensor_{sensor_id}_z'

            if col_x not in df.columns:
                print(f"[WARN] Sensor {sensor_id} columns not found")
                continue

            b_raw = df[[col_x, col_y, col_z]].values.astype(float)
            valid_mask = ~(np.isnan(b_raw).any(axis=1) | np.isinf(b_raw).any(axis=1))
            b_raw = b_raw[valid_mask]

            if len(b_raw) < 100:
                print(f"[WARN] Sensor {sensor_id}: insufficient valid data ({len(b_raw)} points)")
                continue

            result = ellipsoid_fit(b_raw, sensor_id, self.csv_name)
            self.results.append(result)

            print(f"  {sensor_id:2d}   "
                  f"{result.eigenvalue_ratio:6.2f}  {result.improvement_ratio:8.2f}x  "
                  f"{result.radius_raw_std:.4f} → {result.radius_corr_std:.4f}")

        self._save_intrinsic_params()
        self._save_calibration_results()
        self._print_summary()

    def _save_intrinsic_params(self):
        """Save intrinsic parameters to config/intrinsic_params.json (overwrite)."""
        from sensor_array_config.base import IntrinsicParamsSet, IntrinsicParams
        params_set = IntrinsicParamsSet(params={
            r.sensor_id: IntrinsicParams(o_i=r.o_i, C_i=r.C_i)
            for r in self.results
        })
        config_file = self.config_dir / 'intrinsic_params.json'
        params_set.to_json(str(config_file))
        print(f"\n内参已保存: {config_file}")

    def _save_calibration_results(self):
        """Save full calibration results to report/ directory."""
        calibration_params = {
            'version': '1.0',
            'description': f'{self.sensor_type} Sensor Array Phase 1 Calibration Full Results',
            'source_file': f'{self.csv_name}.csv',
            'calibration_type': self.calibration_type,
            'sensors': [r.to_dict() for r in self.results]
        }

        result_file = self.report_dir / f'calibration_params_{self.csv_name}.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_params, f, indent=2, ensure_ascii=False)

        print(f"结果已保存: {result_file}")

    def _print_summary(self):
        """Print calibration summary to terminal."""
        print("\n" + "=" * 70)
        print("校准结果汇总")
        print("=" * 70)

        if not self.results:
            print("\n无有效校准结果")
            return

        improvements = [r.improvement_ratio for r in self.results if r.improvement_ratio < float('inf')]
        if improvements:
            print(f"\n改善比率统计:")
            print(f"  Mean:   {np.mean(improvements):.2f}x")
            print(f"  Median: {np.median(improvements):.2f}x")
            print(f"  Min:    {np.min(improvements):.2f}x")
            print(f"  Max:    {np.max(improvements):.2f}x")

        print("\n" + "=" * 70)
        print("Phase 1 校准完成!")
        print(f"内参文件: {self.config_dir / 'intrinsic_params.json'}")
        print(f"结果目录: report/")
        print("=" * 70 + "\n")


# ============== Phase 2: Consistency Post-Processor ==============

class ConsistencyPostProcessor:
    """
    Post-processor for Phase 2 consistency calibration.

    Reads CSV files from consistency_calibration, performs consistency fitting,
    and outputs results to sensor_array_config/config/{sensor_type}/consistency_params.json.
    """

    def __init__(self, csv_dir: str = None, calibration_type: str = 'consistency', sensor_type: str = 'QMC6309'):
        """
        Initialize the post-processor.

        Args:
            csv_dir: Path to directory containing consistency CSV files.
                     Defaults to calibration/data/consistency/
            calibration_type: 'consistency' (only option for now)
            sensor_type: Sensor type name, used as subdirectory under sensor_array_config/config/
        """
        self.project_dir = script_dir.parent
        self.csv_dir = Path(csv_dir) if csv_dir else self.project_dir / 'data' / 'consistency'
        self.calibration_type = calibration_type
        self.sensor_type = sensor_type
        self._sensor_config = get_config(sensor_type)

        # Output: sensor_array_config/config/{sensor_type}/consistency_params.json
        self.config_dir = src_root / 'sensor_array_config' / 'config' / self.sensor_type
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.results = []

    def run(self):
        """Execute the full post-processing pipeline."""
        print("\n" + "=" * 70)
        print("Phase 2: 一致性校准 (Consistency Calibration)")
        print("=" * 70)
        print(f"\n数据来源: {self.csv_dir}")
        print()

        # Check CSV files exist
        required_files = [
            'consistency_calib_background.csv',
            'consistency_calib_ch1_positive.csv',
            'consistency_calib_ch1_negative.csv',
            'consistency_calib_ch2_positive.csv',
            'consistency_calib_ch2_negative.csv',
            'consistency_calib_ch3_positive.csv',
            'consistency_calib_ch3_negative.csv',
        ]

        missing = [f for f in required_files if not (self.csv_dir / f).exists()]
        if missing:
            print(f"[ERROR] Missing CSV files: {missing}")
            return

        # ========== Step 1: Consistency fitting ==========
        print("\n[Step 1] 执行一致性校准...")
        print("-" * 70)

        D_list, e_list, fit_info = consistency_fit(self.csv_dir, sensor_config=self._sensor_config)

        # ========== Step 2: Print results ==========
        print("\n  拟合参数:")
        print("  " + "-" * 65)
        print(f"  {'Sensor':<8} {'D_ix':<10} {'D_iy':<10} {'D_iz':<10} {'e_ix':<9} {'e_iy':<9} {'e_iz':<9}")
        print("  " + "-" * 65)

        results = []
        for i, (D, e) in enumerate(zip(D_list, e_list)):
            result = ConsistencyResult(
                sensor_id=i + 1,
                csv_file=str(self.csv_dir),
                D_i=D.tolist(),
                e_i=e.tolist(),
                d_i={'x': float(D[0,0]), 'y': float(D[1,1]), 'z': float(D[2,2])},
                fit_info=fit_info
            )
            results.append(result)
            print(f"  {i+1:<8} {D[0,0]:<10.4f} {D[1,1]:<10.4f} {D[2,2]:<10.4f} "
                  f"{e[0]:<+9.4f} {e[1]:<+9.4f} {e[2]:<+9.4f}")

        # ========== Step 3: Validation ==========
        print("\n[Step 2] 验证校正效果...")
        print("-" * 55)
        print(f"  {'Condition':<8} {'Axis':<6} {'校正前':<12} {'校正后':<12} {'改善'}")
        print("  " + "-" * 55)

        try:
            validation = validate_consistency(self.csv_dir, D_list, e_list, sensor_config=self._sensor_config)
            for i in range(len(validation['conditions'])):
                cond = validation['conditions'][i]
                axis = validation['axes'][i]
                std_b = validation['before'][i]
                std_a = validation['after'][i]
                imp = validation['improvement_pct'][i]
                print(f"  {cond:<8} {axis:<6} {std_b:<12.6f} {std_a:<12.6f} {imp:>+6.1f}%")
        except Exception as e:
            print(f"  [WARN] 验证失败: {e}")

        # ========== Step 4: Save params ==========
        print("\n[Step 3] 保存参数...")
        output_file = self.config_dir / 'consistency_params.json'
        save_consistency_params(results, output_file, sensor_config=self._sensor_config)

        print("\n" + "=" * 70)
        print("Phase 2 校准完成!")
        print(f"一致性参数: {output_file}")
        print("=" * 70 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calibration Post-Processor')
    subparsers = parser.add_subparsers(dest='phase', help='Phase to process')

    # Phase 1: ellipsoid
    ellipsoid_parser = subparsers.add_parser('ellipsoid', aliases=['s1'],
                                               help='Phase 1: Ellipsoid calibration')
    ellipsoid_parser.add_argument('csv_path', help='Path to the CSV file from sampling')
    ellipsoid_parser.add_argument('--type', '-t', default='ellipsoid',
                                  choices=['ellipsoid', 'handheld'],
                                  help='Calibration type')
    ellipsoid_parser.add_argument('--sensor-type', '-s', default='QMC6309',
                                  help='Sensor array type (default: QMC6309)')

    # Phase 2: consistency
    consistency_parser = subparsers.add_parser('consistency', aliases=['s2'],
                                                help='Phase 2: Consistency calibration')
    consistency_parser.add_argument('--csv-dir', '-d', default=None,
                                     help='Directory containing consistency CSV files')
    consistency_parser.add_argument('--sensor-type', '-s', default='QMC6309',
                                     help='Sensor array type (default: QMC6309)')

    args = parser.parse_args()

    if args.phase in ('ellipsoid', 's1'):
        processor = CalibrationPostProcessor(args.csv_path, args.type, args.sensor_type)
        processor.run()
    elif args.phase in ('consistency', 's2'):
        processor = ConsistencyPostProcessor(csv_dir=args.csv_dir, sensor_type=args.sensor_type)
        processor.run()
    else:
        parser.print_help()
