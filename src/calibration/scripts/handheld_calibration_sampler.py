#!/usr/bin/env python3
"""
Handheld Calibration Sampler

Two independent workflow switches:
    skip_sampling  : bypass data collection, use existing CSV
    skip_calibration: only evaluate, don't compute/save params

Mode Matrix:
    skip_sampling=false, skip_calibration=false  →  collect → fit → evaluate → save params
    skip_sampling=false, skip_calibration=true   →  collect → evaluate only
    skip_sampling=true,  skip_calibration=false  →  load CSV → fit → evaluate → save params
    skip_sampling=true,  skip_calibration=true   →  load CSV → evaluate only

See handheld_calibration.launch for usage examples.
"""
import sys
import rospy
import csv
import os
from pathlib import Path

# Add src to Python path for calibration.lib.ellipsoid_fit import
_src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from serial_processor.msg import StmUplink
from sensor_array_config import get_config, SensorArrayConfig


class HandheldCalibrationSampler:
    def __init__(self):
        rospy.init_node('handheld_calibration_sampler', anonymous=True)

        # ROS parameters
        self.samples_per_avg = rospy.get_param('~samples_per_avg', 10)
        self.max_samples = rospy.get_param('~max_samples', 500)
        self.skip_sampling = rospy.get_param('~skip_sampling', False)
        self.skip_calibration = rospy.get_param('~skip_calibration', False)
        self.csv_path_param = rospy.get_param('~csv_path', None)

        # Load sensor array configuration
        self._sensor_type = rospy.get_param('~sensor_type', 'QMC6309')
        self._sensor_config: SensorArrayConfig = get_config(self._sensor_type)
        self._n_sensors = self._sensor_config.manifest.n_sensors
        self.sample_count = 0

        rospy.loginfo(f"[Mode] skip_sampling={self.skip_sampling}, skip_calibration={self.skip_calibration}")
        rospy.loginfo(f"[Config] sensor_type={self._sensor_type}, n_sensors={self._n_sensors}")

        # Determine CSV path
        if self.skip_sampling:
            self.csv_path = Path(self.csv_path_param) if self.csv_path_param else self._get_default_csv_path()
            if not self.csv_path.exists():
                rospy.logerr(f"CSV file not found: {self.csv_path}")
                sys.exit(1)
            rospy.loginfo(f"Using existing CSV: {self.csv_path}")
            # Skip sampling: run calibration/evaluation immediately
            self._run_phase1()
        else:
            # Collect samples
            self.csv_path = None
            self._setup_csv()
            self._start_collection()
            rospy.on_shutdown(self.cleanup)
            rospy.loginfo("Starting data collection. Move sensor slowly by hand...")
            rospy.loginfo("Press Ctrl+C to stop and save data.")
            rospy.spin()

    def _start_collection(self):
        """Start data collection mode."""
        self.packages = {i: [] for i in range(1, self._n_sensors + 1)}
        self.current_package = {}
        self.csv_file = None
        self.csv_writer = None
        self._setup_csv()
        rospy.loginfo("Subscribing to stm_uplink_raw topic...")
        self.sub = rospy.Subscriber('stm_uplink_raw', StmUplink, self._uplink_callback)

    def _uplink_callback(self, msg):
        """Store sensor data - one message may contain 1 or more sensors."""
        for sensor in msg.sensor_data:
            if 1 <= sensor.id <= self._n_sensors:
                self.current_package[sensor.id] = (sensor.x, sensor.y, sensor.z)

        if len(self.current_package) == self._n_sensors:
            for sensor_id in range(1, self._n_sensors + 1):
                self.packages[sensor_id].append(self.current_package[sensor_id])
            self.current_package = {}

            if len(self.packages[1]) >= self.samples_per_avg:
                self._write_averaged_sample()
                for sensor_id in range(1, self._n_sensors + 1):
                    self.packages[sensor_id] = []

    def _write_averaged_sample(self):
        """Average all packages and write to CSV."""
        row = []
        for sensor_id in range(1, self._n_sensors + 1):
            packages = self.packages[sensor_id]
            avg_x = sum(p[0] for p in packages) / len(packages)
            avg_y = sum(p[1] for p in packages) / len(packages)
            avg_z = sum(p[2] for p in packages) / len(packages)
            row.extend([f"{avg_x:.7g}", f"{avg_y:.7g}", f"{avg_z:.7g}"])

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.sample_count += 1
        rospy.loginfo(f"Sample {self.sample_count}/{self.max_samples} written")

        if self.sample_count >= self.max_samples:
            rospy.loginfo(f"Collected {self.max_samples} samples. Stopping...")
            rospy.signal_shutdown("Completed")

    def _setup_csv(self):
        """Setup CSV file with headers."""
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, "handheld_calib.csv")
        self.csv_path = Path(csv_path)

        try:
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            header = []
            for i in range(1, self._n_sensors + 1):
                header.extend([f'sensor_{i}_x', f'sensor_{i}_y', f'sensor_{i}_z'])
            self.csv_writer.writerow(header)

            rospy.loginfo(f"CSV file created: {csv_path}")
            rospy.loginfo(f"Averaging {self.samples_per_avg} samples per row")

        except Exception as e:
            rospy.logerr(f"Failed to create CSV file: {e}")
            sys.exit(1)

    def _get_default_csv_path(self) -> Path:
        """Get default CSV path."""
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        return Path(output_dir) / "handheld_calib.csv"

    def _run_phase1(self):
        """Run Phase 1: ellipsoid calibration or evaluation on CSV."""
        if self.skip_calibration:
            rospy.loginfo("Running Phase 1 (ellipsoid) evaluation only...")
        else:
            rospy.loginfo("Running Phase 1 (ellipsoid) calibration...")

        try:
            from calibration.lib.ellipsoid_fit import batch_ellipsoid_fit, save_calibration_params

            results = batch_ellipsoid_fit(
                csv_path=str(self.csv_path),
                sensor_type=self._sensor_type,
                evaluate_only=self.skip_calibration
            )

            if not self.skip_calibration:
                # Save params to sensor_array_config
                sensor_type_dir = (
                    Path(__file__).parent.parent.parent /
                    'sensor_array_config' / 'sensor_array_config' / 'config' / self._sensor_type.lower()
                )
                sensor_type_dir.mkdir(parents=True, exist_ok=True)
                output_path = sensor_type_dir / 'intrinsic_params.json'
                save_calibration_params(results, output_path, sensor_config=self._sensor_config)
                rospy.loginfo(f"Intrinsic params saved to: {output_path}")

            rospy.loginfo("Phase 1 complete.")
        except Exception as e:
            rospy.logerr(f"Phase 1 failed: {e}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """Cleanup on shutdown."""
        if self.csv_file is not None:
            self.csv_file.close()
            rospy.loginfo(f"CSV closed. Total samples: {self.sample_count}")
            rospy.loginfo(f"CSV saved to: {self.csv_path}")

        # Run Phase 1 after collection
        if self.csv_path is not None and self.sample_count > 0:
            self._run_phase1()

        rospy.loginfo("Handheld calibration complete.")


if __name__ == '__main__':
    try:
        sampler = HandheldCalibrationSampler()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass

