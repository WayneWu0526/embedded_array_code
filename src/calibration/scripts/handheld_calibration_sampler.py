#!/usr/bin/env python3
"""
Handheld calibration sampler - manual data collection without robot arm.

Usage:
    roslaunch calibration handheld_calibration.launch

Simply hold and slowly move the sensor array by hand. The script will
continuously collect and average samples.
"""
import sys
import rospy
import csv
import os
import datetime
from serial_processor.msg import StmUplink
from sensor_array_config import get_config, SensorArrayConfig

class HandheldCalibrationSampler:
    def __init__(self):
        rospy.init_node('handheld_calibration_sampler', anonymous=True)

        self.samples_per_avg = rospy.get_param('~samples_per_avg', 10)
        self.max_samples = rospy.get_param('~max_samples', 500)
        self.sample_count = 0

        # Load sensor array configuration
        self._sensor_type = rospy.get_param('~sensor_type', 'QMC6309')
        self._sensor_config: SensorArrayConfig = get_config(self._sensor_type)
        self._n_sensors = self._sensor_config.manifest.n_sensors
        rospy.loginfo(f"Using sensor type: {self._sensor_type}, n_sensors={self._n_sensors}")

        # Accumulated packages: each package is one complete set of all 12 sensors
        # packages[sensor_id] = list of 10 (x, y, z) tuples
        self.packages = {i: [] for i in range(1, self._n_sensors + 1)}

        # Current package being accumulated
        self.current_package = {}  # sensor_id -> (x, y, z)

        # Subscribe to stm_uplink_raw for raw sensor data (not calibrated)
        rospy.loginfo("Subscribing to stm_uplink_raw topic...")
        self.sub = rospy.Subscriber('stm_uplink_raw', StmUplink, self._uplink_callback)

        # Setup CSV
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None  # Will be set by _setup_csv()
        self._setup_csv()

        # Register shutdown hook for cleanup
        rospy.on_shutdown(self.cleanup)

        # Collect continuously
        rospy.loginfo("Starting handheld calibration. Move sensor slowly by hand...")
        rospy.loginfo("Press Ctrl+C to stop and save data.")

        rospy.spin()

    def _uplink_callback(self, msg):
        """Store sensor data - one message may contain 1 or more sensors."""
        for sensor in msg.sensor_data:
            if 1 <= sensor.id <= self._n_sensors:
                self.current_package[sensor.id] = (sensor.x, sensor.y, sensor.z)

        # Check if current package is complete (all 12 sensors received)
        if len(self.current_package) == self._n_sensors:
            # Save this complete package to all_packages
            for sensor_id in range(1, self._n_sensors + 1):
                self.packages[sensor_id].append(self.current_package[sensor_id])
            self.current_package = {}  # Reset for next package

            # Check if we have enough packages to average and write
            if len(self.packages[1]) >= self.samples_per_avg:
                self._write_averaged_sample()
                # Clear all packages after writing
                for sensor_id in range(1, self._n_sensors + 1):
                    self.packages[sensor_id] = []

    def _write_averaged_sample(self):
        """Average all packages and write to CSV."""
        timestamp = datetime.datetime.now().isoformat()

        row = [timestamp]
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

        # Check if we've collected enough samples
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

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"handheld_calib_{timestamp}.csv")
        self.csv_path = csv_path  # Save for post-processing

        try:
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # Header: timestamp, sensor_1_x, sensor_1_y, sensor_1_z, ...
            header = ['timestamp']
            for i in range(1, self._n_sensors + 1):
                header.extend([f'sensor_{i}_x', f'sensor_{i}_y', f'sensor_{i}_z'])
            self.csv_writer.writerow(header)

            rospy.loginfo(f"CSV file created: {csv_path}")
            rospy.loginfo(f"Averaging {self.samples_per_avg} samples per row")

        except Exception as e:
            rospy.logerr(f"Failed to create CSV file: {e}")
            sys.exit(1)

    def cleanup(self):
        """Cleanup on shutdown."""
        if self.csv_file is not None:
            self.csv_file.close()
            rospy.loginfo(f"CSV file closed. Total samples: {self.sample_count}")
            self.csv_file = None  # Prevent double close

            # Run Phase 1 calibration (s1) - only if we have valid data
            if self.csv_path is not None:
                rospy.loginfo("Running s1 calibration...")
                from calibration_postprocessor import CalibrationPostProcessor
                post_processor = CalibrationPostProcessor(
                    csv_path=self.csv_path,
                    calibration_type='handheld',
                    sensor_type=self._sensor_type
                )
                post_processor.run()

        rospy.loginfo("Handheld calibration complete.")


if __name__ == '__main__':
    try:
        sampler = HandheldCalibrationSampler()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass  # cleanup() is called via rospy.on_shutdown() hook
