#!/usr/bin/env python3
"""Manual calibration recorder for the MagGrad STM32 firmware."""

import os
import json
import threading
from datetime import datetime

import rospy
from serial_processor.msg import StmUplink
from sensor_array_config import get_hardware_config, resolve_runtime_value
from std_msgs.msg import Bool, String


DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "manual_calibration")
)


class ManualAverageRecorder:
    def __init__(self, output_dir, frames_to_average, n_sensors):
        self.output_dir = os.path.expanduser(output_dir)
        self.frames_to_average = int(frames_to_average)
        self.n_sensors = int(n_sensors)
        self.file = None
        self.path = None
        self.buffer = []
        os.makedirs(self.output_dir, exist_ok=True)
        if self.frames_to_average <= 0:
            raise ValueError("frames_to_average must be positive")

    @property
    def is_recording(self):
        return self.file is not None

    def start(self):
        self.stop()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.output_dir, f"manual_record_{ts}.csv")
        self.file = open(self.path, "w")
        self.file.write(",".join(self._header()) + "\n")
        self.buffer = []
        rospy.loginfo(f"[MagGradManualRecord] Recording started: {self.path}")

    def stop(self):
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            rospy.loginfo("[MagGradManualRecord] Recording stopped")
        self.file = None
        self.buffer = []

    def _header(self):
        header = []
        for sid in range(1, self.n_sensors + 1):
            header.extend([f"sensor_{sid}_x", f"sensor_{sid}_y", f"sensor_{sid}_z"])
        return header

    def add(self, msg):
        if self.file is None:
            return False
        self.buffer.append(msg)
        if len(self.buffer) < self.frames_to_average:
            return False
        row = self._average_row()
        if row is None:
            rospy.logwarn("[MagGradManualRecord] Dropping incomplete averaged row")
            self.buffer = []
            return False
        self.file.write(",".join(f"{value:.6f}" for value in row) + "\n")
        self.buffer = []
        return True

    def _average_row(self):
        sensor_sums = {
            sid: [0.0, 0.0, 0.0, 0]
            for sid in range(1, self.n_sensors + 1)
        }
        for msg in self.buffer:
            for sensor in msg.sensor_data:
                sid = int(sensor.id)
                if sid not in sensor_sums:
                    continue
                sums = sensor_sums[sid]
                sums[0] += float(sensor.x)
                sums[1] += float(sensor.y)
                sums[2] += float(sensor.z)
                sums[3] += 1

        row = []
        for sid in range(1, self.n_sensors + 1):
            sx, sy, sz, count = sensor_sums[sid]
            if count == 0:
                return None
            row.extend([sx / count, sy / count, sz / count])
        return row


class MagGradManualRecordNode:
    def __init__(self):
        rospy.init_node("maggrad_manual_record_node", anonymous=True)

        self.output_dir = rospy.get_param("~output_dir", DEFAULT_OUTPUT_DIR)
        self.frames_to_average = int(rospy.get_param("~frames_to_average", 10))
        self.hardware_config_name = resolve_runtime_value(
            rospy.get_param("~hardware_config", "runtime"),
            "hardware_config",
            "qmc6309",
        )
        n_sensors_param = rospy.get_param("~n_sensors", "runtime")
        if resolve_runtime_value(n_sensors_param, "n_sensors", "runtime") == "runtime":
            self.n_sensors = int(get_hardware_config(self.hardware_config_name).magnetometer.n_sensors)
        else:
            self.n_sensors = int(resolve_runtime_value(n_sensors_param, "n_sensors", 12))
        self.input_topic = rospy.get_param("~input_topic", "stm_uplink_raw")

        self.lock = threading.Lock()
        self.rows_written = 0
        self.recorder = ManualAverageRecorder(
            self.output_dir,
            self.frames_to_average,
            self.n_sensors,
        )

        self.sub_data = rospy.Subscriber(self.input_topic, StmUplink, self._on_data)
        self.sub_trigger = rospy.Subscriber("~record_trigger", Bool, self._on_trigger)
        self.pub_status = rospy.Publisher("~status", String, queue_size=10, latch=True)
        rospy.on_shutdown(self._on_shutdown)
        self._publish_status()

        rospy.loginfo(
            f"MagGrad manual recorder initialized: input_topic={self.input_topic}, "
            f"hardware_config={self.hardware_config_name}, n_sensors={self.n_sensors}, "
            f"frames_to_average={self.frames_to_average}, output_dir={os.path.expanduser(self.output_dir)}"
        )

    def _on_trigger(self, msg):
        with self.lock:
            if bool(msg.data):
                self.rows_written = 0
                self.recorder.start()
            else:
                self.recorder.stop()
            self._publish_status()

    def _on_data(self, msg):
        with self.lock:
            if self.recorder.add(msg):
                self.rows_written += 1
                self._publish_status()

    def _publish_status(self):
        state = {
            "recording": self.recorder.is_recording,
            "path": self.recorder.path,
            "rows_written": self.rows_written,
        }
        self.pub_status.publish(String(data=json.dumps(state, sort_keys=True)))

    def _on_shutdown(self):
        with self.lock:
            self.recorder.stop()

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        MagGradManualRecordNode().run()
    except rospy.ROSInterruptException:
        pass
