#!/usr/bin/env python3
"""Serial bridge for STM32 magnetometer/IMU binary streams.

This node parses the USB CDC binary stream used by MagGrad_Dual_V1 and
STM32H7_Sensor_V1:
  sync(0xA5 0x5A) + version + type + seq + tick_ms + payload_len + payload + crc16

It publishes magnetometer array snapshots as the existing StmUplink messages
and ICM42670 samples as MagGradImuRaw plus optional sensor_msgs/Imu.
"""

import csv
import glob
import math
import os
import struct
from datetime import datetime

import numpy as np
import rospy
import serial
import serial.tools.list_ports
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray, Header, String

from serial_processor.msg import MagGradImuRaw, SensorData, StmUplink
from sensor_array_config import (
    HardwareConfig,
    get_hardware_config,
    get_runtime_config,
    resolve_runtime_value,
)


class MagGradProtocol:
    SYNC = b"\xA5\x5A"
    VERSION = 1
    TYPE_AK = 0x01
    TYPE_TMAG = 0x02
    TYPE_ICM = 0x03
    TYPE_AK_ARRAY = 0x11
    TYPE_TMAG_ARRAY = 0x12
    TYPE_QMC_ARRAY = 0x21
    TYPE_ERR = 0xE0
    TYPE_STATS = 0xF0
    MAX_PAYLOAD_LEN = 512

    @staticmethod
    def crc16_ccitt_false(data):
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc


class StreamRecorder:
    def __init__(self, output_dir, n_sensors):
        self.output_dir = output_dir
        self.n_sensors = n_sensors
        self.state = "idle"
        self.path = None
        self.file = None
        self.writer = None
        os.makedirs(self.output_dir, exist_ok=True)

    def trigger(self, enable):
        if enable:
            self.start()
        else:
            self.stop()

    def start(self):
        if self.file is not None:
            self.stop()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.output_dir, f"maggrad_stream_{ts}.csv")
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        header = ["pc_time", "seq", "tick_ms", "bitmap"]
        for sid in range(1, self.n_sensors + 1):
            header.extend([f"sensor_{sid}_x_gs", f"sensor_{sid}_y_gs", f"sensor_{sid}_z_gs"])
        header.extend(["imu_seq", "imu_tick_ms", "ax", "ay", "az", "gx", "gy", "gz", "temp"])
        self.writer.writerow(header)
        self.state = "recording"
        rospy.loginfo(f"[MagGradRecorder] Recording started: {self.path}")

    def stop(self):
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
        self.file = None
        self.writer = None
        self.state = "idle"
        rospy.loginfo("[MagGradRecorder] Recording stopped")

    def write_snapshot(self, seq, tick_ms, bitmap, sensors, latest_imu):
        if self.state != "recording" or self.writer is None:
            return
        sensor_map = {s.id: s for s in sensors}
        row = [f"{rospy.Time.now().to_sec():.6f}", seq, tick_ms, f"0x{bitmap:04X}"]
        for sid in range(1, self.n_sensors + 1):
            sensor = sensor_map.get(sid)
            if sensor is None:
                row.extend(["", "", ""])
            else:
                row.extend([f"{sensor.x:.6f}", f"{sensor.y:.6f}", f"{sensor.z:.6f}"])
        if latest_imu is None:
            row.extend([""] * 9)
        else:
            row.extend([
                latest_imu.seq, latest_imu.tick_ms,
                latest_imu.ax, latest_imu.ay, latest_imu.az,
                latest_imu.gx, latest_imu.gy, latest_imu.gz,
                latest_imu.temp,
            ])
        self.writer.writerow(row)


class SerialNodeMagGrad:
    def __init__(self):
        rospy.init_node("serial_node_maggrad", anonymous=True)

        self.runtime_config = get_runtime_config()
        self.port = self._resolve_serial_port(self._runtime_param("~port", "port", "auto"))
        self.baudrate = int(self._runtime_param("~baudrate", "baudrate", 115200))
        self.post_open_delay = float(rospy.get_param("~post_open_delay", 2.2))
        self.hardware_config_name = self._runtime_param("~hardware_config", "hardware_config", "qmc6309")
        self.hardware_config: HardwareConfig = get_hardware_config(self.hardware_config_name)
        self.magnetometer = self.hardware_config.magnetometer
        self.imu_config = self.hardware_config.imu
        self.adu_to_gs = float(self.magnetometer.adu_to_gs)
        self.n_sensors = int(self.magnetometer.n_sensors)
        default_output_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "data",
            "serial_processor",
            "maggrad",
        ))
        self.output_dir = os.path.expanduser(
            rospy.get_param(
                "~output_dir",
                default_output_dir,
            )
        )
        self.startup_strategy = str(
            self._runtime_param("~startup_strategy", "startup_strategy", "cont")
        ).strip().lower()
        self.startup_sensors = str(
            self._runtime_param("~startup_sensors", "startup_sensors", "auto")
        ).strip().upper()
        self.trigger_rate_hz = int(self._runtime_param("~trigger_rate_hz", "trigger_rate_hz", 100))

        self.publish_scaled_imu = self._param_bool(
            rospy.get_param("~publish_scaled_imu", self.imu_config.publish_scaled_imu)
        )
        self.accel_lsb_per_g = float(self.imu_config.accel_lsb_per_g)
        self.gyro_lsb_per_dps = float(self.imu_config.gyro_lsb_per_dps)
        self.imu_frame_id = self.imu_config.imu_frame_id
        self.imu_axis_transform = self.imu_config.axis_transform_numpy()

        self.R_CORR = self._load_r_corr()
        self.D_matrix, self.e_bias = self._load_affine()
        self.latest_imu = None

        self.pub = rospy.Publisher("stm_uplink", StmUplink, queue_size=100)
        self.pub_raw = rospy.Publisher("stm_uplink_raw", StmUplink, queue_size=100)
        self.pub_magnitude = rospy.Publisher("stm_magnitude", Float32MultiArray, queue_size=100)
        self.pub_magnitude_raw = rospy.Publisher("stm_magnitude_raw", Float32MultiArray, queue_size=100)
        self.pub_imu_raw = rospy.Publisher("maggrad/imu_raw", MagGradImuRaw, queue_size=100)
        self.pub_imu = rospy.Publisher("maggrad/imu", Imu, queue_size=100)
        self.pub_status = rospy.Publisher("maggrad/record/status", String, queue_size=10)

        self.recorder = StreamRecorder(self.output_dir, self.n_sensors)
        self.sub_trigger = rospy.Subscriber("~record_trigger", Bool, self._on_record_trigger)
        rospy.on_shutdown(self._on_shutdown)

        self.ser = None
        self.connect()
        self._configure_firmware_stream()
        rospy.loginfo(
            f"SerialNodeMagGrad initialized: port={self.port}, baudrate={self.baudrate}, "
            f"hardware_config={self.hardware_config_name}, adu_to_gs={self.adu_to_gs:.8f}"
        )

    def _runtime_param(self, ros_param, runtime_key, fallback):
        value = rospy.get_param(ros_param, "runtime")
        return resolve_runtime_value(value, runtime_key, fallback)

    def _resolve_serial_port(self, configured_port):
        port = str(configured_port).strip()
        if port and port.lower() != "auto":
            if port == "/dev/ttyACM":
                candidates = sorted(glob.glob("/dev/ttyACM*"))
                if candidates:
                    return candidates[0]
            return port

        candidates = []
        patterns = ["/dev/ttyACM*", "/dev/ttyUSB*", "/dev/cu.usbmodem*", "/dev/tty.usbmodem*"]
        for pattern in patterns:
            candidates.extend(glob.glob(pattern))
        if candidates:
            return sorted(candidates)[0]

        for item in serial.tools.list_ports.comports():
            dev = item.device
            text = f"{item.device} {item.description}".lower()
            if "usb" in text or "modem" in text:
                return dev
        return port

    def _param_bool(self, value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def _load_r_corr(self):
        r_corr = {}
        for entry in self.magnetometer.R_CORR:
            mat = np.array(entry.matrix).reshape(3, 3, order="F")
            for sid in entry.sensor_ids:
                r_corr[int(sid)] = mat
        rospy.loginfo(f"Loaded R_CORR for {len(r_corr)} sensors")
        return r_corr

    def _load_affine(self):
        d_matrix = {}
        e_bias = {}
        for sid, params in self.magnetometer.affine_model.params.items():
            d_matrix[int(sid)] = np.array(params.D_i)
            e_bias[int(sid)] = np.array(params.e_i).squeeze()
        rospy.loginfo(f"Loaded affine calibration for {len(d_matrix)} sensors")
        return d_matrix, e_bias

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
            self.ser.reset_input_buffer()
            rospy.loginfo(f"Connected to {self.port} at {self.baudrate} baud")
            if self.post_open_delay > 0.0:
                rospy.loginfo(f"Waiting {self.post_open_delay:.2f}s for USB CDC after opening serial port")
                rospy.sleep(self.post_open_delay)
                self.ser.reset_input_buffer()
        except Exception as exc:
            rospy.logerr(f"Error connecting to serial port: {exc}")
            rospy.signal_shutdown("Serial connection failed")

    def _write_command(self, command):
        if not self.ser or not self.ser.is_open:
            return
        line = (command.strip() + "\r\n").encode("ascii")
        self.ser.write(line)
        self.ser.flush()
        rospy.loginfo(f"STM32 command sent: {command.strip()}")

    def _startup_sensor_arg(self):
        requested = self.startup_sensors
        if requested in ("", "AUTO"):
            return "ALL"
        return requested

    def _configure_firmware_stream(self):
        """Put firmware into the requested runtime mode.

        Supported firmwares boot in IDLE and only start binary output after
        receiving a CR/LF-terminated MODE command over USB CDC.
        """
        if self.startup_strategy in ("", "none", "manual"):
            rospy.loginfo("STM32 startup command disabled")
            return

        strategy_map = {
            "idle": "IDLE",
            "cont": "CONT",
            "continuous": "CONT",
            "trig": "TRIG",
            "trigger": "TRIG",
            "trig_auto": "TRIG_AUTO",
            "trig-auto": "TRIG_AUTO",
            "trigger_auto": "TRIG_AUTO",
        }
        mode = strategy_map.get(self.startup_strategy)
        if mode is None:
            rospy.logwarn(f"Unknown startup_strategy={self.startup_strategy}; leaving firmware mode unchanged")
            return

        sensors = self._startup_sensor_arg()
        if mode == "IDLE":
            command = "MODE IDLE"
        elif mode == "TRIG_AUTO":
            command = f"MODE TRIG_AUTO {sensors} {self.trigger_rate_hz}"
        else:
            command = f"MODE {mode} {sensors}"

        self._write_command(command)

    def _on_record_trigger(self, msg):
        self.recorder.trigger(msg.data)
        self.pub_status.publish(String(data=self.recorder.state))

    def _on_shutdown(self):
        self.recorder.stop()
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def _parse_frames(self, buffer):
        frames = []
        while len(buffer) >= 2:
            start = buffer.find(MagGradProtocol.SYNC)
            if start < 0:
                buffer = buffer[-1:]
                break
            if start > 0:
                buffer = buffer[start:]
            if len(buffer) < 16:
                break

            version = buffer[2]
            frame_type = buffer[3]
            seq = struct.unpack_from("<I", buffer, 4)[0]
            tick_ms = struct.unpack_from("<I", buffer, 8)[0]
            payload_len = struct.unpack_from("<H", buffer, 12)[0]
            if payload_len > MagGradProtocol.MAX_PAYLOAD_LEN:
                rospy.logwarn(f"Dropping frame with invalid payload_len={payload_len}")
                buffer = buffer[1:]
                continue

            total_len = 14 + payload_len + 2
            if len(buffer) < total_len:
                break

            payload_start = 14
            payload_end = payload_start + payload_len
            payload = buffer[payload_start:payload_end]
            got_crc = struct.unpack_from("<H", buffer, payload_end)[0]
            calc_crc = MagGradProtocol.crc16_ccitt_false(buffer[2:payload_end])
            if version != MagGradProtocol.VERSION or got_crc != calc_crc:
                rospy.logwarn(
                    f"Dropping invalid frame: version={version}, type=0x{frame_type:02X}, "
                    f"got_crc=0x{got_crc:04X}, calc_crc=0x{calc_crc:04X}"
                )
                buffer = buffer[1:]
                continue

            frames.append((frame_type, seq, tick_ms, payload))
            buffer = buffer[total_len:]
        return frames, buffer

    def _make_header(self, seq, frame_id):
        header = Header()
        header.seq = seq
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return header

    def _publish_magnetometer_array(self, seq, tick_ms, bitmap, chip_sensors, frame_id):
        raw_sensors = []
        for sensor in chip_sensors:
            vec = np.array([sensor.x, sensor.y, sensor.z])
            if sensor.id in self.R_CORR:
                vec = self.R_CORR[sensor.id] @ vec
            raw_sensors.append(SensorData(
                id=sensor.id,
                x=float(vec[0]),
                y=float(vec[1]),
                z=float(vec[2]),
            ))

        header = self._make_header(seq, frame_id)
        raw_msg = StmUplink()
        raw_msg.header = header
        raw_msg.cycle_id = seq & 0xFFFF
        raw_msg.slot = 0
        raw_msg.bitmap = bitmap
        raw_msg.timestamp = tick_ms
        raw_msg.sensor_data = raw_sensors
        raw_msg.cycle_end = 1
        self.pub_raw.publish(raw_msg)

        raw_magnitudes = Float32MultiArray()
        raw_magnitudes.data = [math.sqrt(s.x * s.x + s.y * s.y + s.z * s.z) for s in raw_sensors]
        self.pub_magnitude_raw.publish(raw_magnitudes)

        corrected = []
        for sensor in raw_sensors:
            vec = np.array([sensor.x, sensor.y, sensor.z])
            if sensor.id in self.D_matrix and sensor.id in self.e_bias:
                vec = self.D_matrix[sensor.id] @ vec + self.e_bias[sensor.id]
            corrected.append(SensorData(
                id=sensor.id,
                x=float(vec[0]),
                y=float(vec[1]),
                z=float(vec[2]),
            ))

        msg = StmUplink()
        msg.header = header
        msg.cycle_id = raw_msg.cycle_id
        msg.slot = raw_msg.slot
        msg.bitmap = raw_msg.bitmap
        msg.timestamp = raw_msg.timestamp
        msg.sensor_data = corrected
        msg.cycle_end = raw_msg.cycle_end
        self.pub.publish(msg)

        magnitudes = Float32MultiArray()
        magnitudes.data = [math.sqrt(s.x * s.x + s.y * s.y + s.z * s.z) for s in corrected]
        self.pub_magnitude.publish(magnitudes)

        self.recorder.write_snapshot(seq, tick_ms, bitmap, raw_sensors, self.latest_imu)

    def _publish_ak_array(self, seq, tick_ms, payload):
        if len(payload) < 3:
            rospy.logwarn("AK_ARRAY payload too short")
            return
        count = payload[0]
        bitmap = struct.unpack_from("<H", payload, 1)[0]
        expected_len = 3 + count * 12
        if len(payload) != expected_len:
            rospy.logwarn(f"Invalid AK_ARRAY length: got={len(payload)}, expected={expected_len}")
            return

        chip_sensors = []
        offset = 3
        for _ in range(count):
            sid, bus, mask, hx, hy, hz, status, err, dor = struct.unpack_from("<BBBhhhBBB", payload, offset)
            offset += 12
            if err or dor:
                rospy.logdebug(
                    f"AK sensor status: sid={sid}, bus={bus}, mask=0x{mask:02X}, "
                    f"status={status}, err={err}, dor={dor}"
                )
            chip_sensors.append(SensorData(
                id=sid,
                x=float(hx) * self.adu_to_gs,
                y=float(hy) * self.adu_to_gs,
                z=float(hz) * self.adu_to_gs,
            ))

        self._publish_magnetometer_array(
            seq, tick_ms, bitmap, chip_sensors, self.magnetometer.frame_id
        )

    def _publish_qmc_array(self, seq, tick_ms, payload):
        if len(payload) < 3:
            rospy.logwarn("QMC_ARRAY payload too short")
            return
        count = payload[0]
        bitmap = struct.unpack_from("<H", payload, 1)[0]
        expected_len = 3 + count * 11
        if len(payload) != expected_len:
            rospy.logwarn(f"Invalid QMC_ARRAY length: got={len(payload)}, expected={expected_len}")
            return

        chip_sensors = []
        offset = 3
        for _ in range(count):
            sid, bus, mask, x, y, z, ok, err = struct.unpack_from("<BBBhhhBB", payload, offset)
            offset += 11
            if not ok or err:
                rospy.logdebug(
                    f"QMC sensor status: sid={sid}, bus={bus}, mask=0x{mask:02X}, ok={ok}, err={err}"
                )
            chip_sensors.append(SensorData(
                id=sid,
                x=float(x) * self.adu_to_gs,
                y=float(y) * self.adu_to_gs,
                z=float(z) * self.adu_to_gs,
            ))

        self._publish_magnetometer_array(
            seq, tick_ms, bitmap, chip_sensors, self.magnetometer.frame_id
        )

    def _publish_tmag_array(self, seq, tick_ms, payload):
        if len(payload) < 3:
            rospy.logwarn("TMAG_ARRAY payload too short")
            return
        count = payload[0]
        bitmap = struct.unpack_from("<H", payload, 1)[0]
        expected_len = 3 + count * 12
        if len(payload) != expected_len:
            rospy.logwarn(f"Invalid TMAG_ARRAY length: got={len(payload)}, expected={expected_len}")
            return

        chip_sensors = []
        offset = 3
        for _ in range(count):
            sid, ch_mask, addr, x, y, z, status, err, flags = struct.unpack_from(
                "<BBBhhhBBB", payload, offset
            )
            offset += 12
            if err or flags:
                rospy.logdebug(
                    f"TMAG sensor status: sid={sid}, ch_mask=0x{ch_mask:02X}, addr=0x{addr:02X}, "
                    f"status={status}, err={err}, flags=0x{flags:02X}"
                )
            chip_sensors.append(SensorData(
                id=sid,
                x=float(x) * self.adu_to_gs,
                y=float(y) * self.adu_to_gs,
                z=float(z) * self.adu_to_gs,
            ))

        self._publish_magnetometer_array(
            seq, tick_ms, bitmap, chip_sensors, self.magnetometer.frame_id
        )

    def _publish_imu(self, seq, tick_ms, payload):
        if len(payload) != 14:
            rospy.logwarn(f"Invalid ICM payload length: {len(payload)}")
            return
        ax, ay, az, gx, gy, gz, temp = struct.unpack("<hhhhhhh", payload)
        msg = MagGradImuRaw()
        msg.header = self._make_header(seq, self.imu_frame_id)
        msg.seq = seq
        msg.tick_ms = tick_ms
        msg.ax = ax
        msg.ay = ay
        msg.az = az
        msg.gx = gx
        msg.gy = gy
        msg.gz = gz
        msg.temp = temp
        self.latest_imu = msg
        self.pub_imu_raw.publish(msg)

        if self.publish_scaled_imu:
            imu = Imu()
            imu.header = msg.header
            imu.orientation_covariance[0] = -1.0
            gyro_scale = math.pi / 180.0 / self.gyro_lsb_per_dps
            accel_scale = 9.80665 / self.accel_lsb_per_g
            angular_velocity = self.imu_axis_transform @ np.array([gx, gy, gz], dtype=float)
            linear_acceleration = self.imu_axis_transform @ np.array([ax, ay, az], dtype=float)
            imu.angular_velocity.x = angular_velocity[0] * gyro_scale
            imu.angular_velocity.y = angular_velocity[1] * gyro_scale
            imu.angular_velocity.z = angular_velocity[2] * gyro_scale
            imu.linear_acceleration.x = linear_acceleration[0] * accel_scale
            imu.linear_acceleration.y = linear_acceleration[1] * accel_scale
            imu.linear_acceleration.z = linear_acceleration[2] * accel_scale
            self.pub_imu.publish(imu)

    def _handle_frame(self, frame):
        frame_type, seq, tick_ms, payload = frame
        if frame_type == MagGradProtocol.TYPE_AK_ARRAY:
            self._publish_ak_array(seq, tick_ms, payload)
        elif frame_type == MagGradProtocol.TYPE_TMAG_ARRAY:
            self._publish_tmag_array(seq, tick_ms, payload)
        elif frame_type == MagGradProtocol.TYPE_QMC_ARRAY:
            self._publish_qmc_array(seq, tick_ms, payload)
        elif frame_type == MagGradProtocol.TYPE_ICM:
            self._publish_imu(seq, tick_ms, payload)
        elif frame_type == MagGradProtocol.TYPE_ERR:
            rospy.logwarn(f"STM32 error frame: seq={seq}, tick_ms={tick_ms}, payload={payload.hex()}")
        elif frame_type == MagGradProtocol.TYPE_STATS:
            rospy.logdebug(f"STM32 stats frame: seq={seq}, tick_ms={tick_ms}, payload={payload.hex()}")
        else:
            rospy.logdebug(f"Ignoring frame type=0x{frame_type:02X}, seq={seq}, len={len(payload)}")

    def run(self):
        rospy.loginfo("SerialNodeMagGrad is ready.")
        buffer = b""
        while not rospy.is_shutdown():
            try:
                if self.ser and self.ser.in_waiting > 0:
                    buffer += self.ser.read(self.ser.in_waiting)
                    frames, buffer = self._parse_frames(buffer)
                    for frame in frames:
                        self._handle_frame(frame)
                else:
                    rospy.sleep(0.001)
            except serial.SerialException as exc:
                rospy.logerr(f"Serial error: {exc}")
                break

        self._on_shutdown()


if __name__ == "__main__":
    try:
        SerialNodeMagGrad().run()
    except rospy.ROSInterruptException:
        pass
