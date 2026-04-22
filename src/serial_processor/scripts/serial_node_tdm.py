#!/usr/bin/env python3
"""
TDM (Time Division Multiplexing) Serial Node for STM32 communication.

Binary Protocol:
- Downlink (PC -> STM32): 13 bytes
    Header(2) + Version(1) + Mode(1) + Bitmap(2) + SettlingTime(2) + CycleTime(4) + CycleNum(1)
- Uplink (STM32 -> PC): Variable
  Header(2) + Version(1) + cycle_id(2) + slot(1) + Bitmap(2) + Timestamp(8) + sensor_data(N*7) + cycle_end(1)
  sensor_data: SensorID(1 byte) + X(2 bytes) + Y(2 bytes) + Z(2 bytes), signed int, scale: raw * 32/32768
"""

import rospy
import serial
import struct
import threading
import math
import glob
import os
import json
import csv
import numpy as np
from std_msgs.msg import Header, Float32MultiArray, Empty
from serial_processor.msg import SensorData, StmUplink, StmDownlink
from sensor_array_config import get_config, SensorArrayConfig
from calibration.forward import CalibrationProcessor


class ManualCaptureModule:
    """Subscribes to trigger topic, collects N samples of corrected stm_uplink, averages, saves to CSV."""

    SENSOR_COUNT = 12

    def __init__(self, output_filename, sample_count=10):
        """
        Args:
            output_filename: CSV filename (not full path). Saved to serial_processor/data/.
            sample_count: Number of samples to average (default 10).
        """
        self._sample_count = sample_count
        self._buffer = []  # list of stm_uplink messages
        self._lock = threading.Lock()
        self._capturing = False

        # Resolve output path: ~/embedded_array_ws/src/serial_processor/data/<filename>
        ws_root = os.path.expanduser('~/embedded_array_ws/src/serial_processor')
        self._data_dir = os.path.join(ws_root, 'data')
        os.makedirs(self._data_dir, exist_ok=True)
        self._output_path = os.path.join(self._data_dir, output_filename)

        # Ensure CSV header exists
        if not os.path.exists(self._output_path):
            header = ['timestamp'] + [
                f'sensor_{i}_x,sensor_{i}_y,sensor_{i}_z'
                for i in range(1, self.SENSOR_COUNT + 1)
            ]
            with open(self._output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(','.join(header).split(','))  # split to flatten

        # Subscribers
        self._trigger_sub = rospy.Subscriber(
            '/manual_capture/trigger', Empty, self._on_trigger
        )
        self._uplink_sub = rospy.Subscriber(
            'stm_uplink', StmUplink, self._on_uplink
        )

        rospy.loginfo(
            f"ManualCaptureModule: output={self._output_path}, samples={sample_count}"
        )

    def _on_trigger(self, _msg):
        """Reset buffer and start collecting."""
        with self._lock:
            self._buffer = []
            self._capturing = True
        rospy.loginfo("ManualCapture: trigger received, collecting samples...")

    def _on_uplink(self, msg):
        """Buffer messages while capturing."""
        with self._lock:
            if not self._capturing:
                return
            self._buffer.append(msg)
            if len(self._buffer) >= self._sample_count:
                self._write_average()
                self._capturing = False
                self._buffer = []

    def _write_average(self):
        """Compute per-sensor averages and append to CSV."""
        with self._lock:
            if not self._buffer:
                return
            # buffer: list of StmUplink, each has sensor_data[0..11]
            # Build per-sensor accumulators
            sums = [[0.0, 0.0, 0.0] for _ in range(self.SENSOR_COUNT)]
            for uplink_msg in self._buffer:
                for idx, sd in enumerate(uplink_msg.sensor_data):
                    sums[idx][0] += sd.x
                    sums[idx][1] += sd.y
                    sums[idx][2] += sd.z
            avg = [
                [s[0] / len(self._buffer), s[1] / len(self._buffer), s[2] / len(self._buffer)]
                for s in sums
            ]
            # Timestamp
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            row = [ts] + [coord for sensor_avg in avg for coord in sensor_avg]
            with open(self._output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            rospy.loginfo(f"ManualCapture: wrote {len(self._buffer)} samples to {self._output_path}")


class SerialNodeTDM:
    # Protocol constants
    HEADER = 0xAA55
    TERMINATOR = b'\r\n'
    UPLINK_MIN_SIZE = 17  # Header(2) + Version(1) + cycle_id(2) + slot(1) + Bitmap(2) + Timestamp(8) + cycle_end(1)
    SENSOR_DATA_SIZE = 7  # SensorID(1) + X(2) + Y(2) + Z(2)

    def __init__(self):
        rospy.init_node('serial_node_tdm', anonymous=True)

        # Parameters
        self.port = self._resolve_serial_port(rospy.get_param('~port', '/dev/ttyACM'))
        self.baudrate = rospy.get_param('~baudrate', 921600)
        float_endian = str(rospy.get_param('~sensor_float_endian', 'little')).lower()
        self.float_endian = '<' if float_endian in ('little', 'le', '<') else '>'
        self.float_endian_name = 'little' if self.float_endian == '<' else 'big'

        # Load sensor array configuration
        self._sensor_type = rospy.get_param('~sensor_type', 'QMC6309')
        self._sensor_config: SensorArrayConfig = get_config(self._sensor_type)
        self._adu_to_gs = self._sensor_config.manifest.adu_to_gs
        rospy.loginfo(f"Using sensor type: {self._sensor_type}")

        # Calibration processor (ellipsoid + R_CORR + consistency)
        self._calib = CalibrationProcessor(self._sensor_config)

        # Serial connection
        self.ser = None
        self.connect()

        # Publisher for uplink data (fully corrected: ellipsoid + R_CORR + consistency)
        self.pub = rospy.Publisher('stm_uplink', StmUplink, queue_size=100)
        # Publisher for raw uplink data (uncorrected, for archive/Phase 2 calibration)
        self.pub_raw = rospy.Publisher('stm_uplink_raw', StmUplink, queue_size=100)
        # Publisher for magnetic field magnitude of corrected data
        self.pub_magnitude = rospy.Publisher('stm_magnitude', Float32MultiArray, queue_size=100)
        # Publisher for raw magnetic field magnitude (uncorrected)
        self.pub_magnitude_raw = rospy.Publisher('stm_magnitude_raw', Float32MultiArray, queue_size=100)

        # Subscriber for downlink commands
        self.sub = rospy.Subscriber('stm_downlink', StmDownlink, self._downlink_callback)

        # Manual capture module
        capture_filename = rospy.get_param('~capture_output_filename', 'manual_capture.csv')
        self._capture = ManualCaptureModule(capture_filename, sample_count=10)

        rospy.loginfo(
            f"SerialNodeTDM initialized: {self.port} at {self.baudrate} baud, "
            f"sensor_float_endian={self.float_endian_name}"
        )

    def _resolve_serial_port(self, configured_port):
        """
        Resolve serial port with auto-detection for '/dev/ttyACM*'.

        Behavior:
        - If configured as exact device path like '/dev/ttyACM0', use it directly.
        - If configured as '/dev/ttyACM' (no index), auto-select from '/dev/ttyACM*'.
        - If no candidate is found, return original configured value (connect() will report error).
        """
        port = str(configured_port).strip()

        # Explicit indexed port, keep user preference.
        if port.startswith('/dev/ttyACM') and len(port) > len('/dev/ttyACM'):
            return port

        if port == '/dev/ttyACM':
            candidates = sorted(glob.glob('/dev/ttyACM*'))
            if candidates:
                rospy.loginfo(f"Auto-detected serial port: {candidates[0]} (candidates={candidates})")
                return candidates[0]
            rospy.logwarn("No /dev/ttyACM* device found, fallback to /dev/ttyACM")

        return port

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
            rospy.loginfo(f"Connected to {self.port} at {self.baudrate} baud")
        except Exception as e:
            rospy.logerr(f"Error connecting to serial port: {e}")
            rospy.signal_shutdown("Serial connection failed")

    # Initialization reply status codes
    INIT_STATUS_SUCCESS = 0x00
    INIT_STATUS_PARAM_ERROR = 0x01
    INIT_STATUS_SENSOR_INVALID = 0x02
    INIT_STATUS_UNKNOWN = 0xFF

    def _downlink_callback(self, msg):
        """Send downlink command to STM32 and wait for initialization reply"""
        if not self.ser or not self.ser.is_open:
            rospy.logerr("Serial port not connected or open")
            return

        try:
            # Pack 13-byte command: Header(2) + Version(1) + Mode(1) + Bitmap(2) + SettlingTime(2) + CycleTime(4) + CycleNum(1)
            # STM32 uses big-endian
            # Mode: 0x00=MANUAL, 0x01=CVT, 0x02=CCI
            # Bitmap: bit0=sensor1, bit11=sensor12
            # SettlingTime: 0.01ms units
            # CycleTime: 0.01ms units, max 10000.00ms
            # CycleNum: total cycle count, 1 byte
            data = struct.pack('>HBBHHIB',
                self.HEADER,
                0x01,  # Version
                msg.mode,
                msg.bitmap,
                msg.settling_time,
                msg.cycle_time,
                msg.cycle_num
            )

            rospy.sleep(0.01)
            num_written = self.ser.write(bytes(data))
            self.ser.flush()

            if num_written != 13:
                rospy.logwarn(f"Serial write incomplete: only {num_written} bytes sent")

            # rospy.sleep(0.1)

            reply = self.ser.read(3)
            if len(reply) == 3 and reply[0:2] == b'\xAA\x55':
                status = reply[2]
                if status == 0:
                    rospy.loginfo("STM32 initialized successfully")
                elif status == 1:
                    rospy.logwarn("STM32: Parameter error in downlink command")
                elif status == self.INIT_STATUS_SENSOR_INVALID:
                    rospy.logwarn("STM32: Invalid sensor bitmap")
                else:
                    rospy.logwarn(f"STM32: Unknown error (status=0x{status:02X})")
            elif len(reply) > 0:
                rospy.logwarn(f"STM32: Unexpected reply length or header: {reply.hex()}")
            else:
                rospy.logwarn("STM32: No initialization reply received (Timeout)")
        except Exception as e:
            rospy.logerr(f"Error sending downlink: {e}")

    def _parse_uplink(self, raw_data):
        """
        Parse binary uplink data from STM32.
        Returns StmUplink message or None if parsing fails.
        """
        if len(raw_data) < self.UPLINK_MIN_SIZE:
            rospy.logwarn(f"Uplink data too short: {len(raw_data)} bytes")
            return None

        try:
            # Unpack header (STM32 uses big-endian)
            header = struct.unpack('>H', raw_data[0:2])[0]
            if header != self.HEADER:
                rospy.logwarn(f"Invalid header: 0x{header:04X}")
                return None

            version = raw_data[2]
            cycle_id, slot = struct.unpack('>HB', raw_data[3:6])
            bitmap = struct.unpack('>H', raw_data[6:8])[0]
            timestamp = struct.unpack('>Q', raw_data[8:16])[0]

            sensor_payload_len = len(raw_data) - self.UPLINK_MIN_SIZE
            if sensor_payload_len < 0 or sensor_payload_len % self.SENSOR_DATA_SIZE != 0:
                rospy.logwarn(f"Invalid uplink payload length: {len(raw_data)}")
                return None

            sensors = []
            sensor_count = sensor_payload_len // self.SENSOR_DATA_SIZE
            rospy.logdebug(f"Parsing uplink: cycle_id={cycle_id}, slot={slot}, bitmap=0x{bitmap:04X}, "
                           f"timestamp={timestamp}, sensor_count={sensor_count}")
            offset = 16
            for _ in range(sensor_count):
                # sensor_data format: SensorID(1 byte) + X(2 bytes) + Y(2 bytes) + Z(2 bytes), signed int16
                sensor_chunk = raw_data[offset:offset+self.SENSOR_DATA_SIZE]
                sid = sensor_chunk[0]
                # STM32 sends big-endian 16-bit signed integers
                raw_x, raw_y, raw_z = struct.unpack('>hhh', sensor_chunk[1:7])
                x = raw_x * self._adu_to_gs
                y = raw_y * self._adu_to_gs
                z = raw_z * self._adu_to_gs
                if not all(math.isfinite(value) for value in (x, y, z)):
                    rospy.logwarn(
                        f"Non-finite sensor payload detected: cycle_id={cycle_id}, slot={slot}, "
                        f"sensor_id={sid}, raw=({raw_x}, {raw_y}, {raw_z})"
                    )
                    return None
                sensors.append(SensorData(id=sid, x=x, y=y, z=z))
                offset += self.SENSOR_DATA_SIZE

            cycle_end = raw_data[offset]

            bitmap_sensor_count = bin(bitmap).count('1')
            if bitmap_sensor_count != sensor_count:
                rospy.logdebug(
                    f"Bitmap/sensor count mismatch: bitmap={bitmap_sensor_count}, parsed={sensor_count}"
                )

            # Create message
            msg = StmUplink()
            msg.header = Header(stamp=rospy.Time.now(), frame_id='hall_array_frame')
            msg.cycle_id = cycle_id
            msg.slot = slot
            msg.bitmap = bitmap
            msg.timestamp = timestamp
            msg.sensor_data = sensors
            msg.cycle_end = cycle_end

            return msg

        except struct.error as e:
            rospy.logerr(f"Struct unpacking error: {e}")
            return None

    def run(self):
        rospy.loginfo("SerialNodeTDM is ready.")
        buffer = b''

        while not rospy.is_shutdown():
            try:
                if self.ser and self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    buffer += data

                    # Look for header 0xAA55 in buffer (big-endian: \xAA\x55)
                    while len(buffer) >= 2:
                        # Find header position
                        idx = buffer.find(b'\xAA\x55')
                        if idx == -1:
                            # No header found, keep last byte in case it's start of header
                            if len(buffer) > 1:
                                buffer = buffer[-1:]
                            break
                        elif idx > 0:
                            # Discard bytes before header
                            buffer = buffer[idx:]

                        end_idx = buffer.find(self.TERMINATOR, 2)
                        if end_idx == -1:
                            # Wait for a full frame terminated by \r\n
                            break

                        frame = buffer[:end_idx]
                        msg = self._parse_uplink(frame)
                        if msg is not None:
                            # Publish raw data first (for archive/Phase 2-5)
                            self.pub_raw.publish(msg)

                            # Publish raw magnetic field magnitude
                            raw_magnitudes = Float32MultiArray()
                            raw_magnitudes.data = [math.sqrt(s.x**2 + s.y**2 + s.z**2) for s in msg.sensor_data]
                            self.pub_magnitude_raw.publish(raw_magnitudes)

                            # Apply full correction: ellipsoid + R_CORR + consistency
                            corrected_sensors = []
                            for s in msg.sensor_data:
                                cx, cy, cz = self._calib.apply(s.id, s.x, s.y, s.z)
                                corrected_sensors.append(SensorData(id=s.id, x=cx, y=cy, z=cz))

                            # Create corrected message
                            corrected_msg = StmUplink()
                            corrected_msg.header = msg.header
                            corrected_msg.cycle_id = msg.cycle_id
                            corrected_msg.slot = msg.slot
                            corrected_msg.bitmap = msg.bitmap
                            corrected_msg.timestamp = msg.timestamp
                            corrected_msg.sensor_data = corrected_sensors
                            corrected_msg.cycle_end = msg.cycle_end

                            # Publish corrected data
                            self.pub.publish(corrected_msg)

                            # Publish magnetic field magnitude (based on corrected data)
                            magnitudes = Float32MultiArray()
                            magnitudes.data = [math.sqrt(s.x**2 + s.y**2 + s.z**2) for s in corrected_msg.sensor_data]
                            self.pub_magnitude.publish(magnitudes)
                        else:
                            rospy.logwarn(f"Dropping invalid uplink frame (len={len(frame)}): {frame[:16].hex()}...")

                        buffer = buffer[end_idx + len(self.TERMINATOR):]
                else:
                    rospy.sleep(0.001)
            except serial.SerialException as e:
                rospy.logerr(f"Serial error: {e}")
                break

        if self.ser:
            self.ser.close()


if __name__ == '__main__':
    try:
        node = SerialNodeTDM()
        node.run()
    except rospy.ROSInterruptException:
        pass
