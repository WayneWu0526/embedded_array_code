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
import numpy as np
from std_msgs.msg import Header, Float32MultiArray, Bool, String
from datetime import datetime
from serial_processor.msg import SensorData, StmUplink, StmDownlink
from sensor_array_config import get_config, SensorArrayConfig


class ManualRecorder:
    """Records stm_uplink_raw data to CSV on trigger."""

    STATE_IDLE = 'idle'
    STATE_RECORDING = 'recording'
    STATE_PAUSED = 'paused'

    def __init__(self, output_dir, frames_to_average, n_sensors=12):
        self.output_dir = output_dir
        self.frames_to_average = frames_to_average
        self.n_sensors = n_sensors
        self._state = self.STATE_IDLE
        self._file = None
        self._buffer = []  # list of StmUplink messages
        self._csv_path = None
        if self.frames_to_average <= 0:
            raise ValueError("frames_to_average must be positive")

    @property
    def state(self):
        return self._state

    def _write_header(self, f):
        header = []
        for i in range(1, self.n_sensors + 1):
            header.extend([f'sensor_{i}_x', f'sensor_{i}_y', f'sensor_{i}_z'])
        f.write(','.join(header) + '\n')

    def _average_buffer(self):
        """Compute per-sensor average from buffered StmUplink messages."""
        if not self._buffer:
            return None
        n = len(self._buffer)
        sum_x = [0.0] * self.n_sensors
        sum_y = [0.0] * self.n_sensors
        sum_z = [0.0] * self.n_sensors
        for msg in self._buffer:
            for idx, s in enumerate(msg.sensor_data):
                sum_x[idx] += s.x
                sum_y[idx] += s.y
                sum_z[idx] += s.z
        avg = [0.0] * self.n_sensors
        for i in range(self.n_sensors):
            avg[i] = (sum_x[i] / n, sum_y[i] / n, sum_z[i] / n)
        timestamp = self._buffer[-1].timestamp
        self._buffer.clear()
        return timestamp, avg

    def trigger(self, enable):
        """Handle trigger message."""
        if enable:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if self._state == self.STATE_IDLE:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._csv_path = os.path.join(self.output_dir, f'manual_record_{ts}.csv')
            try:
                self._file = open(self._csv_path, 'w')
                self._write_header(self._file)
                rospy.loginfo(f"[ManualRecorder] Recording started: {self._csv_path}")
            except IOError as e:
                rospy.logerr(f"[ManualRecorder] Failed to open file for writing: {e}")
                self._state = self.STATE_IDLE
                return
        self._state = self.STATE_RECORDING

    def _stop_recording(self):
        """Stop recording and close file, returning to IDLE state."""
        if self._state in (self.STATE_RECORDING, self.STATE_PAUSED):
            self.flush_and_close()
            self._state = self.STATE_IDLE
            rospy.loginfo("[ManualRecorder] Recording stopped")

    def on_uplink_raw(self, msg):
        """Process incoming stm_uplink_raw message. Returns True if row was written."""
        if self._state != self.STATE_RECORDING:
            return False
        self._buffer.append(msg)
        if len(self._buffer) >= self.frames_to_average:
            result = self._average_buffer()
            if result is None:
                return False
            timestamp, avg = result
            row = []
            for (x, y, z) in avg:
                row.extend([f'{x:.6f}', f'{y:.6f}', f'{z:.6f}'])
            self._file.write(','.join(row) + '\n')
            return True
        return False

    def flush_and_close(self):
        """Flush and close file. Called on shutdown."""
        if self._file:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
        self._buffer.clear()
        self._state = self.STATE_IDLE
        rospy.loginfo("[ManualRecorder] File closed")


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

        # Manual record parameters
        self.output_dir = os.path.expanduser(rospy.get_param('~output_dir', '~/embedded_array_ws/src/sensor_data_collection/data'))
        self.frames_to_average = int(rospy.get_param('~frames_to_average', 10))
        os.makedirs(self.output_dir, exist_ok=True)
        rospy.loginfo(f"Manual record output directory: {self.output_dir}")

        # Serial connection
        self.ser = None
        self.connect()

        # Publisher for calibrated uplink data (orientation-aligned raw + affine D_i, e_i)
        self.pub = rospy.Publisher('stm_uplink', StmUplink, queue_size=100)
        # Publisher for raw uplink data: ADU->Gs plus R_CORR orientation alignment, no affine calibration
        self.pub_raw = rospy.Publisher('stm_uplink_raw', StmUplink, queue_size=100)
        # Publisher for magnetic field magnitude of calibrated data
        self.pub_magnitude = rospy.Publisher('stm_magnitude', Float32MultiArray, queue_size=100)
        # Publisher for raw magnetic field magnitude (orientation-aligned raw)
        self.pub_magnitude_raw = rospy.Publisher('stm_magnitude_raw', Float32MultiArray, queue_size=100)

        # Load R_CORR orientation alignment matrices
        self._load_sensor_array_params()
        # Load affine model calibration D_i, e_i from sensor config
        affine = self._sensor_config.affine_model
        self.D_matrix = {}
        self.e_bias = {}
        for sid, params in affine.params.items():
            self.D_matrix[sid] = np.array(params.D_i)
            self.e_bias[sid] = np.array(params.e_i).squeeze()
        rospy.loginfo(f"Loaded affine model calibration for {len(self.D_matrix)} sensors from config")

        # Initialize manual recorder
        self.recorder = ManualRecorder(self.output_dir, self.frames_to_average)
        # Subscribe to own stm_uplink_raw for recording (self-subscribe)
        self.sub_record = rospy.Subscriber('stm_uplink_raw', StmUplink, self._on_uplink_raw_record)
        # Subscribe to trigger
        self.sub_trigger = rospy.Subscriber('~record_trigger', Bool, self._on_record_trigger)
        # Status publisher
        self.pub_status = rospy.Publisher('stm_manual_record/status', String, queue_size=10)

        # Subscriber for downlink commands
        self.sub = rospy.Subscriber('stm_downlink', StmDownlink, self._downlink_callback)

        rospy.loginfo(
            f"SerialNodeTDM initialized: {self.port} at {self.baudrate} baud, "
            f"sensor_float_endian={self.float_endian_name}"
        )

        # Register shutdown handler
        rospy.on_shutdown(self._on_shutdown_record)

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

    def _load_sensor_array_params(self):
        """Load d_list and R_CORR orientation alignment from SensorArrayConfig."""
        try:
            hw = self._sensor_config.hardware
            self._d_list = np.array(hw.d_list)
            manifest = self._sensor_config.manifest
            self._n_sensors = manifest.n_sensors
            self._n_groups = manifest.n_groups
            self._sensors_per_group = manifest.sensors_per_group
            r_corr_entries = hw.R_CORR
            # Build R_CORR dict: sensor_id -> np.array(3x3)
            # Each R_CORR entry has sensor_ids and a 9-element matrix
            self.R_CORR = {}
            for entry in r_corr_entries:
                mat = np.array(entry.matrix).reshape(3, 3, order='F')
                for sid in entry.sensor_ids:
                    self.R_CORR[sid] = mat
            self._sensor_to_group = {}
            for idx, entry in enumerate(r_corr_entries):
                for sid in entry.sensor_ids:
                    self._sensor_to_group[sid] = idx + 1
            rospy.loginfo(f"Loaded sensor array params: {self._n_sensors} sensors, {self._n_groups} groups")
        except Exception as e:
            rospy.logwarn(f"Failed to load sensor array params: {e}")
            # Fallback: will use empty configs
            self._d_list = np.array([])
            self._n_sensors = 0
            self._n_groups = 0
            self._sensors_per_group = 0
            self.R_CORR = {}
            self._sensor_to_group = {}

    def _on_record_trigger(self, msg):
        """Handle record trigger (Bool)."""
        enable = msg.data
        self.recorder.trigger(enable)
        self.pub_status.publish(String(data=self.recorder.state))

    def _on_uplink_raw_record(self, msg):
        """Process stm_uplink_raw for recording."""
        self.recorder.on_uplink_raw(msg)

    def _on_shutdown_record(self):
        self.recorder.flush_and_close()

    def _align_raw_axes(self, sensor_id, x, y, z):
        """Apply R_CORR orientation alignment to raw Gs readings."""
        R = self.R_CORR.get(sensor_id)
        if R is None:
            return x, y, z
        b_aligned = R @ np.array([x, y, z])
        return float(b_aligned[0]), float(b_aligned[1]), float(b_aligned[2])

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
                x, y, z = self._align_raw_axes(sid, x, y, z)
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
                            # Publish raw data first. Raw means ADU->Gs and R_CORR orientation-aligned,
                            # but not affine-calibrated.
                            self.pub_raw.publish(msg)

                            # Publish raw magnetic field magnitude
                            raw_magnitudes = Float32MultiArray()
                            raw_magnitudes.data = [math.sqrt(s.x**2 + s.y**2 + s.z**2) for s in msg.sensor_data]
                            self.pub_magnitude_raw.publish(raw_magnitudes)

                            # Apply affine calibration: D_i @ b + e_i
                            corrected_sensors = []
                            for s in msg.sensor_data:
                                cx, cy, cz = float(s.x), float(s.y), float(s.z)
                                if s.id in self.D_matrix and s.id in self.e_bias:
                                    b_cons = self.D_matrix[s.id] @ np.array([cx, cy, cz]) + self.e_bias[s.id]
                                    cx, cy, cz = float(b_cons[0]), float(b_cons[1]), float(b_cons[2])
                                corrected_sensors.append(SensorData(id=s.id, x=cx, y=cy, z=cz))

                            # Create calibrated message
                            corrected_msg = StmUplink()
                            corrected_msg.header = msg.header
                            corrected_msg.cycle_id = msg.cycle_id
                            corrected_msg.slot = msg.slot
                            corrected_msg.bitmap = msg.bitmap
                            corrected_msg.timestamp = msg.timestamp
                            corrected_msg.sensor_data = corrected_sensors
                            corrected_msg.cycle_end = msg.cycle_end

                            # Publish calibrated data
                            self.pub.publish(corrected_msg)

                            # Publish magnetic field magnitude (based on calibrated data)
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
