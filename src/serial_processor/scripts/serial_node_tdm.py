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
from std_msgs.msg import Header, Float32MultiArray
from serial_processor.msg import SensorData, StmUplink, StmDownlink
from sensor_array_config import get_config, SensorArrayConfig


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

        # Serial connection
        self.ser = None
        self.connect()

        # Publisher for uplink data (corrected: ellipsoid + R_CORR)
        self.pub = rospy.Publisher('stm_uplink', StmUplink, queue_size=100)
        # Publisher for intermediate data: ellipsoid + R_CORR correction, before consistency correction
        self.pub_ellipsoid_rotated = rospy.Publisher('stm_uplink_ellipsoid_rotated', StmUplink, queue_size=100)
        # Publisher for raw uplink data (uncorrected, for archive/Phase 2-5)
        self.pub_raw = rospy.Publisher('stm_uplink_raw', StmUplink, queue_size=100)
        # Publisher for magnetic field magnitude of each sensor
        self.pub_magnitude = rospy.Publisher('stm_magnitude', Float32MultiArray, queue_size=100)
        # Publisher for ellipsoid_rotated magnetic field magnitude (intermediate)
        self.pub_magnitude_ellipsoid_rotated = rospy.Publisher('stm_magnitude_ellipsoid_rotated', Float32MultiArray, queue_size=100)
        # Publisher for raw magnetic field magnitude (uncorrected)
        self.pub_magnitude_raw = rospy.Publisher('stm_magnitude_raw', Float32MultiArray, queue_size=100)

        # Load ellipsoid calibration parameters
        self._load_calibration_params()
        # Load R_CORR rotation correction matrices
        self._load_sensor_array_params()

        # Subscriber for downlink commands
        self.sub = rospy.Subscriber('stm_downlink', StmDownlink, self._downlink_callback)

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

    def _load_calibration_params(self):
        """Load Phase 1 (ellipsoid) and Phase 2 (consistency) calibration params from SensorArrayConfig."""
        # Reset params
        self.offset = {}
        self.correction = {}
        self.D_matrix = {}
        self.e_bias = {}

        # Load Phase 1 (ellipsoid)
        try:
            intrinsic = self._sensor_config.intrinsic
            if intrinsic and intrinsic.params:
                for sid, params in intrinsic.params.items():
                    self.offset[sid] = np.array(params.o_i)
                    self.correction[sid] = np.array(params.C_i)
                rospy.loginfo(f"Loaded ellipsoid calibration for {len(self.offset)} sensors")
            else:
                rospy.logwarn("No intrinsic (ellipsoid) parameters found.")
        except Exception as e:
            rospy.logwarn(f"Failed to load ellipsoid params: {e}")

        # Load Phase 2 (consistency)
        try:
            consistency = self._sensor_config.consistency
            if consistency and consistency.params:
                for sid, params in consistency.params.items():
                    self.D_matrix[sid] = np.array(params.D_i)
                    self.e_bias[sid] = np.array(params.e_i)
                rospy.loginfo(f"Loaded consistency calibration for {len(self.D_matrix)} sensors")
            else:
                rospy.logwarn("No consistency parameters found.")
        except Exception as e:
            rospy.logwarn(f"Failed to load consistency params: {e}")

    def _load_sensor_array_params(self):
        """Load d_list and R_CORR from SensorArrayConfig."""
        try:
            hw = self._sensor_config.hardware
            self._d_list = np.array(hw.d_list)
            manifest = self._sensor_config.manifest
            self._n_sensors = manifest.n_sensors
            self._n_groups = manifest.n_groups
            self._sensors_per_group = manifest.sensors_per_group
            # Build R_CORR dict: sensor_id -> np.array(3x3)
            # Each R_CORR entry has sensor_ids and a 9-element matrix
            self.R_CORR = {}
            for entry in hw.R_CORR:
                mat = np.array(entry.matrix).reshape(3, 3, order='F')
                for sid in entry.sensor_ids:
                    self.R_CORR[sid] = mat
            rospy.loginfo(f"Loaded sensor array params: {self._n_sensors} sensors, {self._n_groups} groups")
        except Exception as e:
            rospy.logwarn(f"Failed to load sensor array params: {e}")
            # Fallback: will use empty configs
            self._d_list = np.array([])
            self._n_sensors = 0
            self._n_groups = 0
            self._sensors_per_group = 0
            self.R_CORR = {}

        # Build sensor -> group lookup (group index is position in R_CORR list)
        self._sensor_to_group = {}
        for idx, entry in enumerate(hw.R_CORR):
            for sid in entry.sensor_ids:
                self._sensor_to_group[sid] = idx + 1

    def _apply_ellipsoid_correction(self, raw_x, raw_y, raw_z, sensor_id):
        """Apply ellipsoid correction to raw sensor data.

        Args:
            raw_x, raw_y, raw_z: Raw sensor readings (after scale)
            sensor_id: Sensor ID (1-12)

        Returns:
            Tuple of (corrected_x, corrected_y, corrected_z)
        """
        if sensor_id not in self.offset or sensor_id not in self.correction:
            return raw_x, raw_y, raw_z

        o_i = self.offset[sensor_id]
        C_i = self.correction[sensor_id]
        b_raw = np.array([raw_x, raw_y, raw_z])
        b_corr = (b_raw - o_i) @ C_i.T
        return b_corr[0], b_corr[1], b_corr[2]

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

                            # Apply ellipsoid correction to sensor data
                            ellipsoid_rotated_sensors = []
                            corrected_sensors = []
                            for s in msg.sensor_data:
                                cx, cy, cz = self._apply_ellipsoid_correction(s.x, s.y, s.z, s.id)
                                # Apply R_CORR rotation (transform from sensor-local to reference frame)
                                if s.id in self.R_CORR:
                                    b_rot = self.R_CORR[s.id] @ np.array([cx, cy, cz])
                                    cx, cy, cz = b_rot[0], b_rot[1], b_rot[2]
                                ellipsoid_rotated_sensors.append(SensorData(id=s.id, x=cx, y=cy, z=cz))

                                # Apply consistency correction: D_i * b + e_i
                                if s.id in self.D_matrix and s.id in self.e_bias:
                                    b_cons = self.D_matrix[s.id] @ np.array([cx, cy, cz]) + self.e_bias[s.id]
                                    cx, cy, cz = b_cons[0], b_cons[1], b_cons[2]
                                
                                corrected_sensors.append(SensorData(id=s.id, x=cx, y=cy, z=cz))

                            # Create ellipsoid_rotated message (intermediate: after ellipsoid + R_CORR, before consistency)
                            ellipsoid_rotated_msg = StmUplink()
                            ellipsoid_rotated_msg.header = msg.header
                            ellipsoid_rotated_msg.cycle_id = msg.cycle_id
                            ellipsoid_rotated_msg.slot = msg.slot
                            ellipsoid_rotated_msg.bitmap = msg.bitmap
                            ellipsoid_rotated_msg.timestamp = msg.timestamp
                            ellipsoid_rotated_msg.sensor_data = ellipsoid_rotated_sensors
                            ellipsoid_rotated_msg.cycle_end = msg.cycle_end

                            # Publish ellipsoid_rotated data
                            self.pub_ellipsoid_rotated.publish(ellipsoid_rotated_msg)

                            # Publish magnetic field magnitude (based on ellipsoid_rotated data)
                            magnitudes_ellipsoid_rotated = Float32MultiArray()
                            magnitudes_ellipsoid_rotated.data = [math.sqrt(s.x**2 + s.y**2 + s.z**2) for s in ellipsoid_rotated_msg.sensor_data]
                            self.pub_magnitude_ellipsoid_rotated.publish(magnitudes_ellipsoid_rotated)

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
