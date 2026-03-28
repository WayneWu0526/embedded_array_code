#!/usr/bin/env python3
"""
TDM (Time Division Multiplexing) Serial Node for STM32 communication.

Binary Protocol:
- Downlink (PC -> STM32): 12 bytes
  Header(2) + Version(1) + Mode(1) + Bitmap(2) + SettlingTime(2) + CycleTime(4)
- Uplink (STM32 -> PC): Variable
  Header(2) + Version(1) + cycle_id(2) + slot(1) + Bitmap(2) + Timestamp(8) + sensor_data(N*13) + cycle_end(1)
"""

import rospy
import serial
import struct
import threading
from std_msgs.msg import Header
from serial_processor.msg import SensorData, StmUplink, StmDownlink


class SerialNodeTDM:
    # Protocol constants
    HEADER = 0xAA55
    UPLINK_MIN_SIZE = 17  # Header(2) + Version(1) + cycle_id(2) + slot(1) + Bitmap(2) + Timestamp(8) + cycle_end(1)

    def __init__(self):
        rospy.init_node('serial_node_tdm', anonymous=True)

        # Parameters
        self.port = rospy.get_param('~port', '/dev/ttyACM0')
        self.baudrate = rospy.get_param('~baudrate', 921600)

        # Serial connection
        self.ser = None
        self.data_lock = threading.Lock()
        self.connect()

        # Publisher for uplink data
        self.pub = rospy.Publisher('stm_uplink', StmUplink, queue_size=100)

        # Subscriber for downlink commands
        self.sub = rospy.Subscriber('stm_downlink', StmDownlink, self._downlink_callback)

        rospy.loginfo(f"SerialNodeTDM initialized: {self.port} at {self.baudrate} baud")

    def connect(self):
        try:
            # 禁用 dsrdtr 防止串口打开时导致单片机重启，并显式设置 DTR/RTS 为 False
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
        rospy.loginfo(f"!!!!!!!!!!!!!!!Received downlink command!!!!!!!!!!!!!")

        if not self.ser or not self.ser.is_open:
            rospy.logerr("!!!!!!!Error: Serial port not connected or open!!!!!!!!!!!!!")
            return

        try:
            # Pack 12-byte command: Header(2) + Version(1) + Mode(1) + Bitmap(2) + SettlingTime(2) + CycleTime(4)
            # STM32 uses big-endian
            # Mode: 0x01=CVT, 0x02=CCI
            # Bitmap: bit0=sensor1, bit11=sensor12
            # SettlingTime: 0.01ms units
            # CycleTime: 0.01ms units, max 10000.00ms
            data = struct.pack('>HBBHHI',
                self.HEADER,
                0x01,  # Version
                msg.mode,
                msg.bitmap,
                msg.settling_time,
                msg.cycle_time
            )
            # 锁定并合并发送：确保数据尽可能作为一个整体发往 USB 链路层
            # 禁用缓冲区以获取更及时的控制，但一次性写入所有内容
            self.ser.write_timeout = 1.0  # 设置写入超时
            
            rospy.loginfo(f"--- ATOMIC SERIAL SEND ---")
            rospy.loginfo(f"Raw Bytes: {' '.join(f'{b:02x}' for b in data)}")
            
            # 使用 bytes() 确保数据是不可变的字节对象，并一次性调用 write
            # 增加一点点写之前的静默时间，清理可能还在进行的读取流
            rospy.sleep(0.01)
            
            # 写入并强制刷新到物理分片
            num_written = self.ser.write(bytes(data))
            self.ser.flush()
            
            if num_written != 12:
                rospy.logwarn(f"Serial write incomplete: only {num_written} bytes sent")

            # 等待回复逻辑保持不变
            rospy.sleep(0.1)  # 稍微延长等待 STM32 处理的时间

            # 使用阻塞读取（带 timeout）等待 STM32 初始化回复
            reply = self.ser.read(3)
            if len(reply) == 3 and reply[0:2] == b'\xAA\x55':
                status = reply[2]
                if status == self.INIT_STATUS_SUCCESS:
                    rospy.loginfo("STM32 initialized successfully")
                elif status == self.INIT_STATUS_PARAM_ERROR:
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

            # Parse sensor data based on bitmap
            sensor_count = bin(bitmap).count('1')
            expected_len = self.UPLINK_MIN_SIZE + sensor_count * 13

            if len(raw_data) < expected_len:
                rospy.logwarn(f"Data too short: need {expected_len}, got {len(raw_data)}")
                return None

            sensors = []
            offset = 16
            for _ in range(sensor_count):
                # Unpack 13 bytes: id(1) + x(4) + y(4) + z(4) (big-endian floats)
                sid, x, y, z = struct.unpack('>Bfff', raw_data[offset:offset+13])
                sensors.append(SensorData(id=sid, x=x, y=y, z=z))
                offset += 13

            cycle_end = raw_data[offset]

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

                        # Try to parse
                        msg = self._parse_uplink(buffer)
                        if msg is not None:
                            self.pub.publish(msg)
                            # Remove parsed data from buffer
                            sensor_count = bin(msg.bitmap).count('1')
                            parsed_len = self.UPLINK_MIN_SIZE + sensor_count * 13
                            buffer = buffer[parsed_len:]
                        else:
                            # Need more data, wait for next read
                            break
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
