#!/usr/bin/env python3
import rospy
import serial
import threading
from std_msgs.msg import String, Header
from geometry_msgs.msg import Vector3
from serial_processor.srv import GetHallData, GetHallDataResponse

class SerialNode:
    def __init__(self):
        rospy.init_node('serial_processor_node', anonymous=True)

        # 获取参数
        self.port = rospy.get_param('~port', '/dev/ttyACM0')
        self.baudrate = rospy.get_param('~baudrate', 921600)
        self.pub_rate = rospy.get_param('~pub_rate', 100)  # 发布频率上限 Hz

        # 转换系数: 32768 -> 32Gs => 1 LSB = 32/32768 Gs
        self.scale = 32.0 / 32768.0

        # 内部存储最新数据和时间戳
        self.latest_sensors = [Vector3() for _ in range(12)]
        self.latest_timestamp = rospy.Time.now()
        self.data_lock = threading.Lock()

        # 初始化服务（保留，兼容旧代码）
        self.srv = rospy.Service('get_hall_data', GetHallData, self.handle_get_hall_data)

        # 初始化Topic发布者 - 新增：持续发布最新数据
        # 注意：这里我们发布的是 Response 类型，虽然不规范，但这是目前的实现方式
        self.pub = rospy.Publisher('hall_data', GetHallDataResponse, queue_size=100)

        # 初始化订阅者，用于向下位机发送数据
        self.sub = rospy.Subscriber('serial_send', String, self.send_callback)

        # 控制发布频率
        self.last_pub_time = rospy.Time.now()
        self.min_interval = rospy.Duration(1.0 / self.pub_rate)
        
        self.ser = None
        self.connect()

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            rospy.loginfo(f"Connected to {self.port} at {self.baudrate} baud.")
        except Exception as e:
            rospy.logerr(f"Error connecting to serial port: {e}")
            rospy.signal_shutdown("Serial connection failed")

    def handle_get_hall_data(self, req):
        """
        服务回调函数：返回最新的一组 12 个 Vector3 数据
        """
        res = GetHallDataResponse()
        with self.data_lock:
            res.header.stamp = self.latest_timestamp
            res.header.frame_id = "hall_array_frame"
            res.sensors = self.latest_sensors
        return res

    def _publish_hall_data(self):
        """Topic方式发布Hall传感器数据"""
        msg = GetHallDataResponse()
        with self.data_lock:
            msg.header.stamp = self.latest_timestamp
            msg.header.frame_id = "hall_array_frame"
            msg.sensors = self.latest_sensors
        self.pub.publish(msg)

    def send_callback(self, msg):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(msg.data.encode('utf-8'))
            except Exception as e:
                rospy.logerr(f"Error sending to serial: {e}")

    def unpack_data(self, raw_line):
        try:
            line = raw_line.decode('utf-8').strip()
            if not line:
                return None
            
            parts = line.replace(',', ' ').split()
            if len(parts) >= 36:
                # 解析为 12 个 Vector3
                sensors = []
                for i in range(12):
                    v = Vector3()
                    v.x = int(parts[i*3]) * self.scale
                    v.y = int(parts[i*3 + 1]) * self.scale
                    v.z = int(parts[i*3 + 2]) * self.scale
                    sensors.append(v)
                return sensors
        except (ValueError, UnicodeDecodeError):
            pass
        return None

    def run(self):
        rospy.loginfo("Serial Service Node (Structured) is ready.")
        while not rospy.is_shutdown():
            try:
                if self.ser and self.ser.in_waiting > 0:
                    raw_line = self.ser.readline()
                    sensors = self.unpack_data(raw_line)
                    
                    if sensors:
                        with self.data_lock:
                            self.latest_sensors = sensors
                            self.latest_timestamp = rospy.Time.now()

                        # Topic发布：控制频率不超过 pub_rate
                        now = rospy.Time.now()
                        if now - self.last_pub_time >= self.min_interval:
                            self._publish_hall_data()
                            self.last_pub_time = now
                else:
                    import time
                    time.sleep(0.001)
            except serial.SerialException as e:
                rospy.logerr(f"Serial error: {e}")
                break

        if self.ser:
            self.ser.close()

if __name__ == '__main__':
    try:
        node = SerialNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    try:
        node = SerialNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
