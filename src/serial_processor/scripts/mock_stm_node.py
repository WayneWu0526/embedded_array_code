#!/usr/bin/env python3
"""
Mock STM32 Node for testing data_collection_node without real hardware.
Publishes fake StmUplink data simulating TDM slot-based sensor readings.
"""

import rospy
import time
from std_msgs.msg import Header
from serial_processor.msg import SensorData, StmUplink

# Mode constants
MODE_CVT = 0x01  # 4 slots
MODE_CCI = 0x02  # 3 slots


class MockStmNode:
    def __init__(self):
        rospy.init_node('mock_stm_node', anonymous=True)

        # Parameters
        self.mode = int(rospy.get_param('~mode', MODE_CVT))
        self.bitmap = int(rospy.get_param('~bitmap', 0x000F))  # sensors 1-4
        self.cycle_time = rospy.get_param('~cycle_time', 1.0)  # seconds between cycles
        self.slot_interval = rospy.get_param('~slot_interval', 0.2)  # seconds between slots

        # Calculate number of sensors from bitmap
        self.num_sensors = bin(self.bitmap).count('1')
        self.num_slots = 4 if self.mode == MODE_CVT else 3

        # Publisher
        self.pub = rospy.Publisher('stm_uplink', StmUplink, queue_size=10)

        rospy.loginfo(f"MockStmNode: mode={'CVT' if self.mode == MODE_CVT else 'CCI'}, "
                      f"slots={self.num_slots}, sensors={self.num_sensors}, "
                      f"cycle_time={self.cycle_time}s")

    def create_sensor_data(self, slot):
        """Create fake sensor data for a slot"""
        sensors = []
        sensor_id = 1
        for i in range(self.num_sensors):
            # Simulate some variation based on slot
            sensors.append(SensorData(
                id=sensor_id,
                x=float(slot * 0.01),
                y=float(slot * 0.02),
                z=float(slot * 0.03)
            ))
            sensor_id += 1
        return sensors

    def publish_slot(self, cycle_id, slot, cycle_end=False):
        """Publish a single slot message"""
        msg = StmUplink()
        msg.header = Header(stamp=rospy.Time.now(), frame_id='mock_stm')
        msg.cycle_id = cycle_id
        msg.slot = slot
        msg.bitmap = self.bitmap
        msg.timestamp = int(time.time() * 1e6)  # microseconds
        msg.sensor_data = self.create_sensor_data(slot)
        msg.cycle_end = 1 if cycle_end else 0

        self.pub.publish(msg)
        rospy.loginfo(f"Mock: cycle_id={cycle_id}, slot={slot}, "
                      f"sensors={len(msg.sensor_data)}, cycle_end={cycle_end}")

    def run(self):
        """Run the mock node - publish continuous cycles"""
        cycle_id = 0

        while not rospy.is_shutdown():
            # Publish each slot in the cycle
            for slot in range(self.num_slots):
                is_last_slot = (slot == self.num_slots - 1)
                self.publish_slot(cycle_id, slot, is_last_slot)
                rospy.sleep(self.slot_interval)

            cycle_id += 1
            rospy.sleep(self.cycle_time)  # Wait before next cycle


if __name__ == '__main__':
    try:
        node = MockStmNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
