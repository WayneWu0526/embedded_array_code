#!/usr/bin/env python3
"""
Data Collection Node for TDM-based sensor array system.

Workflow:
1. Send downlink command (mode, bitmap, settling_time) to STM32
2. Receive uplink data (stm_uplink) with sensor readings per slot
3. Query TF for pose corresponding to each slot
4. Build cycle package and save to JSON
5. Call GELS service when cycle is complete
"""

import rospy
import json
import os
from datetime import datetime

from geometry_msgs.msg import TransformStamped
from serial_processor.msg import StmUplink, StmDownlink
from serial_processor.srv import GetHallData, GetHallDataResponse
from std_msgs.msg import Bool


# Mode constants
MODE_CVT = 0x01  # Constant Voltage Mode: 4 slots
MODE_CCI = 0x02  # Constant Current Mode: 3 slots

# Mode to slot count
MODE_SLOT_COUNT = {
    MODE_CVT: 4,
    MODE_CCI: 3,
}


class SlotBuffer:
    """Buffer for a single slot's data"""

    def __init__(self, slot, bitmap, timestamp, sensor_data):
        self.slot = slot
        self.bitmap = bitmap
        self.timestamp = timestamp
        self.sensor_data = sensor_data
        self.pose = None  # Set by DataCollector


class CycleBuffer:
    """Buffer for a complete cycle's data"""

    def __init__(self, cycle_id, mode):
        self.cycle_id = cycle_id
        self.mode = mode
        self.num_slots = MODE_SLOT_COUNT.get(mode, 4)
        self.slots = {}  # slot -> SlotBuffer
        self.ground_truth_pose = None
        self.stm_timestamp = None
        self.pc_timestamp = None

    def add_slot(self, slot_buffer):
        self.slots[slot_buffer.slot] = slot_buffer
        if self.stm_timestamp is None or slot_buffer.timestamp > self.stm_timestamp:
            self.stm_timestamp = slot_buffer.timestamp

    def is_complete(self):
        """Check if all expected slots have been received"""
        if len(self.slots) < self.num_slots:
            return False
        # Check slot 3 is excluded for CCI mode
        if self.mode == MODE_CCI and 3 in self.slots:
            return False
        return True

    def to_dict(self):
        """Convert to JSON-serializable dict"""
        slot_data = []
        for slot in range(self.num_slots):
            if slot == 3 and self.mode == MODE_CCI:
                continue  # CCI mode has no slot 3
            if slot not in self.slots:
                continue
            sb = self.slots[slot]
            entry = {
                'slot': slot,
                'sensor_data': [
                    {'id': s.id, 'x': s.x, 'y': s.y, 'z': s.z}
                    for s in sb.sensor_data
                ]
            }
            if sb.pose is not None:
                entry['pose'] = sb.pose
            slot_data.append(entry)

        mode_str = 'CVT' if self.mode == MODE_CVT else 'CCI'
        return {
            'header': {
                'cycle_id': self.cycle_id,
                'mode': mode_str,
                'num_slots': self.num_slots,
            },
            'stm_timestamp': self.stm_timestamp,
            'pc_timestamp': self.pc_timestamp,
            'slot_data': slot_data,
            'ground_truth_pose': self.ground_truth_pose,
        }


class DataCollector:
    """Main data collection node"""

    def __init__(self):
        # Parameters
        self.num_cycles = int(rospy.get_param('~num_cycles', 10))
        self.output_dir = rospy.get_param('~output_dir',
                                          os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
        # Convert mode to integer: rosparam may load as string "CVT"/"CCI" or int
        mode_raw = rospy.get_param('~mode', MODE_CVT)
        if mode_raw == 'CVT' or mode_raw == 1 or mode_raw == '1':
            self.mode = MODE_CVT
        elif mode_raw == 'CCI' or mode_raw == 2 or mode_raw == '2':
            self.mode = MODE_CCI
        else:
            self.mode = MODE_CVT
        self.bitmap = int(rospy.get_param('~bitmap', 0x0FFF))  # All 12 sensors
        self.settling_time = int(rospy.get_param('~settling_time', 10000))  # 100ms = 10000 units
        self.cycle_time = int(rospy.get_param('~cycle_time', 100000))  # 1000ms = 100000 units

        # Load slot->frame mapping from config
        self.slot_frame_map = rospy.get_param('~slot_frame_mapping', {})

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Current cycle being built
        self.current_cycle = None
        self.completed_cycles = 0

        # TF listener
        self.tf_buffer = None
        self._init_tf()

        # Publishers
        self.pub_stm_downlink = rospy.Publisher('stm_downlink', StmDownlink, queue_size=10)

        # FY8300 signal generator output control (enable after STM32 init)
        self.pub_fy8300_ch = [
            rospy.Publisher(f'/fy8300/ch{i}/output_en', Bool, queue_size=1)
            for i in range(1, 4)
        ]

        # Subscribers
        self.sub_stm_uplink = rospy.Subscriber('stm_uplink', StmUplink, self._uplink_callback)

        # Subscribe to old hall_data for backward compatibility (optional)
        rospy.Subscriber('/serial_processor/hall_data', GetHallDataResponse, self._hall_callback)

        rospy.sleep(0.5)  # 等待 publisher/subscriber 连接建立

        # Send initial downlink command to STM32
        self._send_downlink()

        rospy.loginfo(f"DataCollector initialized: mode={'CVT' if self.mode == MODE_CVT else 'CCI'}, "
                      f"num_cycles={self.num_cycles}, bitmap=0x{self.bitmap:03X}, "
                      f"settling_time={self.settling_time}, cycle_time={self.cycle_time}")

    def _init_tf(self):
        """Initialize TF2 listener and wait for required frames to be available"""
        try:
            from tf2_ros import Buffer, TransformListener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer)

            # Wait for required TF frames to be published
            required_frames = ['diana7_em_tcp_filt', 'arm1_em_tcp_filt',
                               'arm2_em_tcp_filt', 'sensor_array_filt']
            rospy.loginfo("Waiting for TF frames to become available...")
            for frame in required_frames:
                timeout = 60.0  # max 60s per frame
                rate = rospy.Rate(10)
                t0 = rospy.Time.now()
                while not rospy.is_shutdown():
                    try:
                        self.tf_buffer.lookup_transform('lab_table', frame, rospy.Time(0))
                        rospy.loginfo(f"  {frame}: available")
                        break
                    except Exception:
                        if (rospy.Time.now() - t0).to_sec() > timeout:
                            rospy.logwarn(f"  {frame}: timeout, proceeding anyway")
                            break
                    rate.sleep()
            rospy.loginfo("TF2 listener initialized")
        except ImportError as e:
            rospy.logwarn(f"tf2_ros not available: {e}")
            self.tf_buffer = None

    def _send_downlink(self):
        """Send initial configuration to STM32"""
        msg = StmDownlink()
        # self.mode is already an integer from __init__, but double-check
        msg.mode = int(self.mode)
        msg.bitmap = self.bitmap
        msg.settling_time = self.settling_time
        msg.cycle_time = self.cycle_time
        self.pub_stm_downlink.publish(msg)
        rospy.loginfo(f"Sent downlink: mode=0x{msg.mode:02X}, bitmap=0x{msg.bitmap:03X}, "
                      f"settling_time={msg.settling_time}, cycle_time={msg.cycle_time}")

        # Enable FY8300 signal generator outputs after STM32 is initialized
        rospy.sleep(0.2)
        enable_msg = Bool()
        enable_msg.data = True
        for pub in self.pub_fy8300_ch:
            pub.publish(enable_msg)
        rospy.loginfo("FY8300 outputs enabled")

    def _lookup_pose(self, frame):
        """Look up transform from lab_table to frame"""
        if self.tf_buffer is None:
            return None
        try:
            t = self.tf_buffer.lookup_transform('lab_table', frame, rospy.Time(0))
            return {
                'position': {
                    'x': t.transform.translation.x,
                    'y': t.transform.translation.y,
                    'z': t.transform.translation.z,
                },
                'rotation': {
                    'x': t.transform.rotation.x,
                    'y': t.transform.rotation.y,
                    'z': t.transform.rotation.z,
                    'w': t.transform.rotation.w,
                }
            }
        except Exception as e:
            rospy.logwarn(f"TF lookup failed for {frame}: {e}")
            return None

    def _uplink_callback(self, msg):
        """Handle incoming uplink data from STM32"""
        slot = msg.slot
        cycle_id = msg.cycle_id
        is_cycle_end = msg.cycle_end == 1

        # Check if we need to start a new cycle
        if self.current_cycle is None or self.current_cycle.cycle_id != cycle_id:
            self.current_cycle = CycleBuffer(cycle_id, self.mode)

        # Create slot buffer
        slot_buffer = SlotBuffer(
            slot=slot,
            bitmap=msg.bitmap,
            timestamp=msg.timestamp,
            sensor_data=msg.sensor_data
        )

        # Look up pose for this slot
        frame = self.slot_frame_map.get(slot)
        if frame and frame != 'null':
            slot_buffer.pose = self._lookup_pose(frame)

        # If cycle end, also look up ground truth pose
        if is_cycle_end:
            self.current_cycle.ground_truth_pose = self._lookup_pose('sensor_array_filt')
            self.current_cycle.pc_timestamp = rospy.Time.now().to_sec()

        self.current_cycle.add_slot(slot_buffer)

        # Log
        rospy.logdebug(f"Received: cycle_id={cycle_id}, slot={slot}, "
                       f"cycle_end={is_cycle_end}, sensors={len(msg.sensor_data)}")

        # Check if cycle is complete
        if is_cycle_end and self.current_cycle.is_complete():
            self._finalize_cycle()

    def _hall_callback(self, msg):
        """Legacy hall data callback for backward compatibility"""
        # Can be used for monitoring or logging
        pass

    def _finalize_cycle(self):
        """Process completed cycle: save to JSON and call GELS service"""
        if self.current_cycle is None:
            return

        cycle_data = self.current_cycle.to_dict()

        # Save to JSON file
        filename = os.path.join(self.output_dir, f"cycle_{cycle_data['header']['cycle_id']:04d}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(cycle_data, f, indent=2)
            rospy.loginfo(f"Saved cycle {cycle_data['header']['cycle_id']} to {filename}")
        except Exception as e:
            rospy.logerr(f"Failed to save cycle JSON: {e}")

        self.completed_cycles += 1
        rospy.loginfo(f"Completed {self.completed_cycles}/{self.num_cycles} cycles")

        # Reset current cycle
        self.current_cycle = None

        # Check if all cycles are done
        if self.completed_cycles >= self.num_cycles:
            rospy.loginfo("All cycles completed, shutting down")
            rospy.signal_shutdown('Data collection complete')


def main():
    rospy.init_node('data_collection_node')
    collector = DataCollector()
    rospy.spin()


if __name__ == '__main__':
    main()
