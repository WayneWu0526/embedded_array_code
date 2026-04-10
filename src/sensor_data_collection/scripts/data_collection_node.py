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
import math
import threading
from datetime import datetime
from collections import deque

from geometry_msgs.msg import TransformStamped, Pose
from serial_processor.msg import StmUplink, StmDownlink
from serial_processor.srv import GetHallData, GetHallDataResponse
from std_msgs.msg import Bool, Float64
from sensor_data_collection.msg import SlotData, SensorReading
from sensor_data_collection.srv import LocalizeCycle, LocalizeCycleRequest


# Mode constants
MODE_CVT = 0x01  # Constant Voltage Mode: 4 slots
MODE_CCI = 0x02  # Constant Current Mode: 3 slots
MODE_MANUAL = 0x00  # Manual trigger mode (continuous data)

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

    def __init__(self, cycle_id, mode, bitmap=0x0FFF):
        self.cycle_id = cycle_id
        self.mode = mode
        self.bitmap = bitmap
        self.num_slots = MODE_SLOT_COUNT.get(mode, 4)
        self.slots = {}  # slot -> SlotBuffer
        self.ground_truth_pose = None
        self.stm_timestamp = None
        self.pc_timestamp = None

    @staticmethod
    def _parse_bitmap(bitmap):
        """Parse bitmap to list of sensor IDs (1-indexed)"""
        sensor_ids = []
        for i in range(12):
            if bitmap & (1 << i):
                sensor_ids.append(i + 1)
        return sensor_ids

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
                'sensor_ids': self._parse_bitmap(self.bitmap),
            },
            'stm_timestamp': self.stm_timestamp,
            'pc_timestamp': self.pc_timestamp,
            'slot_data': slot_data,
            'ground_truth_pose': self.ground_truth_pose,
        }


class DataCollector:
    """Main data collection node supporting both auto and manual trigger modes.

    Auto mode (manual_trigger=false): FY8300 triggers cycles, STM32 sends slot data with cycle_end.
    Manual mode (manual_trigger=true): User presses Enter to save each position, STM32 sends continuous data.
    """

    def __init__(self):
        # Check if running in manual trigger mode
        self.manual_trigger = rospy.get_param('~manual_trigger', False)

        # Parameters
        self.num_cycles = int(rospy.get_param('~num_cycles', 10))
        if self.num_cycles < 0:
            rospy.logwarn("num_cycles is negative, clamping to 0 for uint8 cycle_num")
            self.num_cycles = 0
        elif self.num_cycles > 255:
            rospy.logwarn("num_cycles exceeds uint8 range, clamping to 255 for cycle_num")
            self.num_cycles = 255
        self.output_dir = rospy.get_param('~output_dir',
                                          os.path.join(os.path.dirname(os.path.dirname(__file__)), '/home/zhang/embedded_array_ws/src/sensor_data_collection/result'))

        # Convert mode to integer: rosparam may load as string "CVT"/"CCI" or int
        mode_raw = rospy.get_param('~mode', MODE_CVT)
        if mode_raw == 'CVT' or mode_raw == 1 or mode_raw == '1':
            self.mode = MODE_CVT
        elif mode_raw == 'CCI' or mode_raw == 2 or mode_raw == '2':
            self.mode = MODE_CCI
        else:
            self.mode = MODE_CVT
        self.bitmap = int(rospy.get_param('~bitmap', 0x0FFF))  # All 12 sensors
        self.settling_time = int(rospy.get_param('~settling_time', 10000))  # 0.01ms units
        self.sampling_time = int(rospy.get_param('~sampling_time', 1400))  # 0.01ms units
        if self.sampling_time < 1400:
            rospy.logwarn("sampling_time is too short, may cause issues with sensor readings, setting to minimum 1400 (14ms)")
            self.sampling_time = 1400

        # Manual mode specific parameters
        if self.manual_trigger:
            # num_positions: CVT=4 slots, CCI=3 slots, can be overridden
            self.num_positions = int(rospy.get_param('~num_positions', 4 if self.mode == MODE_CVT else 3))
            self.num_frames_to_average = int(rospy.get_param('~num_frames_to_average', 10))
            # Load manual mode slot->frame mapping
            raw_manual_map = rospy.get_param('~manual_slot_frame_mapping', {})
            self.manual_slot_frame_map = {}
            for k, v in raw_manual_map.items():
                key = int(k)
                if v in ('none', 'null', None):
                    self.manual_slot_frame_map[key] = None
                else:
                    self.manual_slot_frame_map[key] = v
            # Manual mode uses Mode 0x00 (continuous)
            self.stm_mode = MODE_MANUAL
            # Manual mode cycle tracking
            self.cycle_id = 0
            self.cycle_slot_data = []  # List of slot data dicts for current cycle
            self.collected_positions = []  # List of slot indices that have been saved
            self.slot_buffers = [deque(maxlen=self.num_frames_to_average) for _ in range(self.num_positions)]
        else:
            # Calculate cycle_time: CVT=4 slots, CCI=3 slots
            num_slots = 4 if self.mode == MODE_CVT else 3
            self.cycle_time = num_slots * (self.settling_time + self.sampling_time)
            # Auto mode uses CVT/CCI mode for STM32
            self.stm_mode = self.mode

        # Load slot->frame mapping from config and normalize key types
        # YAML keys may be strings or ints; rosparam always returns strings
        raw_map = rospy.get_param('~slot_frame_mapping', {})
        self.slot_frame_map = {}
        for k, v in raw_map.items():
            key = int(k)  # normalize key to int
            if v in ('none', 'null', None):
                self.slot_frame_map[key] = None
            else:
                self.slot_frame_map[key] = v

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Current cycle being built
        self.current_cycle = None
        self.completed_cycles = 0

        # Shutdown handling
        self._shutdown_requested = threading.Event()
        rospy.on_shutdown(self._on_shutdown)

        # TF listener
        self.tf_buffer = None
        self._init_tf()

        # Publishers
        self.pub_stm_downlink = rospy.Publisher('stm_downlink', StmDownlink, queue_size=10)

        # Manual mode: publish sensor data for external monitoring
        if self.manual_trigger:
            self.pub_hall_data = rospy.Publisher('~hall_data', StmUplink, queue_size=10)

        # FY8300 signal generator output control (only for auto mode)
        if not self.manual_trigger:
            self.pub_fy8300_ch = [
                rospy.Publisher(f'/fy8300/ch{i}/output_en', Bool, queue_size=1)
                for i in range(1, 4)
            ]
            # FY8300 frequency publisher per channel
            self.pub_fy8300_freq = [
                rospy.Publisher(f'/fy8300/ch{i}/frequency', Float64, queue_size=1)
                for i in range(1, 4)
            ]

        # Subscribers
        self.sub_stm_uplink = rospy.Subscriber('stm_uplink', StmUplink, self._uplink_callback)

        # Subscribe to old hall_data for backward compatibility (optional)
        rospy.Subscriber('/serial_processor/hall_data', GetHallDataResponse, self._hall_callback)

        # Service client for localization
        rospy.loginfo("Waiting for localization service...")
        rospy.wait_for_service('localize_cycle')
        self.localize_client = rospy.ServiceProxy('localize_cycle', LocalizeCycle)
        rospy.loginfo("Localization service connected")

        rospy.sleep(0.5)  # 等待 publisher/subscriber 连接建立

        if self.manual_trigger:
            # Manual mode: start keyboard listener thread
            rospy.loginfo(f"DataCollector initialized: mode={self._get_mode_str()}, manual_trigger=true, "
                          f"num_positions={self.num_positions}, num_frames_to_average={self.num_frames_to_average}, "
                          f"manual_slot_frame_mapping={self.manual_slot_frame_map}")
            self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            self.keyboard_thread.start()
        else:
            # Auto mode: FY8300 + STM32 initialization
            # 初始化顺序（按 README）：
            # 1. ZED2i TF 已就绪（由 localization_tagslam.launch 保证）
            # 2. FY8300 设置目标频率并使能输出，等待稳定
            # 3. STM32 下行指令最后下发

            # 计算频率
            frequency = 100000.0 / self.cycle_time
            rospy.loginfo(f"FY8300 frequency: {frequency:.3f} Hz (cycle_time={self.cycle_time})")

            # FY8300 直接设置目标频率并使能输出，无需先置0
            self._publish_fy8300_frequency(frequency)
            self._set_fy8300_output_enabled(True)
            rospy.loginfo("FY8300 outputs enabled, waiting for stabilization...")
            rospy.sleep(10.0)  # 等待 FY8300 输出稳定

            rospy.loginfo(f"DataCollector initialized: mode={self._get_mode_str()}, manual_trigger=false, "
                          f"num_cycles={self.num_cycles}, bitmap=0x{self.bitmap:03X}, "
                          f"settling_time={self.settling_time}, cycle_time={self.cycle_time}")

        # 最后发送 STM32 下行指令
        self._send_downlink()

    def _get_mode_str(self):
        """Get mode as string for logging"""
        if self.manual_trigger:
            return f"CVT/CCI+MANUAL"
        return 'CVT' if self.mode == MODE_CVT else 'CCI'

    def _init_tf(self):
        """Initialize TF2 listener and wait for required frames to be available"""
        try:
            from tf2_ros import Buffer, TransformListener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer)

            # Determine which frames to wait for based on mode
            if self.manual_trigger:
                # Manual mode: only wait for frames that are actually used
                required_frames = set(self.manual_slot_frame_map.values())
                required_frames.discard(None)  # Remove null entries
                required_frames.add('sensor_array_filt')  # Always need ground truth
            else:
                # Auto mode: wait for all frames
                required_frames = ['diana7_em_tcp_filt', 'arm1_em_tcp_filt',
                                   'arm2_em_tcp_filt', 'sensor_array_filt']

            required_frames = list(required_frames)
            rospy.loginfo(f"Waiting for TF frames: {required_frames}...")
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
        """Send configuration to STM32"""
        msg = StmDownlink()
        msg.mode = int(self.stm_mode)
        msg.bitmap = self.bitmap
        if self.manual_trigger:
            # Manual mode: continuous data, timing params set to 0
            msg.settling_time = 0
            msg.cycle_time = 0
            msg.cycle_num = 0
        else:
            msg.settling_time = self.settling_time
            msg.cycle_time = self.cycle_time
            msg.cycle_num = self.num_cycles
        self.pub_stm_downlink.publish(msg)
        mode_str = 'MANUAL' if self.manual_trigger else self._get_mode_str()
        rospy.loginfo(f"Sent downlink: mode=0x{msg.mode:02X} ({mode_str}), bitmap=0x{msg.bitmap:03X}, "
                      f"settling_time={msg.settling_time}, cycle_time={msg.cycle_time}, "
                      f"cycle_num={msg.cycle_num}")

    def _publish_fy8300_frequency(self, frequency):
        """Publish the same FY8300 frequency to all channels."""
        freq_msg = Float64()
        freq_msg.data = frequency
        for pub in self.pub_fy8300_freq:
            pub.publish(freq_msg)

    def _set_fy8300_output_enabled(self, enabled):
        """Enable or disable all FY8300 outputs."""
        enable_msg = Bool()
        enable_msg.data = enabled
        for pub in self.pub_fy8300_ch:
            pub.publish(enable_msg)

    def _shutdown_fy8300(self):
        """Stop FY8300 output on node shutdown."""
        if not hasattr(self, 'pub_fy8300_freq') or not hasattr(self, 'pub_fy8300_ch'):
            return
        self._publish_fy8300_frequency(0.0)
        self._set_fy8300_output_enabled(False)

    def _on_shutdown(self):
        """Handle node shutdown gracefully."""
        rospy.loginfo("Shutdown initiated...")
        self._shutdown_requested.set()

        # Shutdown FY8300 if in auto mode
        if not self.manual_trigger:
            self._shutdown_fy8300()

        # Wait for keyboard thread to finish (with timeout)
        if hasattr(self, 'keyboard_thread') and self.keyboard_thread.is_alive():
            rospy.loginfo("Waiting for keyboard thread to finish...")
            self.keyboard_thread.join(timeout=2.0)
            if self.keyboard_thread.is_alive():
                rospy.logwarn("Keyboard thread did not finish in time")

        # Save any pending data
        if hasattr(self, 'slot_buffers') and self.slot_buffers:
            for i, buf in enumerate(self.slot_buffers):
                if buf and len(buf) > 0 and i not in self.collected_positions:
                    rospy.logwarn(f"Slot {i} has unsaved data ({len(buf)} frames)")

        rospy.loginfo("Shutdown complete")

    # ========== Manual Mode Methods ==========

    def _get_active_slot(self):
        """Get the next slot index to collect, or None if all slots are complete"""
        for i in range(self.num_positions):
            if i not in self.collected_positions:
                return i
        return None

    def _average_sensor_data(self, buffer):
        """Average N frames of sensor data"""
        if not buffer:
            return []
        first = buffer[0]['sensor_data']
        if not first:
            return []

        num_sensors = len(first)
        avg_data = []
        for sensor_idx in range(num_sensors):
            sum_x = sum(frame['sensor_data'][sensor_idx].x for frame in buffer)
            sum_y = sum(frame['sensor_data'][sensor_idx].y for frame in buffer)
            sum_z = sum(frame['sensor_data'][sensor_idx].z for frame in buffer)
            count = len(buffer)
            reading = SensorReading()
            reading.id = first[sensor_idx].id
            reading.x = sum_x / count
            reading.y = sum_y / count
            reading.z = sum_z / count
            avg_data.append(reading)
        return avg_data

    def _keyboard_loop(self):
        """Listen for Enter key presses in separate thread, print status"""
        import sys
        import tty
        import termios

        rospy.loginfo("\n========================================")
        rospy.loginfo("  MANUAL TRIGGER MODE - 手动采集模式")
        rospy.loginfo("  Press ENTER to save each position")
        rospy.loginfo("  按 Enter 键保存每个位置")
        rospy.loginfo("  Press 'q' to quit immediately")
        rospy.loginfo("  按 q 立即退出")
        rospy.loginfo("  Press Ctrl+C to exit")
        rospy.loginfo("========================================\n")

        old_settings = None
        is_tty = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False

        if sys.platform == 'linux' and is_tty:
            try:
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setraw(sys.stdin.fileno())
            except termios.error:
                is_tty = False

        try:
            while not self._shutdown_requested.is_set():
                self._print_status()
                if sys.platform == 'linux' and is_tty:
                    try:
                        ch = sys.stdin.read(1)
                        if ch in ('q', 'Q'):
                            rospy.loginfo("[MANUAL] Quit requested by keyboard input 'q'.")
                            self._shutdown_requested.set()
                            rospy.signal_shutdown("Manual quit requested")
                            break
                        if ch == '\r' or ch == '\n':  # Enter key
                            rospy.sleep(0.1)
                            self._on_enter()
                    except (IOError, OSError):
                        break
                else:
                    rospy.sleep(0.5)
        finally:
            # Restore terminal settings
            if old_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            rospy.loginfo("[MANUAL] Keyboard listener stopped.")

    def _print_status(self):
        """Print current status to stdout"""
        active_slot = self._get_active_slot()
        if active_slot is None:
            rospy.loginfo("[MANUAL] All positions collected, waiting for localization...")
        else:
            frames = len(self.slot_buffers[active_slot])
            buf = self.slot_buffers[active_slot]
            if buf and frames > 0:
                latest = buf[-1]['sensor_data']
                if latest:
                    s = latest[0]
                    rospy.logdebug(f"[MANUAL] Slot:{active_slot} Frames:{frames}/{self.num_frames_to_average} "
                                   f"Sensors: latest(id={s.id}, x={s.x:.4f}, y={s.y:.4f}, z={s.z:.4f}) "
                                   f"Collected:{self.collected_positions}")
                else:
                    rospy.logdebug(f"[MANUAL] Slot:{active_slot} Frames:{frames}/{self.num_frames_to_average} "
                                   f"Collected:{self.collected_positions}")
            else:
                rospy.logdebug(f"[MANUAL] Slot:{active_slot} Frames:0/{self.num_frames_to_average} "
                               f"Collected:{self.collected_positions}")

    def _on_enter(self):
        """Handle Enter key press: save current position data"""
        active_slot = self._get_active_slot()
        if active_slot is None:
            rospy.logwarn("[MANUAL] All positions already collected!")
            return

        buffer = self.slot_buffers[active_slot]
        if len(buffer) < self.num_frames_to_average:
            rospy.logwarn(f"[MANUAL] Not enough frames: {len(buffer)}/{self.num_frames_to_average}")
            return

        # Average the sensor data
        avg_sensor_data = self._average_sensor_data(buffer)

        # Look up pose for this slot using manual_slot_frame_mapping
        frame = self.manual_slot_frame_map.get(active_slot)
        pose = self._lookup_pose(frame) if frame else None

        # Save slot data
        slot_entry = {
            'slot': active_slot,
            'sensor_data': [
                {'id': s.id, 'x': s.x, 'y': s.y, 'z': s.z}
                for s in avg_sensor_data
            ],
            'pose': pose
        }
        self.cycle_slot_data.append(slot_entry)
        self.collected_positions.append(active_slot)

        remaining = self.num_positions - len(self.collected_positions)
        rospy.loginfo(f"[MANUAL] Slot {active_slot} saved! {remaining} positions remaining.")
        if remaining == 0:
            self._finalize_cycle_manual()

    def _finalize_cycle_manual(self):
        """Process completed manual cycle: call localization service and save to JSON"""
        rospy.loginfo("[MANUAL] All positions collected! Calling localization service...")

        # Determine mode string for JSON
        mode_str = 'CVT' if self.mode == MODE_CVT else 'CCI'

        # Build cycle data dict
        # Get true sensor_array_filt pose as ground truth (not slot[0].pose)
        ground_truth_pose = self._lookup_pose('sensor_array_filt') if self.cycle_slot_data else None
        cycle_data = {
            'header': {
                'cycle_id': self.cycle_id,
                'mode': mode_str,
                'num_slots': self.num_positions,
                'sensor_ids': list(range(1, 13)),  # All 12 sensors
                'num_frames_averaged': self.num_frames_to_average
            },
            'pc_timestamp': rospy.Time.now().to_sec(),
            'slot_data': self.cycle_slot_data,
            'ground_truth_pose': ground_truth_pose,
        }
        cycle_data = self._sanitize_json_value(cycle_data)

        # Call localization service
        try:
            req = self._build_localize_request(cycle_data)
            resp = self.localize_client(req)
            if resp.success:
                cycle_data['localization'] = {
                    'pose': {
                        'position': {
                            'x': resp.localization_pose.position.x,
                            'y': resp.localization_pose.position.y,
                            'z': resp.localization_pose.position.z,
                        },
                        'orientation': {
                            'x': resp.localization_pose.orientation.x,
                            'y': resp.localization_pose.orientation.y,
                            'z': resp.localization_pose.orientation.z,
                            'w': resp.localization_pose.orientation.w,
                        }
                    },
                    'position_error': resp.position_error,
                    'orientation_error': resp.orientation_error,
                }
                rospy.loginfo(f"Localization succeeded for cycle {cycle_data['header']['cycle_id']}")
            else:
                rospy.logwarn(f"Localization failed for cycle {cycle_data['header']['cycle_id']}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Localization service call failed: {e}")

        # Save to JSON file
        filename = os.path.join(self.output_dir, f"cycle_{cycle_data['header']['cycle_id']:04d}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(cycle_data, f, indent=2, allow_nan=False)
            rospy.loginfo(f"Saved cycle {cycle_data['header']['cycle_id']} to {filename}")
        except Exception as e:
            rospy.logerr(f"Failed to save cycle JSON: {e}")

        # Increment completed cycles
        self.completed_cycles += 1
        rospy.loginfo(f"Completed {self.completed_cycles}/{self.num_cycles} cycles")

        # Check if all cycles are done
        if self.completed_cycles >= self.num_cycles:
            rospy.loginfo("All cycles completed, shutting down")
            self._shutdown_requested.set()
            rospy.signal_shutdown('Data collection complete')
            return

        # Reset for next cycle
        self.cycle_id += 1
        self.cycle_slot_data = []
        self.collected_positions = []
        self.slot_buffers = [deque(maxlen=self.num_frames_to_average) for _ in range(self.num_positions)]
        rospy.loginfo("[MANUAL] Cycle saved! Ready for next round.")

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
        if self.manual_trigger:
            # Manual mode: store continuous data in buffer for later averaging
            active_slot = self._get_active_slot()
            if active_slot is None:
                return  # All positions already collected

            self.slot_buffers[active_slot].append({
                'timestamp': msg.timestamp,
                'sensor_data': msg.sensor_data,
                'bitmap': msg.bitmap
            })
            # Publish for external monitoring (rostopic echo /data_collection_node/hall_data)
            self.pub_hall_data.publish(msg)
            return

        # Auto mode: FY8300 triggered, slot-based data
        slot = msg.slot
        cycle_id = msg.cycle_id
        is_cycle_end = msg.cycle_end == 1

        # Check if we need to start a new cycle
        if self.current_cycle is None or self.current_cycle.cycle_id != cycle_id:
            self.current_cycle = CycleBuffer(cycle_id, self.mode, self.bitmap)

        # Create slot buffer
        slot_buffer = SlotBuffer(
            slot=slot,
            bitmap=msg.bitmap,
            timestamp=msg.timestamp,
            sensor_data=msg.sensor_data
        )

        # Look up pose for this slot (slot_frame_map keys are normalized to int)
        frame = self.slot_frame_map.get(slot)
        if frame:  # frame is None for slots without pose
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

    def _sanitize_json_value(self, value, path='root'):
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            rospy.logwarn(f"Replacing non-finite JSON value at {path}: {value}")
            return None
        if isinstance(value, dict):
            return {
                key: self._sanitize_json_value(subvalue, f'{path}.{key}')
                for key, subvalue in value.items()
            }
        if isinstance(value, list):
            return [
                self._sanitize_json_value(item, f'{path}[{index}]')
                for index, item in enumerate(value)
            ]
        return value

    def _build_localize_request(self, cycle_data):
        """Convert cycle_data dict to LocalizeCycle service request"""
        req = LocalizeCycleRequest()
        req.cycle_id = cycle_data['header']['cycle_id']
        req.mode = cycle_data['header']['mode']
        req.num_slots = cycle_data['header']['num_slots']
        req.sensor_ids = cycle_data['header']['sensor_ids']
        req.ground_truth_pose = Pose()

        # slot_data
        for slot_entry in cycle_data['slot_data']:
            slot_msg = SlotData()
            slot_msg.slot = slot_entry['slot']
            # sensor_data
            for sensor in slot_entry['sensor_data']:
                reading = SensorReading()
                reading.id = sensor['id']
                reading.x = sensor['x']
                reading.y = sensor['y']
                reading.z = sensor['z']
                slot_msg.sensor_data.append(reading)
            # pose
            if 'pose' in slot_entry and slot_entry['pose'] is not None:
                slot_msg.pose.position.x = slot_entry['pose']['position']['x']
                slot_msg.pose.position.y = slot_entry['pose']['position']['y']
                slot_msg.pose.position.z = slot_entry['pose']['position']['z']
                slot_msg.pose.orientation.x = slot_entry['pose']['rotation']['x']
                slot_msg.pose.orientation.y = slot_entry['pose']['rotation']['y']
                slot_msg.pose.orientation.z = slot_entry['pose']['rotation']['z']
                slot_msg.pose.orientation.w = slot_entry['pose']['rotation']['w']
            else:
                slot_msg.pose = Pose()
            req.slot_data.append(slot_msg)

        # ground_truth_pose
        if cycle_data.get('ground_truth_pose'):
            req.ground_truth_pose.position.x = cycle_data['ground_truth_pose']['position']['x']
            req.ground_truth_pose.position.y = cycle_data['ground_truth_pose']['position']['y']
            req.ground_truth_pose.position.z = cycle_data['ground_truth_pose']['position']['z']
            req.ground_truth_pose.orientation.x = cycle_data['ground_truth_pose']['rotation']['x']
            req.ground_truth_pose.orientation.y = cycle_data['ground_truth_pose']['rotation']['y']
            req.ground_truth_pose.orientation.z = cycle_data['ground_truth_pose']['rotation']['z']
            req.ground_truth_pose.orientation.w = cycle_data['ground_truth_pose']['rotation']['w']

        return req

    def _finalize_cycle(self):
        """Process completed cycle: call localization service and save to JSON"""
        if self.current_cycle is None:
            return

        cycle_data = self._sanitize_json_value(self.current_cycle.to_dict())

        # Call localization service
        try:
            req = self._build_localize_request(cycle_data)
            resp = self.localize_client(req)
            if resp.success:
                cycle_data['localization'] = {
                    'pose': {
                        'position': {
                            'x': resp.localization_pose.position.x,
                            'y': resp.localization_pose.position.y,
                            'z': resp.localization_pose.position.z,
                        },
                        'orientation': {
                            'x': resp.localization_pose.orientation.x,
                            'y': resp.localization_pose.orientation.y,
                            'z': resp.localization_pose.orientation.z,
                            'w': resp.localization_pose.orientation.w,
                        }
                    },
                    'position_error': resp.position_error,
                    'orientation_error': resp.orientation_error,
                }
                rospy.loginfo(f"Localization succeeded for cycle {cycle_data['header']['cycle_id']}")
            else:
                rospy.logwarn(f"Localization failed for cycle {cycle_data['header']['cycle_id']}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Localization service call failed: {e}")

        # Save to JSON file (with or without localization result)
        filename = os.path.join(self.output_dir, f"cycle_{cycle_data['header']['cycle_id']:04d}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(cycle_data, f, indent=2, allow_nan=False)
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
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received")
    finally:
        collector._shutdown_requested.set()
        collector._on_shutdown()


if __name__ == '__main__':
    main()
