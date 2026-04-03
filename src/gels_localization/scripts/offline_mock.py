#!/usr/bin/env python3
"""
Mock classes for offline processing without ROS.

These classes simulate ROS message types to allow offline processing
of cycle JSON files without requiring the ROS framework.
"""


class MockPose:
    """Mock geometry_msgs/Pose"""
    def __init__(self, x=0, y=0, z=0, qx=0, qy=0, qz=0, qw=1):
        self.position = type('obj', (object,), {'x': x, 'y': y, 'z': z})()
        self.orientation = type('obj', (object,), {'x': qx, 'y': qy, 'z': qz, 'w': qw})()


class MockSensorReading:
    """Mock sensor reading"""
    def __init__(self, id=1, x=0.0, y=0.0, z=0.0):
        self.id = id
        self.x = x
        self.y = y
        self.z = z


class MockSlotData:
    """Mock slot data container"""
    def __init__(self, slot=0, sensor_data=None, pose=None):
        self.slot = slot
        self.sensor_data = sensor_data or []
        self.pose = pose or MockPose()


class MockLocalizationResponse:
    """Mock LocalizeCycle service response"""
    def __init__(self):
        self.success = False
        self.localization_pose = MockPose()
        self.position_error = 0.0
        self.orientation_error = 0.0
        self.details = None


class MockLocalizeCycleRequest:
    """Mock LocalizeCycle service request"""
    def __init__(self, cycle_id=1, mode='CCI', sensor_ids=None, slot_data=None, ground_truth_pose=None):
        self.cycle_id = cycle_id
        self.mode = mode
        self.num_slots = len(slot_data) if slot_data else 0
        self.sensor_ids = sensor_ids or [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.slot_data = slot_data or []
        self.ground_truth_pose = ground_truth_pose or MockPose()
