#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
import tf2_geometry_msgs

def move_example():
    rospy.init_node('visual_servo_example_move')

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # Publisher for the trajectory
    traj_pub = rospy.Publisher('/servo/diana7/trajectory', PoseArray, queue_size=1)

    rospy.loginfo("Waiting for TF 'world' -> 'lab_table' to become available...")
    
    rate = rospy.Rate(1) # Check every 1 second
    calibration_done = False
    
    while not rospy.is_shutdown() and not calibration_done:
        try:
            # Check if calibration is done by looking for the lab_table frame (published after calibrate())
            # We use a short timeout inside the loop
            tf_buffer.lookup_transform('world', 'lab_table', rospy.Time(0), rospy.Duration(0.5))
            rospy.loginfo("Calibration detected! T_world_vision (lab_table) is now available.")
            calibration_done = True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("Still waiting for calibration (lab_table frame)...")
            rate.sleep()
            continue

    if rospy.is_shutdown():
        return

    try:
        # Get current pose of diana7 in world frame
        trans = tf_buffer.lookup_transform('world', 'diana7_ee_link', rospy.Time(0), rospy.Duration(5.0))
        
        current_pose = Pose()
        current_pose.position.x = trans.transform.translation.x
        current_pose.position.y = trans.transform.translation.y
        current_pose.position.z = trans.transform.translation.z
        current_pose.orientation = trans.transform.rotation

        rospy.loginfo(f"Current Pose X: {current_pose.position.x:.4f}")

        # Calculate target: move in X positive direction by 0.1 * current_x
        # Note: If x is 0, we might want a fixed offset, but following user order:
        # offset = 0.3 * abs(current_pose.position.x)
        # if offset == 0:
        offset = 0.1 # Fallback 5cm if at origin
            
        target_pose = Pose()
        target_pose.position.x = current_pose.position.x + offset
        target_pose.position.y = current_pose.position.y 
        target_pose.position.z = current_pose.position.z
        target_pose.orientation = current_pose.orientation

        rospy.loginfo(f"Target Pose X: {target_pose.position.x:.4f} (Offset: {offset:.4f})")

        # Create PoseArray
        msg = PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        # args: diana7
        msg.poses.append(target_pose)

        # Wait a bit for publisher connection
        rospy.sleep(1.0)
        traj_pub.publish(msg)
        rospy.loginfo("Trajectory published to /servo/diana7/trajectory")

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"TF Lookup failed: {e}")

if __name__ == '__main__':
    try:
        move_example()
    except rospy.ROSInterruptException:
        pass
