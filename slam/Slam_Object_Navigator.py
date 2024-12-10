#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import tf2_ros
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion
from gesturebot.slam.path_planner import PathPlanner
from gesturebot.slam.pure_pursuit import PurePursuit


class SLAMObjectNavigator:
    def __init__(self):
        rospy.init_node("slam_object_navigator")

        # Subscribers
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("my_odom", Point, self.my_odom_cb)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # TF Components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # SLAM Components
        self.path_planner = PathPlanner()
        self.pure_pursuit = PurePursuit()
        
        # State Variables
        self.map = None
        self.curr_dist = 0.0
        self.curr_yaw = 0.0
        self.dist_tolerance = 0.2
        self.angle_tolerance = 0.05
        self.detected_fiducials = []

    def my_odom_cb(self, msg):
        """Callback function for `my_odom`."""
        self.curr_dist = msg.x
        self.curr_yaw = msg.y

    def map_callback(self, msg):
        """Updates the map when a new occupancy grid is received."""
        self.map = msg

    def normalize_angle(self, angle):
        """Normalizes an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def scan_for_fiducials(self):
        """Scans for fiducials by rotating in place."""
        rospy.loginfo("Scanning for fiducials...")
        rate = rospy.Rate(10)
        detected_fiducials = []

        for _ in range(150):  # Scan for 15 seconds (150 iterations at 10 Hz)
            try:
                frames = self.tf_buffer.all_frames_as_string()
                for line in frames.splitlines():
                    if "pin_" in line:
                        fiducial_id = int(line.split("_")[1])
                        if fiducial_id not in detected_fiducials:
                            detected_fiducials.append(fiducial_id)
                            rospy.loginfo(f"Detected fiducial: pin_{fiducial_id}")
            except Exception as e:
                rospy.logwarn(f"Error scanning for fiducials: {e}")
            rate.sleep()

        self.detected_fiducials = detected_fiducials
        rospy.loginfo(f"Detected fiducials: {self.detected_fiducials}")

    def plan_and_navigate_to_fiducial(self, fiducial_id):
        """Plans a path to a fiducial and navigates around obstacles."""
        rate = rospy.Rate(10)
        frame_id = f"pin_{fiducial_id}"

        while not rospy.is_shutdown():
            try:
                trans = self.tf_buffer.lookup_transform("odom", frame_id, rospy.Time(0))
                fid_x = trans.transform.translation.x
                fid_y = trans.transform.translation.y
                target_position = Point(x=fid_x, y=fid_y)

                # Plan the path using PathPlanner
                start = self.path_planner.world_to_grid(self.map, self.pure_pursuit.pose.position)
                goal = self.path_planner.world_to_grid(self.map, target_position)
                path, _, _, _ = self.path_planner.a_star(self.map, None, start, goal)

                if path is None:
                    rospy.logwarn(f"No path found to fiducial {fiducial_id}. Retrying...")
                    rate.sleep()
                    continue

                # Convert the path to a ROS Path message
                path_msg = self.path_planner.path_to_message(self.map, path)
                self.pure_pursuit.update_path(path_msg)

                rospy.loginfo(f"Following path to fiducial {fiducial_id}...")
                while not rospy.is_shutdown():
                    distance_to_target = math.sqrt(fid_x**2 + fid_y**2)
                    if distance_to_target < self.dist_tolerance:
                        rospy.loginfo(f"Reached fiducial {fiducial_id}.")
                        self.pure_pursuit.stop()
                        return

                    rate.sleep()
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn(f"Unable to get TF for {frame_id}. Retrying...")
                rate.sleep()

    def run(self):
        """Main execution loop."""
        rospy.loginfo("SLAM Object Navigator is running...")

        # Step 1: Scan for fiducials
        self.scan_for_fiducials()

        # Step 2: Navigate to fiducial 108 (or other detected fiducials)
        if 108 in self.detected_fiducials:
            rospy.loginfo("Navigating to fiducial 108...")
            self.plan_and_navigate_to_fiducial(108)
        else:
            rospy.logwarn("Fiducial 108 not detected.")

        rospy.loginfo("SLAM Object Navigator has completed navigation.")


if __name__ == "__main__":
    try:
        navigator = SLAMObjectNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("SLAM Object Navigator shutting down.")
