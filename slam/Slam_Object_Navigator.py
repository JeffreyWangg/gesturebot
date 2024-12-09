#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from tf import TransformBroadcaster, TransformListener
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from nav_msgs.msg import OccupancyGrid, Path
from path_planner import PathPlanner
from frontier_exploration import FrontierExploration
from pure_pursuit import PurePursuit
from tf.transformations import quaternion_from_euler
import math

class SLAMObjectNavigator:
    def __init__(self):
        rospy.init_node("slam_object_navigator")

        # Subscribers
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # SLAM components
        self.frontier_exploration = FrontierExploration()
        self.pure_pursuit = PurePursuit()
        self.map = None

        # TF Components
        self.tf_broadcaster = TransformBroadcaster()
        self.tf_listener = TransformListener()
        self.object_tf_frames = {}

        # Control Parameters
        self.pose = None
        self.goal_frame = None

        # Fiducial Parameters
        self.fiducial_ids = []
        self.current_fiducial = None
        self.dist_tolerance = 0.2
        self.angle_tolerance = 0.05

    def map_callback(self, msg):
        """
        Updates the map when a new occupancy grid is received.
        """
        self.map = msg

    def assign_object_tf(self, object_name, object_position):
        """
        Assigns a TF frame to a detected object or fiducial marker.
        :param object_name: Name of the object.
        :param object_position: Position of the object in the map frame (x, y).
        """
        frame_id = f"object_{object_name}"
        self.object_tf_frames[frame_id] = object_position

        # Broadcast the TF for the object
        self.tf_broadcaster.sendTransform(
            (object_position.x, object_position.y, 0),
            (0, 0, 0, 1),  # Orientation quaternion (identity)
            rospy.Time.now(),
            frame_id,
            "map"
        )
        rospy.loginfo(f"Assigned TF frame '{frame_id}' to object at {object_position}")

    def scan_for_fiducials(self, scan_duration=15):
        """
        Rotates the robot to scan for fiducials. The `mapper` node maps fiducials into the TF tree.
        """
        twist = Twist()
        twist.angular.z = 0.5
        rospy.loginfo("Scanning for fiducials...")
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and rospy.Time.now() - start_time < rospy.Duration(scan_duration):
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo("Scan complete.")

    def navigate_to_fiducial(self, fiducial_id):
        """
        Navigates to a specific fiducial using the TF frame published by the `mapper` node.
        :param fiducial_id: ID of the fiducial to navigate to.
        """
        frame_id = f"fiducial_{fiducial_id}"

        rospy.loginfo(f"Navigating to fiducial '{frame_id}'...")
        rate = rospy.Rate(10)

        # Wait for the fiducial transform to be available
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.tf_listener.lookupTransform("/map", frame_id, rospy.Time(0))
                goal_position = Point(x=trans[0], y=trans[1])
                break
            except Exception as e:
                rospy.logwarn(f"Unable to get TF for {frame_id}: {e}")
                rate.sleep()

        # Plan path to the fiducial
        start = PathPlanner.world_to_grid(self.map, self.pure_pursuit.pose.position)
        goal = PathPlanner.world_to_grid(self.map, goal_position)

        path, _, _, _ = PathPlanner.a_star(self.map, None, start, goal)

        if path is None:
            rospy.logwarn(f"No path found to fiducial '{fiducial_id}'.")
            return

        # Send path to Pure Pursuit
        path_msg = PathPlanner.path_to_message(self.map, path)
        self.pure_pursuit.update_path(path_msg)

        # Follow the path
        rospy.loginfo(f"Following path to fiducial '{fiducial_id}'...")
        while not rospy.is_shutdown():
            if self.pure_pursuit.get_distance_to_waypoint_index(len(path) - 1) < self.pure_pursuit.DISTANCE_TOLERANCE:
                rospy.loginfo(f"Reached fiducial '{fiducial_id}'.")
                self.pure_pursuit.stop()
                break
            rate.sleep()

    def turn_to_heading(self, target_yaw):
        """
        Rotates the robot to the desired yaw heading.
        """
        twist = Twist()
        rate = rospy.Rate(10)
        rospy.loginfo(f"Turning to heading: {target_yaw:.2f} radians.")

        while not rospy.is_shutdown():
            # Get the robot's current yaw
            try:
                (trans, rot) = self.tf_listener.lookupTransform("/map", "/base_footprint", rospy.Time(0))
                _, _, current_yaw = euler_from_quaternion(rot)
            except Exception as e:
                rospy.logwarn(f"Unable to get current yaw: {e}")
                rate.sleep()
                continue

            yaw_error = target_yaw - current_yaw

            if abs(yaw_error) < self.angle_tolerance:
                rospy.loginfo("Turn completed.")
                break

            twist.angular.z = 0.5 * yaw_error
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def run(self):
        """
        Main execution loop.
        """
        rospy.loginfo("SLAM Object Navigator is running...")
        rate = rospy.Rate(10)

        # Scan for fiducials first
        self.scan_for_fiducials()

        # Navigate to each fiducial in sequence
        for fiducial_id in self.fiducial_ids:
            rospy.loginfo(f"Navigating to fiducial {fiducial_id}...")
            self.navigate_to_fiducial(fiducial_id)
            rospy.sleep(1)

        rospy.loginfo("SLAM Object Navigator has completed all tasks.")

if __name__ == "__main__":
    navigator = SLAMObjectNavigator()
    navigator.fiducial_ids = [102]  
    try:
        navigator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down SLAM Object Navigator.")
