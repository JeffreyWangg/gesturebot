#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import tf2_ros
from geometry_msgs.msg import Point, Twist, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion
from path_planner import PathPlanner
from pure_pursuit import PurePursuit
from frontier_exploration import FrontierExploration


class SLAMObjectNavigator:
    def __init__(self):
        rospy.init_node("slam_object_navigator")

        # Subscribers
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # TF Components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # SLAM Components
        self.path_planner = PathPlanner()
        self.pure_pursuit = PurePursuit()
        self.frontier_exploration = FrontierExploration()

        # State Variables
        self.map = None
        self.current_pose = None
        self.dist_tolerance = 0.2
        self.detected_fiducials = []
        self.exploration_active = True  # Tracks whether the robot is exploring

    def map_callback(self, msg):
        """Updates the map when a new occupancy grid is received."""
        self.map = msg
        self.frontier_exploration.update_map(msg)  # Keep Frontier Exploration updated
    def update_odometry(self):
        """   
        Updates the robot's pose.
        """
        try:
            # Try to get the transform from 'map' to 'base_link'
            transform = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0))
            rospy.loginfo("Using 'map' frame for localization.")
        except tf2_ros.LookupException:
            # Fallback to 'odom' if 'map' is unavailable
            rospy.logwarn("Map frame not available. Falling back to odom frame.")
            transform = self.tf_buffer.lookup_transform("odom", "base_link", rospy.Time(0))

        # Update the robot's pose
        self.pose = Pose(
            position=Point(
                x=transform.transform.translation.x,
                y=transform.transform.translation.y,
            ),
            orientation=Quaternion(
                x=transform.transform.rotation.x,
                y=transform.transform.rotation.y,
                z=transform.transform.rotation.z,
                w=transform.transform.rotation.w,
            ),
        )

    def update_pose(self):
        """
        Updates the current pose of the robot by querying the transform from 'map' to 'base_link'.
        """
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0))
            trans = transform.transform.translation
            rot = transform.transform.rotation
            self.current_pose = Pose(
                position=Point(x=trans.x, y=trans.y, z=trans.z),
                orientation=Quaternion(x=rot.x, y=rot.y, z=rot.z, w=rot.w),
            )
            self.frontier_exploration.update_odometry(self.current_pose)  # Update Frontier Exploration
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            rospy.logwarn(f"Failed to update pose: {e}")

    def scan_for_fids(self):
        """
        Scans for fiducials by rotating in place.
        """
        twist = Twist()
        twist.angular.z = 0.5
        twist.linear.x = 0.0
        scan_duration = rospy.Duration(15)  # Adjust scan duration as needed
        start_time = rospy.Time.now()

        rate = rospy.Rate(10)
        rospy.loginfo("Scanning for fiducials...")
        while not rospy.is_shutdown() and rospy.Time.now() - start_time < scan_duration:
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo("Scan complete")

    def navigate_to_fiducial(self, fiducial_id):
        """Navigates to a specific fiducial."""
        rospy.loginfo(f"Navigating to fiducial {fiducial_id}...")
        rate = rospy.Rate(10)
        frame_id = f"pin_{fiducial_id}"

        while not rospy.is_shutdown():
            self.update_pose()
            if self.current_pose is None:
                rospy.logwarn("Waiting for current pose...")
                rate.sleep()
                continue

            try:
                # Get transform to the fiducial's `pin_i` frame
            
                fid_x = trans.transform.translation.x
                fid_y = trans.transform.translation.y
                target_position = Point(x=fid_x, y=fid_y)

                # Plan path using PathPlanner
                start = self.path_planner.world_to_grid(self.map, self.current_pose.position)
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

    def explore_environment(self):
        """Performs frontier-based exploration."""
        rospy.loginfo("Starting frontier exploration...")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and self.exploration_active:
            frontier_goal = self.frontier_exploration.find_next_frontier(self.map)
            if not frontier_goal:
                rospy.loginfo("No more frontiers to explore.")
                break

            rospy.loginfo(f"Found frontier goal: {frontier_goal}")
            self.navigate_to_goal(frontier_goal)
            rate.sleep()

    def run(self):
        """Main execution loop."""
        rospy.loginfo("SLAM Object Navigator is running...")

        # Step 1: Perform frontier exploration
        self.explore_environment()

        # Step 2: Navigate to fiducials
        rospy.loginfo("Scanning for fiducials...")
        self.scan_for_fids()

        rospy.loginfo("Navigating to fiducial 108...")
        self.navigate_to_fiducial(108)

        rospy.loginfo("SLAM Object Navigator has completed navigation.")

    def navigate_to_goal(self, goal):
        """Navigates to a specific goal."""
        rospy.loginfo(f"Navigating to goal: {goal}...")
        # Implementation to navigate to a specified (x, y) goal.


if __name__ == "__main__":
    try:
        navigator = SLAMObjectNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("SLAM Object Navigator shutting down.")
