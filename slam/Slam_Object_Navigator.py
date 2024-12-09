#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from tf import TransformBroadcaster, TransformListener
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from nav_msgs.msg import OccupancyGrid, Path
from path_planner import PathPlanner
from frontier_exploration import FrontierExploration
from pure_pursuit import PurePursuit


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

    def map_callback(self, msg):
        """
        Updates the map when a new occupancy grid is received.
        """
        self.map = msg

    def assign_object_tf(self, object_name, object_position):
        """
        Assigns a TF frame to a detected object.
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

    def navigate_to_object(self, object_name):
        """
        Navigates to an object using its assigned TF frame.
        :param object_name: Name of the object to navigate to.
        """
        frame_id = f"object_{object_name}"

        if frame_id not in self.object_tf_frames:
            rospy.logwarn(f"Object '{object_name}' does not have an assigned TF frame.")
            return

        rospy.loginfo(f"Navigating to object '{object_name}'...")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.tf_listener.lookupTransform("/map", frame_id, rospy.Time(0))
                goal_position = Point(x=trans[0], y=trans[1])
                break
            except Exception as e:
                rospy.logwarn(f"Unable to get TF for {frame_id}: {e}")
                rate.sleep()

        # Plan path to the goal
        start = PathPlanner.world_to_grid(self.map, self.pure_pursuit.pose.position)
        goal = PathPlanner.world_to_grid(self.map, goal_position)

        path, _, _, _ = PathPlanner.a_star(self.map, None, start, goal)

        if path is None:
            rospy.logwarn(f"No path found to object '{object_name}'.")
            return

        # Send path to Pure Pursuit
        path_msg = PathPlanner.path_to_message(self.map, path)
        self.pure_pursuit.update_path(path_msg)

        rospy.loginfo(f"Following path to '{object_name}'...")
        while not rospy.is_shutdown():
            if self.pure_pursuit.get_distance_to_waypoint_index(len(path) - 1) < self.pure_pursuit.DISTANCE_TOLERANCE:
                rospy.loginfo(f"Reached object '{object_name}'.")
                self.pure_pursuit.stop()
                break
            rate.sleep()

    def run(self):
        """
        Main execution loop.
        """
        rospy.loginfo("SLAM Object Navigator is running...")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # Add control logic or ROS service handling for new objects
            rate.sleep()


if __name__ == "__main__":
    navigator = SLAMObjectNavigator()
    try:
        navigator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down SLAM Object Navigator.")
