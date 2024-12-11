#!/usr/bin/env python3

import os
import rospy
import rospkg
import threading
import subprocess
import numpy as np
from typing import Union
from path_planner import PathPlanner
from frontier_search import FrontierSearch
from nav_msgs.msg import OccupancyGrid, Path, GridCells, Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
from gesture_cam.msg import FrontierList
from tf import TransformListener
from tf.transformations import euler_from_quaternion


class FrontierExploration:
    def __init__(self):
        """
        Class constructor
        """
        # Removed rospy.init_node("frontier_exploration") to avoid multiple init_node calls
        rospy.loginfo("Initializing FrontierExploration...")

        # Set if in debug mode
        self.is_in_debug_mode = (
            rospy.has_param("~debug") and rospy.get_param("~debug") == "true"
        )

        # Publishers
        self.pure_pursuit_pub = rospy.Publisher(
            "/pure_pursuit/path", Path, queue_size=10
        )

        if self.is_in_debug_mode:
            self.frontier_cells_pub = rospy.Publisher(
                "/frontier_exploration/frontier_cells", GridCells, queue_size=10
            )
            self.start_pub = rospy.Publisher(
                "/frontier_exploration/start", GridCells, queue_size=10
            )
            self.goal_pub = rospy.Publisher(
                "/frontier_exploration/goal", GridCells, queue_size=10
            )
            self.cspace_pub = rospy.Publisher("/cspace", GridCells, queue_size=10)
            self.cost_map_pub = rospy.Publisher(
                "/cost_map", OccupancyGrid, queue_size=10
            )

        # Subscribers
        rospy.Subscriber("/odom", Odometry, self.update_odometry)
        rospy.Subscriber("/map", OccupancyGrid, self.update_map)

        self.tf_listener = TransformListener()
        self.lock = threading.Lock()
        self.pose = None
        self.map = None

        self.NUM_EXPLORE_FAILS_BEFORE_FINISH = 30
        self.no_path_found_counter = 0
        self.no_frontiers_found_counter = 0
        self.is_finished_exploring = False

    def update_odometry(self, msg=None):
        """
        Updates the current pose of the robot.
        """
        try:
            # Try to get the transform from 'map' to 'base_footprint'
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_footprint", rospy.Time(0))
            rospy.loginfo("Using 'map' frame for odometry.")
        except tf.Exception:
            # Fallback to 'odom' if 'map' is unavailable
            rospy.logwarn("Map frame not available. Falling back to odom frame.")
            (trans, rot) = self.tf_listener.lookupTransform("odom", "base_footprint", rospy.Time(0))

        # Update the robot's pose
        self.pose = Pose(
            position=Point(x=trans[0], y=trans[1]),
            orientation=Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3]),
        )

    def update_map(self, msg: OccupancyGrid):
        """
        Updates the current map.
        This method is a callback bound to a Subscriber.
        :param msg [OccupancyGrid] The current map information.
        """
        self.map = msg

    def find_next_frontier(self, map_data: OccupancyGrid):
        """
        Finds the next unexplored frontier goal.
        :param map_data: OccupancyGrid map data.
        :return: (x, y) tuple of the next frontier goal or None if no frontier found.
        """
        if map_data is None:
            rospy.logwarn("Map data is None. Cannot find frontiers.")
            return None

        start = PathPlanner.world_to_grid(map_data, self.pose.position)

        # Get frontiers using FrontierSearch
        frontier_list, _ = FrontierSearch.search(map_data, start, self.is_in_debug_mode)

        if frontier_list is None or not frontier_list.frontiers:
            rospy.logwarn("No frontiers found.")
            return None

        # Sort frontiers by size and return the largest one
        largest_frontier = max(
            frontier_list.frontiers, key=lambda frontier: frontier.size
        )
        return largest_frontier.centroid

    def explore_frontier(self, frontier_list: FrontierList):
        """
        Explores the provided frontiers and publishes a path to the best one.
        """
        if self.is_finished_exploring or self.pose is None or self.map is None:
            return

        frontiers = frontier_list.frontiers

        if not frontiers:
            rospy.loginfo("No frontiers found.")
            self.no_frontiers_found_counter += 1
            self.check_if_finished_exploring()
            return
        else:
            self.no_frontiers_found_counter = 0

        A_STAR_COST_WEIGHT = 10.0
        FRONTIER_SIZE_COST_WEIGHT = 1.0

        # Calculate the C-space
        cspace, cspace_cells = PathPlanner.calc_cspace(self.map, self.is_in_debug_mode)

        # Calculate the cost map
        cost_map = PathPlanner.calc_cost_map(self.map)
        if self.is_in_debug_mode:
            self.publish_cost_map(self.map, cost_map)

        # Get the start
        start = PathPlanner.world_to_grid(self.map, self.pose.position)

        # Execute A* for every frontier
        lowest_cost = float("inf")
        best_path = None

        for frontier in frontiers:
            goal = PathPlanner.world_to_grid(self.map, frontier.centroid)

            path, a_star_cost, _, _ = PathPlanner.a_star(cspace, cost_map, start, goal)

            if path is None or a_star_cost is None:
                continue

            cost = (A_STAR_COST_WEIGHT * a_star_cost) + (
                FRONTIER_SIZE_COST_WEIGHT / frontier.size
            )

            if cost < lowest_cost:
                lowest_cost = cost
                best_path = path

        if best_path:
            rospy.loginfo(f"Best path found with cost {lowest_cost}.")
            path_msg = PathPlanner.path_to_message(self.map, best_path)
            self.pure_pursuit_pub.publish(path_msg)
        else:
            rospy.loginfo("No valid path found.")
            self.no_path_found_counter += 1
            self.check_if_finished_exploring()

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.pose is None or self.map is None:
                continue

            start = PathPlanner.world_to_grid(self.map, self.pose.position)

            frontier_list, _ = FrontierSearch.search(
                self.map, start, self.is_in_debug_mode
            )

            if frontier_list is None:
                continue

            self.explore_frontier(frontier_list)

            rate.sleep()


if __name__ == "__main__":
    FrontierExploration().run()
