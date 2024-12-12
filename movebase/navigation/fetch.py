# Adding functionality to save and return to poses

#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf
pose_1 = None
pose_2 = None

# debugging
def get_current_pose():
    """Get the current pose of the robot from the odometry topic."""
    try:
        msg = rospy.wait_for_message('/odom', Odometry, timeout=5)
        return msg.pose.pose
    except rospy.ROSException:
        print("can't get pose :(((((")
        return None


def save_pose(pose_num):
    """Save the robot's current pose into one of the three slots."""
    global pose1, pose2
    pose = get_current_pose()
    if pose:
        if pose_num == 1:
            pose1 = pose
            print("Pose 1 saved.")
            print(f"Saved Pose: {pose}")
        elif pose_num == 2:
            pose2 = pose
            print("Pose 2 saved.")
        else:
            print("Invalid pose number. Use 1, 2")
    else:
        print("no pose. Check odometry.")

def get_goal_pose(pose_num):
    """Retrieve a saved pose and convert it into a MoveBaseGoal."""
    global pose1, pose2, pose3
    pose = None
    if pose_num == 1:
        pose = pose1
    elif pose_num == 2:
        pose = pose2

    if not pose:
        print(f"Pose {pose_num} has not been saved")
        return None

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'  
    # or 'odom' try both tbh
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = pose
    return goal


def move_to_saved_pose(client, pose_num):
    """Send the robot to one of the saved poses."""
    goal = get_goal_pose(pose_num)
    if goal:
        print(f"Navigating to Pose {pose_num}...")
        client.send_goal(goal)
        client.wait_for_result()
        print(f"Arrived at Pose {pose_num}.")
    else:
        print("invalid goal")

def main():
    rospy.init_node('fetch_node')
    rospy.loginfo("Robot navigation node initialized.")

    # Initialize action client for move_base
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base server.")

    # Command loop
    while not rospy.is_shutdown():
        command = input("Enter command ('save1', 'save2', 'go1', 'go2', or 'exit'): ").strip().lower()
        if command == 'save1':
            save_pose(1)
        elif command == 'save2':
            save_pose(2)
        elif command == 'go1':
            move_to_saved_pose(client, 1)
        elif command == 'go2':
            move_to_saved_pose(client, 2)
        elif command == 'exit':
            print("Exiting...")
            break
        else:
            print("Unknown command. Use 'save1', 'save2', 'save3', 'go1', 'go2', 'go3', or 'exit'.")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


