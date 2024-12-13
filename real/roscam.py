#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point, Pose, Twist, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import mediapipe as mp
import copy
import itertools
import csv
from model import KeyPointClassifier  # Import KeyPointClassifier
from mmdet.apis import DetInferencer
import torch
import actionlib
import tf
import time

class GestureRecognizer:
    def __init__(self):

        #object store
        self.objects = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        self.banned_objects = [
            "person", "chair", "tvmonitor", "diningtable", "bench"
        ]

        #===================
        # POSE LOCATION
        self.object_poses = {}

        # ROS setup
        self.bridge = CvBridge()
        self.cv_image = None

        # Subscribe to camera feed
        # self.image_sub = rospy.Subscriber('/cv_camera/image_raw', Image, self.image_cb) 
        # <= for branbot
        self.image_sub = rospy.Subscriber('/raspicam_node/image/compressed',
                                          CompressedImage,
                                          self.image_cb)

        #Publishers
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.image_pub = rospy.Publisher("/image", Image, queue_size=1)
        self.depth_image_pub = rospy.Publisher("/depth_image", Image, queue_size=1)

        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.gesture_count = 0
        self.previous_gesture = None
        self.gesture = None

        # Keypoint Classifier setup
        self.keypoint_classifier = KeyPointClassifier()
        self.keypoint_classifier_labels = ['Open', 'Close', 'Pointer', 'OK', 'Peace Sign']

        #================ obj seg
        self.force_recog = False
        config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
        checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
        self.inferencer = DetInferencer(config_file, checkpoint_file, device="cpu")

        self.predictions = None
        self.keep = None

        #================ depth 
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.device = torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        self.debug_Box = None

        self.transform = midas_transforms.small_transform

        # MOVE BASE SETUP
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base server.")
        
        while self.transform is None:
            pass

        while self.cv_image is None:
            pass

    #get robot pose
    def get_current_pose(self):
        """Get the current pose of the robot from the odometry topic."""
        try:
            msg = rospy.wait_for_message('/odom', Odometry, timeout=5)
            return msg.pose.pose
        except rospy.ROSException:
            print("can't get pose :(((((")
            return None

    def get_goal_pose(self, object_label):
        """Retrieve a saved pose and convert it into a MoveBaseGoal."""
        pose = self.object_poses[object_label]
        if not pose:
            print(f"Pose {object_label} has not been saved")
            return None

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        # or 'odom' try both tbh
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = pose
        return goal

    def image_cb(self, msg):
        try:
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")

    def depth_callback(self, event):
        input_batch = self.transform(self.cv_image).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=self.cv_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # to draw image
        self.disparity_map = prediction.cpu().numpy()
        output_image = self.disparity_map / 1200

        if self.debug_Box:
            bottom = (int(self.debug_Box[0]), int(self.debug_Box[1]))
            top = (int(self.debug_Box[2]), int(self.debug_Box[3]))
            output_image = cv.rectangle(output_image, bottom, top, (0, 255, 0), 2)

        image_msg = self.bridge.cv2_to_imgmsg(output_image)
        self.depth_image_pub.publish(image_msg)

    def segmentation_callback(self, event):
        if self.gesture == 'Open' or self.force_recog:
            # obj seg
            output = self.inferencer(self.cv_image)
            self.predictions = output['predictions'][0]

            #keep is list of indices for output['predictions']
            keep = self.non_max_suppression(self.predictions['bboxes'], self.predictions['scores'], 0.5)[:10]

            #===============================================
            # We need to filter out objects like chairs, tables, tv monitors and people
            for index in keep:
                keep = list(filter(lambda index: self.objects[self.predictions['labels'][index]] not in self.banned_objects, keep))
            self.keep = keep
            
            labels = [self.objects[i] for i in [self.predictions['labels'][i] for i in self.keep]]

            print([self.predictions['scores'][i] for i in self.keep])
            print(labels)

            for label in labels:
                pose = self.get_current_pose()
                if pose:
                    self.object_poses[label] = pose

            image = self.cv_image
            bboxes = [self.predictions['bboxes'][i] for i in self.keep]

            for box in bboxes:
                bottom = (int(box[0]), int(box[1]))
                top = (int(box[2]), int(box[3]))
                image = cv.rectangle(image, bottom, top, (0, 0, 255), 2)
            image_msg = self.bridge.cv2_to_imgmsg(image)
            self.image_pub.publish(image_msg)

    def run(self):
        rospy.Timer(rospy.Duration(1.5), self.segmentation_callback) #start object recognition callback
        rospy.Timer(rospy.Duration(1.5), self.depth_callback) #start depth estimation callback

        mode = 0
        print("Gesture Recognition Running")
        while not rospy.is_shutdown():
            key = cv.waitKey(10)
            if key == 27:  # esc key
                break
            number, mode = self.select_mode(key, mode)

            #testing
            if self.gesture == 'Peace Sign':
                self.move_to_object("laptop")

            if self.gesture == 'Close':
                self.forward()

            if self.gesture == 'Pointer':
                self.right()

            if self.cv_image is not None:
                # Process the current frame
                self.process_frame(number, mode)
            rospy.sleep(0.01)  # Small delay to prevent high CPU usage

    def forward(self):
        twist = Twist()
        twist.linear.x = 0.1
        self.cmd_vel_pub.publish(twist)

    def right(self):
        twist = Twist()
        twist.angular.z = 0.3
        self.cmd_vel_pub.publish(twist)

    def move_to_saved_pose(self, client, object_label):
        """Send the robot to one of the saved poses."""
        goal = self.get_goal_pose(object_label)
        if goal:
            print(f"Navigating to Pose {object_label}...")
            client.send_goal(goal)
            client.wait_for_result()
            print(f"Arrived at Pose {object_label}.")
        else:
            print("invalid goal")

    def move_to_object(self, object_label):
        # move to saved pose
        self.move_to_saved_pose(self.client, object_label)


        #if the object is not in predictions, wait for 6 seconds until it appears or give up
        # if it appears, enter while loop and begin walking towards the box
        rate = rospy.Rate(5)
        twist = Twist()
        self.force_recog = True
        miss_counter = 0

        print("get out of the way")
        time.sleep(3)
        print("starting")

        # while True:
        #     if self.predictions is None or self.keep is None:
        #         print("no predictions")
        #         pass
        #     object_list = [self.objects[i] for i in [self.predictions['labels'][i] for i in self.keep]] #same size as keep
        #     # print(object_list)
        #     if object_label not in object_list:
        #         miss_counter += 1
        #         if miss_counter >= 30: #for 6 seconds
        #             print("missed too much, break")
        #             break
        #         continue
        #     miss_counter = 0
        #     object_prediction_index = self.keep[object_list.index(object_label)]
        #     print(self.objects[self.predictions['labels'][object_prediction_index]])
        #     box = self.predictions['bboxes'][object_prediction_index]
        #     self.debug_Box = box

        #     x_min = int(box[0])
        #     y_min = int(box[1])
        #     x_max = int(box[2])
        #     y_max = int(box[3])
        #     subarray = self.disparity_map[x_min:x_max, y_min:y_max]
        #     median = np.median(subarray)
        #     mask = subarray >= median
        #     subarray = subarray[mask]
        #     depth = np.mean(subarray)
        #     print(f"depth: {depth}")
        #     if depth > 800:
        #         print("too close, break")
        #         break
        #     twist.linear.x = 0.02
        #     self.cmd_vel_pub.publish(twist)
        #     rate.sleep()

        self.force_recog = False
        twist.linear.x = 0
        self.cmd_vel_pub.publish(twist)
        self.gesture = None
        print("move_to is done")

    #for object segmentation
    def non_max_suppression(self, boxes, scores, threshold):
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            for j in order:
                # iou between the two boxes
                intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                            max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
                union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                        (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
                iou = intersection / union

                if iou > threshold:
                    order.remove(j)
        return keep

    def process_frame(self, number, mode):
        image = cv.flip(self.cv_image, 1)  # Mirror image for convenience
        debug_image = copy.deepcopy(image)

        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = self.calc_landmark_list(image, hand_landmarks)
                preprocessed_landmarks = self.preprocess_landmarks(landmark_list)

                #log to csv
                self.logging_csv(number, mode, preprocessed_landmarks)

                # Classify gesture
                hand_sign_id = self.keypoint_classifier(preprocessed_landmarks)
                hand_sign_label = self.keypoint_classifier_labels[hand_sign_id]
                
                #multiple gestures are required for recognition
                if self.previous_gesture is None or self.previous_gesture == hand_sign_label:
                    self.gesture_count += 1
                else:
                    self.gesture_count = 0
                if self.gesture_count > 30:
                    print(f"Gesture is now {hand_sign_label}")
                    self.gesture = hand_sign_label
                self.previous_gesture = hand_sign_label

                # debug image
                debug_image = self.draw_landmarks(debug_image, landmark_list)

        cv.imshow("Gesture Recognition", debug_image)
        cv.waitKey(1)

    def calc_landmark_list(self, image, hand_landmarks): #get pixel location
        image_width, image_height = image.shape[1], image.shape[0]
        return [
            [
                int(landmark.x * image_width),
                int(landmark.y * image_height),
            ]
            for landmark in hand_landmarks.landmark
        ]

    def preprocess_landmarks(self, landmark_list): # need to normalize
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Normalize based on the wrist (first landmark)
        base_x, base_y = temp_landmark_list[0]
        for landmark in temp_landmark_list:
            landmark[0] -= base_x
            landmark[1] -= base_y

        # Flatten the list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        # Normalize the scale
        max_value = max(map(abs, temp_landmark_list))
        temp_landmark_list = [n / max_value for n in temp_landmark_list]

        return temp_landmark_list

    def draw_landmarks(self, image, landmark_list):
        for point in landmark_list:
            cv.circle(image, tuple(point), 5, (255, 0, 0), -1)
        return image
    
    def select_mode(self, key, mode):
        number = -1
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
        if key == 110:  # n, to change back to no logging
            mode = 0
        if key == 107:  # k, to change to logging
            print("start keypoints")
            mode = 1
        return number, mode
    
    def logging_csv(self, number, mode, landmark_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            csv_path = '/my_ros_data/catkin_ws/src/cosi119_src/gesture_cam/gesturebot/real/model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
                print(f"logged {number}")
        return

if __name__ == "__main__":
    rospy.init_node("gesture_recognizer")
    recognizer = GestureRecognizer()
    try:
        recognizer.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Gesture Recognizer.")
    cv.destroyAllWindows()
