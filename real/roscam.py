#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import mediapipe as mp
import copy
import itertools
import csv
from model import KeyPointClassifier  # Import KeyPointClassifier

class GestureRecognizer:
    def __init__(self):
        # ROS setup
        self.bridge = CvBridge()
        self.cv_image = None

        # Subscribe to camera feed
        rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.image_cb)

        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Keypoint Classifier setup
        self.keypoint_classifier = KeyPointClassifier()
        with open('/my_ros_data/catkin_ws/src/cosi119_src/gesture_cam/gesturebot/real/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    def image_cb(self, msg):
        """Process incoming images from the camera."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.cv_image = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")

    def run(self):
        """Main loop to process frames continuously."""
        rospy.loginfo("Gesture recognizer is running...")
        while not rospy.is_shutdown():
            if self.cv_image is not None:
                # Process the current frame
                self.process_frame()
            else:
                rospy.loginfo_once("Waiting for camera feed...")
            rospy.sleep(0.01)  # Small delay to prevent high CPU usage

    def process_frame(self):
        image = cv.flip(self.cv_image, 1)  # Mirror image for convenience
        debug_image = copy.deepcopy(image)

        # Convert image to RGB for MediaPipe
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Process hand landmarks
                landmark_list = self.calc_landmark_list(image, hand_landmarks)
                preprocessed_landmarks = self.preprocess_landmarks(landmark_list)

                # Classify gesture
                hand_sign_id = self.keypoint_classifier(preprocessed_landmarks)
                hand_sign_label = self.keypoint_classifier_labels[hand_sign_id]

                # Annotate image
                brect = self.calc_bounding_rect(image, hand_landmarks)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_bounding_rect(debug_image, brect)
                debug_image = self.draw_info_text(debug_image, brect, hand_sign_label)

        cv.imshow("Gesture Recognition", debug_image)
        cv.waitKey(1)

    def calc_landmark_list(self, image, hand_landmarks):
        """Calculate pixel coordinates of landmarks."""
        image_width, image_height = image.shape[1], image.shape[0]
        return [
            [
                int(landmark.x * image_width),
                int(landmark.y * image_height),
            ]
            for landmark in hand_landmarks.landmark
        ]

    def preprocess_landmarks(self, landmark_list):
        """Normalize landmarks to relative coordinates and scale."""
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

    def calc_bounding_rect(self, image, hand_landmarks):
        """Calculate bounding rectangle for hand."""
        image_width, image_height = image.shape[1], image.shape[0]
        points = [
            (int(landmark.x * image_width), int(landmark.y * image_height))
            for landmark in hand_landmarks.landmark
        ]
        x, y, w, h = cv.boundingRect(np.array(points))
        return [x, y, x + w, y + h]

    def draw_landmarks(self, image, landmark_list):
        """Draw landmarks on the image."""
        for point in landmark_list:
            cv.circle(image, tuple(point), 5, (255, 0, 0), -1)
        return image

    def draw_bounding_rect(self, image, brect):
        """Draw bounding rectangle on the image."""
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
        return image

    def draw_info_text(self, image, brect, hand_sign_label):
        """Draw gesture information on the image."""
        cv.putText(
            image,
            hand_sign_label,
            (brect[0], brect[1] - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        return image


if __name__ == "__main__":
    rospy.init_node("gesture_recognizer")
    recognizer = GestureRecognizer()
    try:
        recognizer.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Gesture Recognizer.")
    cv.destroyAllWindows()
