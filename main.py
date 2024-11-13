import numpy as np
import math
import time
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import rospy
from geometry_msgs.msg import Point, Pose, Twist
from sensor_msgs.msg import CompressedImage, Image
import cv_bridge

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
model_path = "/my_ros_data/gesture/src/hand_landmarker.task"

class Gesture:
    def __init__(self):
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber('/raspicam_node/image/compressed',
                                          CompressedImage,
                                          self.image_callback)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a hand landmarker instance with the image mode:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE, 
            num_hands=2)
        self.landmarker = HandLandmarker.create_from_options(options)

        self.rgb_image = None
        while self.rgb_image is None:
            pass

    def image_callback(self, msg):
        self.rgb_image = cv_bridge.CvBridge().compressed_imgmsg_to_cv2(msg)

    def run(self):
        start = time.time()
        while not rospy.is_shutdown():
            if self.landmarker is None:
                return
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb_image)
            hand_landmarker_result = self.landmarker.detect(mp_image)

            annotated_image = self.draw_landmarks_on_image(self.rgb_image, hand_landmarker_result)
            # print(annotated_image)
            # print(hand_landmarker_result)
            cv.waitKey(50)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            handedness = f"{handedness[0].category_name}"
            cv.putText(annotated_image, handedness,
                        (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)
            print(handedness)
            twist = Twist()
            if handedness == "Left":
                twist.linear.x = 0.2
            elif handedness == "Right":
                twist.linear.x = 0
            self.cmd_vel_pub.publish(twist)
            
        cv.imshow('main window', cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

        return annotated_image

if __name__ == '__main__':
    rospy.init_node('gesture')
    Gesture().run()