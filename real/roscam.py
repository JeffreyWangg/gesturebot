#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point, Pose, Twist
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import mediapipe as mp
import copy
import itertools
import csv
from model import KeyPointClassifier  # Import KeyPointClassifier
from mmdet.apis import DetInferencer

class GestureRecognizer:
    def __init__(self):

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

        # ROS setup
        self.bridge = CvBridge()
        self.cv_image = None

        # Subscribe to camera feed
        # rospy.Subscriber('/cv_camera/image_raw', Image, self.image_cb)
        self.image_sub = rospy.Subscriber('/raspicam_node/image/compressed',
                                          CompressedImage,
                                          self.image_cb)

        #Publishers
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.image_pub = rospy.Publisher("/image", Image, queue_size=1)

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
        # with open('/my_ros_data/catkin_ws/src/cosi119_src/gesture_cam/gesturebot/real/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        with open('/my_ros_data/gesture/src/real/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

        # obj seg
        config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
        checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
        self.inferencer = DetInferencer(config_file, checkpoint_file, device="cpu")

        while self.cv_image is None:
            pass

    def image_cb(self, msg):
        """Process incoming images from the camera."""
        try:
            # np_arr = np.frombuffer(msg.data, np.uint8)
            # self.cv_image = cv.imdecode(np_arr, cv.IMREAD_COLOR)
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")

    def segmentation_callback(self, event):
        # Convert the ROS Image message to a CV2 image
        # cv_image = self.bridge.compressed_imgmsg_to_cv2(self.cv_image) #rgb
        # cv_image = self.bridge.imgmsg_to_cv2(self.image) #rgb

        # obj seg
        output = self.inferencer(self.cv_image)

        #keep is list of indices for output['predictions']
        keep = self.non_max_suppression(output['predictions'][0]['bboxes'], output['predictions'][0]['scores'], 0.5)[:5]
        print([self.objects[i] for i in [output['predictions'][0]['labels'][i] for i in keep]])

        image = self.cv_image

        for box in bboxes:
            # print(box)
            bottom = (int(box[0]), int(box[1]))
            top = (int(box[2]), int(box[3]))
            image = cv.rectangle(image, bottom, top, (0, 0, 255), 2)
        image_msg = self.bridge.cv2_to_imgmsg(image)
        self.image_pub.publish(image_msg)

    def run(self):
        """Main loop to process frames continuously."""
        rospy.Timer(rospy.Duration(1.5), self.segmentation_callback) #start object recognition callback

        mode = 0
        rospy.loginfo("Gesture recognizer is running...")
        while not rospy.is_shutdown():
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = self.select_mode(key, mode)

            if self.cv_image is not None:
                # Process the current frame
                self.process_frame(number, mode)
            else:
                rospy.loginfo_once("Waiting for camera feed...")
            rospy.sleep(0.01)  # Small delay to prevent high CPU usage

    #for object segmentation
    def non_max_suppression(self, boxes, scores, threshold):
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            for j in order:
                # Calculate the IoU between the two boxes
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

        # Convert image to RGB for MediaPipe
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Process hand landmarks
                landmark_list = self.calc_landmark_list(image, hand_landmarks)
                preprocessed_landmarks = self.preprocess_landmarks(landmark_list)

                #log to csv
                self.logging_csv(number, mode, preprocessed_landmarks)

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
    
    def select_mode(self, key, mode):
        number = -1
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
        if key == 110:  # n
            mode = 0
        if key == 107:  # k
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
