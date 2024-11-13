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

image_callback(self, msg):
        rgb_image = self.bridge.imgmsg_to_cv2(msg)

image_sub = rospy.Subscriber('/camera/rgb/image_raw',
                                          Image,
                                          self.image_callback)

