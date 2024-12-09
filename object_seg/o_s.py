#!/usr/bin/env python
import torch
import rospy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point, Pose, Twist
import cv_bridge
import cv2
from mmdet.apis import DetInferencer


class ObjectSegmentation:
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

        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        # self.image_sub = rospy.Subscriber('/raspicam_node/image/compressed',
        #                                   CompressedImage,
        #                                   self.image_callback)
        self.image_sub = rospy.Subscriber('/cv_camera/image_raw',
                                           Image,
                                           self.image_callback)
        self.image_pub = rospy.Publisher("/image", Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()

        config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
        checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
        self.inferencer = DetInferencer(config_file, checkpoint_file, device="cpu")

        self.object_pose = {}

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg) #rgb
        output = self.inferencer(cv_image)

        keep = self.non_max_suppression(output['predictions'][0]['bboxes'], output['predictions'][0]['scores'], 0.5)[:3]
        print([self.objects[i] for i in [output['predictions'][0]['labels'][i] for i in keep]])
        
        # image_msg = self.bridge.cv2_to_imgmsg(output)
        self.image_pub.publish(msg)

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

    def main(self):
        rospy.loginfo("Image Processor Node Started")
        rospy.spin()

        # Close OpenCV windows on exit
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node("object_seg")
    ObjectSegmentation().main()
