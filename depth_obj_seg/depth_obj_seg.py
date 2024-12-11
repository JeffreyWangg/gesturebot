#!/usr/bin/env python
import torch
import rospy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point, Pose, Twist
from mmdet.apis import DetInferencer
import cv_bridge
import cv2
import numpy as np

#ros timer for obj => seg

class DepthObjSeg:
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

        self.image = None

        # obj seg
        config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
        checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
        self.inferencer = DetInferencer(config_file, checkpoint_file, device="cpu")

        # depth
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.device = torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()

        self.transform = midas_transforms.small_transform

        while self.transform and self.image is None:
            pass

    def image_callback(self, msg):
        self.image = msg

    #depth MIGHT be 1305.2767 * 0.9575^x

    def timer_callback(self, event):
        # Convert the ROS Image message to a CV2 image
        # cv_image = self.bridge.compressed_imgmsg_to_cv2(msg) #rgb
        cv_image = self.bridge.imgmsg_to_cv2(self.image) #rgb

        # obj seg
        output = self.inferencer(cv_image)

        #keep is list of indices for output['predictions']
        keep = self.non_max_suppression(output['predictions'][0]['bboxes'], output['predictions'][0]['scores'], 0.5)[:8]
        print([self.objects[i] for i in [output['predictions'][0]['labels'][i] for i in keep]])

        # depth 
        input_batch = self.transform(cv_image).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=cv_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # print([output['predictions'][0]['bboxes'][i] for i in keep])
        # to draw image
        output_image_raw = prediction.cpu().numpy()
        output_image = output_image_raw / 1200
        bboxes = [output['predictions'][0]['bboxes'][i] for i in keep]

        #print depths for test
        # self.print_depths(bboxes, output_image_raw)
        depths = self.calc_depth(bboxes, output_image_raw)
        # for i in range(len(keep) - 1):
        #     if self.objects[output['predictions'][0]['labels'][keep[i]]] == 'bottle':
        #         print(depths[i])
        #         print(65.5181 * (0.9978 ** depths[i]))


        for box in bboxes:
            # print(box)
            bottom = (int(box[0]), int(box[1]))
            top = (int(box[2]), int(box[3]))
            output_image = cv2.rectangle(output_image, bottom, top, (0, 0, 255), 2)
        image_msg = self.bridge.cv2_to_imgmsg(output_image)
        self.image_pub.publish(image_msg)

    def calc_depth(self, bboxes, disparity_map):
        result = []
        for box in bboxes:
            #x is column y is row, so do [y][x]
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            subarray = disparity_map[x_min:x_max, y_min:y_max]
            median = np.median(subarray)
            mask = subarray >= median
            subarray = subarray[mask]
            result.append(np.mean(subarray))
        return result

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
        rospy.Timer(rospy.Duration(1.5), self.timer_callback)
        rospy.spin()

        # Close OpenCV windows on exit
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node("depth_obj_seg")
    DepthObjSeg().main()
