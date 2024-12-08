#!/usr/bin/env python
import torch
import rospy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point, Pose, Twist
import cv_bridge
import cv2


class Depth:
    def __init__(self):
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        # self.image_sub = rospy.Subscriber('/raspicam_node/image/compressed',
        #                                   CompressedImage,
        #                                   self.image_callback)
        self.image_sub = rospy.Subscriber('/cv_camera/image_raw',
                                           Image,
                                           self.image_callback)
        self.image_pub = rospy.Publisher("/image", Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()

        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.device = torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()

        self.transform = midas_transforms.small_transform

        while self.transform is None:
            pass

    def image_callback(self, msg):
        # Convert the ROS Image message to a CV2 image
        # cv_image = self.bridge.compressed_imgmsg_to_cv2(msg) #rgb
        cv_image = self.bridge.imgmsg_to_cv2(msg) #rgb
        # rospy.loginfo("Image received and converted.")
        # image_msg = self.bridge.cv2_to_compressed_imgmsg(cv_image)
        # img = cv2.imread(filename)
        # img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(cv_image).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=cv_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy() / 1200
        # h, w = output.shape
        # mat = cv2.CreateMat(h, w, cv2.CV_32FC3)
        # img = cv2.CvtColor(output, cv2.CV_GRAY2BGR)
        # print(output)
        # image_msg = self.bridge.cv2_to_compressed_imgmsg(img)
        image_msg = self.bridge.cv2_to_imgmsg(output)
        self.image_pub.publish(image_msg)

    def main(self):
        rospy.loginfo("Image Processor Node Started")
        rospy.spin()

        # Close OpenCV windows on exit
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node("depth")
    Depth().main()
