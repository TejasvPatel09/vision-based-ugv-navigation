import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class FrameSaver(Node):

    def __init__(self):
        super().__init__('frame_saver')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10
        )

        self.saved = False

    def callback(self, msg):

        if self.saved:
            return

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        cv2.imwrite(
            '/tmp/ugv_test_frame.jpg',
            frame
        )

        self.get_logger().info(
            'Saved camera frame to /tmp/ugv_test_frame.jpg'
        )

        self.saved = True


def main():

    rclpy.init()

    node = FrameSaver()

    while rclpy.ok() and not node.saved:
        rclpy.spin_once(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
