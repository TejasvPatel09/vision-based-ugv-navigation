import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class DrivableMaskNode(Node):

    def __init__(self):
        super().__init__('drivable_mask_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/yolop/drivable_area',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            '/navigation/drivable_mask',
            10
        )

        self.get_logger().info(
            'Drivable mask node started'
        )

        self.get_logger().info(
            'Subscribed: /yolop/drivable_area'
        )

        self.get_logger().info(
            'Publishing: /navigation/drivable_mask'
        )

    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough'
        )

        # YOLOP mask is expected to contain
        # drivable-area pixels as non-zero values.
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
        else:
            gray = image

        # Convert to binary mask
        mask = np.where(
            gray > 0,
            255,
            0
        ).astype(np.uint8)

        # Remove small noise
        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        output = self.bridge.cv2_to_imgmsg(
            mask,
            encoding='mono8'
        )

        output.header = msg.header

        self.publisher.publish(output)


def main(args=None):

    rclpy.init(args=args)

    node = DrivableMaskNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
