import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge

import cv2
import numpy as np


class NavigationGridNode(Node):

    def __init__(self):
        super().__init__('navigation_grid_node')

        self.bridge = CvBridge()

        self.drivable_mask = None
        self.detections = []

        # --------------------------------------------------
        # DRIVABLE MASK
        # --------------------------------------------------

        self.mask_subscription = self.create_subscription(
            Image,
            '/navigation/drivable_mask',
            self.mask_callback,
            10
        )

        # --------------------------------------------------
        # YOLO DETECTIONS
        # --------------------------------------------------

        self.detection_subscription = self.create_subscription(
            String,
            '/yolo/detections',
            self.detection_callback,
            10
        )

        # --------------------------------------------------
        # LOCAL GRID
        # --------------------------------------------------

        self.grid_publisher = self.create_publisher(
            Image,
            '/navigation/local_grid',
            10
        )

        self.get_logger().info(
            'Navigation grid node started'
        )

        self.get_logger().info(
            'Subscribed: /navigation/drivable_mask'
        )

        self.get_logger().info(
            'Subscribed: /yolo/detections'
        )

        self.get_logger().info(
            'Publishing: /navigation/local_grid'
        )

        self.timer = self.create_timer(
            0.2,
            self.create_grid
        )

    # ==================================================
    # MASK CALLBACK
    # ==================================================

    def mask_callback(self, msg):

        try:

            self.drivable_mask = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='mono8'
            )

        except Exception as e:

            self.get_logger().error(
                f'Mask conversion error: {e}'
            )

    # ==================================================
    # DETECTION CALLBACK
    # ==================================================

    def detection_callback(self, msg):

        self.detections = []

        if not msg.data.strip():
            return

        for detection in msg.data.split(';'):

            parts = detection.split(',')

            if len(parts) < 6:
                continue

            try:

                label = parts[0].strip().lower()
                confidence = float(parts[1])

                x1 = int(parts[2])
                y1 = int(parts[3])
                x2 = int(parts[4])
                y2 = int(parts[5])

            except ValueError:

                continue

            if confidence < 0.50:
                continue

            self.detections.append(
                (
                    label,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

    # ==================================================
    # CREATE LOCAL GRID
    # ==================================================

    def create_grid(self):

        if self.drivable_mask is None:
            return

        mask = self.drivable_mask.copy()

        height, width = mask.shape

        # ------------------------------------------------
        # START WITH UNKNOWN / OBSTACLE
        # ------------------------------------------------

        grid = np.full(
            (height, width),
            128,
            dtype=np.uint8
        )

        # ------------------------------------------------
        # DRIVABLE AREA = FREE
        # ------------------------------------------------

        grid[mask > 0] = 0

        # ------------------------------------------------
        # OBSTACLES = OCCUPIED
        # ------------------------------------------------

        dangerous_objects = [
            'person',
            'car',
            'truck',
            'bus',
            'motorcycle',
            'bicycle'
        ]

        for detection in self.detections:

            label, confidence, x1, y1, x2, y2 = detection

            if label not in dangerous_objects:
                continue

            # Keep coordinates inside image
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))

            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            # Draw obstacle
            grid[y1:y2, x1:x2] = 255

        # ------------------------------------------------
        # KEEP ONLY NAVIGATION-RELEVANT LOWER AREA
        # ------------------------------------------------

        navigation_grid = np.zeros_like(grid)

        roi_start = int(height * 0.50)

        navigation_grid[roi_start:height, :] = (
            grid[roi_start:height, :]
        )

        # ------------------------------------------------
        # PUBLISH GRID
        # ------------------------------------------------

        output = self.bridge.cv2_to_imgmsg(
            navigation_grid,
            encoding='mono8'
        )

        output.header.stamp = (
            self.get_clock().now().to_msg()
        )

        output.header.frame_id = 'camera_link'

        self.grid_publisher.publish(output)


def main(args=None):

    rclpy.init(args=args)

    node = NavigationGridNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':

    main()
