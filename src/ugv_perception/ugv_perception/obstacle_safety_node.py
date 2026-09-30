import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

import numpy as np
import math


class ObstacleSafetyNode(Node):

    def __init__(self):
        super().__init__('obstacle_safety_node')

        self.bridge = CvBridge()
        self.drivable_mask = None
        self.scan = None

        # Safety status publisher
        self.publisher = self.create_publisher(
            String,
            '/navigation/safety_status',
            10
        )

        # YOLO detections
        self.create_subscription(
            String,
            '/yolo/detections',
            self.detection_callback,
            10
        )

        # Drivable mask
        self.create_subscription(
            Image,
            '/navigation/drivable_mask',
            self.mask_callback,
            10
        )

        # LiDAR
        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publish safety status continuously
        self.timer = self.create_timer(
            0.2,
            self.safety_timer_callback
        )

        self.get_logger().info(
            'Obstacle safety node started - LiDAR + YOLO'
        )

    # --------------------------------------------------
    # LiDAR callback
    # --------------------------------------------------

    def scan_callback(self, msg):

        self.scan = msg

    # --------------------------------------------------
    # Camera mask callback
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Get minimum distance from LiDAR sector
    # --------------------------------------------------

    def get_sector_distance(
        self,
        start_angle,
        end_angle
    ):

        if self.scan is None:

            return float('inf')

        values = []

        for i, distance in enumerate(
            self.scan.ranges
        ):

            if not math.isfinite(distance):
                continue

            if distance < 0.25:
                continue

            if distance > self.scan.range_max:
                continue

            angle = math.degrees(
                self.scan.angle_min +
                i * self.scan.angle_increment
            ) % 360

            if start_angle <= angle <= end_angle:

                values.append(distance)

        if not values:

            return float('inf')

        return min(values)

    # --------------------------------------------------
    # LiDAR decision
    # --------------------------------------------------

    def lidar_decision(self):

        if self.scan is None:

            return 'SAFE'

        # Forward sector
        front_left = self.get_sector_distance(
            330,
            360
        )

        front_right = self.get_sector_distance(
            0,
            30
        )

        front = min(
            front_left,
            front_right
        )

        # Left side
        left = self.get_sector_distance(
            30,
            90
        )

        # Right side
        right = self.get_sector_distance(
            270,
            330
        )

        self.get_logger().info(
            f'LiDAR | '
            f'Front: {front:.2f} m | '
            f'Left: {left:.2f} m | '
            f'Right: {right:.2f} m'
        )

        # Emergency stop
        if front < 0.45:

            return 'STOP'

        # Avoid obstacle
        if front < 0.80:

            if left > right:

                return 'TURN_LEFT'

            elif right > left:

                return 'TURN_RIGHT'

            else:

                return 'STOP'

        return 'SAFE'

    # --------------------------------------------------
    # Continuous safety timer
    # --------------------------------------------------

    def safety_timer_callback(self):

        status = self.lidar_decision()

        self.publish_status(status)

        if status != 'SAFE':

            self.get_logger().warning(
                f'LiDAR decision: {status}'
            )

    # --------------------------------------------------
    # Drivable area calculation
    # --------------------------------------------------

    def check_region(
        self,
        x1,
        x2
    ):

        if self.drivable_mask is None:

            return 0.0

        mask = self.drivable_mask

        height, width = mask.shape

        x1 = max(
            0,
            min(x1, width)
        )

        x2 = max(
            0,
            min(x2, width)
        )

        y1 = int(
            height * 0.55
        )

        region = mask[
            y1:height,
            x1:x2
        ]

        if region.size == 0:

            return 0.0

        return (
            np.count_nonzero(region) /
            region.size
        ) * 100.0

    # --------------------------------------------------
    # YOLO callback
    # --------------------------------------------------

    def detection_callback(self, msg):

        # LiDAR has priority
        lidar_status = self.lidar_decision()

        if lidar_status != 'SAFE':

            self.publish_status(
                lidar_status
            )

            return

        # No YOLO detections
        if not msg.data.strip():

            self.publish_status(
                'SAFE'
            )

            return

        detections = msg.data.split(';')

        dangerous_objects = [
            'person',
            'car',
            'truck',
            'bus',
            'motorcycle',
            'bicycle'
        ]

        status = 'SAFE'

        for detection in detections:

            parts = detection.split(',')

            if len(parts) < 6:

                continue

            label = parts[0].strip().lower()

            try:

                confidence = float(parts[1])

                x1 = int(parts[2])
                y1 = int(parts[3])
                x2 = int(parts[4])
                y2 = int(parts[5])

            except ValueError:

                continue

            if label not in dangerous_objects:

                continue

            if confidence < 0.50:

                continue

            center_x = (
                x1 + x2
            ) / 2

            left_free = self.check_region(
                0,
                213
            )

            center_free = self.check_region(
                213,
                427
            )

            right_free = self.check_region(
                427,
                640
            )

            if center_x < 213:

                if right_free >= 3:

                    status = 'TURN_RIGHT'

                else:

                    status = 'STOP'

            elif center_x > 427:

                if left_free >= 3:

                    status = 'TURN_LEFT'

                else:

                    status = 'STOP'

            else:

                if (
                    left_free >= 3 and
                    right_free >= 3
                ):

                    if left_free >= right_free:

                        status = 'TURN_LEFT'

                    else:

                        status = 'TURN_RIGHT'

                elif left_free >= 3:

                    status = 'TURN_LEFT'

                elif right_free >= 3:

                    status = 'TURN_RIGHT'

                else:

                    status = 'STOP'

        self.publish_status(status)

    # --------------------------------------------------
    # Publish status
    # --------------------------------------------------

    def publish_status(self, status):

        msg = String()

        msg.data = status

        self.publisher.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = ObstacleSafetyNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
