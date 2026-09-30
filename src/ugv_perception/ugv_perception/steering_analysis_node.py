import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class SteeringAnalysisNode(Node):

    def __init__(self):
        super().__init__('steering_analysis_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/navigation/drivable_mask',
            self.image_callback,
            10
        )

        # Previous decisions for temporal smoothing
        self.decision_history = []

        # Number of recent frames used for smoothing
        self.history_size = 5

        self.get_logger().info(
            'Steering analysis node started'
        )

        self.get_logger().info(
            'Near-field weighting enabled'
        )

        self.get_logger().info(
            'Temporal smoothing enabled'
        )

    def calculate_score(self, region):

        if region.size == 0:
            return 0.0

        return (
            np.count_nonzero(region)
            / region.size
        ) * 100.0

    def image_callback(self, msg):

        try:

            mask = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='mono8'
            )

            height, width = mask.shape

            # -------------------------------------------------
            # Use lower 50% of image
            # -------------------------------------------------

            roi = mask[
                int(height * 0.50):height,
                :
            ]

            roi_height = roi.shape[0]

            # -------------------------------------------------
            # Split into FAR and NEAR regions
            # -------------------------------------------------

            far_region = roi[
                :int(roi_height * 0.40),
                :
            ]

            near_region = roi[
                int(roi_height * 0.40):,
                :
            ]

            # -------------------------------------------------
            # Split LEFT / CENTER / RIGHT
            # -------------------------------------------------

            third = width // 3

            far_left = far_region[:, :third]
            far_center = far_region[:, third:2 * third]
            far_right = far_region[:, 2 * third:]

            near_left = near_region[:, :third]
            near_center = near_region[:, third:2 * third]
            near_right = near_region[:, 2 * third:]

            # -------------------------------------------------
            # Calculate scores
            #
            # Near region gets 70% importance.
            # Far region gets 30%.
            # -------------------------------------------------

            left_score = (
                0.30 * self.calculate_score(far_left)
                + 0.70 * self.calculate_score(near_left)
            )

            center_score = (
                0.30 * self.calculate_score(far_center)
                + 0.70 * self.calculate_score(near_center)
            )

            right_score = (
                0.30 * self.calculate_score(far_right)
                + 0.70 * self.calculate_score(near_right)
            )

            # -------------------------------------------------
            # Safety threshold
            # -------------------------------------------------

            minimum_score = 10.0

            if (
                left_score < minimum_score
                and
                center_score < minimum_score
                and
                right_score < minimum_score
            ):

                raw_decision = 'STOP'

            elif center_score >= 15.0:

                raw_decision = 'FORWARD'

            elif (
                left_score > right_score
                and
                left_score >= minimum_score
            ):

                raw_decision = 'TURN_LEFT'

            elif (
                right_score > left_score
                and
                right_score >= minimum_score
            ):

                raw_decision = 'TURN_RIGHT'

            else:

                raw_decision = 'STOP'

            # -------------------------------------------------
            # Temporal smoothing
            # -------------------------------------------------

            self.decision_history.append(
                raw_decision
            )

            if len(self.decision_history) > self.history_size:

                self.decision_history.pop(0)

            # Count decisions
            unique, counts = np.unique(
                self.decision_history,
                return_counts=True
            )

            decision_counts = dict(
                zip(unique, counts)
            )

            final_decision = max(
                decision_counts,
                key=decision_counts.get
            )

            # -------------------------------------------------
            # Print result
            # -------------------------------------------------

            self.get_logger().info(
                f'LEFT: {left_score:.1f}% | '
                f'CENTER: {center_score:.1f}% | '
                f'RIGHT: {right_score:.1f}% | '
                f'RAW: {raw_decision} | '
                f'SMOOTH: {final_decision}'
            )

        except Exception as e:

            self.get_logger().error(
                f'Steering analysis error: {e}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = SteeringAnalysisNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':

    main()
