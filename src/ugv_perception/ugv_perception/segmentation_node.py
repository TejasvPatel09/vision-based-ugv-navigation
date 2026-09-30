import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO


class SegmentationNode(Node):

    def __init__(self):
        super().__init__('segmentation_node')

        self.bridge = CvBridge()

        # Load YOLO segmentation model
        self.model = YOLO('yolo11n-seg.pt')

        # Subscribe to ROS camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publisher for annotated segmentation image
        self.publisher = self.create_publisher(
            Image,
            '/segmentation/image',
            10
        )

        self.get_logger().info(
            'Segmentation node started'
        )

        self.get_logger().info(
            'Subscribed to /camera/image_raw'
        )

        self.get_logger().info(
            'Publishing segmentation on /segmentation/image'
        )

    def image_callback(self, msg):

        try:
            # Convert ROS Image to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            # Run YOLO segmentation
            results = self.model(
                frame,
                verbose=False
            )

            # Create annotated image with segmentation masks
            annotated_frame = results[0].plot()

            # Make sure image is uint8
            annotated_frame = annotated_frame.astype('uint8')

            # Make sure image has 3 color channels
            if len(annotated_frame.shape) == 2:
                annotated_frame = cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_GRAY2BGR
                )

            # Manually construct ROS Image message
            # This avoids the cv_bridge error we encountered.
            output_msg = Image()

            output_msg.header = msg.header
            output_msg.height = annotated_frame.shape[0]
            output_msg.width = annotated_frame.shape[1]
            output_msg.encoding = 'bgr8'
            output_msg.is_bigendian = 0
            output_msg.step = annotated_frame.shape[1] * 3
            output_msg.data = annotated_frame.tobytes()

            # Publish annotated segmentation image
            self.publisher.publish(output_msg)

            # Log number of detected objects
            self.get_logger().info(
                f'Segmentation processed: '
                f'{len(results[0].boxes)} objects'
            )

        except Exception as e:
            self.get_logger().error(
                f'Segmentation error: {e}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = SegmentationNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
