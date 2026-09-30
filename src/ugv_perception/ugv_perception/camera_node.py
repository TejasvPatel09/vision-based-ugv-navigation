import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.srv import SetCameraInfo

from cv_bridge import CvBridge


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        self.bridge = CvBridge()
        
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if not self.camera.isOpened():
            self.get_logger().error(
                'Could not open camera /dev/video0'
            )
            return

        # Camera settings
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------

        self.image_publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )

        self.camera_info_publisher = self.create_publisher(
            CameraInfo,
            '/camera/camera_info',
            10
        )

        # --------------------------------------------------
        # Calibrated Camera Information
        # --------------------------------------------------

        self.camera_info = CameraInfo()

        self.camera_info.width = 640
        self.camera_info.height = 480

        self.camera_info.distortion_model = 'plumb_bob'

        # Distortion coefficients
        self.camera_info.d = [
            0.203867,
            -0.160047,
            0.033008,
            -0.004111,
            0.0
        ]

        # Camera matrix K
        self.camera_info.k = [
            464.29937, 0.0, 300.63136,
            0.0, 468.03359, 284.13941,
            0.0, 0.0, 1.0
        ]

        # Rectification matrix R
        self.camera_info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        # Projection matrix P
        self.camera_info.p = [
            504.32433, 0.0, 298.56552, 0.0,
            0.0, 491.19239, 299.03611, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        # --------------------------------------------------
        # Camera calibration service
        # --------------------------------------------------

        self.calibration_service = self.create_service(
            SetCameraInfo,
            '/camera/set_camera_info',
            self.set_camera_info_callback
        )

        # --------------------------------------------------
        # Timer
        # --------------------------------------------------

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.publish_frame
        )

        self.get_logger().info(
            'Camera node started'
        )

        self.get_logger().info(
            'Resolution: 640x480'
        )

        self.get_logger().info(
            'Publishing: /camera/image_raw'
        )

        self.get_logger().info(
            'Publishing calibrated: /camera/camera_info'
        )

        self.get_logger().info(
            'Service: /camera/set_camera_info'
        )

    def set_camera_info_callback(
        self,
        request,
        response
    ):

        self.camera_info = request.camera_info

        response.success = True

        response.status_message = (
            'Camera calibration information stored'
        )

        self.get_logger().info(
            'Camera calibration information received'
        )

        return response

    def publish_frame(self):

        ret, frame = self.camera.read()

        if not ret:

            self.get_logger().warning(
                'Could not read camera frame'
            )

            return

        # Create image message
        image_msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding='bgr8'
        )

        image_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        image_msg.header.frame_id = 'camera_link'

        # Give CameraInfo the same timestamp/frame
        self.camera_info.header = image_msg.header

        # Publish
        self.image_publisher.publish(
            image_msg
        )

        self.camera_info_publisher.publish(
            self.camera_info
        )

    def destroy_node(self):

        if hasattr(self, 'camera'):

            self.camera.release()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = CameraNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
