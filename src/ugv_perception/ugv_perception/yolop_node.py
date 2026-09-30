import cv2
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class YOLOPNode(Node):

    def __init__(self):
        super().__init__('yolop_node')

        self.bridge = CvBridge()

        # -------------------------------------------------
        # Load YOLOP ONNX model
        # -------------------------------------------------

        self.model_path = (
            '/home/tejasvpatel/YOLOP/weights/yolop-640-640.onnx'
        )

        self.session = ort.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider']
        )

        self.get_logger().info(
            'YOLOP ONNX model loaded successfully'
        )

        # -------------------------------------------------
        # ROS subscriber
        # -------------------------------------------------

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # -------------------------------------------------
        # ROS publishers
        # -------------------------------------------------

        self.drive_publisher = self.create_publisher(
            Image,
            '/yolop/drivable_area',
            10
        )

        self.visualization_publisher = self.create_publisher(
            Image,
            '/yolop/visualization',
            10
        )

        self.get_logger().info(
            'Subscribed to /camera/image_raw'
        )

        self.get_logger().info(
            'Publishing drivable mask on /yolop/drivable_area'
        )

        self.get_logger().info(
            'Publishing visualization on /yolop/visualization'
        )

    # =====================================================
    # YOLOP letterbox preprocessing
    # =====================================================

    def resize_unscale(
        self,
        image,
        new_shape=(640, 640),
        color=114
    ):

        shape = image.shape[:2]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Create canvas
        canvas = np.zeros(
            (new_shape[0], new_shape[1], 3),
            dtype=np.uint8
        )

        canvas.fill(color)

        # Scale ratio
        r = min(
            new_shape[0] / shape[0],
            new_shape[1] / shape[1]
        )

        # New image size
        new_unpad = (
            int(round(shape[1] * r)),
            int(round(shape[0] * r))
        )

        new_unpad_w = new_unpad[0]
        new_unpad_h = new_unpad[1]

        # Padding
        pad_w = new_shape[1] - new_unpad_w
        pad_h = new_shape[0] - new_unpad_h

        dw = pad_w // 2
        dh = pad_h // 2

        # Resize
        if shape[::-1] != new_unpad:

            image = cv2.resize(
                image,
                new_unpad,
                interpolation=cv2.INTER_AREA
            )

        # Place image on canvas
        canvas[
            dh:dh + new_unpad_h,
            dw:dw + new_unpad_w,
            :
        ] = image

        return (
            canvas,
            r,
            dw,
            dh,
            new_unpad_w,
            new_unpad_h
        )

    # =====================================================
    # Image callback
    # =====================================================

    def image_callback(self, msg):

        try:

            # -------------------------------------------------
            # ROS Image → OpenCV
            # -------------------------------------------------

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            original_height, original_width = frame.shape[:2]

            # -------------------------------------------------
            # Convert BGR → RGB
            # -------------------------------------------------

            img_rgb = frame[:, :, ::-1].copy()

            # -------------------------------------------------
            # YOLOP letterbox resize
            # -------------------------------------------------

            (
                canvas,
                r,
                dw,
                dh,
                new_unpad_w,
                new_unpad_h
            ) = self.resize_unscale(
                img_rgb,
                (640, 640)
            )

            # -------------------------------------------------
            # YOLOP normalization
            # -------------------------------------------------

            image = canvas.astype(
                np.float32
            )

            image /= 255.0

            image[:, :, 0] -= 0.485
            image[:, :, 1] -= 0.456
            image[:, :, 2] -= 0.406

            image[:, :, 0] /= 0.229
            image[:, :, 1] /= 0.224
            image[:, :, 2] /= 0.225

            # HWC → CHW
            image = image.transpose(
                2,
                0,
                1
            )

            # Add batch dimension
            image = np.expand_dims(
                image,
                axis=0
            )

            # -------------------------------------------------
            # YOLOP inference
            # -------------------------------------------------

            (
                det_out,
                drive_area_seg,
                lane_line_seg
            ) = self.session.run(
                [
                    'det_out',
                    'drive_area_seg',
                    'lane_line_seg'
                ],
                input_feed={
                    'images': image
                }
            )

            # -------------------------------------------------
            # Remove letterbox padding
            # -------------------------------------------------

            # -------------------------------------------------
            # YOLOP segmentation masks
            # Match official test_onnx.py implementation
            # -------------------------------------------------

            drive_area_seg = drive_area_seg[
                :,
                :,
                dh:dh + new_unpad_h,
                dw:dw + new_unpad_w
            ]

            lane_line_seg = lane_line_seg[
                :,
                :,
                dh:dh + new_unpad_h,
                dw:dw + new_unpad_w
            ]

            # Select segmentation class
            drive_mask = np.argmax(
                drive_area_seg,
                axis=1
            )[0]

            lane_mask = np.argmax(
                lane_line_seg,
                axis=1
            )[0]

            # Convert to 0/255
            drive_mask = (
                drive_mask * 255
            ).astype(np.uint8)

            lane_mask = (
                lane_mask * 255
            ).astype(np.uint8)

            # Resize to original camera resolution
            drive_mask = cv2.resize(
                drive_mask,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR
            )

            lane_mask = cv2.resize(
                lane_mask,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR
            )

            # -------------------------------------------------
            # Get segmentation classes
            # -------------------------------------------------
            self.get_logger().info(
                f"YOLOP drive output: shape={drive_area_seg.shape}, "
                f"min={drive_area_seg.min():.4f}, "
                f"max={drive_area_seg.max():.4f}"
            )
            self.get_logger().info(
                f"Drive channel 0: min={drive_area_seg[0,0].min():.4f}, "
                f"max={drive_area_seg[0,0].max():.4f}, "
                f"mean={drive_area_seg[0,0].mean():.4f}"
            )

            self.get_logger().info(
                f"Drive channel 1: min={drive_area_seg[0,1].min():.4f}, "
                f"max={drive_area_seg[0,1].max():.4f}, "
                f"mean={drive_area_seg[0,1].mean():.4f}"
            ) 
            drive_mask = np.argmax(
                drive_area_seg,
                axis=1
            )[0]

            self.get_logger().info(
                f"Drive classes: {np.unique(drive_mask, return_counts=True)}"
            )

            lane_mask = np.argmax(
                lane_line_seg,
                axis=1
            )[0]

            # -------------------------------------------------
            # Convert masks to 0/255
            # -------------------------------------------------

            drive_mask = (
                drive_mask * 255
            ).astype(np.uint8)

            lane_mask = (
                lane_mask * 255
            ).astype(np.uint8)

            # -------------------------------------------------
            # Resize masks to original camera resolution
            # -------------------------------------------------

            drive_mask = cv2.resize(
                drive_mask,
                (
                    original_width,
                    original_height
                ),
                interpolation=cv2.INTER_LINEAR
            )

            lane_mask = cv2.resize(
                lane_mask,
                (
                    original_width,
                    original_height
                ),
                interpolation=cv2.INTER_LINEAR
            )

            # -------------------------------------------------
            # Create visualization
            # -------------------------------------------------

            visualization = frame.copy()

            # Drivable area = green
            drive_region = drive_mask > 127

            green_overlay = np.zeros_like(
                frame
            )

            green_overlay[
                drive_region
            ] = (0, 255, 0)

            # Lane area = blue
            lane_region = lane_mask > 127

            blue_overlay = np.zeros_like(
                frame
            )

            blue_overlay[
                lane_region
            ] = (255, 0, 0)

            # Blend drivable area
            visualization = cv2.addWeighted(
                visualization,
                0.65,
                green_overlay,
                0.35,
                0
            )

            # Blend lane area
            visualization = cv2.addWeighted(
                visualization,
                0.75,
                blue_overlay,
                0.25,
                0
            )

            # -------------------------------------------------
            # Publish drivable-area mask
            # -------------------------------------------------

            drive_mask_bgr = cv2.cvtColor(
                drive_mask,
                cv2.COLOR_GRAY2BGR
            )

            drive_msg = Image()

            drive_msg.header = msg.header
            drive_msg.height = drive_mask.shape[0]
            drive_msg.width = drive_mask.shape[1]
            drive_msg.encoding = 'mono8'
            drive_msg.is_bigendian = 0
            drive_msg.step = drive_mask.shape[1]
            drive_msg.data = drive_mask.tobytes()

            self.drive_publisher.publish(
                drive_msg
            )

            # -------------------------------------------------
            # Publish visualization
            # -------------------------------------------------

            visualization_msg = Image()

            visualization_msg.header = msg.header
            visualization_msg.height = visualization.shape[0]
            visualization_msg.width = visualization.shape[1]
            visualization_msg.encoding = 'bgr8'
            visualization_msg.is_bigendian = 0
            visualization_msg.step = visualization.shape[1] * 3
            visualization_msg.data = visualization.tobytes()

            self.visualization_publisher.publish(
                visualization_msg
            )

            self.get_logger().info(
                'YOLOP processed frame successfully'
            )

        except Exception as e:

            self.get_logger().error(
                f'YOLOP error: {e}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = YOLOPNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
