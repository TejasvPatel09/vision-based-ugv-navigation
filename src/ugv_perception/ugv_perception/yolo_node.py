import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')

        self.bridge = CvBridge()
        self.model = YOLO('yolo11n.pt')

        # Subscribe to camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publish annotated image
        self.image_publisher = self.create_publisher(
            Image,
            '/yolo/image_annotated',
            10
        )

        # Publish detections
        self.detection_publisher = self.create_publisher(
            String,
            '/yolo/detections',
            10
        )

        self.get_logger().info('YOLO perception node started')
        self.get_logger().info(
            'Publishing annotated images on /yolo/image_annotated'
        )
        self.get_logger().info(
            'Publishing detections on /yolo/detections'
        )

    def image_callback(self, msg):

        # ROS Image -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        # Run YOLO
        results = self.model(
            frame,
            verbose=False
        )

        # Draw bounding boxes
        annotated_frame = results[0].plot()
        annotated_frame = annotated_frame.astype('uint8')

        # Make sure image has 3 channels
        if len(annotated_frame.shape) == 2:
            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_GRAY2BGR
            )

        # Create ROS image message manually
        output_msg = Image()

        output_msg.header = msg.header
        output_msg.height = annotated_frame.shape[0]
        output_msg.width = annotated_frame.shape[1]
        output_msg.encoding = 'bgr8'
        output_msg.is_bigendian = 0
        output_msg.step = annotated_frame.shape[1] * 3
        output_msg.data = annotated_frame.tobytes()

        # Publish annotated image
        self.image_publisher.publish(output_msg)

        # Create detection message
        detection_text = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = self.model.names[class_id]

                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x = int(x1)
                y = int(y1)
                width = int(x2 - x1)
                height = int(y2 - y1)

                # Format:
                # class,confidence,x,y,width,height
                detection_text.append(
                    f'{class_name},{confidence:.2f},'
                    f'{x},{y},{width},{height}'
                )

        # Publish detections
        detection_msg = String()

        if detection_text:
            detection_msg.data = ' | '.join(detection_text)
        else:
            detection_msg.data = 'No objects detected'

        self.detection_publisher.publish(detection_msg)


def main(args=None):

    rclpy.init(args=args)

    node = YoloNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
