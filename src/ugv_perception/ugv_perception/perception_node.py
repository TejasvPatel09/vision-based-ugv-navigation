import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        self.publisher_ = self.create_publisher(
            String,
            'perception_status',
            10
        )

        self.timer = self.create_timer(1.0, self.publish_status)

        self.get_logger().info('UGV Perception Node Started')

    def publish_status(self):
        msg = String()
        msg.data = 'Camera perception system running'

        self.publisher_.publish(msg)

        self.get_logger().info(
            'Publishing: "%s"' % msg.data
        )


def main(args=None):
    rclpy.init(args=args)

    node = PerceptionNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
