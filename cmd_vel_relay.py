import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Twist


class CmdVelRelay(Node):
    def __init__(self):
        super().__init__('cmd_vel_relay')

        self.sub = self.create_subscription(
            TwistStamped,
            '/cmd_vel_nav',
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            Twist,
            '/cmd_vel_gz',
            10
        )

    def callback(self, msg):
        twist = Twist()

        twist.linear.x = msg.twist.linear.x
        twist.linear.y = msg.twist.linear.y
        twist.linear.z = msg.twist.linear.z

        twist.angular.x = msg.twist.angular.x
        twist.angular.y = msg.twist.angular.y
        twist.angular.z = msg.twist.angular.z

        self.pub.publish(twist)


def main():
    rclpy.init()
    node = CmdVelRelay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
