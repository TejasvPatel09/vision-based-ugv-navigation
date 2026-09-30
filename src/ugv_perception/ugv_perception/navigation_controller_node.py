import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped

from cv_bridge import CvBridge


class NavigationControllerNode(Node):

    def __init__(self):
        super().__init__('navigation_controller_node')

        self.bridge = CvBridge()

        self.drivable_mask = None
        self.safety_status = 'STOP'
        self.astar_direction = 'FORWARD'

        # Controlled avoidance state
        self.avoidance_command = None
        self.avoidance_end_time = None

        # -----------------------------
        # Subscribers
        # -----------------------------

        self.mask_subscription = self.create_subscription(
            Image,
            '/navigation/drivable_mask',
            self.mask_callback,
            10
        )

        self.safety_subscription = self.create_subscription(
            String,
            '/navigation/safety_status',
            self.safety_callback,
            10
        )

        self.astar_subscription = self.create_subscription(
            String,
            '/navigation/path_direction',
            self.astar_callback,
            10
        )

        # -----------------------------
        # Publisher
        # -----------------------------

        self.cmd_publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel_nav',
            10
        )

        # -----------------------------
        # Timer
        # -----------------------------

        self.timer = self.create_timer(
            0.2,
            self.control_loop
        )

        self.get_logger().info(
            'Navigation controller started'
        )

    # =========================================================
    # DRIVABLE MASK
    # =========================================================

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

    # =========================================================
    # SAFETY
    # =========================================================

    def safety_callback(self, msg):

        new_status = msg.data.strip()

        # Start a short controlled turn
        if new_status == 'TURN_LEFT':

            if self.avoidance_command != 'TURN_LEFT':

                self.avoidance_command = 'TURN_LEFT'

                self.avoidance_end_time = (
                    self.get_clock().now().nanoseconds
                    + 1_500_000_000
                )

                self.get_logger().warning(
                    'Starting controlled LEFT avoidance turn'
                )

        elif new_status == 'TURN_RIGHT':

            if self.avoidance_command != 'TURN_RIGHT':

                self.avoidance_command = 'TURN_RIGHT'

                self.avoidance_end_time = (
                    self.get_clock().now().nanoseconds
                    + 1_500_000_000
                )

                self.get_logger().warning(
                    'Starting controlled RIGHT avoidance turn'
                )

        elif new_status == 'STOP':

            self.avoidance_command = None
            self.avoidance_end_time = None

        self.safety_status = new_status

    # =========================================================
    # A*
    # =========================================================

    def astar_callback(self, msg):

        self.astar_direction = msg.data.strip()

    # =========================================================
    # CONTROL LOOP
    # =========================================================

    def control_loop(self):

        now = self.get_clock().now().nanoseconds

        # -----------------------------------------
        # HARD STOP
        # -----------------------------------------

        if self.safety_status == 'STOP':

            self.avoidance_command = None
            self.avoidance_end_time = None

            self.publish_command('STOP')

            return

        # -----------------------------------------
        # CONTROLLED AVOIDANCE TURN
        # -----------------------------------------

        if self.avoidance_command is not None:

            if (
                self.avoidance_end_time is not None
                and now < self.avoidance_end_time
            ):

                self.publish_command(
                    self.avoidance_command
                )

                return

            # Turn finished
            self.get_logger().info(
                'Avoidance turn completed'
            )

            self.avoidance_command = None
            self.avoidance_end_time = None

        # -----------------------------------------
        # A* PATH PLANNING
        # -----------------------------------------

        if self.astar_direction == 'STOP':

            self.publish_command('STOP')

            return

        if self.astar_direction == 'TURN_LEFT':

            self.publish_command('TURN_LEFT')

            return

        if self.astar_direction == 'TURN_RIGHT':

            self.publish_command('TURN_RIGHT')

            return

        # -----------------------------------------
        # FORWARD
        # -----------------------------------------

        if self.astar_direction == 'FORWARD':

            self.publish_command('FORWARD')

            return

        # -----------------------------------------
        # FALLBACK
        # -----------------------------------------

        self.publish_command('STOP')

    # =========================================================
    # COMMAND
    # =========================================================

    def publish_command(self, command):

        msg = TwistStamped()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.header.frame_id = 'base_link'

        # -----------------------------------------
        # FORWARD
        # -----------------------------------------

        if command == 'FORWARD':

            msg.twist.linear.x = 0.20
            msg.twist.angular.z = 0.0

        # -----------------------------------------
        # LEFT
        # -----------------------------------------

        elif command == 'TURN_LEFT':

            msg.twist.linear.x = 0.00
            msg.twist.angular.z = 0.40

        # -----------------------------------------
        # RIGHT
        # -----------------------------------------

        elif command == 'TURN_RIGHT':

            msg.twist.linear.x = 0.00
            msg.twist.angular.z = -0.40

        # -----------------------------------------
        # STOP
        # -----------------------------------------

        elif command == 'STOP':

            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0

        self.cmd_publisher.publish(msg)

        self.get_logger().info(
            f'Safety: {self.safety_status} | '
            f'A*: {self.astar_direction} | '
            f'CMD: {command}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = NavigationControllerNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
