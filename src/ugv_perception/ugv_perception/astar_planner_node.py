import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge

import numpy as np
import heapq


class AStarPlannerNode(Node):

    def __init__(self):
        super().__init__('astar_planner_node')

        self.bridge = CvBridge()
        self.grid = None
        self.last_direction = None

        self.subscription = self.create_subscription(
            Image,
            '/navigation/local_grid',
            self.grid_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            '/navigation/path_direction',
            10
        )

        self.timer = self.create_timer(
            0.5,
            self.plan_path
        )

        self.get_logger().info('A* planner started')
        self.get_logger().info('Subscribed: /navigation/local_grid')
        self.get_logger().info(
            'Publishing: /navigation/path_direction'
        )

    # =========================================================
    # GRID CALLBACK
    # =========================================================

    def grid_callback(self, msg):

        try:
            self.grid = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='mono8'
            )

        except Exception as e:

            self.get_logger().error(
                f'Grid conversion error: {e}'
            )

    # =========================================================
    # HEURISTIC
    # =========================================================

    def heuristic(self, a, b):

        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # =========================================================
    # A*
    # =========================================================

    def astar(self, grid, start, goal):

        rows, cols = grid.shape

        open_set = []

        heapq.heappush(
            open_set,
            (0, start)
        )

        came_from = {}

        g_score = {
            start: 0
        }

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        while open_set:

            _, current = heapq.heappop(
                open_set
            )

            if current == goal:

                path = [current]

                while current in came_from:

                    current = came_from[current]

                    path.append(current)

                path.reverse()

                return path

            for dr, dc in directions:

                nr = current[0] + dr
                nc = current[1] + dc

                if nr < 0 or nr >= rows:
                    continue

                if nc < 0 or nc >= cols:
                    continue

                # Anything brighter than 200 is considered
                # a hard obstacle.
                if grid[nr, nc] >= 200:
                    continue

                neighbor = (nr, nc)

                new_cost = (
                    g_score[current] + 1
                )

                if (
                    neighbor not in g_score
                    or new_cost < g_score[neighbor]
                ):

                    came_from[neighbor] = current

                    g_score[neighbor] = new_cost

                    f_score = (
                        new_cost
                        + self.heuristic(
                            neighbor,
                            goal
                        )
                    )

                    heapq.heappush(
                        open_set,
                        (f_score, neighbor)
                    )

        return None

    # =========================================================
    # FIND NEAREST FREE CELL
    # =========================================================

    def nearest_free(
        self,
        grid,
        row,
        col,
        max_radius=80
    ):

        rows, cols = grid.shape

        row = max(
            0,
            min(row, rows - 1)
        )

        col = max(
            0,
            min(col, cols - 1)
        )

        if grid[row, col] < 200:
            return (row, col)

        for radius in range(
            1,
            max_radius
        ):

            for dr in range(
                -radius,
                radius + 1
            ):

                for dc in range(
                    -radius,
                    radius + 1
                ):

                    r = row + dr
                    c = col + dc

                    if r < 0 or r >= rows:
                        continue

                    if c < 0 or c >= cols:
                        continue

                    if grid[r, c] < 200:
                        return (r, c)

        return None

    # =========================================================
    # PLAN
    # =========================================================

    def plan_path(self):

        if self.grid is None:
            return

        grid = self.grid.copy()

        height, width = grid.shape

        # Use lower 60% of camera/grid.
        roi_start = int(
            height * 0.40
        )

        navigation = grid[
            roi_start:height,
            :
        ]

        nav_height, nav_width = navigation.shape

        # -----------------------------------------------------
        # START = bottom center
        # -----------------------------------------------------

        start_row = nav_height - 5

        start_col = nav_width // 2

        # -----------------------------------------------------
        # GOAL = upper center of navigation area
        # -----------------------------------------------------

        goal_row = int(
            nav_height * 0.20
        )

        goal_col = nav_width // 2

        start = self.nearest_free(
            navigation,
            start_row,
            start_col
        )

        goal = self.nearest_free(
            navigation,
            goal_row,
            goal_col
        )

        if start is None:

            self.publish_direction(
                'STOP'
            )

            self.log_direction(
                'STOP',
                'No free START cell'
            )

            return

        if goal is None:

            self.publish_direction(
                'STOP'
            )

            self.log_direction(
                'STOP',
                'No free GOAL cell'
            )

            return

        # -----------------------------------------------------
        # RUN A*
        # -----------------------------------------------------

        path = self.astar(
            navigation,
            start,
            goal
        )

        if path is None:

            self.publish_direction(
                'STOP'
            )

            self.log_direction(
                'STOP',
                f'No path | start={start} goal={goal}'
            )

            return

        # -----------------------------------------------------
        # PATH FOUND
        # -----------------------------------------------------

        if len(path) < 2:

            self.publish_direction(
                'FORWARD'
            )

            return

        # Look ahead along path.
        look_ahead = min(
            20,
            len(path) - 1
        )

        target = path[
            look_ahead
        ]

        target_row = target[0]
        target_col = target[1]

        center = nav_width / 2

        # Wider dead-band prevents unnecessary turning.
        left_limit = center - 35
        right_limit = center + 35

        if target_col < left_limit:

            direction = 'TURN_LEFT'

        elif target_col > right_limit:

            direction = 'TURN_RIGHT'

        else:

            direction = 'FORWARD'

        self.publish_direction(
            direction
        )

        self.log_direction(
            direction,
            (
                f'path={len(path)} '
                f'start={start} '
                f'goal={goal} '
                f'target={target}'
            )
        )

    # =========================================================
    # PUBLISH
    # =========================================================

    def publish_direction(
        self,
        direction
    ):

        msg = String()

        msg.data = direction

        self.publisher.publish(
            msg
        )

    # =========================================================
    # LOG
    # =========================================================

    def log_direction(
        self,
        direction,
        reason
    ):

        # Avoid flooding terminal with identical messages.
        if direction == self.last_direction:
            return

        self.last_direction = direction

        if direction == 'STOP':

            self.get_logger().warning(
                f'A*: {direction} | {reason}'
            )

        else:

            self.get_logger().info(
                f'A*: {direction} | {reason}'
            )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = AStarPlannerNode()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
