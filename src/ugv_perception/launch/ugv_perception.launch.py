from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():

    camera_node = Node(
        package='ugv_perception',
        executable='camera_node',
        name='camera_node',
        output='screen'
    )

    yolo_node = ExecuteProcess(
        cmd=[
            '/home/tejasvpatel/ugv_ai/ros_venv/bin/python',
            '/home/tejasvpatel/ugv_ws/src/ugv_perception/ugv_perception/yolo_node.py'
        ],
        output='screen'
    )

    yolop_node = ExecuteProcess(
        cmd=[
            '/home/tejasvpatel/ugv_ai/ros_venv/bin/python',
            '/home/tejasvpatel/ugv_ws/src/ugv_perception/ugv_perception/yolop_node.py'
        ],
        output='screen'
    )

    segmentation_node = ExecuteProcess(
        cmd=[
            '/home/tejasvpatel/ugv_ai/ros_venv/bin/python',
            '/home/tejasvpatel/ugv_ws/src/ugv_perception/ugv_perception/segmentation_node.py'
        ],
        output='screen'
    )

    drivable_mask_node = Node(
        package='ugv_perception',
        executable='drivable_mask_node',
        name='drivable_mask_node',
        output='screen'
    )

    obstacle_safety_node = Node(
        package='ugv_perception',
        executable='obstacle_safety_node',
        name='obstacle_safety_node',
        output='screen'
    )

    navigation_grid_node = Node(
        package='ugv_perception',
        executable='navigation_grid_node',
        name='navigation_grid_node',
        output='screen'
    )

    navigation_controller_node = Node(
        package='ugv_perception',
        executable='navigation_controller_node',
        name='navigation_controller_node',
        output='screen'
    )

    astar_node = ExecuteProcess(
        cmd=[
            '/usr/bin/python3',
            '/home/tejasvpatel/ugv_ws/src/ugv_perception/ugv_perception/astar_planner_node.py'
        ],
        output='screen'
    )

    return LaunchDescription([
        camera_node,
        yolo_node,
        segmentation_node,
        yolop_node,
        drivable_mask_node,
        obstacle_safety_node,
        navigation_grid_node,
        astar_node,
        navigation_controller_node
    ])
