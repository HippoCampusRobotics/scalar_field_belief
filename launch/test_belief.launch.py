from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("vehicle_name"),
            DeclareLaunchArgument("frame_id", default_value="map"),
            GroupAction(
                [
                    Node(
                        package="scalar_field_belief",
                        executable="scalar_field_belief_node.py",
                        namespace=LaunchConfiguration("vehicle_name"),
                        name="scalar_field_belief",
                        parameters=[
                            {
                                "frame_id": LaunchConfiguration("frame_id"),
                                "publish_visualization": True,
                                "visualization_grid_step": 0.1,
                                "refit_policy": "every_measurement",
                                "training_iter": 25,
                            }
                        ],
                        output="screen",
                        emulate_tty=True,
                    ),
                    Node(
                        package="scalar_field_belief",
                        executable="test_measurement_publisher_node.py",
                        namespace=LaunchConfiguration("vehicle_name"),
                        name="test_measurement_publisher",
                        parameters=[
                            {
                                "frame_id": LaunchConfiguration("frame_id"),
                                "period_sec": 1.0,
                            }
                        ],
                        output="screen",
                        emulate_tty=True,
                    ),
                ]
            ),
        ]
    )