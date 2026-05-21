from ament_index_python import get_package_share_path
from hippo_common import launch_helper
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration


def declare_launch_args(launch_description: LaunchDescription):
    pkg_path = get_package_share_path('scalar_field_belief')
    params_file = str(pkg_path / 'config/belief_params_default.yaml')

    action = DeclareLaunchArgument(
        'belief_params',
        default_value=params_file,
    )
    launch_description.add_action(action)

    launch_helper.declare_vehicle_name_and_sim_time(
        launch_description=launch_description,
        use_sim_time_default='true',
    )


def create_scalar_field_belief_node():
    return Node(
        package='scalar_field_belief',
        executable='scalar_field_belief_node.py',
        namespace=launch_helper.LaunchConfiguration('vehicle_name'),
        name='scalar_field_belief',
        parameters=[
            LaunchConfiguration('belief_params'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            },
        ],
        output='screen',
        emulate_tty=True,
    )


def generate_launch_description():
    launch_description = LaunchDescription()
    declare_launch_args(launch_description=launch_description)

    action = GroupAction(
        [
            create_scalar_field_belief_node(),
        ]
    )
    launch_description.add_action(action)

    return launch_description