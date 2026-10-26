# Copyright (C) 2026 Nathalie Bauschmann
# SPDX-License-Identifier: GPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from ament_index_python import get_package_share_path
from hippo_common import launch_helper
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def declare_launch_args(launch_description: LaunchDescription):
    pkg_path = get_package_share_path('scalar_field_belief')
    params_file = str(pkg_path / 'config/belief_params_default.yaml')

    action = DeclareLaunchArgument(
        'belief_params',
        default_value=params_file,
    )
    launch_description.add_action(action)

    action = DeclareLaunchArgument(
        'scenario_path',
        default_value='',
        description='scalar_field_sim scenario whose geometry is used with '
        'belief_geometry: scenario; absolute or relative to the installed '
        'scalar_field_sim share directory. Overrides scenario_path in '
        'belief_params.',
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
                'scenario_path': LaunchConfiguration('scenario_path'),
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
