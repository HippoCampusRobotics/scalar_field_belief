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

"""Launch only the GMRF belief node (gmrf_belief_node).

Arguments:
  belief_params  parameter YAML (default: installed config/belief_params_gmrf.yaml)
  namespace      node namespace (default: ''); use different namespaces or
                 remappings to run the GP and GMRF beliefs side by side.
  use_sim_time   'true' / 'false' (default: 'false')
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('scalar_field_belief'),
        'config',
        'belief_params_gmrf.yaml',
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'belief_params',
                default_value=default_params,
                description='GMRF belief parameter file.',
            ),
            DeclareLaunchArgument(
                'namespace',
                default_value='',
                description='Namespace of the GMRF belief node.',
            ),
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description='Use simulation time.',
            ),
            Node(
                package='scalar_field_belief',
                executable='gmrf_belief_node',
                name='gmrf_belief',
                namespace=LaunchConfiguration('namespace'),
                parameters=[
                    LaunchConfiguration('belief_params'),
                    {'use_sim_time': LaunchConfiguration('use_sim_time')},
                ],
                output='screen',
                emulate_tty=True,
            ),
        ]
    )
