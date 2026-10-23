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

cmake_minimum_required(VERSION 3.5)
project(scalar_field_belief)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(scalar_field_interfaces REQUIRED)

# include_directories(include)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Belief playground (tools/belief_playground, Qt GUI, no ROS node) and the
# GMRF Python module it uses (gmrf/python, needs pybind11-dev + python3-dev).
#   colcon build --cmake-args -DSCALAR_FIELD_BELIEF_BUILD_PLAYGROUND=ON
#   ros2 run scalar_field_belief belief_playground
option(SCALAR_FIELD_BELIEF_BUILD_PLAYGROUND
  "Build/install the belief playground and the GMRF Python module" OFF)
if(SCALAR_FIELD_BELIEF_BUILD_PLAYGROUND)
  ament_python_install_package(belief_playground
    PACKAGE_DIR tools/belief_playground/belief_playground)
  install(PROGRAMS tools/belief_playground/scripts/belief_playground
    DESTINATION lib/${PROJECT_NAME})
  # Same interpreter as ament_cmake_python (ABI of the installed module),
  # installed next to the other Python packages of this package.
  if(NOT DEFINED Python_EXECUTABLE)
    get_executable_path(_sfb_python Python3::Interpreter CONFIGURE)
    set(Python_EXECUTABLE "${_sfb_python}")
  endif()
  ament_get_python_install_dir(GMRF_PYTHON_INSTALL_DIR)
  set(GMRF_BUILD_PYTHON ON)
endif()

# Install Python executables
install(PROGRAMS
  nodes/scalar_field_belief_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Optional C++ GMRF backend (gmrf/): core library + gmrf_belief_node.
# Enable with: colcon build --cmake-args -DSCALAR_FIELD_BELIEF_BUILD_GMRF=ON
option(SCALAR_FIELD_BELIEF_BUILD_GMRF "Build the C++ GMRF belief and gmrf_belief_node" OFF)

# gmrf/ is needed for the node and for the GMRF Python module
# (GMRF_BUILD_PYTHON, set by the belief playground option).
if(SCALAR_FIELD_BELIEF_BUILD_GMRF OR GMRF_BUILD_PYTHON)
  add_subdirectory(gmrf)
  install(FILES gmrf/ros/belief_params_gmrf.yaml DESTINATION share/${PROJECT_NAME}/config)
endif()

if(SCALAR_FIELD_BELIEF_BUILD_GMRF)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(ament_index_cpp REQUIRED)
  if(NOT TARGET scalar_field_belief::gmrf_toml)
    message(FATAL_ERROR "gmrf_belief_node needs toml++ (apt: libtomlplusplus-dev).")
  endif()
  add_executable(gmrf_belief_node gmrf/ros/gmrf_belief_node.cpp)
  target_link_libraries(gmrf_belief_node PRIVATE
    scalar_field_belief::gmrf
    scalar_field_belief::gmrf_toml
    rclcpp::rclcpp
    ament_index_cpp::ament_index_cpp
    ${sensor_msgs_TARGETS}
    ${std_srvs_TARGETS}
    ${geometry_msgs_TARGETS}
    ${scalar_field_interfaces_TARGETS}
  )
  install(TARGETS gmrf_belief_node DESTINATION lib/${PROJECT_NAME})
  install(FILES gmrf/ros/gmrf_belief.launch.py DESTINATION share/${PROJECT_NAME}/launch)
endif()


ament_package()
