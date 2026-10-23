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

# Opt-in source-parity harness (see README.md in this folder).
#
#   cmake -S gmrf/tests/source_parity -B <scratch>/parity \
#         -DGMRF_BUILD_SOURCE_PARITY=ON \
#         -DGMRF_REFERENCE_DIR=<repo>/reference/UnderwaterMultiagentExploration/gmrf
#   cmake --build <scratch>/parity -j && ctest --test-dir <scratch>/parity -V
#
# This is a separate CMake project; it is NOT part of the normal gmrf build or
# of colcon. The reference tree is only read; all generated files go to the
# build directory.
cmake_minimum_required(VERSION 3.18)
project(gmrf_source_parity LANGUAGES CXX)

option(GMRF_BUILD_SOURCE_PARITY "Build the source-parity harness against the original GMRF" OFF)
set(GMRF_REFERENCE_DIR "" CACHE PATH
  "Path to the original UnderwaterMultiagentExploration/gmrf directory (read-only)")

if(NOT GMRF_BUILD_SOURCE_PARITY)
  message(STATUS "gmrf_source_parity: disabled (configure with -DGMRF_BUILD_SOURCE_PARITY=ON)")
  return()
endif()

get_filename_component(_ref "${GMRF_REFERENCE_DIR}" ABSOLUTE)
if(NOT EXISTS "${_ref}/src/gmrf/gmrf.cpp")
  message(FATAL_ERROR "GMRF_REFERENCE_DIR='${GMRF_REFERENCE_DIR}' does not contain src/gmrf/gmrf.cpp")
endif()
string(FIND "${CMAKE_CURRENT_BINARY_DIR}/" "${_ref}/" _inside)
if(_inside EQUAL 0)
  message(FATAL_ERROR "Do not build inside the reference tree; use a scratch build directory.")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(SFML 2.6 COMPONENTS graphics REQUIRED)
find_package(tomlplusplus CONFIG REQUIRED)

# res_loc.h is generated into the build tree (the reference's own CMake writes
# it into the source tree, which we must not do) and put first on the path.
set(RES_LOC "${CMAKE_CURRENT_BINARY_DIR}/resources_build")
file(MAKE_DIRECTORY "${RES_LOC}/configs")
configure_file("${_ref}/include/res_loc.h.in" "${CMAKE_CURRENT_BINARY_DIR}/generated/res_loc.h" @ONLY)
set(GMRF_PARITY_TOML "${RES_LOC}/configs/gmrf_conf.toml")
file(WRITE "${GMRF_PARITY_TOML}"
  "# Minimal config for the parity harness; all values are overridden in code.\n"
  "[general]\nsize_x = 2.0\nsize_y = 2.0\nN_X = 3\nN_Y = 3\n")

# Original sources, unmodified (warnings silenced; asserts kept on).
add_library(student_gmrf STATIC
  "${_ref}/src/gmrf/gmrf.cpp"
  "${_ref}/src/viz.cpp"
  "${_ref}/src/color_mixer.cpp"
  "${_ref}/src/configs.cpp"
  "${_ref}/src/scenario.cpp"
)
target_include_directories(student_gmrf BEFORE PUBLIC
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
  "${_ref}/include"
)
target_compile_features(student_gmrf PUBLIC cxx_std_20)
target_compile_options(student_gmrf PRIVATE -w -UNDEBUG)
target_link_libraries(student_gmrf PUBLIC Eigen3::Eigen sfml-graphics tomlplusplus::tomlplusplus)
# Shim: configs.cpp uses std::ofstream without #include <fstream> (the
# FetchContent toml++ v3.3.0 it was written against pulled it in
# transitively; the system toml++ 3.4 does not). Force-include it for that TU.
set_source_files_properties("${_ref}/src/configs.cpp" PROPERTIES COMPILE_OPTIONS "-include;fstream")

# The port, built from this checkout (tests/examples/install off when embedded).
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." gmrf_port)

add_executable(gmrf_source_parity parity_main.cpp student_driver.cpp)
target_link_libraries(gmrf_source_parity PRIVATE student_gmrf scalar_field_belief::gmrf)
target_compile_definitions(gmrf_source_parity PRIVATE GMRF_PARITY_TOML="${GMRF_PARITY_TOML}")
set_source_files_properties(student_driver.cpp PROPERTIES COMPILE_OPTIONS "-UNDEBUG")

enable_testing()
add_test(NAME gmrf_source_parity COMMAND gmrf_source_parity)
