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

# GMRF scalar-field belief core (ROS-independent).
#
# Standalone:   cmake -S gmrf -B <build> && cmake --build <build> && ctest --test-dir <build>
# Embedded:     add_subdirectory(gmrf) from the scalar_field_belief ROS package
#               (tests, examples and install rules are off by default then).
cmake_minimum_required(VERSION 3.18)
project(scalar_field_belief_gmrf VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(GMRF_IS_TOP_LEVEL ON)
else()
  set(GMRF_IS_TOP_LEVEL OFF)
endif()

option(GMRF_WITH_TOML "Build the optional TOML config/wall loader (needs toml++)" ON)
option(GMRF_BUILD_TESTS "Build the GTest unit tests" ${GMRF_IS_TOP_LEVEL})
option(GMRF_BUILD_EXAMPLES "Build the standalone examples" ${GMRF_IS_TOP_LEVEL})
option(GMRF_INSTALL "Generate install rules and a CMake package config" ${GMRF_IS_TOP_LEVEL})
option(GMRF_BUILD_PYTHON "Build the optional pybind11 module (python/)" OFF)

if(GMRF_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# --- Core library: C++ standard library + Eigen only --------------------------
add_library(scalar_field_belief_gmrf STATIC
  src/config.cpp
  src/gmrf_belief.cpp
)
add_library(scalar_field_belief::gmrf ALIAS scalar_field_belief_gmrf)
set_target_properties(scalar_field_belief_gmrf PROPERTIES
  EXPORT_NAME gmrf
  POSITION_INDEPENDENT_CODE ON
)
target_compile_features(scalar_field_belief_gmrf PUBLIC cxx_std_17)
target_include_directories(scalar_field_belief_gmrf
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(scalar_field_belief_gmrf PUBLIC Eigen3::Eigen)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(scalar_field_belief_gmrf PRIVATE -Wall -Wextra -Wpedantic)
  # Sparse Eigen code is very slow unoptimised. When the including project
  # sets no build type (e.g. plain colcon), optimise this target only.
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    target_compile_options(scalar_field_belief_gmrf PRIVATE -O2)
  endif()
endif()

set(GMRF_HAS_TOML OFF)
# --- Optional TOML loader -----------------------------------------------------
if(GMRF_WITH_TOML)
  find_package(tomlplusplus CONFIG QUIET)
  if(tomlplusplus_FOUND)
    add_library(scalar_field_belief_gmrf_toml STATIC src/toml_loader.cpp)
    add_library(scalar_field_belief::gmrf_toml ALIAS scalar_field_belief_gmrf_toml)
    set_target_properties(scalar_field_belief_gmrf_toml PROPERTIES
      EXPORT_NAME gmrf_toml
      POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(scalar_field_belief_gmrf_toml
      PUBLIC scalar_field_belief_gmrf
      PRIVATE tomlplusplus::tomlplusplus
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(scalar_field_belief_gmrf_toml PRIVATE -Wall -Wextra)
    endif()
    set(GMRF_HAS_TOML ON)
  else()
    message(STATUS "scalar_field_belief_gmrf: toml++ not found, TOML loader disabled")
  endif()
endif()

# --- Examples and tests (top-level builds only by default) --------------------
if(GMRF_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(GMRF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(GMRF_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# --- Install + CMake package ---------------------------------------------------
if(GMRF_INSTALL)
  include(CMakePackageConfigHelpers)
  set(GMRF_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/scalar_field_belief_gmrf)

  set(_gmrf_targets scalar_field_belief_gmrf)
  if(GMRF_HAS_TOML)
    list(APPEND _gmrf_targets scalar_field_belief_gmrf_toml)
  endif()
  install(TARGETS ${_gmrf_targets}
    EXPORT scalar_field_belief_gmrfTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(FILES
    include/scalar_field_belief/gmrf/config.hpp
    include/scalar_field_belief/gmrf/gmrf_belief.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/scalar_field_belief/gmrf
  )
  if(GMRF_HAS_TOML)
    install(FILES include/scalar_field_belief/gmrf/toml_loader.hpp
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/scalar_field_belief/gmrf)
  endif()
  install(EXPORT scalar_field_belief_gmrfTargets
    NAMESPACE scalar_field_belief::
    DESTINATION ${GMRF_CMAKE_DIR}
  )
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scalar_field_belief_gmrfConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/scalar_field_belief_gmrfConfig.cmake
    INSTALL_DESTINATION ${GMRF_CMAKE_DIR}
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/scalar_field_belief_gmrfConfigVersion.cmake
    COMPATIBILITY SameMinorVersion
  )
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/scalar_field_belief_gmrfConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/scalar_field_belief_gmrfConfigVersion.cmake
    DESTINATION ${GMRF_CMAKE_DIR}
  )
endif()
