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

# One GTest executable per tests/test_*.cpp file. Add a file and re-run CMake.
# Files named test_toml_*.cpp additionally link the TOML loader and are
# skipped when it is not available. Test data lives in tests/data/
# (available to the code as GMRF_TEST_DATA_DIR).
find_package(GTest REQUIRED)
include(GoogleTest)

file(GLOB GMRF_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
list(SORT GMRF_TEST_SOURCES)

foreach(test_src IN LISTS GMRF_TEST_SOURCES)
  get_filename_component(test_name ${test_src} NAME_WE)
  if(test_name MATCHES "^test_toml_" AND NOT TARGET scalar_field_belief::gmrf_toml)
    message(STATUS "Skipping ${test_name} (TOML loader not built)")
    continue()
  endif()
  set(test_target gmrf_${test_name})
  add_executable(${test_target} ${test_src})
  target_link_libraries(${test_target} PRIVATE scalar_field_belief::gmrf GTest::gtest_main)
  if(test_name MATCHES "^test_toml_")
    target_link_libraries(${test_target} PRIVATE scalar_field_belief::gmrf_toml)
  endif()
  target_include_directories(${test_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${test_target} PRIVATE
    GMRF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  gtest_discover_tests(${test_target} DISCOVERY_MODE PRE_TEST)
endforeach()
