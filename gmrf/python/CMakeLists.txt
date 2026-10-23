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

# Optional pybind11 bindings for the GMRF core (GMRF_BUILD_PYTHON=ON).
#
# Builds the extension module `scalar_field_belief_gmrf_py` into this binary
# directory, plus `gmrf_py_reference`, a tiny C++ executable that runs a fixed
# deterministic case through the C++ API directly. tests/test_gmrf_bindings.py
# compares the module against it. Nothing here is installed unless the
# including project sets GMRF_PYTHON_INSTALL_DIR (the scalar_field_belief ROS
# package does, for the belief playground); the reference executable is only
# built for standalone/test builds.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 2.6 CONFIG REQUIRED)

pybind11_add_module(scalar_field_belief_gmrf_py MODULE gmrf_bindings.cpp)
target_link_libraries(scalar_field_belief_gmrf_py PRIVATE scalar_field_belief::gmrf)
set_target_properties(scalar_field_belief_gmrf_py PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(scalar_field_belief_gmrf_py PRIVATE -Wall -Wextra)
endif()

if(GMRF_PYTHON_INSTALL_DIR)
  install(TARGETS scalar_field_belief_gmrf_py
    LIBRARY DESTINATION ${GMRF_PYTHON_INSTALL_DIR})
endif()

if(GMRF_IS_TOP_LEVEL OR GMRF_BUILD_TESTS)
  add_executable(gmrf_py_reference reference_case.cpp)
  target_link_libraries(gmrf_py_reference PRIVATE scalar_field_belief::gmrf)
  set_target_properties(gmrf_py_reference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gmrf_py_reference PRIVATE -Wall -Wextra)
  endif()
endif()

# Binding test via ctest, only if pytest is importable by the build interpreter.
if(GMRF_BUILD_TESTS)
  execute_process(COMMAND ${Python_EXECUTABLE} -c "import pytest, numpy"
    RESULT_VARIABLE _gmrf_py_has_pytest OUTPUT_QUIET ERROR_QUIET)
  if(_gmrf_py_has_pytest EQUAL 0)
    add_test(NAME gmrf_python_bindings
      COMMAND ${Python_EXECUTABLE} -m pytest -q -p no:cacheprovider
              ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    set_tests_properties(gmrf_python_bindings PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
  else()
    message(STATUS "scalar_field_belief_gmrf: pytest/numpy not found for "
                   "${Python_EXECUTABLE}; Python binding test not registered")
  endif()
endif()
