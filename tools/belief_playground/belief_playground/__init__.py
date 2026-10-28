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

"""Interactive GP/GMRF scalar-field belief playground (no ROS) and live
belief viewer.

Modules: scenario (scalar_field_sim access), backends (adapters),
session (observation log, worker, save/load), gui (PySide6 + PyQtGraph),
launcher (scenario/config lookup, installed entry point), __main__ (command
line), visgp_diagnostics (headless visGP checks). Live viewer: live_data
(shared state, display resampling), live_ros (rclpy subscriber node),
live_gui (Qt window), live_main (command line).
"""
