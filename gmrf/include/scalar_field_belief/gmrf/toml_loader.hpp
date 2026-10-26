// Copyright (C) 2026 Nathalie Bauschmann
// SPDX-License-Identifier: GPL-3.0-only
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3 of the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// Optional TOML loaders (target scalar_field_belief::gmrf_toml, needs toml++).
#ifndef SCALAR_FIELD_BELIEF_GMRF_TOML_LOADER_HPP
#define SCALAR_FIELD_BELIEF_GMRF_TOML_LOADER_HPP

#include <string>
#include <string_view>
#include <vector>

#include "scalar_field_belief/gmrf/config.hpp"

namespace scalar_field_belief::gmrf {

/// Rectangular bounds read from an optional [bounds] table [m].
struct DomainBounds {
  double x_min = 0.0;
  double x_max = 0.0;
  double y_min = 0.0;
  double y_max = 0.0;
};

/// Content of a wall geometry file.
struct WallFile {
  std::string name;  ///< [general] name, empty if absent
  std::vector<WallSegment>
      walls;                ///< [objects.walls] wall_vertices, file order
  bool has_bounds = false;  ///< true if [bounds] was present
  DomainBounds bounds;
};

/// Loads a GmrfConfig from the source's gmrf_conf.toml layout:
///   [general]            N_X, N_Y, size_x, size_y (+ optional x_min, y_min,
///                        x_max, y_max, wall_endpoint_extension)
///   [field_params]       prior_mean, prior_sigma, sigma_zeta, sigma_p
///                        (= sigma_p_per_m), sigma_s, sigma_flow,
///                        enable_measurement_aging
///   [flow_field_params]  prior_mean_x, prior_mean_y, prior_sigma, sigma_zeta,
///                        sigma_p (= ff_sigma_p_per_m), sigma_s, sigma_c,
///                        sigma_o
///   [energies]           field_neighbors, flow_neighbors, flow_conservation,
///                        obstacle_flow, flow_field, force_joint_state
///   [solver]             max_gauss_newton_iterations, step_tolerance,
///                        relative_cost_tolerance, min_step_fraction
/// Missing keys keep the GmrfConfig defaults (stationary scalar model; NOT the
/// source's value_or fallbacks). Source keys without an effect here
/// (edge_prior_var, [source_loc]) are accepted and ignored. Unknown tables or
/// keys and wrong value types are errors. The result is validated.
/// Throws std::runtime_error (file/syntax) or std::invalid_argument.
GmrfConfig loadGmrfConfigToml(const std::string& path);
GmrfConfig parseGmrfConfigToml(std::string_view text,
                               const std::string& source_name = "<string>");

/// Loads walls from [objects.walls] wall_vertices = [[x0, y0, x1, y1], ...]
/// (source map format, scalar_field_sim geometry files and self-contained
/// scenarios). Optional: [general] name, [bounds] x_min/x_max/y_min/y_max,
/// [objects.walls] num_walls (must match). Other tables are ignored. A file
/// without walls yields an empty list.
///
/// If the file has a top-level string `geometry_file` (scalar_field_sim
/// scenario in reference form), name, bounds and walls are read from that
/// file instead, resolved relative to the directory of `path` (absolute paths
/// are used as given); all other content of `path` is ignored. Only one level
/// of reference is followed: a referenced file containing `geometry_file` or
/// `field_file` throws std::invalid_argument, as does a file with `field_file`
/// but without `geometry_file`.
/// Throws std::runtime_error (missing file, unreadable file, syntax) or
/// std::invalid_argument.
WallFile loadWallsToml(const std::string& path);
/// Same as loadWallsToml for TOML text. A `geometry_file` or `field_file` key
/// throws std::invalid_argument, since there is no file location to resolve
/// it.
WallFile parseWallsToml(std::string_view text,
                        const std::string& source_name = "<string>");

}  // namespace scalar_field_belief::gmrf

#endif  // SCALAR_FIELD_BELIEF_GMRF_TOML_LOADER_HPP
