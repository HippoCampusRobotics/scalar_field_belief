// Copyright (C) 2023 Finn Lukas Busch
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
#ifndef SCALAR_FIELD_BELIEF_GMRF_CONFIG_HPP
#define SCALAR_FIELD_BELIEF_GMRF_CONFIG_HPP

namespace scalar_field_belief::gmrf {

/// A physical 2D point [m].
struct Point2 {
  double x = 0.0;
  double y = 0.0;
};

/// A straight wall segment between (x0, y0) and (x1, y1) [m].
/// Endpoint order does not matter. Zero-length walls are rejected.
struct WallSegment {
  double x0 = 0.0;
  double y0 = 0.0;
  double x1 = 0.0;
  double y1 = 0.0;
};

/// Flow-field state component (joint mode only).
enum class FlowComponent { X, Y };

/// Configuration of the GMRF belief.
///
/// All `sigma*` values are STANDARD DEVIATIONS; each enters the model squared
/// exactly once (information = 1 / sigma^2).
///
/// Grid: nodes at (x_min + i*dx, y_min + j*dy), i = 0..nx-1, j = 0..ny-1,
/// dx = (x_max - x_min)/(nx - 1), dy = (y_max - y_min)/(ny - 1). Domain
/// endpoints are nodes. Node index n = j*nx + i (x index fastest).
///
/// The default configuration is the stationary scalar model: field-neighbour
/// coupling on, all flow energies off, no measurement aging.
struct GmrfConfig {
  // --- Domain and grid -----------------------------------------------------
  double x_min = 0.0;   ///< [m]
  double x_max = 10.0;  ///< [m], > x_min
  double y_min = 0.0;   ///< [m]
  double y_max = 10.0;  ///< [m], > y_min
  int nx = 10;          ///< number of nodes in x, >= 2
  int ny = 10;          ///< number of nodes in y, >= 2

  // --- Scalar field ---------------------------------------------------------
  /// Prior mean of every node [measurement units].
  double prior_mean = 0.0;
  /// Std of the independent per-node prior factor [measurement units].
  double prior_sigma = 100.0;
  /// Field-neighbour std coefficient. The std of the difference between two
  /// neighbouring nodes is sigma_p = sigma_p_per_m * dx^2 (source formula;
  /// the x spacing dx is used for horizontal AND vertical edges).
  /// Units: [measurement units / m^2] despite the source name.
  double sigma_p_per_m = 1.0;
  /// Measurement noise std [measurement units].
  double sigma_s = 1.0;
  /// Measurement aging: noise variance grows by sigma_zeta^2 per unit of
  /// caller-provided time. Only used if enable_measurement_aging is true.
  double sigma_zeta = 0.0;
  bool enable_measurement_aging = false;
  /// Std of the flow/field coupling residual (flow_field energy).
  double sigma_flow = 1.0;

  // --- Flow field (joint mode only) ----------------------------------------
  double ff_prior_mean_x = 0.0;
  double ff_prior_mean_y = 0.0;
  double ff_prior_sigma = 100.0;
  double ff_sigma_zeta = 0.0;
  /// Flow-neighbour std = ff_sigma_p_per_m * dx^2 (same formula as the field).
  double ff_sigma_p_per_m = 1.0;
  double ff_sigma_s = 1.0;  ///< flow measurement noise std
  double ff_sigma_c = 1.0;  ///< flow conservation std
  double ff_sigma_o = 1.0;  ///< obstacle/flow std

  // --- Energy terms ---------------------------------------------------------
  bool field_neighbors = true;
  bool flow_neighbors = false;
  bool flow_conservation = false;
  bool flow_field = false;  ///< nonlinear (Gauss-Newton) when enabled
  bool obstacle_flow = false;
  /// Keep the full joint state [c, w_x, w_y] even if all flow energies are
  /// disabled (used to demonstrate equivalence with the scalar-only state).
  bool force_joint_state = false;

  // --- Walls ----------------------------------------------------------------
  /// Each wall is extended by this length [m] beyond both endpoints along its
  /// own direction before it is intersected with the grid (source tolerance).
  double wall_endpoint_extension = 0.01;

  // --- Gauss-Newton (only used when flow_field is enabled) -------------------
  int max_gauss_newton_iterations = 50;
  /// Converged when ||step||_2 < gn_step_tolerance (absolute, state units).
  double gn_step_tolerance = 1e-5;
  /// Converged when the relative cost decrease of an accepted step is below
  /// this.
  double gn_relative_cost_tolerance = 1e-5;
  /// Backtracking halves the step until this fraction; then the update stops.
  double gn_min_step_fraction = 1e-4;

  /// Throws std::invalid_argument with a descriptive message if invalid.
  void validate() const;

  /// True if any flow energy is enabled or force_joint_state is set.
  bool usesJointState() const;

  /// True unless the (bilinear) flow_field energy is enabled.
  bool isLinear() const { return !flow_field; }
};

}  // namespace scalar_field_belief::gmrf

#endif  // SCALAR_FIELD_BELIEF_GMRF_CONFIG_HPP
