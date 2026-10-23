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

#include "scalar_field_belief/gmrf/config.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace scalar_field_belief::gmrf {
namespace {

void requireFinite(double v, const char* name) {
  if (!std::isfinite(v)) {
    throw std::invalid_argument(std::string("GmrfConfig: ") + name +
                                " must be finite.");
  }
}

void requirePositive(double v, const char* name) {
  requireFinite(v, name);
  if (!(v > 0.0)) {
    throw std::invalid_argument(std::string("GmrfConfig: ") + name +
                                " must be > 0 (got " + std::to_string(v) +
                                ").");
  }
}

void requireNonNegative(double v, const char* name) {
  requireFinite(v, name);
  if (v < 0.0) {
    throw std::invalid_argument(std::string("GmrfConfig: ") + name +
                                " must be >= 0 (got " + std::to_string(v) +
                                ").");
  }
}

}  // namespace

bool GmrfConfig::usesJointState() const {
  return force_joint_state || flow_neighbors || flow_conservation ||
         flow_field || obstacle_flow;
}

void GmrfConfig::validate() const {
  requireFinite(x_min, "x_min");
  requireFinite(x_max, "x_max");
  requireFinite(y_min, "y_min");
  requireFinite(y_max, "y_max");
  if (!(x_max > x_min)) {
    throw std::invalid_argument(
        "GmrfConfig: x_max must be greater than x_min.");
  }
  if (!(y_max > y_min)) {
    throw std::invalid_argument(
        "GmrfConfig: y_max must be greater than y_min.");
  }
  if (!std::isfinite(x_max - x_min) || !std::isfinite(y_max - y_min)) {
    throw std::invalid_argument("GmrfConfig: domain extent must be finite.");
  }
  if (nx < 2 || ny < 2) {
    throw std::invalid_argument("GmrfConfig: nx and ny must be >= 2.");
  }
  // Eigen sparse matrices use int indices; the joint state has 3*nx*ny entries.
  const long long state = 3LL * nx * ny;
  if (state > static_cast<long long>(std::numeric_limits<int>::max() / 8)) {
    throw std::invalid_argument("GmrfConfig: nx * ny is too large.");
  }

  requireFinite(prior_mean, "prior_mean");
  requirePositive(prior_sigma, "prior_sigma");
  requirePositive(sigma_s, "sigma_s");
  requireNonNegative(sigma_zeta, "sigma_zeta");
  if (field_neighbors) {
    requirePositive(sigma_p_per_m, "sigma_p_per_m");
  } else {
    requireFinite(sigma_p_per_m, "sigma_p_per_m");
  }
  if (flow_field) {
    requirePositive(sigma_flow, "sigma_flow");
  } else {
    requireFinite(sigma_flow, "sigma_flow");
  }

  requireFinite(ff_prior_mean_x, "ff_prior_mean_x");
  requireFinite(ff_prior_mean_y, "ff_prior_mean_y");
  requireNonNegative(ff_sigma_zeta, "ff_sigma_zeta");
  if (usesJointState()) {
    requirePositive(ff_prior_sigma, "ff_prior_sigma");
    requirePositive(ff_sigma_s, "ff_sigma_s");
  } else {
    requireFinite(ff_prior_sigma, "ff_prior_sigma");
    requireFinite(ff_sigma_s, "ff_sigma_s");
  }
  if (flow_neighbors) {
    requirePositive(ff_sigma_p_per_m, "ff_sigma_p_per_m");
  } else {
    requireFinite(ff_sigma_p_per_m, "ff_sigma_p_per_m");
  }
  if (flow_conservation) {
    requirePositive(ff_sigma_c, "ff_sigma_c");
  } else {
    requireFinite(ff_sigma_c, "ff_sigma_c");
  }
  if (obstacle_flow) {
    requirePositive(ff_sigma_o, "ff_sigma_o");
  } else {
    requireFinite(ff_sigma_o, "ff_sigma_o");
  }

  requireNonNegative(wall_endpoint_extension, "wall_endpoint_extension");

  if (max_gauss_newton_iterations < 1) {
    throw std::invalid_argument(
        "GmrfConfig: max_gauss_newton_iterations must be >= 1.");
  }
  requireNonNegative(gn_step_tolerance, "gn_step_tolerance");
  requireNonNegative(gn_relative_cost_tolerance, "gn_relative_cost_tolerance");
  requirePositive(gn_min_step_fraction, "gn_min_step_fraction");
  if (gn_min_step_fraction > 1.0) {
    throw std::invalid_argument(
        "GmrfConfig: gn_min_step_fraction must be <= 1.");
  }

  // Derived standard deviations must stay positive and finite after squaring.
  const double dx = (x_max - x_min) / (nx - 1);
  if (field_neighbors) {
    const double sp = sigma_p_per_m * dx * dx;
    if (!(sp > 0.0) || !std::isfinite(1.0 / (sp * sp))) {
      throw std::invalid_argument(
          "GmrfConfig: sigma_p_per_m * dx^2 underflows/overflows; adjust "
          "sigma_p_per_m.");
    }
  }
  if (flow_neighbors) {
    const double sp = ff_sigma_p_per_m * dx * dx;
    if (!(sp > 0.0) || !std::isfinite(1.0 / (sp * sp))) {
      throw std::invalid_argument(
          "GmrfConfig: ff_sigma_p_per_m * dx^2 underflows/overflows; adjust "
          "ff_sigma_p_per_m.");
    }
  }
}

}  // namespace scalar_field_belief::gmrf
