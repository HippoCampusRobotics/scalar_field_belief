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

// Small shared helpers for the GMRF unit tests.
#ifndef SCALAR_FIELD_BELIEF_GMRF_TESTS_UTILS_HPP
#define SCALAR_FIELD_BELIEF_GMRF_TESTS_UTILS_HPP

#include <Eigen/Core>
#include <initializer_list>
#include <utility>

#include "scalar_field_belief/gmrf/gmrf_belief.hpp"

namespace gmrf_test {

namespace gm = scalar_field_belief::gmrf;

/// Small stationary scalar configuration on [0, 2] x [0, 4] (0.25 m spacing).
inline gm::GmrfConfig smallConfig() {
  gm::GmrfConfig c;
  c.x_min = 0.0;
  c.x_max = 2.0;
  c.y_min = 0.0;
  c.y_max = 4.0;
  c.nx = 9;
  c.ny = 17;
  c.prior_mean = 0.5;
  c.prior_sigma = 2.0;
  c.sigma_p_per_m = 4.0;  // neighbour std = 4 * 0.25^2 = 0.25
  c.sigma_s = 0.1;
  return c;
}

inline Eigen::MatrixX2d points(
    std::initializer_list<std::pair<double, double>> pts) {
  Eigen::MatrixX2d xy(static_cast<Eigen::Index>(pts.size()), 2);
  Eigen::Index r = 0;
  for (const auto& p : pts) {
    xy(r, 0) = p.first;
    xy(r, 1) = p.second;
    ++r;
  }
  return xy;
}

/// All grid node positions (row-major over j, then i).
inline Eigen::MatrixX2d nodePoints(const gm::GmrfBelief& b) {
  Eigen::MatrixX2d xy(b.nodeCount(), 2);
  for (int j = 0; j < b.ny(); ++j) {
    for (int i = 0; i < b.nx(); ++i) {
      const auto p = b.nodePosition(i, j);
      xy(b.nodeIndex(i, j), 0) = p.x;
      xy(b.nodeIndex(i, j), 1) = p.y;
    }
  }
  return xy;
}

}  // namespace gmrf_test

#endif  // SCALAR_FIELD_BELIEF_GMRF_TESTS_UTILS_HPP
