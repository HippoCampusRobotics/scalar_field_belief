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

// Plain-C++ interface to the student's original GMRF (compiled unmodified
// from the reference tree). Kept free of student and port headers so the two
// code bases live in separate translation units (the student headers put
// `using namespace Eigen` and unscoped names like `field` in the global
// namespace).
#ifndef GMRF_SOURCE_PARITY_STUDENT_DRIVER_HPP
#define GMRF_SOURCE_PARITY_STUDENT_DRIVER_HPP

#include <array>
#include <vector>

namespace parity {

struct Case {
  double size_x = 1.0,
         size_y = 1.0;  // student domain is [0, size_x] x [0, size_y]
  int nx = 2, ny = 2;
  double prior_mean = 0.0, prior_sigma = 1.0, sigma_p_per_m = 1.0,
         sigma_s = 1.0;
  bool flow_neighbors =
      false;  // only to keep the student's wall row offsets in range
  std::vector<std::array<double, 4>> walls;  // x0, y0, x1, y1
  std::vector<std::vector<std::array<double, 3>>>
      batches;  // per update: (x, y, v)
};

struct NodalResult {
  std::vector<double> mean;  // node order n = j*nx + i
  std::vector<double>
      var;  // student's covariances[field] (nodal marginal variance)
  std::vector<std::array<double, 4>>
      walls_as_stored;  // after sf::Vertex float storage
};

NodalResult runStudent(const Case& c, const char* toml_path);

}  // namespace parity

#endif
