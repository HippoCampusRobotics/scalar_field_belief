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

// Drives the student's unmodified GMRF class. Only public API is used:
// GmrfParams(path) + direct field overrides, a tiny Scenario subclass that
// supplies walls, GMRF::addMeasurement(..., field), GMRF::updateEstimates,
// GMRF::getEstimates() and GMRF::getCovariances().
#include "student_driver.hpp"

#include <stdexcept>

#include "gmrf/gmrf.h"
#include "scenario.h"

namespace {

/// Minimal scenario: only walls (no sources, no ground truth field).
class WallScenario : public Scenario {
 public:
  explicit WallScenario(const std::vector<std::array<double, 4>>& walls) {
    for (const auto& w : walls) {
      Eigen::Matrix<double, 1, 4> v;
      v << w[0], w[1], w[2], w[3];
      objects_.addObject(wallType::wall, v);
    }
  }
  void getFieldValue(double, double, double& v, double) override { v = 0.0; }
  double getFieldValue(double, double, double) override { return 0.0; }
  void getFlowValue(double, double, double& fx, double& fy, double) override {
    fx = 0.0;
    fy = 0.0;
  }
  Objects* getObjects() override { return &objects_; }

 private:
  Objects objects_;
};

}  // namespace

namespace parity {

NodalResult runStudent(const Case& c, const char* toml_path) {
  GmrfParams p(
      toml_path);  // file sets nothing essential; all fields overridden below
  p.N_X = c.nx;
  p.N_Y = c.ny;
  p.size_x = c.size_x;
  p.size_y = c.size_y;
  p.prior_mean = c.prior_mean;
  p.prior_sigma = c.prior_sigma;
  p.sigma_p_per_m = c.sigma_p_per_m;
  p.sigma_zeta = 0.0;
  p.sigma_s = c.sigma_s;
  p.sigma_flow = 1.0;
  p.ff_prior_mean_x = 0.0;
  p.ff_prior_mean_y = 0.0;
  p.ff_prior_sigma = 1.0;
  p.ff_sigma_zeta = 0.0;
  p.ff_sigma_p_per_m = 1.0;
  p.ff_sigma_s = 1.0;
  p.ff_sigma_c = 1.0;
  p.ff_sigma_o = 1.0;
  p.field_neighbors = true;
  p.flow_neighbors = c.flow_neighbors;
  p.flow_conservation = false;
  p.obstacle_flow = false;
  p.flow_field = false;
  p.all_neighbors = false;

  WallScenario scenario(c.walls);
  NodalResult out;
  for (std::size_t k = 0;
       k + 1 < scenario.getObjects()->objects[wallType::wall].size(); k += 2) {
    const auto& vs = scenario.getObjects()->objects[wallType::wall];
    out.walls_as_stored.push_back({vs[k].position.x, vs[k].position.y,
                                   vs[k + 1].position.x, vs[k + 1].position.y});
  }

  GMRF g(p, &scenario);
  for (const auto& batch : c.batches) {
    for (const auto& m : batch) {
      g.addMeasurement(m[0], m[1], m[2], 0.0, field);
    }
    g.updateEstimates(0.0);
  }
  const auto est = g.getEstimates();     // (i, j) map over node order j*N_X + i
  const auto& cov = g.getCovariances();  // (i, j) nodal variances
  out.mean.resize(static_cast<std::size_t>(c.nx * c.ny));
  out.var.resize(out.mean.size());
  for (int j = 0; j < c.ny; ++j) {
    for (int i = 0; i < c.nx; ++i) {
      out.mean[static_cast<std::size_t>(j * c.nx + i)] = est(i, j);
      out.var[static_cast<std::size_t>(j * c.nx + i)] = cov(i, j);
    }
  }
  return out;
}

}  // namespace parity
