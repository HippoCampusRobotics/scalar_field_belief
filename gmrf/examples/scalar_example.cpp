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

// Minimal standalone use of the GMRF scalar-field belief (no ROS).
#include <Eigen/Core>
#include <cstdio>
#include <exception>

#include "scalar_field_belief/gmrf/gmrf_belief.hpp"

namespace gm = scalar_field_belief::gmrf;

int main() {
  try {
    gm::GmrfConfig cfg;  // stationary scalar model by default
    cfg.x_min = 0.0;
    cfg.x_max = 2.0;
    cfg.y_min = 0.0;
    cfg.y_max = 4.0;
    cfg.nx = 41;  // 0.05 m spacing
    cfg.ny = 81;
    cfg.prior_mean = 0.0;
    cfg.prior_sigma = 0.01;   // std [units]
    cfg.sigma_p_per_m = 0.8;  // neighbour std = 0.8 * 0.05^2 = 0.002 [units]
    cfg.sigma_s = 0.0005;     // measurement noise std [units]

    // A wall at x = 1.0 from y = 0.6 to 3.4 (same as
    // config/walls/simple_wall_case.toml).
    gm::GmrfBelief belief(cfg, {gm::WallSegment{1.0, 0.6, 1.0, 3.4}});

    belief.addMeasurement(0.8, 2.0, 0.010);
    belief.addMeasurement(0.9, 2.1, 0.012);
    const gm::UpdateReport rep = belief.update();
    std::printf("update: linear=%d iterations=%d assimilated=%zu\n", rep.linear,
                rep.iterations, rep.num_assimilated);

    Eigen::MatrixX2d q(4, 2);
    q << 0.85, 2.05,  // next to the measurements (west of the wall)
        1.15, 2.05,   // just east of the wall
        1.0, 0.2,     // below the wall end
        2.0, 4.0;     // domain corner
    Eigen::VectorXd mean;
    Eigen::VectorXd var;
    belief.queryMeanVariance(q, mean, var);
    Eigen::MatrixXd cov;
    belief.queryMeanCovariance(q, mean, cov);
    for (Eigen::Index k = 0; k < q.rows(); ++k) {
      std::printf("(%.2f, %.2f): mean=% .6f  var=%.3e  cov[0,k]=% .3e\n",
                  q(k, 0), q(k, 1), mean(k), var(k), cov(0, k));
    }
    std::printf("measurement noise variance: %.3e\n",
                belief.measurementNoiseVariance());
    belief.reset();
    std::printf("after reset: mean at (0.85, 2.05) = %.6f\n",
                belief.queryMean(q.topRows(1))(0));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
