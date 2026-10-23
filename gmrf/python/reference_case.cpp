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

// Deterministic reference case for the Python binding test. Runs a small
// fixed scenario through the C++ API directly and prints all results as
// "key v0 v1 ..." lines with full double precision.
// tests/test_gmrf_bindings.py runs the same case through the bindings and
// compares. Keep both definitions of the case in sync.
#include <Eigen/Core>
#include <cstdio>
#include <vector>

#include "scalar_field_belief/gmrf/gmrf_belief.hpp"

namespace gm = scalar_field_belief::gmrf;

namespace {

void print(const char* key, const double* v, Eigen::Index n) {
  std::printf("%s", key);
  for (Eigen::Index i = 0; i < n; ++i) {
    std::printf(" %.17g", v[i]);
  }
  std::printf("\n");
}

void print(const char* key, const Eigen::VectorXd& v) {
  print(key, v.data(), v.size());
}

void print(const char* key, const Eigen::MatrixXd& m) {
  // Row-major flattening.
  const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      r = m;
  print(key, r.data(), r.size());
}

}  // namespace

int main() {
  gm::GmrfConfig cfg;
  cfg.x_min = -1.0;
  cfg.x_max = 1.5;
  cfg.y_min = 0.5;
  cfg.y_max = 2.0;
  cfg.nx = 11;
  cfg.ny = 7;
  cfg.prior_mean = 0.1;
  cfg.prior_sigma = 0.5;
  cfg.sigma_p_per_m = 2.0;
  cfg.sigma_s = 0.05;
  const std::vector<gm::WallSegment> walls = {{0.2, 0.4, 0.2, 1.6}};

  gm::GmrfBelief belief(cfg, walls);

  Eigen::MatrixX2d q(5, 2);
  q << -0.9, 0.6,  //
      -0.6, 1.0,   //
      0.1, 1.2,    //
      0.35, 1.2,   //
      1.45, 1.95;

  Eigen::VectorXd mean, var;
  Eigen::MatrixXd cov;
  belief.queryMeanVariance(q, mean, var);
  print("prior_mean", mean);
  print("prior_var", var);

  belief.addMeasurement(-0.6, 1.0, 0.3);
  belief.addMeasurement(0.7, 1.2, -0.2);
  belief.addMeasurement(0.0, 1.8, 0.5);
  const gm::UpdateReport rep = belief.update();
  std::printf("num_assimilated %zu\n", rep.num_assimilated);
  std::printf("num_measurements %zu\n", belief.numMeasurements());

  belief.queryMeanVariance(q, mean, var);
  print("post_mean", mean);
  print("post_var", var);
  belief.queryMeanCovariance(q, mean, cov);
  print("post_cov_mean", mean);
  print("post_cov", cov);

  std::printf("noise_var %.17g\n", belief.measurementNoiseVariance());
  std::printf("grid %d %d %.17g %.17g\n", belief.nx(), belief.ny(), belief.dx(),
              belief.dy());

  belief.reset();
  belief.queryMeanVariance(q, mean, var);
  print("reset_mean", mean);
  print("reset_var", var);
  return 0;
}
