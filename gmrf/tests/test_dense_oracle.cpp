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

// Independent dense-reference checks of the scalar GMRF (no walls):
// prior/posterior mean and FULL query covariance against Q assembled from the
// model equations (see dense_oracle.hpp), on a tiny non-square grid with a
// negative origin; covariance properties; monotone variance; const queries;
// reset == prior.
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <random>
#include <vector>

#include "dense_oracle.hpp"
#include "gmrf_test_utils.hpp"

namespace gm = scalar_field_belief::gmrf;
using gmrf_test::points;

namespace {

constexpr double kRel = 1e-8;

// 4 x 3 nodes on [-1.5, 0.9] x [-0.7, 0.5]: dx = 0.8, dy = 0.6 (non-square).
gm::GmrfConfig tinyConfig(bool force_joint = false) {
  gm::GmrfConfig c;
  c.x_min = -1.5;
  c.x_max = 0.9;
  c.y_min = -0.7;
  c.y_max = 0.5;
  c.nx = 4;
  c.ny = 3;
  c.prior_mean = 0.3;
  c.prior_sigma = 1.7;
  c.sigma_p_per_m = 2.0;  // sigma_p = 2 * 0.8^2 = 1.28 for both directions
  c.sigma_s = 0.2;
  c.force_joint_state = force_joint;
  return c;
}

using Obs = std::vector<std::pair<std::pair<double, double>, double>>;

// Off-grid points, duplicates, corners, upper edges, a node, cell borders.
Eigen::MatrixX2d queryPoints() {
  return points({{-1.23, -0.41},
                 {0.11, 0.37},
                 {-0.6, -0.2},
                 {-1.23, -0.41},  // duplicate of row 0
                 {-1.5, -0.7},
                 {0.9, 0.5},
                 {-1.5, 0.5},
                 {0.9, -0.7},  // corners
                 {0.9, -0.05},
                 {-0.33, 0.5},  // upper x / y edges
                 {-0.7, -0.1},
                 {0.1, 0.5},
                 {-0.7, 0.2},
                 {0.55, -0.69}});  // node, on-node edge
}

Obs observations() {
  return {{{-1.1, -0.3}, 1.4}, {{0.4, 0.2}, -0.6},
          {{0.9, 0.5}, 2.0},  // corner
          {{-0.25, 0.5}, 0.8}, {{0.9, -0.33}, -1.1},
          {{0.4, 0.2}, -0.4}};  // upper edges, repeat
}

void checkAgainstOracle(const gm::GmrfBelief& b, const oracle::Model& m,
                        const Obs& obs) {
  const oracle::Posterior p = oracle::scalarPosterior(m, obs);
  const Eigen::MatrixX2d xy = queryPoints();
  const Eigen::MatrixXd H = m.H(xy);
  const Eigen::VectorXd mean_ref = H * p.mean;
  const Eigen::MatrixXd cov_ref = H * p.Sigma * H.transpose();

  Eigen::VectorXd mean, var, mean2;
  Eigen::MatrixXd cov;
  b.queryMeanCovariance(xy, mean, cov);
  b.queryMeanVariance(xy, mean2, var);
  const Eigen::VectorXd mean3 = b.queryMean(xy);

  const double dm = oracle::relDiff(mean, mean_ref);
  const double dc = oracle::relDiff(cov, cov_ref);
  const double dv = oracle::relDiff(var, cov_ref.diagonal());
  std::cout << "[oracle] obs=" << obs.size() << " rel mean " << dm
            << " rel cov " << dc << " rel var " << dv << "\n";
  EXPECT_LT(dm, kRel);
  EXPECT_LT(oracle::relDiff(mean2, mean_ref), kRel);
  EXPECT_LT(oracle::relDiff(mean3, mean_ref), kRel);
  EXPECT_LT(dc, kRel);  // includes all off-diagonals
  EXPECT_LT(dv, kRel);

  // Nodal mean (state) vs oracle.
  EXPECT_LT(oracle::relDiff(b.fieldNodeMean(), p.mean), kRel);

  // Structural properties of the returned covariance.
  EXPECT_EQ(cov.rows(), xy.rows());
  EXPECT_EQ(cov.cols(), xy.rows());
  EXPECT_LE((cov - cov.transpose()).cwiseAbs().maxCoeff(),
            1e-14 * cov.cwiseAbs().maxCoeff());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
  EXPECT_GE(es.eigenvalues().minCoeff(), -1e-10 * es.eigenvalues().maxCoeff());
  EXPECT_LT(oracle::relDiff(var, cov.diagonal()), 1e-10);
  // Duplicate query rows 0 and 3 are identical in mean, variance and
  // covariance.
  EXPECT_DOUBLE_EQ(mean(0), mean(3));
  EXPECT_DOUBLE_EQ(var(0), var(3));
  EXPECT_NEAR(cov(0, 3), cov(0, 0), 1e-12 * cov(0, 0));
  EXPECT_LT((cov.row(0) - cov.row(3)).cwiseAbs().maxCoeff(),
            1e-12 * cov.cwiseAbs().maxCoeff());
}

}  // namespace

TEST(DenseOracle, OracleHMatchesExpectedWeightsAndPortH) {
  // Sanity of the oracle itself on a hand-computed point, then port H.
  const gm::GmrfConfig c = tinyConfig();
  const oracle::Model m(c);
  // (-1.1, -0.4): cell (0,0), u = 0.4/0.8 = 0.5, v = 0.3/0.6 = 0.5.
  const Eigen::RowVectorXd h = m.hRow(-1.1, -0.4);
  EXPECT_NEAR(h(m.id(0, 0)), 0.25, 1e-15);
  EXPECT_NEAR(h(m.id(1, 0)), 0.25, 1e-15);
  EXPECT_NEAR(h(m.id(0, 1)), 0.25, 1e-15);
  EXPECT_NEAR(h(m.id(1, 1)), 0.25, 1e-15);
  const gm::GmrfBelief b(c);
  const Eigen::MatrixXd Hp =
      Eigen::MatrixXd(b.interpolationMatrix(queryPoints()));
  EXPECT_LT(oracle::relDiff(Hp, m.H(queryPoints())), 1e-12);
}

TEST(DenseOracle, PriorMatchesOracle) {
  const gm::GmrfBelief b(tinyConfig());
  checkAgainstOracle(b, oracle::Model(tinyConfig()), {});
  // Prior mean is exactly the constant prior_mean everywhere (Laplacian kills
  // constants).
  EXPECT_LT((b.queryMean(queryPoints()).array() - 0.3).abs().maxCoeff(), 1e-12);
}

TEST(DenseOracle, PosteriorMatchesOracleScalarAndForcedJointState) {
  for (bool joint : {false, true}) {
    SCOPED_TRACE(joint ? "force_joint_state" : "scalar state");
    gm::GmrfBelief b(tinyConfig(joint));
    ASSERT_EQ(b.isJointState(), joint);
    const Obs obs = observations();
    // Two update batches (streaming) must equal one batch.
    for (std::size_t k = 0; k < obs.size(); ++k) {
      b.addMeasurement(obs[k].first.first, obs[k].first.second, obs[k].second);
      if (k == 2) {
        b.update();
      }
    }
    b.update();
    EXPECT_EQ(b.numMeasurements(), obs.size());
    checkAgainstOracle(b, oracle::Model(tinyConfig(joint)), obs);
  }
}

TEST(DenseOracle, EmptyQueryBatch) {
  gm::GmrfBelief b(tinyConfig());
  b.addMeasurement(0.0, 0.0, 1.0);
  b.update();
  const Eigen::MatrixX2d none(0, 2);
  Eigen::VectorXd mean = Eigen::VectorXd::Ones(3),
                  var = Eigen::VectorXd::Ones(3);
  Eigen::MatrixXd cov = Eigen::MatrixXd::Ones(2, 2);
  EXPECT_NO_THROW(b.queryMeanCovariance(none, mean, cov));
  EXPECT_EQ(mean.size(), 0);
  EXPECT_EQ(cov.rows(), 0);
  EXPECT_EQ(cov.cols(), 0);
  EXPECT_NO_THROW(b.queryMeanVariance(none, mean, var));
  EXPECT_EQ(mean.size(), 0);
  EXPECT_EQ(var.size(), 0);
  EXPECT_EQ(b.queryMean(none).size(), 0);
}

TEST(DenseOracle, ObservationsNeverIncreaseVarianceAndRepeatsDecreaseIt) {
  const gm::GmrfConfig c = tinyConfig();
  gm::GmrfBelief b(c);
  // Dense test set (incl. nodes, edges, corners).
  Eigen::MatrixX2d probe(9 * 7, 2);
  for (int j = 0; j < 7; ++j) {
    for (int i = 0; i < 9; ++i) {
      probe(j * 9 + i, 0) = c.x_min + (c.x_max - c.x_min) * i / 8.0;
      probe(j * 9 + i, 1) = c.y_min + (c.y_max - c.y_min) * j / 6.0;
    }
  }
  Eigen::VectorXd mean, var_prev, var;
  b.queryMeanVariance(probe, mean, var_prev);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> ux(c.x_min, c.x_max),
      uy(c.y_min, c.y_max), uv(-3.0, 3.0);
  for (int k = 0; k < 25; ++k) {
    b.addMeasurement(ux(rng), uy(rng), uv(rng));
    b.update();
    b.queryMeanVariance(probe, mean, var);
    for (Eigen::Index r = 0; r < var.size(); ++r) {
      ASSERT_LE(var(r), var_prev(r) * (1.0 + 1e-10) + 1e-15)
          << "step " << k << " probe " << r;
    }
    var_prev = var;
  }
  // Repeated observations at one off-grid point strictly decrease its variance,
  // consistent with the oracle.
  gm::GmrfBelief r(c);
  const Eigen::MatrixX2d p = points({{-0.2, 0.05}});
  Eigen::VectorXd m1, v1;
  r.queryMeanVariance(p, m1, v1);
  double last = v1(0);
  Obs obs;
  for (int k = 0; k < 6; ++k) {
    r.addMeasurement(-0.2, 0.05, 1.0);
    r.update();
    obs.push_back({{-0.2, 0.05}, 1.0});
    r.queryMeanVariance(p, m1, v1);
    EXPECT_LT(v1(0), last * (1.0 - 1e-6)) << "repeat " << k;
    last = v1(0);
    const oracle::Model m(c);
    const auto post = oracle::scalarPosterior(m, obs);
    const Eigen::RowVectorXd h = m.hRow(-0.2, 0.05);
    EXPECT_NEAR(v1(0), (h * post.Sigma * h.transpose())(0, 0), kRel * last);
  }
}

TEST(DenseOracle, QueriesDoNotChangeStateAndResetEqualsPrior) {
  const gm::GmrfConfig c = tinyConfig();
  const gm::GmrfBelief fresh(c);
  gm::GmrfBelief b(c);
  for (const auto& o : observations()) {
    b.addMeasurement(o.first.first, o.first.second, o.second);
  }
  b.update();
  const Eigen::VectorXd state = b.stateMean();
  const Eigen::MatrixXd Q = Eigen::MatrixXd(b.precisionMatrix());
  Eigen::VectorXd m0, v0, m1, v1;
  Eigen::MatrixXd c0, c1;
  b.queryMeanCovariance(queryPoints(), m0, c0);
  for (int k = 0; k < 5; ++k) {
    b.queryMeanVariance(queryPoints(), m1, v1);
    b.queryMeanCovariance(queryPoints(), m1, c1);
    (void)b.queryMean(queryPoints());
  }
  EXPECT_EQ(b.stateMean(), state);  // bitwise
  EXPECT_EQ(Eigen::MatrixXd(b.precisionMatrix()), Q);
  EXPECT_EQ(m0, m1);
  EXPECT_EQ(c0, c1);
  EXPECT_EQ(b.numMeasurements(), observations().size());

  b.reset();
  EXPECT_EQ(b.numMeasurements(), 0u);
  EXPECT_FALSE(b.hasPendingMeasurements());
  Eigen::VectorXd mf, mr;
  Eigen::MatrixXd cf, cr;
  fresh.queryMeanCovariance(queryPoints(), mf, cf);
  b.queryMeanCovariance(queryPoints(), mr, cr);
  EXPECT_EQ(mr, mf);
  EXPECT_EQ(cr, cf);
  checkAgainstOracle(b, oracle::Model(c), {});
  // And it keeps working after reset.
  b.addMeasurement(0.1, 0.1, 5.0);
  b.update();
  checkAgainstOracle(b, oracle::Model(c), {{{0.1, 0.1}, 5.0}});
}
