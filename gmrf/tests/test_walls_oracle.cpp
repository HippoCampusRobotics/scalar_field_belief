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

// Wall checks against the independent dense oracle (dense_oracle.hpp):
// full separator, gap, diagonal wall, reversed and split segments.
// Geometries keep walls clear of nodes, so the tiny tie-break translation is
// irrelevant here.
#include <gtest/gtest.h>

#include <vector>

#include "dense_oracle.hpp"
#include "gmrf_test_utils.hpp"

namespace gm = scalar_field_belief::gmrf;
using gmrf_test::points;

namespace {

constexpr double kRel = 1e-8;
using Obs = std::vector<std::pair<std::pair<double, double>, double>>;

// 5 x 4 nodes on [0, 2] x [0, 1.8]: dx = 0.5, dy = 0.6.
gm::GmrfConfig cfg() {
  gm::GmrfConfig c;
  c.x_min = 0.0;
  c.x_max = 2.0;
  c.y_min = 0.0;
  c.y_max = 1.8;
  c.nx = 5;
  c.ny = 4;
  c.prior_mean = -0.4;
  c.prior_sigma = 1.3;
  c.sigma_p_per_m = 1.5;  // sigma_p = 1.5 * 0.25 = 0.375
  c.sigma_s = 0.15;
  return c;
}

const std::vector<gm::WallSegment> kSeparator{{0.9, -0.5, 0.9, 2.3}};
const std::vector<gm::WallSegment> kGap{{0.9, -0.5, 0.9, 0.9}};
const std::vector<gm::WallSegment> kDiagonal{{0.2, 0.15, 1.8, 1.35}};

// Left of x = 0.9 (incl. near-wall points), right of it, and near the diagonal.
Eigen::MatrixX2d leftPts() {
  return points({{0.3, 0.9}, {0.85, 0.3}, {0.85, 1.7}, {0.0, 0.0}, {0.6, 1.8}});
}
Eigen::MatrixX2d rightPts() {
  return points({{1.7, 0.9}, {0.95, 0.3}, {0.95, 1.7}, {2.0, 1.8}, {2.0, 0.0}});
}
Eigen::MatrixX2d diagPts() {
  return points({{1.0, 0.70},
                 {1.0, 0.80},
                 {0.6, 0.40},
                 {0.6, 0.52},
                 {1.5, 1.08},
                 {1.5, 1.17},
                 {0.3, 0.9},
                 {1.7, 0.4}});
}
Eigen::MatrixX2d allPts() {
  Eigen::MatrixX2d a(15, 2);
  a << leftPts(), rightPts(),
      points({{0.3, 0.2}, {1.95, 1.75}, {1.2, 0.95}, {0.95, 1.1}, {0.85, 1.1}});
  return a;
}

Obs leftObs() {
  return {{{0.3, 0.4}, 2.0}, {{0.7, 1.3}, -1.0}, {{0.85, 0.9}, 1.5}};
}

gm::GmrfBelief withObs(const std::vector<gm::WallSegment>& walls,
                       const Obs& obs) {
  gm::GmrfBelief b(cfg(), walls);
  for (const auto& o : obs) {
    b.addMeasurement(o.first.first, o.first.second, o.second);
  }
  b.update();
  return b;
}

void expectCutsMatchOracle(const gm::GmrfBelief& b, const oracle::Model& m,
                           int expected_cuts) {
  int n = 0;
  for (int j = 0; j < m.c.ny; ++j) {
    for (int i = 0; i + 1 < m.c.nx; ++i) {
      EXPECT_EQ(b.isHorizontalEdgeCut(i, j), m.hCut(i, j))
          << "h edge " << i << "," << j;
      n += m.hCut(i, j) ? 1 : 0;
    }
  }
  for (int j = 0; j + 1 < m.c.ny; ++j) {
    for (int i = 0; i < m.c.nx; ++i) {
      EXPECT_EQ(b.isVerticalEdgeCut(i, j), m.vCut(i, j))
          << "v edge " << i << "," << j;
      n += m.vCut(i, j) ? 1 : 0;
    }
  }
  EXPECT_EQ(n, expected_cuts);  // guards the oracle geometry itself
}

void expectPosteriorMatchesOracle(const gm::GmrfBelief& b,
                                  const std::vector<gm::WallSegment>& walls,
                                  const Obs& obs, const Eigen::MatrixX2d& xy) {
  const oracle::Model m(cfg(), walls);
  const oracle::Posterior p = oracle::scalarPosterior(m, obs);
  const Eigen::MatrixXd H = m.H(xy);
  Eigen::VectorXd mean, var, mean2;
  Eigen::MatrixXd cov;
  b.queryMeanCovariance(xy, mean, cov);
  b.queryMeanVariance(xy, mean2, var);
  const double dm = oracle::relDiff(mean, H * p.mean);
  const double dc = oracle::relDiff(cov, H * p.Sigma * H.transpose());
  std::cout << "[walls oracle] rel mean " << dm << " rel cov " << dc << "\n";
  EXPECT_LT(dm, kRel);
  EXPECT_LT(dc, kRel);
  EXPECT_LT(oracle::relDiff(var, cov.diagonal()), 1e-10);
  EXPECT_LT(oracle::relDiff(b.fieldNodeMean(), p.mean), kRel);
}

}  // namespace

TEST(WallsOracle, FullSeparatorDecouplesCompartments) {
  const oracle::Model m(cfg(), kSeparator);
  const gm::GmrfBelief prior(cfg(), kSeparator);
  expectCutsMatchOracle(prior, m, 4);  // h edges (1,j), j = 0..3
  for (int j = 0; j < 4; ++j) {
    EXPECT_TRUE(m.hCut(1, j));
  }
  expectPosteriorMatchesOracle(prior, kSeparator, {}, allPts());

  const gm::GmrfBelief b = withObs(kSeparator, leftObs());
  expectPosteriorMatchesOracle(b, kSeparator, leftObs(), allPts());

  // Right compartment stays at the prior (mean and variance), both queryable.
  Eigen::VectorXd mr, vr, mp, vp, ml, vl, mlp, vlp;
  b.queryMeanVariance(rightPts(), mr, vr);
  prior.queryMeanVariance(rightPts(), mp, vp);
  EXPECT_LT((mr - mp).cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT((mr.array() - cfg().prior_mean).abs().maxCoeff(), 1e-12);
  EXPECT_LT(oracle::relDiff(vr, vp), 1e-12);
  // ... while the observed compartment did change.
  b.queryMeanVariance(leftPts(), ml, vl);
  prior.queryMeanVariance(leftPts(), mlp, vlp);
  EXPECT_GT((ml - mlp).cwiseAbs().maxCoeff(), 0.1);
  EXPECT_LT(vl.maxCoeff() - vlp.maxCoeff(), 0.0);
  // Right nodal states unchanged.
  const Eigen::VectorXd nodes = b.fieldNodeMean();
  for (int j = 0; j < 4; ++j) {
    for (int i = 2; i < 5; ++i) {
      EXPECT_NEAR(nodes(m.id(i, j)), cfg().prior_mean, 1e-12);
    }
  }
  // Cross-covariance between compartments is zero.
  Eigen::MatrixX2d lr(10, 2);
  lr << leftPts(), rightPts();
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
  b.queryMeanCovariance(lr, mean, cov);
  EXPECT_LE(cov.topRightCorner(5, 5).cwiseAbs().maxCoeff(), 1e-14);
  EXPECT_LE(cov.bottomLeftCorner(5, 5).cwiseAbs().maxCoeff(), 1e-14);
  EXPECT_GT(cov.topLeftCorner(5, 5).cwiseAbs().minCoeff(), 1e-6);

  // An observation in the right compartment only affects the right side.
  gm::GmrfBelief r = withObs(kSeparator, {{{1.6, 0.5}, 3.0}});
  Eigen::VectorXd m2, v2;
  r.queryMeanVariance(leftPts(), m2, v2);
  EXPECT_LT((m2 - mlp).cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT(oracle::relDiff(v2, vlp), 1e-12);
}

TEST(WallsOracle, GapLetsInformationPass) {
  const oracle::Model m(cfg(), kGap);
  const gm::GmrfBelief prior(cfg(), kGap);
  expectCutsMatchOracle(prior, m, 2);  // h edges (1,0), (1,1)
  EXPECT_TRUE(m.hCut(1, 0));
  EXPECT_TRUE(m.hCut(1, 1));
  const gm::GmrfBelief b = withObs(kGap, leftObs());
  expectPosteriorMatchesOracle(b, kGap, leftObs(), allPts());
  Eigen::VectorXd mr, vr, mp, vp;
  b.queryMeanVariance(rightPts(), mr, vr);
  prior.queryMeanVariance(rightPts(), mp, vp);
  EXPECT_GT((mr - mp).cwiseAbs().maxCoeff(), 1e-3);
  EXPECT_GT((vp - vr).maxCoeff(), 1e-4);
  Eigen::MatrixX2d lr(2, 2);
  lr << 0.85, 0.3, 0.95, 0.3;  // across the wall, below the gap
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
  b.queryMeanCovariance(lr, mean, cov);
  EXPECT_GT(std::abs(cov(0, 1)), 1e-6);  // correlated through the gap
}

TEST(WallsOracle, DiagonalWall) {
  const oracle::Model m(cfg(), kDiagonal);
  const gm::GmrfBelief prior(cfg(), kDiagonal);
  // Crosses h edges at y=0.6 (x=0.8) and y=1.2 (x=1.6) and v edges at
  // x=0.5 (y=0.375), x=1.0 (y=0.75), x=1.5 (y=1.125).
  expectCutsMatchOracle(prior, m, 5);
  expectPosteriorMatchesOracle(prior, kDiagonal, {}, diagPts());
  const Obs obs{{{1.0, 0.70}, 2.0},
                {{0.6, 0.52}, -1.0},
                {{1.5, 1.17}, 0.5},
                {{1.7, 0.4}, 1.0}};
  const gm::GmrfBelief b = withObs(kDiagonal, obs);
  expectPosteriorMatchesOracle(b, kDiagonal, obs, diagPts());
  expectPosteriorMatchesOracle(b, kDiagonal, obs, allPts());
}

TEST(WallsOracle, ReversedAndSplitSegmentsGiveSameCutsAndPosterior) {
  struct Case {
    const char* name;
    std::vector<gm::WallSegment> ref;
    std::vector<std::vector<gm::WallSegment>> variants;
    int cuts;
  };
  const std::vector<Case> cases{
      {"separator",
       kSeparator,
       {{{0.9, 2.3, 0.9, -0.5}},
        {{0.9, -0.5, 0.9, 0.6},
         {0.9, 0.6, 0.9, 2.3}},  // split exactly on node row y=0.6
        {{0.9, 2.3, 0.9, 1.0},
         {0.9, -0.5, 0.9, 1.0}}},  // split between rows, mixed order
       4},
      {"diagonal",
       kDiagonal,
       {{{1.8, 1.35, 0.2, 0.15}},
        {{0.2, 0.15, 1.0, 0.75},
         {1.0, 0.75, 1.8, 1.35}},  // split exactly on grid column x=1.0
        {{1.8, 1.35, 1.3, 0.975}, {0.2, 0.15, 1.3, 0.975}}},
       5},
  };
  const Obs obs{{{0.3, 0.4}, 2.0}, {{1.0, 0.8}, -1.0}, {{1.7, 0.4}, 0.7}};
  for (const auto& cs : cases) {
    SCOPED_TRACE(cs.name);
    const gm::GmrfBelief ref = withObs(cs.ref, obs);
    Eigen::VectorXd m0, m1;
    Eigen::MatrixXd c0, c1;
    ref.queryMeanCovariance(allPts(), m0, c0);
    for (std::size_t k = 0; k < cs.variants.size(); ++k) {
      SCOPED_TRACE(k);
      const gm::GmrfBelief b = withObs(cs.variants[k], obs);
      expectCutsMatchOracle(b, oracle::Model(cfg(), cs.ref), cs.cuts);
      b.queryMeanCovariance(allPts(), m1, c1);
      EXPECT_LT(oracle::relDiff(m1, m0), 1e-12);
      EXPECT_LT(oracle::relDiff(c1, c0), 1e-12);
    }
    expectPosteriorMatchesOracle(ref, cs.ref, obs, allPts());
  }
}
