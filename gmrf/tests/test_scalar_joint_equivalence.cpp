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

// Scalar-only state vs. joint [c, w_x, w_y] state, source formulas that are
// preserved (sigma_p uses dx), corrected flow-neighbour cuts, the nonlinear
// flow mode, and optional measurement aging.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "gmrf_test_utils.hpp"

using gmrf_test::nodePoints;
using gmrf_test::points;
using gmrf_test::smallConfig;
namespace gm = scalar_field_belief::gmrf;

namespace {

const std::vector<gm::WallSegment> kWalls{{1.1, 0.6, 1.1, 3.4},
                                          {1.1, 2.1, 1.6, 2.1}};

void feed(gm::GmrfBelief& b) {
  b.addMeasurement(0.3, 0.7, 1.0);
  b.addMeasurement(0.95, 2.05, 2.5);
  b.addMeasurement(1.3, 2.2, -0.5);
  b.addMeasurement(1.9, 3.9, 0.25);
  b.addMeasurement(0.95, 2.05, 2.4);
  b.update();
}

}  // namespace

TEST(ScalarJoint, ScalarStateEqualsJointStateWithFlowEnergiesOff) {
  gm::GmrfConfig cs = smallConfig();
  gm::GmrfConfig cj = cs;
  cj.force_joint_state = true;
  cj.ff_prior_mean_x = 0.3;
  cj.ff_prior_mean_y = -0.2;
  gm::GmrfBelief bs(cs, kWalls);
  gm::GmrfBelief bj(cj, kWalls);
  ASSERT_FALSE(bs.isJointState());
  ASSERT_TRUE(bj.isJointState());
  EXPECT_EQ(bs.stateSize(), bs.nodeCount());
  EXPECT_EQ(bj.stateSize(), 3 * bj.nodeCount());
  feed(bs);
  feed(bj);

  const Eigen::MatrixX2d q = points({{0.3, 0.7},
                                     {1.05, 2.0},
                                     {1.15, 2.0},
                                     {1.3, 2.15},
                                     {1.3, 2.05},
                                     {2.0, 4.0},
                                     {0.0, 0.0},
                                     {1.1, 3.5}});
  Eigen::VectorXd ms;
  Eigen::VectorXd mj;
  Eigen::MatrixXd cs_cov;
  Eigen::MatrixXd cj_cov;
  bs.queryMeanCovariance(q, ms, cs_cov);
  bj.queryMeanCovariance(q, mj, cj_cov);
  const double scale = cs_cov.cwiseAbs().maxCoeff();
  EXPECT_LT((ms - mj).cwiseAbs().maxCoeff(), 1e-10);
  EXPECT_LT((cs_cov - cj_cov).cwiseAbs().maxCoeff(), 1e-10 * scale);

  Eigen::VectorXd vs;
  Eigen::VectorXd vj;
  bs.queryMeanVariance(nodePoints(bs), ms, vs);
  bj.queryMeanVariance(nodePoints(bj), mj, vj);
  EXPECT_LT((ms - mj).cwiseAbs().maxCoeff(), 1e-10);
  EXPECT_LT((vs - vj).cwiseAbs().maxCoeff(), 1e-10 * vs.maxCoeff());

  // Without flow energies the flow blocks keep their prior and decouple.
  EXPECT_TRUE(
      bj.flowNodeMean(gm::FlowComponent::X).isApproxToConstant(0.3, 1e-12));
  EXPECT_TRUE(
      bj.flowNodeMean(gm::FlowComponent::Y).isApproxToConstant(-0.2, 1e-12));
  const Eigen::Index N = bj.nodeCount();
  const Eigen::MatrixXd Qj(bj.precisionMatrix());
  const Eigen::MatrixXd Qs(bs.precisionMatrix());
  EXPECT_LT((Qj.topLeftCorner(N, N) - Qs).cwiseAbs().maxCoeff(),
            1e-12 * Qs.cwiseAbs().maxCoeff());
  EXPECT_EQ(Qj.block(0, N, N, 2 * N).cwiseAbs().maxCoeff(), 0.0);
}

TEST(ScalarJoint, FieldNeighbourSigmaUsesDxForBothDirections) {
  gm::GmrfConfig c = smallConfig();
  c.nx = 5;   // dx = 0.5
  c.ny = 17;  // dy = 0.25
  c.sigma_p_per_m = 3.0;
  const gm::GmrfBelief b(c);
  const double sp = 3.0 * 0.5 * 0.5;
  EXPECT_DOUBLE_EQ(b.fieldNeighborSigma(), sp);
  const auto& Q = b.precisionMatrix();
  const double expected = -1.0 / (sp * sp);
  EXPECT_NEAR(Q.coeff(b.nodeIndex(1, 1), b.nodeIndex(2, 1)), expected,
              1e-12);  // horizontal
  EXPECT_NEAR(Q.coeff(b.nodeIndex(1, 1), b.nodeIndex(1, 2)), expected,
              1e-12);  // vertical
  // Interior diagonal: prior + 4 neighbour terms.
  const double diag = 1.0 / (c.prior_sigma * c.prior_sigma) + 4.0 / (sp * sp);
  EXPECT_NEAR(Q.coeff(b.nodeIndex(2, 5), b.nodeIndex(2, 5)), diag,
              1e-12 * diag);
}

TEST(ScalarJoint, FlowNeighbourCutRemovesBothFlowRowsOfThatEdgeOnly) {
  for (bool field_neighbors : {true, false}) {
    gm::GmrfConfig c = smallConfig();
    c.field_neighbors = field_neighbors;
    c.flow_neighbors = true;
    c.ff_sigma_p_per_m = 2.0;
    const gm::GmrfBelief b(c, kWalls);
    const Eigen::Index N = b.nodeCount();
    const auto& Q = b.precisionMatrix();
    const double sp = 2.0 * b.dx() * b.dx();
    const double coupled = -1.0 / (sp * sp);
    int cuts = 0;
    auto check = [&](Eigen::Index a, Eigen::Index n1, bool cut) {
      for (Eigen::Index off : {N, 2 * N}) {
        const double q = Q.coeff(off + a, off + n1);
        if (cut) {
          EXPECT_EQ(q, 0.0) << "edge " << a << "-" << n1 << " off " << off;
        } else {
          EXPECT_NEAR(q, coupled, 1e-12 * std::abs(coupled)) << a << "-" << n1;
        }
      }
    };
    for (int j = 0; j < b.ny(); ++j) {
      for (int i = 0; i + 1 < b.nx(); ++i) {
        const bool cut = b.isHorizontalEdgeCut(i, j);
        cuts += cut;
        check(b.nodeIndex(i, j), b.nodeIndex(i + 1, j), cut);
      }
    }
    for (int j = 0; j + 1 < b.ny(); ++j) {
      for (int i = 0; i < b.nx(); ++i) {
        const bool cut = b.isVerticalEdgeCut(i, j);
        cuts += cut;
        check(b.nodeIndex(i, j), b.nodeIndex(i, j + 1), cut);
      }
    }
    EXPECT_GT(cuts, 10);
  }
}

TEST(ScalarJoint, NonlinearFlowModeConvergesWithFiniteUncertainty) {
  gm::GmrfConfig c = smallConfig();
  c.nx = 7;
  c.ny = 9;
  c.flow_neighbors = true;
  c.flow_conservation = true;
  c.flow_field = true;
  c.obstacle_flow = true;
  c.sigma_flow = 0.5;
  c.ff_prior_sigma = 1.0;
  c.ff_sigma_s = 0.05;
  c.ff_sigma_c = 0.2;
  c.ff_sigma_o = 0.1;
  c.ff_sigma_p_per_m = 2.0;
  gm::GmrfBelief b(c, {{1.0, 1.0, 1.0, 3.0}, {0.0, 0.0, 0.0, 4.0}});
  EXPECT_FALSE(b.isLinear());
  b.addMeasurement(0.4, 1.7, 1.0);
  b.addMeasurement(1.6, 2.3, 0.2);
  b.addFlowMeasurement(0.4, 1.7, gm::FlowComponent::X, 0.3);
  b.addFlowMeasurement(0.4, 1.7, gm::FlowComponent::Y, -0.1);
  const gm::UpdateReport rep = b.update();
  EXPECT_FALSE(rep.linear);
  EXPECT_TRUE(rep.converged);
  EXPECT_GE(rep.iterations, 1);
  EXPECT_LE(rep.iterations, c.max_gauss_newton_iterations);
  EXPECT_TRUE(std::isfinite(rep.final_cost));

  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
  b.queryMeanCovariance(points({{0.4, 1.7}, {1.6, 2.3}, {0.5, 3.5}}), mean,
                        cov);
  EXPECT_TRUE(mean.allFinite());
  EXPECT_TRUE(cov.allFinite());
  EXPECT_GT(cov.diagonal().minCoeff(), 0.0);
  EXPECT_TRUE(b.flowNodeMean(gm::FlowComponent::X).allFinite());

  // Deterministic: an identical belief gives identical results.
  gm::GmrfBelief b2(c, {{1.0, 1.0, 1.0, 3.0}, {0.0, 0.0, 0.0, 4.0}});
  b2.addMeasurement(0.4, 1.7, 1.0);
  b2.addMeasurement(1.6, 2.3, 0.2);
  b2.addFlowMeasurement(0.4, 1.7, gm::FlowComponent::X, 0.3);
  b2.addFlowMeasurement(0.4, 1.7, gm::FlowComponent::Y, -0.1);
  b2.update();
  EXPECT_EQ(b.stateMean(), b2.stateMean());

  // Iteration cap is respected.
  c.max_gauss_newton_iterations = 1;
  gm::GmrfBelief b3(c);
  b3.addMeasurement(0.4, 1.7, 1.0);
  b3.addFlowMeasurement(0.4, 1.7, gm::FlowComponent::X, 0.3);
  const gm::UpdateReport rep3 = b3.update();
  EXPECT_LE(rep3.iterations, 1);
}

TEST(ScalarJoint, MeasurementAgingIsExplicitAndUsesCallerTime) {
  gm::GmrfConfig c = smallConfig();
  c.enable_measurement_aging = true;
  c.sigma_zeta = 0.2;
  gm::GmrfBelief b(c);
  const Eigen::MatrixX2d q = points({{1.0, 2.0}});
  b.addMeasurement(1.0, 2.0, 3.0, 5.0);
  EXPECT_THROW(b.update(4.0),
               std::invalid_argument);  // before the measurement time
  EXPECT_TRUE(b.hasPendingMeasurements());
  b.update(5.0);
  Eigen::VectorXd m0;
  Eigen::VectorXd v0;
  b.queryMeanVariance(q, m0, v0);
  const gm::UpdateReport rep = b.update(105.0);  // no new data, aging re-solves
  EXPECT_TRUE(rep.solved);
  Eigen::VectorXd m1;
  Eigen::VectorXd v1;
  b.queryMeanVariance(q, m1, v1);
  EXPECT_GT(v1(0), v0(0));
  EXPECT_DOUBLE_EQ(b.lastUpdateTime(), 105.0);

  // Disabled (default): time is ignored entirely.
  gm::GmrfBelief s(smallConfig());
  s.addMeasurement(1.0, 2.0, 3.0, 5.0);
  s.update(0.0);
  s.queryMeanVariance(q, m0, v0);
  s.addMeasurement(1.0, 2.0, 3.0, 1e6);
  s.update(-3.0);
  s.queryMeanVariance(q, m1, v1);
  EXPECT_LT(v1(0), v0(0));
}

TEST(ScalarJoint, StallBeforeAnyAcceptedStepIsReportedAsNotConverged) {
  // Prior-only nonlinear model at its exact optimum (all residuals zero at
  // x=0): with gn_step_tolerance = 0 the zero step is not accepted by the line
  // search, so no step is ever accepted. Aging forces a re-solve without new
  // data.
  gm::GmrfConfig c = smallConfig();
  c.nx = 4;
  c.ny = 5;
  c.prior_mean = 0.0;
  c.flow_field = true;
  c.gn_step_tolerance = 0.0;
  c.enable_measurement_aging = true;
  gm::GmrfBelief b(c);
  const Eigen::VectorXd before = b.stateMean();
  const gm::UpdateReport rep = b.update(1.0);
  EXPECT_TRUE(rep.solved);
  EXPECT_FALSE(rep.linear);
  EXPECT_EQ(rep.iterations, 0);
  EXPECT_FALSE(rep.converged);
  EXPECT_EQ(b.stateMean(), before);  // state kept at the starting point
}
