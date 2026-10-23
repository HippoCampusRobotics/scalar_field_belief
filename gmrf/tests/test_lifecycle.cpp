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

// Lifecycle, prior query, reset, input validation, and domain-boundary
// handling.
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "gmrf_test_utils.hpp"

using gmrf_test::nodePoints;
using gmrf_test::points;
using gmrf_test::smallConfig;
namespace gm = scalar_field_belief::gmrf;

namespace {
const double kNan = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();
}  // namespace

TEST(Lifecycle, PriorQueryBeforeAnyData) {
  const gm::GmrfConfig c = smallConfig();
  const gm::GmrfBelief b(c);
  const Eigen::MatrixX2d q =
      points({{0.0, 0.0}, {0.37, 1.91}, {2.0, 4.0}, {1.0, 2.0}});
  Eigen::VectorXd mean;
  Eigen::VectorXd var;
  b.queryMeanVariance(q, mean, var);
  ASSERT_EQ(mean.size(), 4);
  for (Eigen::Index k = 0; k < 4; ++k) {
    EXPECT_NEAR(mean(k), c.prior_mean, 1e-12);
    EXPECT_GT(var(k), 0.0);
    // Neighbour coupling adds information: prior marginal <= prior_sigma^2.
    EXPECT_LE(var(k), c.prior_sigma * c.prior_sigma * (1.0 + 1e-12));
  }
  Eigen::MatrixXd cov;
  b.queryMeanCovariance(q, mean, cov);
  EXPECT_TRUE(cov.isApprox(cov.transpose(), 0.0));
  for (Eigen::Index k = 0; k < 4; ++k) {
    EXPECT_NEAR(cov(k, k), var(k), 1e-12 * var(k));
  }
  EXPECT_EQ(b.numMeasurements(), 0u);
  EXPECT_FALSE(b.hasPendingMeasurements());
}

TEST(Lifecycle, PendingMeasurementsBlockQueriesUntilUpdate) {
  gm::GmrfBelief b(smallConfig());
  const Eigen::MatrixX2d q = points({{1.0, 1.0}});
  b.addMeasurement(1.0, 1.0, 3.0);
  EXPECT_TRUE(b.hasPendingMeasurements());
  EXPECT_EQ(b.numPendingMeasurements(), 1u);
  EXPECT_THROW(b.queryMean(q), std::logic_error);
  Eigen::VectorXd m;
  Eigen::VectorXd v;
  Eigen::MatrixXd cv;
  EXPECT_THROW(b.queryMeanVariance(q, m, v), std::logic_error);
  EXPECT_THROW(b.queryMeanCovariance(q, m, cv), std::logic_error);

  const gm::UpdateReport rep = b.update();
  EXPECT_TRUE(rep.solved);
  EXPECT_TRUE(rep.linear);
  EXPECT_EQ(rep.iterations, 1);
  EXPECT_EQ(rep.num_assimilated, 1u);
  EXPECT_FALSE(b.hasPendingMeasurements());
  EXPECT_EQ(b.numMeasurements(), 1u);
  EXPECT_NO_THROW(b.queryMean(q));
}

TEST(Lifecycle, DiscardPendingMeasurements) {
  gm::GmrfBelief b(smallConfig());
  const Eigen::VectorXd before = b.fieldNodeMean();
  b.addMeasurement(1.0, 1.0, 3.0);
  b.discardPendingMeasurements();
  EXPECT_FALSE(b.hasPendingMeasurements());
  EXPECT_EQ(b.fieldNodeMean(), before);
}

TEST(Lifecycle, QueriesAreConstAndRepeatable) {
  gm::GmrfBelief b(smallConfig());
  b.addMeasurement(0.6, 1.3, 2.0);
  b.addMeasurement(1.6, 3.3, -1.0);
  b.update();
  const Eigen::MatrixX2d q = points({{0.6, 1.3}, {1.1, 2.2}, {2.0, 0.0}});
  Eigen::VectorXd m1;
  Eigen::VectorXd v1;
  Eigen::VectorXd m2;
  Eigen::VectorXd v2;
  b.queryMeanVariance(q, m1, v1);
  for (int k = 0; k < 3; ++k) {
    Eigen::MatrixXd cov;
    b.queryMeanCovariance(q, m2, cov);
    b.queryMeanVariance(q, m2, v2);
    EXPECT_EQ(m1, m2);
    EXPECT_EQ(v1, v2);
  }
  EXPECT_EQ(b.numMeasurements(), 2u);
  // An update without new data does not change a stationary belief.
  const gm::UpdateReport rep = b.update(1000.0);
  EXPECT_FALSE(rep.solved);
  b.queryMeanVariance(q, m2, v2);
  EXPECT_EQ(m1, m2);
  EXPECT_EQ(v1, v2);
}

TEST(Lifecycle, ResetRestoresPrior) {
  const gm::GmrfConfig c = smallConfig();
  gm::GmrfBelief b(c);
  const Eigen::MatrixX2d q = nodePoints(b);
  Eigen::VectorXd m0;
  Eigen::VectorXd v0;
  b.queryMeanVariance(q, m0, v0);
  b.addMeasurement(0.6, 1.3, 2.0);
  b.update();
  b.addMeasurement(0.7, 1.3, 2.0);  // left pending
  b.reset();
  EXPECT_EQ(b.numMeasurements(), 0u);
  EXPECT_FALSE(b.hasPendingMeasurements());
  Eigen::VectorXd m1;
  Eigen::VectorXd v1;
  b.queryMeanVariance(q, m1, v1);
  EXPECT_EQ(m0, m1);
  EXPECT_EQ(v0, v1);
  EXPECT_EQ(b.lastUpdateTime(), 0.0);
}

TEST(Lifecycle, ObservationsDoNotIncreaseVariance) {
  gm::GmrfBelief b(smallConfig());
  const Eigen::MatrixX2d q = nodePoints(b);
  Eigen::VectorXd m;
  Eigen::VectorXd v_prev;
  b.queryMeanVariance(q, m, v_prev);
  for (int k = 0; k < 4; ++k) {
    b.addMeasurement(0.83, 2.41, 1.5);
    b.update();
    Eigen::VectorXd v;
    b.queryMeanVariance(q, m, v);
    for (Eigen::Index n = 0; n < v.size(); ++n) {
      EXPECT_LE(v(n), v_prev(n) * (1.0 + 1e-10))
          << "node " << n << " step " << k;
    }
    const Eigen::Index nearest = b.nodeIndex(3, 10);  // (0.75, 2.5)
    EXPECT_LT(v(nearest), v_prev(nearest));
    v_prev = v;
  }
}

TEST(Lifecycle, MeanUsesMeasurementUnitsWithoutClipping) {
  gm::GmrfConfig c = smallConfig();
  c.prior_mean = 0.0;
  c.prior_sigma = 1e4;
  c.sigma_s = 1e-3;
  gm::GmrfBelief b(c);
  b.addMeasurement(1.0, 2.0, 250.0);   // exactly at node (4, 8)
  b.addMeasurement(0.25, 0.25, -7.5);  // exactly at node (1, 1)
  b.update();
  const Eigen::VectorXd mean = b.queryMean(points({{1.0, 2.0}, {0.25, 0.25}}));
  EXPECT_NEAR(mean(0), 250.0, 0.05);
  EXPECT_NEAR(mean(1), -7.5, 0.05);
  EXPECT_DOUBLE_EQ(b.measurementNoiseVariance(), 1e-6);
}

TEST(Lifecycle, InvalidMeasurementsAreRejectedWithoutStateChange) {
  gm::GmrfBelief b(smallConfig());
  b.addMeasurement(1.0, 1.0, 1.0);
  b.update();
  const Eigen::VectorXd before = b.fieldNodeMean();
  EXPECT_THROW(b.addMeasurement(2.0 + 1e-9, 1.0, 1.0), std::invalid_argument);
  EXPECT_THROW(b.addMeasurement(-1e-12, 1.0, 1.0), std::invalid_argument);
  EXPECT_THROW(b.addMeasurement(1.0, 4.5, 1.0), std::invalid_argument);
  EXPECT_THROW(b.addMeasurement(kNan, 1.0, 1.0), std::invalid_argument);
  EXPECT_THROW(b.addMeasurement(1.0, kInf, 1.0), std::invalid_argument);
  EXPECT_THROW(b.addMeasurement(1.0, 1.0, kNan), std::invalid_argument);
  EXPECT_THROW(b.addMeasurement(1.0, 1.0, kInf), std::invalid_argument);
  EXPECT_THROW(b.addMeasurement(1.0, 1.0, 1.0, kNan), std::invalid_argument);
  EXPECT_THROW(b.update(kNan), std::invalid_argument);
  EXPECT_THROW(b.addFlowMeasurement(1.0, 1.0, gm::FlowComponent::X, 0.1),
               std::logic_error);
  EXPECT_FALSE(b.hasPendingMeasurements());
  EXPECT_EQ(b.numMeasurements(), 1u);
  EXPECT_EQ(b.fieldNodeMean(), before);
}

TEST(Lifecycle, InvalidQueriesThrowWithoutPartialResults) {
  const gm::GmrfBelief b(smallConfig());
  Eigen::VectorXd mean = Eigen::VectorXd::Constant(3, 42.0);
  Eigen::VectorXd var = Eigen::VectorXd::Constant(3, 42.0);
  EXPECT_THROW(b.queryMeanVariance(points({{1.0, 1.0}, {2.5, 1.0}}), mean, var),
               std::invalid_argument);
  EXPECT_EQ(mean, Eigen::VectorXd::Constant(3, 42.0));
  EXPECT_THROW(b.queryMean(points({{kNan, 1.0}})), std::invalid_argument);
  EXPECT_THROW(b.queryMean(points({{1.0, -0.001}})), std::invalid_argument);
  EXPECT_FALSE(b.isRepresentable(2.1, 1.0));
  EXPECT_FALSE(b.isRepresentable(kNan, 1.0));
  EXPECT_TRUE(b.isRepresentable(2.0, 4.0));
}

TEST(Lifecycle, CornersUpperEdgesAndNodes) {
  gm::GmrfBelief b(smallConfig());
  const Eigen::MatrixX2d q = points(
      {{0.0, 0.0}, {2.0, 0.0}, {0.0, 4.0}, {2.0, 4.0}, {2.0, 2.3}, {1.3, 4.0}});
  for (Eigen::Index k = 0; k < q.rows(); ++k) {
    EXPECT_NO_THROW(b.addMeasurement(q(k, 0), q(k, 1), 1.0));
  }
  b.update();
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
  EXPECT_NO_THROW(b.queryMeanCovariance(q, mean, cov));
  EXPECT_TRUE(mean.allFinite());
  EXPECT_TRUE(cov.allFinite());

  // A point exactly at a node interpolates that node only.
  const auto H = b.interpolationMatrix(points({{2.0, 4.0}, {0.5, 1.75}}));
  EXPECT_EQ(H.row(0).nonZeros(), 1);
  EXPECT_DOUBLE_EQ(H.coeff(0, b.nodeIndex(8, 16)), 1.0);
  EXPECT_DOUBLE_EQ(H.coeff(1, b.nodeIndex(2, 7)), 1.0);
}

TEST(Lifecycle, BilinearWeightsOffGrid) {
  const gm::GmrfBelief b(smallConfig());
  // (0.3, 1.1): cell (1, 4), u = 0.2, v = 0.4
  const auto H = b.interpolationMatrix(points({{0.3, 1.1}}));
  EXPECT_NEAR(H.coeff(0, b.nodeIndex(1, 4)), 0.8 * 0.6, 1e-12);
  EXPECT_NEAR(H.coeff(0, b.nodeIndex(2, 4)), 0.2 * 0.6, 1e-12);
  EXPECT_NEAR(H.coeff(0, b.nodeIndex(1, 5)), 0.8 * 0.4, 1e-12);
  EXPECT_NEAR(H.coeff(0, b.nodeIndex(2, 5)), 0.2 * 0.4, 1e-12);
}

TEST(Lifecycle, EmptyAndDuplicateQueries) {
  gm::GmrfBelief b(smallConfig());
  b.addMeasurement(1.2, 2.2, 1.0);
  b.update();
  Eigen::VectorXd mean;
  Eigen::VectorXd var;
  Eigen::MatrixXd cov;
  b.queryMeanVariance(Eigen::MatrixX2d(0, 2), mean, var);
  EXPECT_EQ(mean.size(), 0);
  EXPECT_EQ(var.size(), 0);
  b.queryMeanCovariance(Eigen::MatrixX2d(0, 2), mean, cov);
  EXPECT_EQ(cov.rows(), 0);
  EXPECT_EQ(cov.cols(), 0);

  b.queryMeanCovariance(points({{0.9, 1.7}, {0.9, 1.7}}), mean, cov);
  EXPECT_DOUBLE_EQ(mean(0), mean(1));
  EXPECT_NEAR(cov(0, 1), cov(0, 0), 1e-14);
  EXPECT_NEAR(cov(1, 1), cov(0, 0), 1e-14);
}

TEST(Lifecycle, NonzeroNegativeOriginAndNonSquareCells) {
  gm::GmrfConfig c = smallConfig();
  c.x_min = -3.0;
  c.x_max = -1.0;
  c.y_min = -5.0;
  c.y_max = -2.0;
  c.nx = 5;   // dx = 0.5
  c.ny = 13;  // dy = 0.25
  gm::GmrfBelief b(c);
  EXPECT_DOUBLE_EQ(b.dx(), 0.5);
  EXPECT_DOUBLE_EQ(b.dy(), 0.25);
  b.addMeasurement(-1.0, -2.0, 2.0);
  b.addMeasurement(-3.0, -5.0, 0.0);
  b.addMeasurement(-2.2, -3.3, 1.0);
  b.update();
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
  b.queryMeanCovariance(points({{-1.0, -2.0}, {-2.2, -3.3}, {-3.0, -2.0}}),
                        mean, cov);
  EXPECT_TRUE(mean.allFinite());
  EXPECT_GT(cov(0, 0), 0.0);
  EXPECT_THROW(b.addMeasurement(0.0, -3.0, 1.0), std::invalid_argument);
}

TEST(Lifecycle, CopiesAreIndependent) {
  gm::GmrfBelief a(smallConfig());
  a.addMeasurement(1.0, 1.0, 2.0);
  a.update();
  gm::GmrfBelief b = a;
  b.addMeasurement(1.5, 3.0, -2.0);
  b.update();
  EXPECT_EQ(a.numMeasurements(), 1u);
  EXPECT_EQ(b.numMeasurements(), 2u);
  EXPECT_NE(a.fieldNodeMean(), b.fieldNodeMean());
}

TEST(Lifecycle, ChunkedQueriesAreConsistentAcrossChunkBoundaries) {
  gm::GmrfBelief b(smallConfig(), {{1.1, 0.6, 1.1, 3.4}});
  b.addMeasurement(0.4, 1.3, 2.0);
  b.addMeasurement(1.7, 2.9, -1.0);
  b.update();
  // 150 points > 2 chunks of 64 columns, deterministic pseudo-random positions.
  Eigen::MatrixX2d q(150, 2);
  for (Eigen::Index k = 0; k < q.rows(); ++k) {
    q(k, 0) = std::fmod(0.137 * static_cast<double>(k * k + 3), 2.0);
    q(k, 1) = std::fmod(0.291 * static_cast<double>(7 * k + 1), 4.0);
  }
  Eigen::VectorXd mean;
  Eigen::VectorXd var;
  Eigen::MatrixXd cov;
  b.queryMeanVariance(q, mean, var);
  b.queryMeanCovariance(q, mean, cov);
  EXPECT_LT((cov.diagonal() - var).cwiseAbs().maxCoeff(),
            1e-12 * var.maxCoeff());
  EXPECT_EQ((cov - cov.transpose()).cwiseAbs().maxCoeff(), 0.0);
  // Same point alone vs. inside the batch (last chunk).
  Eigen::VectorXd m1;
  Eigen::VectorXd v1;
  b.queryMeanVariance(q.row(140), m1, v1);
  EXPECT_NEAR(v1(0), var(140), 1e-14 * var(140));
  EXPECT_DOUBLE_EQ(m1(0), mean(140));
  // PSD within tolerance.
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
  EXPECT_GT(es.eigenvalues().minCoeff(), -1e-12 * es.eigenvalues().maxCoeff());
}
