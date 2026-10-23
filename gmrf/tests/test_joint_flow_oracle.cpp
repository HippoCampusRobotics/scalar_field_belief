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

// Joint mode with the nonlinear flow_field energy (plus flow_neighbors,
// flow_conservation, obstacle_flow and a partial wall) on a tiny grid.
// An independent dense residual/Jacobian oracle (built from the model
// equations in gmrf/README.md, not from precisionMatrix()) checks
//   * the returned state is a Gauss-Newton stationary point of the oracle cost,
//   * the reported final cost equals the oracle cost at that state,
//   * query covariance == H_c (J^T W J)^{-1} H_c^T evaluated at the FINAL state
//     (and differs measurably from the same formula at another point).
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "dense_oracle.hpp"
#include "gmrf_test_utils.hpp"

namespace gm = scalar_field_belief::gmrf;
using gmrf_test::points;

namespace {

// 4 x 3 nodes on [0, 1.5] x [0, 1.2]: dx = 0.5, dy = 0.6.
gm::GmrfConfig jointCfg() {
  gm::GmrfConfig c;
  c.x_min = 0.0;
  c.x_max = 1.5;
  c.y_min = 0.0;
  c.y_max = 1.2;
  c.nx = 4;
  c.ny = 3;
  c.prior_mean = 0.2;
  c.prior_sigma = 2.0;
  c.sigma_p_per_m = 2.0;  // 0.5
  c.sigma_s = 0.1;
  c.sigma_flow = 0.5;
  c.ff_prior_mean_x = 0.3;
  c.ff_prior_mean_y = -0.1;
  c.ff_prior_sigma = 1.0;
  c.ff_sigma_p_per_m = 2.0;
  c.ff_sigma_s = 0.1;
  c.ff_sigma_c = 0.5;
  c.ff_sigma_o = 0.3;
  c.flow_neighbors = true;
  c.flow_conservation = true;
  c.flow_field = true;
  c.obstacle_flow = true;
  c.max_gauss_newton_iterations = 100;
  return c;
}

// Cuts h edges (1,0) and (1,1) (x in [0.5, 1.0] at y = 0, 0.6).
const std::vector<gm::WallSegment> kWall{{0.7, -0.5, 0.7, 0.9}};

struct FieldObs {
  double x, y, v;
};
struct FlowObs {
  double x, y;
  int comp;
  double v;
};  // comp 0 = x, 1 = y

const std::vector<FieldObs> kField{
    {0.2, 0.2, 1.5}, {1.3, 1.0, -0.5}, {0.9, 0.4, 0.8}, {0.45, 1.1, 0.1}};
const std::vector<FlowObs> kFlow{{0.6, 0.9, 0, 0.8}, {1.2, 0.3, 1, 0.4}};

/// Dense residual model r(x) = f(x) - y with weights w (one row per energy
/// term).
struct JointOracle {
  oracle::Model m;
  int N;
  explicit JointOracle(const gm::GmrfConfig& c) : m(c, kWall), N(m.N) {}

  void eval(const Eigen::VectorXd& x, Eigen::VectorXd& f, Eigen::VectorXd& y,
            Eigen::VectorXd& w, Eigen::MatrixXd& J) const {
    const auto& c = m.c;
    std::vector<Eigen::RowVectorXd> rows;
    std::vector<double> fv, yv, wv;
    auto add = [&](const Eigen::RowVectorXd& jrow, double fval, double target,
                   double info) {
      rows.push_back(jrow);
      fv.push_back(fval);
      yv.push_back(target);
      wv.push_back(info);
    };
    auto lin = [&](const Eigen::RowVectorXd& jrow, double target, double info) {
      add(jrow, jrow.dot(x), target, info);
    };
    const int S = 3 * N, ox = N, oy = 2 * N;
    auto zero = [&]() -> Eigen::RowVectorXd {
      return Eigen::RowVectorXd::Zero(S);
    };
    const double sp = c.sigma_p_per_m * m.dx * m.dx;
    const double fsp = c.ff_sigma_p_per_m * m.dx * m.dx;
    // Edge list: (a, b, cut)
    struct E {
      int a, b;
      bool cut;
    };
    std::vector<E> edges;
    for (int j = 0; j < c.ny; ++j) {
      for (int i = 0; i + 1 < c.nx; ++i) {
        edges.push_back({m.id(i, j), m.id(i + 1, j), m.hCut(i, j)});
      }
    }
    for (int j = 0; j + 1 < c.ny; ++j) {
      for (int i = 0; i < c.nx; ++i) {
        edges.push_back({m.id(i, j), m.id(i, j + 1), m.vCut(i, j)});
      }
    }
    for (const E& e : edges) {  // FieldN
      if (e.cut) {
        continue;
      }
      auto r = zero();
      r(e.a) = 1;
      r(e.b) = -1;
      lin(r, 0.0, 1.0 / (sp * sp));
    }
    for (const E& e : edges) {  // FlowN (both rows of the cut edge removed)
      if (e.cut) {
        continue;
      }
      auto r = zero();
      r(ox + e.a) = 1;
      r(ox + e.b) = -1;
      lin(r, 0.0, 1.0 / (fsp * fsp));
      auto q = zero();
      q(oy + e.a) = 1;
      q(oy + e.b) = -1;
      lin(q, 0.0, 1.0 / (fsp * fsp));
    }
    for (int j = 0; j + 1 < c.ny; ++j) {  // FlowC
      for (int i = 0; i + 1 < c.nx; ++i) {
        auto r = zero();
        r(ox + m.id(i, j)) += 1;
        r(oy + m.id(i, j)) += 1;
        r(ox + m.id(i + 1, j)) -= 1;
        r(oy + m.id(i + 1, j)) += 1;
        r(ox + m.id(i, j + 1)) += 1;
        r(oy + m.id(i, j + 1)) -= 1;
        r(ox + m.id(i + 1, j + 1)) -= 1;
        r(oy + m.id(i + 1, j + 1)) -= 1;
        lin(r, 0.0, 1.0 / (c.ff_sigma_c * c.ff_sigma_c));
      }
    }
    for (int j = 0; j < c.ny; ++j) {  // FlowField (bilinear)
      for (int i = 0; i < c.nx; ++i) {
        const bool ax = i + 1 < c.nx && !m.hCut(i, j);
        const bool ay = j + 1 < c.ny && !m.vCut(i, j);
        if (!ax && !ay) {
          continue;
        }
        const int n = m.id(i, j);
        auto r = zero();
        double fval = 0.0;
        if (ax) {
          const int nb = m.id(i + 1, j);
          fval += (x(nb) - x(n)) * x(ox + n);
          r(ox + n) += x(nb) - x(n);
          r(nb) += x(ox + n);
          r(n) -= x(ox + n);
        }
        if (ay) {
          const int nb = m.id(i, j + 1);
          fval += (x(nb) - x(n)) * x(oy + n);
          r(oy + n) += x(nb) - x(n);
          r(nb) += x(oy + n);
          r(n) -= x(oy + n);
        }
        add(r, fval, 0.0, 1.0 / (c.sigma_flow * c.sigma_flow));
      }
    }
    for (int j = 0; j < c.ny;
         ++j) {  // ObstacleFlow (no boundary walls -> flags 0)
      for (int i = 0; i < c.nx; ++i) {
        const int n = m.id(i, j);
        const bool cuts[4] = {
            i + 1 < c.nx && m.hCut(i, j), i > 0 && m.hCut(i - 1, j),
            j + 1 < c.ny && m.vCut(i, j), j > 0 && m.vCut(i, j - 1)};
        for (int k = 0; k < 4; ++k) {
          if (!cuts[k]) {
            continue;
          }
          auto r = zero();
          r((k < 2 ? ox : oy) + n) = 1;
          lin(r, 0.0, 1.0 / (c.ff_sigma_o * c.ff_sigma_o));
        }
      }
    }
    for (int n = 0; n < N; ++n) {  // priors
      auto r = zero();
      r(n) = 1;
      lin(r, c.prior_mean, 1.0 / (c.prior_sigma * c.prior_sigma));
    }
    for (int n = 0; n < N; ++n) {
      auto r = zero();
      r(ox + n) = 1;
      lin(r, c.ff_prior_mean_x, 1.0 / (c.ff_prior_sigma * c.ff_prior_sigma));
      auto q = zero();
      q(oy + n) = 1;
      lin(q, c.ff_prior_mean_y, 1.0 / (c.ff_prior_sigma * c.ff_prior_sigma));
    }
    for (const auto& o : kField) {
      auto r = zero();
      r.head(N) = m.hRow(o.x, o.y);
      lin(r, o.v, 1.0 / (c.sigma_s * c.sigma_s));
    }
    for (const auto& o : kFlow) {
      auto r = zero();
      r.segment(o.comp == 0 ? ox : oy, N) = m.hRow(o.x, o.y);
      lin(r, o.v, 1.0 / (c.ff_sigma_s * c.ff_sigma_s));
    }
    const auto R = static_cast<Eigen::Index>(rows.size());
    J.resize(R, S);
    f.resize(R);
    y.resize(R);
    w.resize(R);
    for (Eigen::Index k = 0; k < R; ++k) {
      J.row(k) = rows[k];
      f(k) = fv[k];
      y(k) = yv[k];
      w(k) = wv[k];
    }
  }
};

}  // namespace

TEST(JointFlowOracle, GaussNewtonStationaryPointAndFinalPointInformation) {
  const gm::GmrfConfig c = jointCfg();
  gm::GmrfBelief b(c, kWall);
  ASSERT_TRUE(b.isJointState());
  ASSERT_FALSE(b.isLinear());
  const JointOracle o(c);
  ASSERT_TRUE(o.m.hCut(1, 0) && o.m.hCut(1, 1) && !o.m.hCut(1, 2));
  for (int j = 0; j < c.ny; ++j) {
    for (int i = 0; i + 1 < c.nx; ++i) {
      ASSERT_EQ(b.isHorizontalEdgeCut(i, j), o.m.hCut(i, j));
    }
  }

  // Streaming: two batches, the second warm-started from the first.
  b.addMeasurement(kField[0].x, kField[0].y, kField[0].v);
  b.addFlowMeasurement(kFlow[0].x, kFlow[0].y, gm::FlowComponent::X,
                       kFlow[0].v);
  const gm::UpdateReport r1 = b.update();
  for (std::size_t k = 1; k < kField.size(); ++k) {
    b.addMeasurement(kField[k].x, kField[k].y, kField[k].v);
  }
  b.addFlowMeasurement(kFlow[1].x, kFlow[1].y, gm::FlowComponent::Y,
                       kFlow[1].v);
  const gm::UpdateReport r2 = b.update();
  std::cout << "[joint] batch1: iterations " << r1.iterations << " converged "
            << r1.converged << " cost " << r1.final_cost
            << "\n[joint] batch2: iterations " << r2.iterations << " converged "
            << r2.converged << " cost " << r2.final_cost << "\n";
  EXPECT_FALSE(r2.linear);
  EXPECT_TRUE(r1.converged);
  EXPECT_TRUE(r2.converged);
  EXPECT_GE(r2.iterations, 1);
  EXPECT_LT(r2.iterations, c.max_gauss_newton_iterations);

  const Eigen::VectorXd xs = b.stateMean();
  ASSERT_EQ(xs.size(), 3 * o.N);
  Eigen::VectorXd f, y, w;
  Eigen::MatrixXd J;
  o.eval(xs, f, y, w, J);
  const double cost = (w.array() * (f - y).array().square()).sum();
  EXPECT_NEAR(r2.final_cost, cost, 1e-8 * cost);
  const Eigen::MatrixXd Q = J.transpose() * w.asDiagonal() * J;
  const Eigen::VectorXd g = J.transpose() * (w.asDiagonal() * (f - y));
  const Eigen::LLT<Eigen::MatrixXd> llt(Q);
  ASSERT_EQ(llt.info(), Eigen::Success);
  const double step = llt.solve(g).norm();
  std::cout << "[joint] oracle cost " << cost << " |grad| " << g.norm()
            << " |GN step| " << step << " flow_x(0) " << xs(o.N)
            << " flow_y(0) " << xs(2 * o.N) << "\n";
  EXPECT_LT(step, 1e-3);  // local GN stationary point (gn tolerances are 1e-5)

  // A full Gauss-Newton restart from the result stays put (cost cannot drop
  // noticeably) -- solved independently with the dense oracle.
  {
    Eigen::VectorXd x2 = xs - llt.solve(g);
    Eigen::VectorXd f2, y2, w2;
    Eigen::MatrixXd J2;
    o.eval(x2, f2, y2, w2, J2);
    const double cost2 = (w2.array() * (f2 - y2).array().square()).sum();
    std::cout << "[joint] cost after one more dense GN step " << cost2 << "\n";
    EXPECT_GT(cost2, cost * (1.0 - 1e-6));
  }

  // Query covariance = H_c Q(x*)^{-1} H_c^T at the final point.
  const Eigen::MatrixX2d xy = points({{0.2, 0.2},
                                      {0.6, 0.3},
                                      {0.8, 0.3},
                                      {1.5, 1.2},
                                      {0.0, 0.0},
                                      {1.1, 0.9},
                                      {0.7, 1.2}});
  Eigen::MatrixXd Hs = Eigen::MatrixXd::Zero(xy.rows(), 3 * o.N);
  Hs.leftCols(o.N) = o.m.H(xy);
  const Eigen::MatrixXd cov_ref = Hs * llt.solve(Hs.transpose());
  Eigen::VectorXd mean, var;
  Eigen::MatrixXd cov;
  b.queryMeanCovariance(xy, mean, cov);
  b.queryMeanVariance(xy, mean, var);
  const double dc = oracle::relDiff(cov, cov_ref);
  std::cout << "[joint] rel cov vs oracle J^T W J at final point " << dc
            << "\n";
  EXPECT_LT(dc, 1e-8);
  EXPECT_LT(oracle::relDiff(var, cov.diagonal()), 1e-10);
  EXPECT_LT(oracle::relDiff(mean, Hs * xs), 1e-12);

  // Sensitivity: the same formula at a different linearisation point
  // (state after the first batch) is measurably different.
  Eigen::VectorXd xp = xs;
  xp.segment(o.N, 2 * o.N) *= 1.5;
  o.eval(xp, f, y, w, J);
  const Eigen::MatrixXd Qp = J.transpose() * w.asDiagonal() * J;
  const Eigen::MatrixXd cov_p = Hs * Qp.llt().solve(Hs.transpose());
  std::cout << "[joint] rel cov at perturbed point "
            << oracle::relDiff(cov_p, cov_ref) << "\n";
  EXPECT_GT(oracle::relDiff(cov_p, cov_ref), 1e-4);
}
