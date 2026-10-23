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

// Independent dense reference ("oracle") for the GMRF tests.
//
// Everything here is rebuilt from the model equations in gmrf/README.md,
// WITHOUT calling the production precisionMatrix(),
// interpolationMatrix(), isHorizontal/VerticalEdgeCut() or nodePosition():
//   * grid nodes  x_i = x_min + i*dx, y_j = y_min + j*dy, n = j*nx + i
//   * FieldN      per uncut grid edge (a,b): info 1/(sigma_p_per_m*dx^2)^2,
//                 Q += info (e_a - e_b)(e_a - e_b)^T  (dx used for BOTH
//                 directions)
//   * prior       Q += I/prior_sigma^2, b += prior_mean/prior_sigma^2
//   * measurement Q += h h^T/sigma_s^2, b += h v/sigma_s^2
//   * H           bilinear weights of the cell's 4 corners; corners hidden by a
//                 wall are dropped and the remaining weights renormalised
//   * edge cut    edge segment properly intersects an (endpoint-extended) wall
// The segment test below is a parametric (Cramer's rule) intersection, which
// is deliberately a different formulation from the production orientation
// predicate. Test geometries keep clear of degenerate (touching) cases.
#ifndef SCALAR_FIELD_BELIEF_GMRF_TESTS_DENSE_ORACLE_HPP
#define SCALAR_FIELD_BELIEF_GMRF_TESTS_DENSE_ORACLE_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "scalar_field_belief/gmrf/gmrf_belief.hpp"

namespace oracle {

namespace gm = scalar_field_belief::gmrf;

struct Seg {
  double x0, y0, x1, y1;
};

/// Parametric intersection of p0-p1 and q0-q1 (both closed, non-parallel).
inline bool segIntersect(const Seg& p, const Seg& q) {
  const double rx = p.x1 - p.x0, ry = p.y1 - p.y0;
  const double sx = q.x1 - q.x0, sy = q.y1 - q.y0;
  const double den = rx * sy - ry * sx;
  if (den == 0.0) {
    return false;  // parallel: test geometries avoid collinear overlaps
  }
  const double qpx = q.x0 - p.x0, qpy = q.y0 - p.y0;
  const double t = (qpx * sy - qpy * sx) / den;  // along p
  const double u = (qpx * ry - qpy * rx) / den;  // along q
  return t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0;
}

/// Wall extended by e beyond both endpoints along its direction.
inline Seg extend(const gm::WallSegment& w, double e) {
  const double len = std::hypot(w.x1 - w.x0, w.y1 - w.y0);
  const double ux = (w.x1 - w.x0) / len, uy = (w.y1 - w.y0) / len;
  return Seg{w.x0 - e * ux, w.y0 - e * uy, w.x1 + e * ux, w.y1 + e * uy};
}

struct Model {
  gm::GmrfConfig c;
  std::vector<Seg> walls;  // extended
  double dx = 0.0, dy = 0.0;
  int N = 0;

  Model(const gm::GmrfConfig& cfg, const std::vector<gm::WallSegment>& w = {})
      : c(cfg) {
    dx = (c.x_max - c.x_min) / (c.nx - 1);
    dy = (c.y_max - c.y_min) / (c.ny - 1);
    N = c.nx * c.ny;
    for (const auto& s : w) {
      walls.push_back(extend(s, c.wall_endpoint_extension));
    }
  }

  int id(int i, int j) const { return j * c.nx + i; }
  double X(int i) const { return c.x_min + i * dx; }
  double Y(int j) const { return c.y_min + j * dy; }

  bool blocked(double px, double py, double qx, double qy) const {
    for (const auto& w : walls) {
      if (segIntersect(Seg{px, py, qx, qy}, w)) {
        return true;
      }
    }
    return false;
  }
  bool hCut(int i, int j) const { return blocked(X(i), Y(j), X(i + 1), Y(j)); }
  bool vCut(int i, int j) const { return blocked(X(i), Y(j), X(i), Y(j + 1)); }

  /// Dense interpolation row (size N) with wall masking + renormalisation.
  Eigen::RowVectorXd hRow(double x, double y) const {
    int i0 = static_cast<int>(std::floor((x - c.x_min) / dx));
    int j0 = static_cast<int>(std::floor((y - c.y_min) / dy));
    i0 = std::clamp(i0, 0, c.nx - 2);
    j0 = std::clamp(j0, 0, c.ny - 2);
    const double u = (x - X(i0)) / dx, v = (y - Y(j0)) / dy;
    const int ci[4] = {i0, i0 + 1, i0, i0 + 1};
    const int cj[4] = {j0, j0, j0 + 1, j0 + 1};
    const double cw[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    Eigen::RowVectorXd h = Eigen::RowVectorXd::Zero(N);
    double sum = 0.0;
    for (int k = 0; k < 4; ++k) {
      if (cw[k] <= 0.0) {
        continue;
      }
      if (blocked(x, y, X(ci[k]), Y(cj[k]))) {
        continue;
      }
      h(id(ci[k], cj[k])) += cw[k];
      sum += cw[k];
    }
    return h / sum;
  }

  Eigen::MatrixXd H(const Eigen::MatrixX2d& xy) const {
    Eigen::MatrixXd out(xy.rows(), N);
    for (Eigen::Index r = 0; r < xy.rows(); ++r) {
      out.row(r) = hRow(xy(r, 0), xy(r, 1));
    }
    return out;
  }

  /// Prior + FieldN precision and information vector.
  void priorSystem(Eigen::MatrixXd& Q, Eigen::VectorXd& b) const {
    Q = Eigen::MatrixXd::Identity(N, N) / (c.prior_sigma * c.prior_sigma);
    b = Eigen::VectorXd::Constant(
        N, c.prior_mean / (c.prior_sigma * c.prior_sigma));
    if (!c.field_neighbors) {
      return;
    }
    const double sp = c.sigma_p_per_m * dx * dx;
    const double info = 1.0 / (sp * sp);
    auto edge = [&](int a, int bb) {
      Q(a, a) += info;
      Q(bb, bb) += info;
      Q(a, bb) -= info;
      Q(bb, a) -= info;
    };
    for (int j = 0; j < c.ny; ++j) {
      for (int i = 0; i + 1 < c.nx; ++i) {
        if (!hCut(i, j)) {
          edge(id(i, j), id(i + 1, j));
        }
      }
    }
    for (int j = 0; j + 1 < c.ny; ++j) {
      for (int i = 0; i < c.nx; ++i) {
        if (!vCut(i, j)) {
          edge(id(i, j), id(i, j + 1));
        }
      }
    }
  }
};

/// Dense scalar posterior from the model equations.
struct Posterior {
  Eigen::MatrixXd Q;
  Eigen::VectorXd mean;
  Eigen::MatrixXd Sigma;  // Q^{-1}
};

inline Posterior scalarPosterior(
    const Model& m,
    const std::vector<std::pair<std::pair<double, double>, double>>& obs) {
  Posterior p;
  Eigen::VectorXd b;
  m.priorSystem(p.Q, b);
  const double is2 = 1.0 / (m.c.sigma_s * m.c.sigma_s);
  for (const auto& o : obs) {
    const Eigen::RowVectorXd h = m.hRow(o.first.first, o.first.second);
    p.Q += is2 * h.transpose() * h;
    b += is2 * o.second * h.transpose();
  }
  Eigen::LLT<Eigen::MatrixXd> llt(p.Q);
  p.mean = llt.solve(b);
  p.Sigma = llt.solve(Eigen::MatrixXd::Identity(m.N, m.N));
  return p;
}

/// max |A - B| / max(|B|_max, floor)
inline double relDiff(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                      double floor = 1e-12) {
  if (A.size() == 0 && B.size() == 0) {
    return 0.0;
  }
  const double scale = std::max(B.cwiseAbs().maxCoeff(), floor);
  return (A - B).cwiseAbs().maxCoeff() / scale;
}

}  // namespace oracle

#endif  // SCALAR_FIELD_BELIEF_GMRF_TESTS_DENSE_ORACLE_HPP
