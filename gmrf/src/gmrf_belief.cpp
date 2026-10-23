// Copyright (C) 2023 Finn Lukas Busch
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

#include "scalar_field_belief/gmrf/gmrf_belief.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "geometry.hpp"

namespace scalar_field_belief::gmrf {

using SpMat = Eigen::SparseMatrix<double>;
using Solver =
    Eigen::SimplicialLLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<int>>;

struct GmrfBelief::Posterior {
  Eigen::VectorXd mean;
  SpMat precision;
  Solver factor;
  double time = 0.0;
};

struct GmrfBelief::System {
  std::vector<Eigen::Triplet<double>> triplets;
  std::vector<double> target;
  std::vector<double> info;
  std::vector<std::pair<Eigen::Index, double>> nonlinear;  // (row, f value)
  SpMat J;
  Eigen::VectorXd f;
  Eigen::VectorXd y;
  Eigen::VectorXd w;
};

namespace {

constexpr Eigen::Index kQueryChunk = 64;
constexpr double kMinWeightSum = 1e-12;

std::string fmtPoint(double x, double y) {
  std::ostringstream s;
  s.precision(9);
  s << "(" << x << ", " << y << ")";
  return s.str();
}

double nodeCoord(double lo, double hi, int k, int n) {
  if (k == n - 1) {
    return hi;
  }
  return lo + (hi - lo) * static_cast<double>(k) / static_cast<double>(n - 1);
}

int clampIndex(double v, int lo, int hi) {
  if (!(v >= lo)) {  // also catches NaN
    return lo;
  }
  if (v > hi) {
    return hi;
  }
  return static_cast<int>(v);
}

SpMat normalMatrix(const SpMat& J, const Eigen::VectorXd& w) {
  const Eigen::VectorXd sw = w.cwiseSqrt();
  const SpMat A = sw.asDiagonal() * J;
  SpMat Q = SpMat(A.transpose()) * A;
  Q.makeCompressed();
  return Q;
}

double costOf(const Eigen::VectorXd& f, const Eigen::VectorXd& y,
              const Eigen::VectorXd& w) {
  return (w.array() * (f - y).array().square()).sum();
}

void factorOrThrow(Solver& solver, const SpMat& Q) {
  solver.compute(Q);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "GmrfBelief: sparse Cholesky factorisation of the precision matrix "
        "failed.");
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction and geometry
// ---------------------------------------------------------------------------

GmrfBelief::GmrfBelief(const GmrfConfig& config, std::vector<WallSegment> walls,
                       std::vector<Point2> free_space_seeds)
    : config_(config), walls_(std::move(walls)) {
  config_.validate();
  for (std::size_t k = 0; k < walls_.size(); ++k) {
    const WallSegment& w = walls_[k];
    if (!std::isfinite(w.x0) || !std::isfinite(w.y0) || !std::isfinite(w.x1) ||
        !std::isfinite(w.y1)) {
      throw std::invalid_argument("GmrfBelief: wall " + std::to_string(k) +
                                  " has non-finite coordinates.");
    }
    const double len = std::hypot(w.x1 - w.x0, w.y1 - w.y0);
    if (!(len > 0.0) || !std::isfinite(len)) {
      throw std::invalid_argument("GmrfBelief: wall " + std::to_string(k) +
                                  " from " + fmtPoint(w.x0, w.y0) + " to " +
                                  fmtPoint(w.x1, w.y1) +
                                  " has zero (or non-finite) length.");
    }
  }
  for (std::size_t k = 0; k < free_space_seeds.size(); ++k) {
    const Point2& s = free_space_seeds[k];
    if (!std::isfinite(s.x) || !std::isfinite(s.y) || s.x < config_.x_min ||
        s.x > config_.x_max || s.y < config_.y_min || s.y > config_.y_max) {
      throw std::invalid_argument(
          "GmrfBelief: free-space seed " + std::to_string(k) + " at " +
          fmtPoint(s.x, s.y) + " is non-finite or outside the domain.");
    }
  }

  dx_ = (config_.x_max - config_.x_min) / (config_.nx - 1);
  dy_ = (config_.y_max - config_.y_min) / (config_.ny - 1);
  num_nodes_ = static_cast<Eigen::Index>(config_.nx) * config_.ny;
  joint_ = config_.usesJointState();
  state_size_ = joint_ ? 3 * num_nodes_ : num_nodes_;

  buildGeometry(free_space_seeds);

  UpdateReport report;
  // Initial linearisation point is zero, as in the source (all_estimates = 0).
  prior_ = solve({}, 0.0, Eigen::VectorXd::Zero(state_size_), report);
  posterior_ = prior_;
}

void GmrfBelief::buildGeometry(const std::vector<Point2>& seeds) {
  const int nx = config_.nx;
  const int ny = config_.ny;
  const double span =
      std::max(config_.x_max - config_.x_min, config_.y_max - config_.y_min);
  const double tol = 1e-9 * span;    // "lies on the domain boundary" tolerance
  const double delta = 1e-9 * span;  // tie-break translation of every wall

  cut_h_.assign(static_cast<std::size_t>(nx - 1) * ny, 0);
  cut_v_.assign(static_cast<std::size_t>(nx) * (ny - 1), 0);
  x_wall_.assign(static_cast<std::size_t>(num_nodes_), 0);
  y_wall_.assign(static_cast<std::size_t>(num_nodes_), 0);
  active_.assign(static_cast<std::size_t>(num_nodes_), 1);
  cutting_.clear();

  // Always extend by a few delta beyond the tie-break translation so that a
  // wall ending exactly on a boundary row/column still reaches it after the
  // shift (otherwise a separator could leak along the domain edge when
  // wall_endpoint_extension = 0).
  const double e = std::max(config_.wall_endpoint_extension, 0.0) + 2.0 * delta;
  for (const WallSegment& w : walls_) {
    const double len = std::hypot(w.x1 - w.x0, w.y1 - w.y0);
    const double ux = (w.x1 - w.x0) / len;
    const double uy = (w.y1 - w.y0) / len;
    const WallSegment ext{w.x0 - e * ux, w.y0 - e * uy, w.x1 + e * ux,
                          w.y1 + e * uy};

    const bool vertical = std::abs(w.x1 - w.x0) <= tol;
    const bool horizontal = std::abs(w.y1 - w.y0) <= tol;
    const bool on_xmin = vertical && std::abs(w.x0 - config_.x_min) <= tol;
    const bool on_xmax = vertical && std::abs(w.x0 - config_.x_max) <= tol;
    const bool on_ymin = horizontal && std::abs(w.y0 - config_.y_min) <= tol;
    const bool on_ymax = horizontal && std::abs(w.y0 - config_.y_max) <= tol;

    if (on_xmin || on_xmax) {
      // Wall along a vertical domain boundary: cuts nothing, flags nodes.
      const int col = on_xmin ? 0 : nx - 1;
      const double lo = std::min(ext.y0, ext.y1) - tol;
      const double hi = std::max(ext.y0, ext.y1) + tol;
      for (int j = 0; j < ny; ++j) {
        const double y = nodeCoord(config_.y_min, config_.y_max, j, ny);
        if (y >= lo && y <= hi) {
          x_wall_[static_cast<std::size_t>(nid(col, j))] = 1;
        }
      }
      continue;
    }
    if (on_ymin || on_ymax) {
      const int row = on_ymin ? 0 : ny - 1;
      const double lo = std::min(ext.x0, ext.x1) - tol;
      const double hi = std::max(ext.x0, ext.x1) + tol;
      for (int i = 0; i < nx; ++i) {
        const double x = nodeCoord(config_.x_min, config_.x_max, i, nx);
        if (x >= lo && x <= hi) {
          y_wall_[static_cast<std::size_t>(nid(i, row))] = 1;
        }
      }
      continue;
    }
    // Tie-break: translate by (delta, 1e-3 * delta). A point or node lying
    // exactly on a wall is therefore on the wall's -x side (-y side for
    // horizontal walls), i.e. the source's floor rule for axis-aligned walls.
    cutting_.push_back(WallSegment{ext.x0 + delta, ext.y0 + 1e-3 * delta,
                                   ext.x1 + delta, ext.y1 + 1e-3 * delta});
  }

  // Edge cuts: an edge is cut iff its segment intersects a (translated) wall.
  for (const WallSegment& w : cutting_) {
    const int i_lo = clampIndex(
        std::floor((std::min(w.x0, w.x1) - config_.x_min) / dx_) - 1.0, 0,
        nx - 1);
    const int i_hi = clampIndex(
        std::floor((std::max(w.x0, w.x1) - config_.x_min) / dx_) + 1.0, 0,
        nx - 1);
    const int j_lo = clampIndex(
        std::floor((std::min(w.y0, w.y1) - config_.y_min) / dy_) - 1.0, 0,
        ny - 1);
    const int j_hi = clampIndex(
        std::floor((std::max(w.y0, w.y1) - config_.y_min) / dy_) + 1.0, 0,
        ny - 1);
    for (int j = j_lo; j <= j_hi; ++j) {
      const double y = nodeCoord(config_.y_min, config_.y_max, j, ny);
      for (int i = i_lo; i <= std::min(i_hi, nx - 2); ++i) {
        auto& c = cut_h_[hEdge(i, j)];
        if (!c &&
            detail::segmentsIntersect(
                nodeCoord(config_.x_min, config_.x_max, i, nx), y,
                nodeCoord(config_.x_min, config_.x_max, i + 1, nx), y, w)) {
          c = 1;
        }
      }
    }
    for (int j = j_lo; j <= std::min(j_hi, ny - 2); ++j) {
      const double y0 = nodeCoord(config_.y_min, config_.y_max, j, ny);
      const double y1 = nodeCoord(config_.y_min, config_.y_max, j + 1, ny);
      for (int i = i_lo; i <= i_hi; ++i) {
        auto& c = cut_v_[vEdge(i, j)];
        const double x = nodeCoord(config_.x_min, config_.x_max, i, nx);
        if (!c && detail::segmentsIntersect(x, y0, x, y1, w)) {
          c = 1;
        }
      }
    }
  }

  // Optional explicit free-space mask (separate from robot reachability).
  if (!seeds.empty()) {
    std::vector<std::uint8_t> reached(static_cast<std::size_t>(num_nodes_), 0);
    std::vector<Eigen::Index> stack;
    for (std::size_t k = 0; k < seeds.size(); ++k) {
      Interp interp;
      const char* reason = nullptr;
      if (!tryInterpolate(seeds[k].x, seeds[k].y, interp, &reason)) {
        throw std::invalid_argument(
            "GmrfBelief: free-space seed " + std::to_string(k) + " at " +
            fmtPoint(seeds[k].x, seeds[k].y) +
            " cannot be associated with a grid node: " + reason);
      }
      for (int c = 0; c < interp.count; ++c) {
        if (!reached[static_cast<std::size_t>(interp.idx[c])]) {
          reached[static_cast<std::size_t>(interp.idx[c])] = 1;
          stack.push_back(interp.idx[c]);
        }
      }
    }
    while (!stack.empty()) {
      const Eigen::Index n = stack.back();
      stack.pop_back();
      const int i = static_cast<int>(n % nx);
      const int j = static_cast<int>(n / nx);
      auto visit = [&](int ii, int jj) {
        const auto m = static_cast<std::size_t>(nid(ii, jj));
        if (!reached[m]) {
          reached[m] = 1;
          stack.push_back(static_cast<Eigen::Index>(m));
        }
      };
      if (i + 1 < nx && !cut_h_[hEdge(i, j)]) {
        visit(i + 1, j);
      }
      if (i > 0 && !cut_h_[hEdge(i - 1, j)]) {
        visit(i - 1, j);
      }
      if (j + 1 < ny && !cut_v_[vEdge(i, j)]) {
        visit(i, j + 1);
      }
      if (j > 0 && !cut_v_[vEdge(i, j - 1)]) {
        visit(i, j - 1);
      }
    }
    active_ = std::move(reached);
    // Decouple inactive nodes (source: edges without a reachable endpoint are
    // cut).
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const bool a = active_[static_cast<std::size_t>(nid(i, j))] != 0;
        if (i + 1 < nx &&
            (!a || !active_[static_cast<std::size_t>(nid(i + 1, j))])) {
          cut_h_[hEdge(i, j)] = 1;
        }
        if (j + 1 < ny &&
            (!a || !active_[static_cast<std::size_t>(nid(i, j + 1))])) {
          cut_v_[vEdge(i, j)] = 1;
        }
      }
    }
  }
}

bool GmrfBelief::visible(double px, double py, double qx, double qy) const {
  if (px == qx && py == qy) {
    return true;
  }
  for (const WallSegment& w : cutting_) {
    if (detail::segmentsIntersect(px, py, qx, qy, w)) {
      return false;
    }
  }
  return true;
}

bool GmrfBelief::tryInterpolate(double x, double y, Interp& out,
                                const char** reason) const {
  out = Interp{};
  if (!std::isfinite(x) || !std::isfinite(y)) {
    *reason = "coordinates are not finite";
    return false;
  }
  if (x < config_.x_min || x > config_.x_max || y < config_.y_min ||
      y > config_.y_max) {
    *reason = "position is outside the domain";
    return false;
  }
  const int nx = config_.nx;
  const int ny = config_.ny;
  const int i0 = clampIndex(std::floor((x - config_.x_min) / dx_), 0, nx - 2);
  const int j0 = clampIndex(std::floor((y - config_.y_min) / dy_), 0, ny - 2);
  const double x0 = nodeCoord(config_.x_min, config_.x_max, i0, nx);
  const double y0 = nodeCoord(config_.y_min, config_.y_max, j0, ny);
  const double u = std::clamp((x - x0) / dx_, 0.0, 1.0);
  const double v = std::clamp((y - y0) / dy_, 0.0, 1.0);

  const int ci[4] = {i0, i0 + 1, i0, i0 + 1};
  const int cj[4] = {j0, j0, j0 + 1, j0 + 1};
  const double cw[4] = {(1.0 - u) * (1.0 - v), u * (1.0 - v), (1.0 - u) * v,
                        u * v};
  double sum = 0.0;
  for (int c = 0; c < 4; ++c) {
    if (!(cw[c] > 0.0)) {
      continue;
    }
    const Eigen::Index n = nid(ci[c], cj[c]);
    if (!active_[static_cast<std::size_t>(n)]) {
      continue;
    }
    if (!visible(x, y, nodeCoord(config_.x_min, config_.x_max, ci[c], nx),
                 nodeCoord(config_.y_min, config_.y_max, cj[c], ny))) {
      continue;
    }
    out.idx[out.count] = n;
    out.w[out.count] = cw[c];
    ++out.count;
    sum += cw[c];
  }
  if (!(sum > kMinWeightSum)) {
    out = Interp{};
    *reason = "all interpolation corners are hidden by walls or inactive";
    return false;
  }
  for (int c = 0; c < out.count; ++c) {
    out.w[c] /= sum;
  }
  return true;
}

GmrfBelief::Interp GmrfBelief::interpolateOrThrow(double x, double y,
                                                  const char* what) const {
  Interp interp;
  const char* reason = nullptr;
  if (!tryInterpolate(x, y, interp, &reason)) {
    throw std::invalid_argument(std::string("GmrfBelief: ") + what + " at " +
                                fmtPoint(x, y) + " rejected: " + reason + ".");
  }
  return interp;
}

bool GmrfBelief::isRepresentable(double x, double y) const {
  Interp interp;
  const char* reason = nullptr;
  return tryInterpolate(x, y, interp, &reason);
}

// ---------------------------------------------------------------------------
// Measurements and lifecycle
// ---------------------------------------------------------------------------

void GmrfBelief::reset() {
  measurements_.clear();
  pending_.clear();
  posterior_ = prior_;
}

void GmrfBelief::addMeasurementImpl(double x, double y, double value, double t,
                                    Kind kind) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("GmrfBelief: measurement value at " +
                                fmtPoint(x, y) + " is not finite.");
  }
  if (!std::isfinite(t)) {
    throw std::invalid_argument("GmrfBelief: measurement time is not finite.");
  }
  Measurement m;
  m.interp = interpolateOrThrow(x, y, "measurement");
  m.value = value;
  m.t = t;
  m.kind = kind;
  pending_.push_back(m);
}

void GmrfBelief::addMeasurement(double x, double y, double value, double t) {
  addMeasurementImpl(x, y, value, t, Kind::Field);
}

void GmrfBelief::addFlowMeasurement(double x, double y, FlowComponent component,
                                    double value, double t) {
  if (!joint_) {
    throw std::logic_error(
        "GmrfBelief: flow measurements require the joint state (enable a flow "
        "energy or "
        "force_joint_state).");
  }
  addMeasurementImpl(x, y, value, t,
                     component == FlowComponent::X ? Kind::FlowX : Kind::FlowY);
}

void GmrfBelief::discardPendingMeasurements() { pending_.clear(); }

UpdateReport GmrfBelief::update(double t) {
  UpdateReport report;
  if (!std::isfinite(t)) {
    throw std::invalid_argument("GmrfBelief: update time is not finite.");
  }
  const bool aging = config_.enable_measurement_aging;
  if (pending_.empty() && !aging) {
    report.linear = isLinear();
    return report;  // stationary model and nothing new: belief unchanged
  }
  if (aging) {
    for (const auto* list : {&measurements_, &pending_}) {
      for (const Measurement& m : *list) {
        if (t < m.t) {
          throw std::invalid_argument(
              "GmrfBelief: update time precedes a measurement time (aging "
              "enabled).");
        }
      }
    }
  }
  std::vector<Measurement> all;
  all.reserve(measurements_.size() + pending_.size());
  all.insert(all.end(), measurements_.begin(), measurements_.end());
  all.insert(all.end(), pending_.begin(), pending_.end());

  std::shared_ptr<const Posterior> post =
      solve(all, t, posterior_->mean, report);

  // Commit (no-throw operations only).
  report.solved = true;
  report.num_assimilated = pending_.size();
  measurements_.swap(all);
  pending_.clear();
  posterior_ = std::move(post);
  return report;
}

// ---------------------------------------------------------------------------
// Energy assembly
// ---------------------------------------------------------------------------

void GmrfBelief::assemble(const Eigen::VectorXd& x,
                          const std::vector<Measurement>& meas, double t,
                          System& sys) const {
  const int nx = config_.nx;
  const int ny = config_.ny;
  const Eigen::Index N = num_nodes_;
  const Eigen::Index ox = N;      // w_x block (joint state only)
  const Eigen::Index oy = 2 * N;  // w_y block (joint state only)

  sys.triplets.clear();
  sys.target.clear();
  sys.info.clear();
  sys.nonlinear.clear();
  Eigen::Index row = 0;
  auto addRow = [&](double target, double info) {
    sys.target.push_back(target);
    sys.info.push_back(info);
    return row++;
  };
  auto entry = [&](Eigen::Index r, Eigen::Index c, double v) {
    sys.triplets.emplace_back(static_cast<int>(r), static_cast<int>(c), v);
  };
  auto n = [&](int i, int j) { return nid(i, j); };

  // FieldN: c_a - c_b per uncut edge, info 1/sigma_p^2, sigma_p =
  // sigma_p_per_m*dx^2.
  if (config_.field_neighbors) {
    const double sp = fieldNeighborSigma();
    const double info = 1.0 / (sp * sp);
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i + 1 < nx; ++i) {
        if (!cut_h_[hEdge(i, j)]) {
          const auto r = addRow(0.0, info);
          entry(r, n(i, j), 1.0);
          entry(r, n(i + 1, j), -1.0);
        }
      }
    }
    for (int j = 0; j + 1 < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        if (!cut_v_[vEdge(i, j)]) {
          const auto r = addRow(0.0, info);
          entry(r, n(i, j), 1.0);
          entry(r, n(i, j + 1), -1.0);
        }
      }
    }
  }

  if (joint_) {
    // FlowN: w_x and w_y differences per uncut edge; a cut removes both rows
    // of THAT edge (corrected source offsets).
    if (config_.flow_neighbors) {
      const double sp = config_.ff_sigma_p_per_m * dx_ * dx_;
      const double info = 1.0 / (sp * sp);
      auto flowEdge = [&](Eigen::Index a, Eigen::Index b) {
        auto r = addRow(0.0, info);
        entry(r, ox + a, 1.0);
        entry(r, ox + b, -1.0);
        r = addRow(0.0, info);
        entry(r, oy + a, 1.0);
        entry(r, oy + b, -1.0);
      };
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i + 1 < nx; ++i) {
          if (!cut_h_[hEdge(i, j)]) {
            flowEdge(n(i, j), n(i + 1, j));
          }
        }
      }
      for (int j = 0; j + 1 < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          if (!cut_v_[vEdge(i, j)]) {
            flowEdge(n(i, j), n(i, j + 1));
          }
        }
      }
    }
    // FlowC: negated, unscaled cell divergence (ignores walls, as the source).
    if (config_.flow_conservation) {
      const double info = 1.0 / (config_.ff_sigma_c * config_.ff_sigma_c);
      for (int j = 0; j + 1 < ny; ++j) {
        for (int i = 0; i + 1 < nx; ++i) {
          const auto r = addRow(0.0, info);
          entry(r, ox + n(i, j), 1.0);
          entry(r, oy + n(i, j), 1.0);
          entry(r, ox + n(i + 1, j), -1.0);
          entry(r, oy + n(i + 1, j), 1.0);
          entry(r, ox + n(i, j + 1), 1.0);
          entry(r, oy + n(i, j + 1), -1.0);
          entry(r, ox + n(i + 1, j + 1), -1.0);
          entry(r, oy + n(i + 1, j + 1), -1.0);
        }
      }
    }
    // FlowField (nonlinear): r_n = [x active](c(i+1,j)-c(i,j)) w_x(n)
    //                            + [y active](c(i,j+1)-c(i,j)) w_y(n),
    // a direction is active if the neighbour exists and the edge is uncut.
    if (config_.flow_field) {
      const double info = 1.0 / (config_.sigma_flow * config_.sigma_flow);
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const bool ax = i + 1 < nx && !cut_h_[hEdge(i, j)];
          const bool ay = j + 1 < ny && !cut_v_[vEdge(i, j)];
          if (!ax && !ay) {
            continue;
          }
          const Eigen::Index c0 = n(i, j);
          const auto r = addRow(0.0, info);
          double f = 0.0;
          if (ax) {
            const Eigen::Index c1 = n(i + 1, j);
            const double dc = x(c1) - x(c0);
            const double w = x(ox + c0);
            f += dc * w;
            entry(r, ox + c0, dc);
            entry(r, c1, w);
            entry(r, c0, -w);
          }
          if (ay) {
            const Eigen::Index c1 = n(i, j + 1);
            const double dc = x(c1) - x(c0);
            const double w = x(oy + c0);
            f += dc * w;
            entry(r, oy + c0, dc);
            entry(r, c1, w);
            entry(r, c0, -w);
          }
          sys.nonlinear.emplace_back(r, f);
        }
      }
    }
    // ObstacleFlow: per node w_x * cut(+x), w_x * cut(-x), w_y * cut(+y),
    // w_y * cut(-y); at the domain boundary the boundary-wall flag is used.
    if (config_.obstacle_flow) {
      const double info = 1.0 / (config_.ff_sigma_o * config_.ff_sigma_o);
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const Eigen::Index c = n(i, j);
          const auto cs = static_cast<std::size_t>(c);
          const bool px =
              i + 1 < nx ? cut_h_[hEdge(i, j)] != 0 : x_wall_[cs] != 0;
          const bool mx =
              i > 0 ? cut_h_[hEdge(i - 1, j)] != 0 : x_wall_[cs] != 0;
          const bool py =
              j + 1 < ny ? cut_v_[vEdge(i, j)] != 0 : y_wall_[cs] != 0;
          const bool my =
              j > 0 ? cut_v_[vEdge(i, j - 1)] != 0 : y_wall_[cs] != 0;
          if (px) {
            entry(addRow(0.0, info), ox + c, 1.0);
          }
          if (mx) {
            entry(addRow(0.0, info), ox + c, 1.0);
          }
          if (py) {
            entry(addRow(0.0, info), oy + c, 1.0);
          }
          if (my) {
            entry(addRow(0.0, info), oy + c, 1.0);
          }
        }
      }
    }
  }

  // Prior (never ages).
  {
    const double info = 1.0 / (config_.prior_sigma * config_.prior_sigma);
    for (Eigen::Index c = 0; c < N; ++c) {
      entry(addRow(config_.prior_mean, info), c, 1.0);
    }
    if (joint_) {
      const double finfo =
          1.0 / (config_.ff_prior_sigma * config_.ff_prior_sigma);
      for (Eigen::Index c = 0; c < N; ++c) {
        entry(addRow(config_.ff_prior_mean_x, finfo), ox + c, 1.0);
      }
      for (Eigen::Index c = 0; c < N; ++c) {
        entry(addRow(config_.ff_prior_mean_y, finfo), oy + c, 1.0);
      }
    }
  }

  // Measurements: H_row * x - value, info 1/(sigma_s^2 [+ sigma_zeta^2 (t -
  // t_k)]).
  const bool aging = config_.enable_measurement_aging;
  for (const Measurement& m : meas) {
    double var = 0.0;
    Eigen::Index off = 0;
    if (m.kind == Kind::Field) {
      var = config_.sigma_s * config_.sigma_s;
      if (aging) {
        var += config_.sigma_zeta * config_.sigma_zeta * (t - m.t);
      }
    } else {
      off = m.kind == Kind::FlowX ? ox : oy;
      var = config_.ff_sigma_s * config_.ff_sigma_s;
      if (aging) {
        var += config_.ff_sigma_zeta * config_.ff_sigma_zeta * (t - m.t);
      }
    }
    const auto r = addRow(m.value, 1.0 / var);
    for (int k = 0; k < m.interp.count; ++k) {
      entry(r, off + m.interp.idx[k], m.interp.w[k]);
    }
  }

  sys.J.resize(row, state_size_);
  sys.J.setFromTriplets(sys.triplets.begin(), sys.triplets.end());
  sys.J.makeCompressed();
  sys.y = Eigen::Map<const Eigen::VectorXd>(sys.target.data(), row);
  sys.w = Eigen::Map<const Eigen::VectorXd>(sys.info.data(), row);
  sys.f = sys.J * x;  // exact for all linear rows
  for (const auto& [r, f] : sys.nonlinear) {
    sys.f(r) = f;
  }
}

std::shared_ptr<const GmrfBelief::Posterior> GmrfBelief::solve(
    const std::vector<Measurement>& meas, double t, const Eigen::VectorXd& x0,
    UpdateReport& report) const {
  auto post = std::make_shared<Posterior>();
  post->time = t;
  report.linear = isLinear();
  System sys;

  if (report.linear) {
    // Linear Gaussian model: one sparse direct solve Q m = J^T W y.
    assemble(x0, meas, t, sys);
    SpMat Q = normalMatrix(sys.J, sys.w);
    const Eigen::VectorXd b = sys.J.transpose() * sys.w.cwiseProduct(sys.y);
    factorOrThrow(post->factor, Q);
    Eigen::VectorXd mean = post->factor.solve(b);
    if (post->factor.info() != Eigen::Success || !mean.allFinite()) {
      throw std::runtime_error(
          "GmrfBelief: sparse solve produced a non-finite mean.");
    }
    report.iterations = 1;
    report.converged = true;
    report.final_cost = costOf(sys.J * mean, sys.y, sys.w);
    post->mean = std::move(mean);
    post->precision = std::move(Q);
    return post;
  }

  // Nonlinear (flow_field): Gauss-Newton with step halving. The returned
  // precision J^T W J is evaluated at the final accepted point (local
  // Gauss-Newton / Laplace approximation, not an exact posterior).
  Eigen::VectorXd x = x0;
  assemble(x, meas, t, sys);
  double cost = costOf(sys.f, sys.y, sys.w);
  if (!std::isfinite(cost)) {
    throw std::runtime_error(
        "GmrfBelief: non-finite cost at the initial point.");
  }
  SpMat Q;
  bool factor_at_x = false;
  bool converged = false;
  int accepted = 0;
  System trial;
  for (int it = 0; it < config_.max_gauss_newton_iterations; ++it) {
    Q = normalMatrix(sys.J, sys.w);
    factorOrThrow(post->factor, Q);
    factor_at_x = true;
    const Eigen::VectorXd g =
        sys.J.transpose() * sys.w.cwiseProduct(sys.f - sys.y);
    const Eigen::VectorXd step = -post->factor.solve(g);
    if (!step.allFinite()) {
      throw std::runtime_error("GmrfBelief: Gauss-Newton step is not finite.");
    }
    if (step.norm() < config_.gn_step_tolerance) {
      converged = true;
      break;
    }
    double alpha = 1.0;
    bool decreased = false;
    double trial_cost = cost;
    Eigen::VectorXd xt;
    while (alpha >= config_.gn_min_step_fraction) {
      xt = x + alpha * step;
      assemble(xt, meas, t, trial);
      trial_cost = costOf(trial.f, trial.y, trial.w);
      if (std::isfinite(trial_cost) && trial_cost < cost) {
        decreased = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!decreased) {
      // No further decrease along the GN direction (source: treated as
      // converged). Stalling before any accepted step is reported as not
      // converged; the state stays at the starting point.
      converged = accepted > 0;
      break;
    }
    const double rel = (cost - trial_cost) /
                       std::max(cost, std::numeric_limits<double>::min());
    x = std::move(xt);
    std::swap(sys, trial);
    cost = trial_cost;
    ++accepted;
    factor_at_x = false;
    if (rel < config_.gn_relative_cost_tolerance) {
      converged = true;
      break;
    }
  }
  if (!factor_at_x) {
    Q = normalMatrix(sys.J, sys.w);
    factorOrThrow(post->factor, Q);
  }
  if (!x.allFinite()) {
    throw std::runtime_error(
        "GmrfBelief: Gauss-Newton produced a non-finite state.");
  }
  report.iterations = accepted;
  report.converged = converged;
  report.final_cost = cost;
  post->mean = std::move(x);
  post->precision = std::move(Q);
  return post;
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

void GmrfBelief::validateQueryState() const {
  if (!pending_.empty()) {
    throw std::logic_error(
        "GmrfBelief: query while measurements are pending; call update() "
        "first.");
  }
}

Eigen::SparseMatrix<double, Eigen::RowMajor> GmrfBelief::buildH(
    const Eigen::Ref<const Eigen::MatrixX2d>& xy, Eigen::Index cols) const {
  std::vector<Eigen::Triplet<double>> trip;
  trip.reserve(static_cast<std::size_t>(xy.rows()) * 4);
  for (Eigen::Index r = 0; r < xy.rows(); ++r) {
    const Interp interp = interpolateOrThrow(xy(r, 0), xy(r, 1), "query point");
    for (int k = 0; k < interp.count; ++k) {
      trip.emplace_back(static_cast<int>(r), static_cast<int>(interp.idx[k]),
                        interp.w[k]);
    }
  }
  Eigen::SparseMatrix<double, Eigen::RowMajor> H(xy.rows(), cols);
  H.setFromTriplets(trip.begin(), trip.end());
  H.makeCompressed();
  return H;
}

Eigen::SparseMatrix<double, Eigen::RowMajor> GmrfBelief::interpolationMatrix(
    const Eigen::Ref<const Eigen::MatrixX2d>& xy) const {
  return buildH(xy, num_nodes_);
}

Eigen::VectorXd GmrfBelief::queryMean(
    const Eigen::Ref<const Eigen::MatrixX2d>& xy) const {
  validateQueryState();
  const auto H = buildH(xy, state_size_);
  return H * posterior_->mean;
}

void GmrfBelief::queryMeanVariance(const Eigen::Ref<const Eigen::MatrixX2d>& xy,
                                   Eigen::VectorXd& mean,
                                   Eigen::VectorXd& variance) const {
  validateQueryState();
  const auto H = buildH(xy, state_size_);
  const Posterior& p = *posterior_;
  const Eigen::Index m = H.rows();
  Eigen::VectorXd out_mean = H * p.mean;
  Eigen::VectorXd out_var(m);
  // var_k = h_k^T Q^{-1} h_k = || L^{-1} P h_k ||^2  (P Q P^T = L L^T)
  Eigen::MatrixXd Z;
  for (Eigen::Index c0 = 0; c0 < m; c0 += kQueryChunk) {
    const Eigen::Index k = std::min(kQueryChunk, m - c0);
    Z.setZero(state_size_, k);
    for (Eigen::Index c = 0; c < k; ++c) {
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(
               H, c0 + c);
           it; ++it) {
        Z(it.col(), c) = it.value();
      }
    }
    if (p.factor.permutationP().size() > 0) {
      Z = p.factor.permutationP() * Z;
    }
    p.factor.matrixL().solveInPlace(Z);
    out_var.segment(c0, k) = Z.colwise().squaredNorm().transpose();
  }
  mean.swap(out_mean);
  variance.swap(out_var);
}

void GmrfBelief::queryMeanCovariance(
    const Eigen::Ref<const Eigen::MatrixX2d>& xy, Eigen::VectorXd& mean,
    Eigen::MatrixXd& covariance) const {
  validateQueryState();
  const auto H = buildH(xy, state_size_);
  const Posterior& p = *posterior_;
  const Eigen::Index m = H.rows();
  Eigen::VectorXd out_mean = H * p.mean;
  Eigen::MatrixXd cov(m, m);
  // X = Q^{-1} H^T in column chunks; cov(:, chunk) = H X.
  Eigen::MatrixXd X;
  for (Eigen::Index c0 = 0; c0 < m; c0 += kQueryChunk) {
    const Eigen::Index k = std::min(kQueryChunk, m - c0);
    X.setZero(state_size_, k);
    for (Eigen::Index c = 0; c < k; ++c) {
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(
               H, c0 + c);
           it; ++it) {
        X(it.col(), c) = it.value();
      }
    }
    X = p.factor.solve(X);
    cov.middleCols(c0, k) = H * X;
  }
  Eigen::MatrixXd sym = 0.5 * (cov + cov.transpose());
  mean.swap(out_mean);
  covariance.swap(sym);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

double GmrfBelief::fieldNeighborSigma() const {
  return config_.sigma_p_per_m * dx_ * dx_;
}

Eigen::Index GmrfBelief::nodeIndex(int i, int j) const {
  if (i < 0 || i >= config_.nx || j < 0 || j >= config_.ny) {
    throw std::out_of_range("GmrfBelief: node index out of range.");
  }
  return static_cast<Eigen::Index>(j) * config_.nx + i;
}

Point2 GmrfBelief::nodePosition(int i, int j) const {
  nodeIndex(i, j);  // range check
  return Point2{nodeCoord(config_.x_min, config_.x_max, i, config_.nx),
                nodeCoord(config_.y_min, config_.y_max, j, config_.ny)};
}

bool GmrfBelief::isNodeActive(int i, int j) const {
  return active_[static_cast<std::size_t>(nodeIndex(i, j))] != 0;
}

bool GmrfBelief::isHorizontalEdgeCut(int i, int j) const {
  if (i < 0 || i >= config_.nx - 1 || j < 0 || j >= config_.ny) {
    throw std::out_of_range("GmrfBelief: horizontal edge index out of range.");
  }
  return cut_h_[hEdge(i, j)] != 0;
}

bool GmrfBelief::isVerticalEdgeCut(int i, int j) const {
  if (i < 0 || i >= config_.nx || j < 0 || j >= config_.ny - 1) {
    throw std::out_of_range("GmrfBelief: vertical edge index out of range.");
  }
  return cut_v_[vEdge(i, j)] != 0;
}

bool GmrfBelief::hasXBoundaryWall(int i, int j) const {
  return x_wall_[static_cast<std::size_t>(nodeIndex(i, j))] != 0;
}

bool GmrfBelief::hasYBoundaryWall(int i, int j) const {
  return y_wall_[static_cast<std::size_t>(nodeIndex(i, j))] != 0;
}

const Eigen::VectorXd& GmrfBelief::stateMean() const {
  return posterior_->mean;
}

Eigen::VectorXd GmrfBelief::fieldNodeMean() const {
  return posterior_->mean.head(num_nodes_);
}

Eigen::VectorXd GmrfBelief::flowNodeMean(FlowComponent component) const {
  if (!joint_) {
    throw std::logic_error("GmrfBelief: flow mean requires the joint state.");
  }
  return posterior_->mean.segment(
      component == FlowComponent::X ? num_nodes_ : 2 * num_nodes_, num_nodes_);
}

const Eigen::SparseMatrix<double>& GmrfBelief::precisionMatrix() const {
  return posterior_->precision;
}

double GmrfBelief::lastUpdateTime() const { return posterior_->time; }

}  // namespace scalar_field_belief::gmrf
