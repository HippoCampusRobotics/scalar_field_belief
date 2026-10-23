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
#ifndef SCALAR_FIELD_BELIEF_GMRF_GMRF_BELIEF_HPP
#define SCALAR_FIELD_BELIEF_GMRF_GMRF_BELIEF_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "scalar_field_belief/gmrf/config.hpp"

namespace scalar_field_belief::gmrf {

/// Result of GmrfBelief::update().
struct UpdateReport {
  /// False if there was nothing to do (no pending data, aging disabled).
  bool solved = false;
  /// True for the linear (direct solve) modes.
  bool linear = true;
  /// Linear: 1 direct solve. Nonlinear: accepted Gauss-Newton steps.
  int iterations = 0;
  /// Nonlinear: false if the iteration cap was reached first.
  bool converged = true;
  /// sum_i info_i * residual_i^2 at the accepted solution.
  double final_cost = 0.0;
  /// Number of pending measurements assimilated by this update.
  std::size_t num_assimilated = 0;
};

/// Gaussian Markov random field belief over a 2D scalar field on a regular
/// grid, optionally jointly with a 2D flow field (see README.md).
///
/// Lifecycle: addMeasurement() validates and queues; update() assimilates all
/// queued measurements (strong exception guarantee). Queries are const, never
/// change the belief, and throw std::logic_error while measurements are
/// pending. A prior posterior exists right after construction and after
/// reset().
///
/// Query semantics (field component, measurement units):
///   mean = H * m,  covariance = H * Q^{-1} * H^T,  variance = diag(covariance)
/// with H the masked, renormalised bilinear interpolation matrix and Q the
/// posterior precision (Gauss-Newton information at the final linearisation
/// point when the nonlinear flow_field energy is enabled). These are latent
/// field uncertainties; measurement noise is reported separately by
/// measurementNoiseVariance().
class GmrfBelief {
 public:
  /// Validates everything before building the belief. Throws
  /// std::invalid_argument for invalid configuration, walls, or seeds and
  /// std::runtime_error if the prior system cannot be factorised.
  ///
  /// free_space_seeds: if non-empty, only nodes connected (via uncut edges) to
  /// a seed are active; all others are excluded from interpolation and
  /// decoupled (they keep their prior). If empty, all nodes are active.
  explicit GmrfBelief(const GmrfConfig& config,
                      std::vector<WallSegment> walls = {},
                      std::vector<Point2> free_space_seeds = {});

  /// Clears all assimilated and pending measurements and restores the prior
  /// posterior (configuration, walls, and free-space mask are kept).
  void reset();

  /// Validates and queues a scalar-field measurement at physical (x, y).
  /// t is only used when measurement aging is enabled. Throws
  /// std::invalid_argument (non-finite input, outside the domain, or a point
  /// that cannot be represented because all interpolation corners are masked);
  /// the belief is unchanged on failure.
  void addMeasurement(double x, double y, double value, double t = 0.0);

  /// Validates and queues a flow measurement (joint state only, else
  /// std::logic_error).
  void addFlowMeasurement(double x, double y, FlowComponent component,
                          double value, double t = 0.0);

  /// Removes all queued (not yet assimilated) measurements.
  void discardPendingMeasurements();

  /// Assimilates all pending measurements. Linear modes: one sparse direct
  /// solve. Nonlinear mode: Gauss-Newton with backtracking from the current
  /// mean. t is the update time used for aging (ignored otherwise).
  /// Strong guarantee: on any exception the belief (incl. pending data) is
  /// unchanged. Throws std::invalid_argument for invalid t and
  /// std::runtime_error on factorisation/numerical failure.
  UpdateReport update(double t = 0.0);

  bool hasPendingMeasurements() const { return !pending_.empty(); }
  std::size_t numPendingMeasurements() const { return pending_.size(); }
  /// Number of assimilated measurements (field and flow).
  std::size_t numMeasurements() const { return measurements_.size(); }

  /// xy: m x 2 physical query points. All throw std::invalid_argument for an
  /// invalid point (no partial results) and std::logic_error while
  /// measurements are pending. Empty input yields empty output.
  Eigen::VectorXd queryMean(const Eigen::Ref<const Eigen::MatrixX2d>& xy) const;
  /// Marginal variances only (no m x m covariance is formed).
  void queryMeanVariance(const Eigen::Ref<const Eigen::MatrixX2d>& xy,
                         Eigen::VectorXd& mean,
                         Eigen::VectorXd& variance) const;
  /// Full symmetric m x m latent covariance H Q^{-1} H^T.
  void queryMeanCovariance(const Eigen::Ref<const Eigen::MatrixX2d>& xy,
                           Eigen::VectorXd& mean,
                           Eigen::MatrixXd& covariance) const;

  /// True if (x, y) is finite, inside the domain and has at least one visible
  /// active interpolation corner (i.e. it can be measured / queried).
  bool isRepresentable(double x, double y) const;

  /// Field interpolation matrix H (m x nodeCount()) used for queries and
  /// measurements, in node index order n = j*nx + i.
  Eigen::SparseMatrix<double, Eigen::RowMajor> interpolationMatrix(
      const Eigen::Ref<const Eigen::MatrixX2d>& xy) const;

  /// Observation noise variance sigma_s^2 [measurement units^2].
  double measurementNoiseVariance() const {
    return config_.sigma_s * config_.sigma_s;
  }

  // --- Grid and model accessors ---------------------------------------------
  const GmrfConfig& config() const { return config_; }
  const std::vector<WallSegment>& walls() const { return walls_; }
  int nx() const { return config_.nx; }
  int ny() const { return config_.ny; }
  Eigen::Index nodeCount() const { return num_nodes_; }
  /// N (scalar-only state) or 3N (joint state [c, w_x, w_y]).
  Eigen::Index stateSize() const { return state_size_; }
  bool isJointState() const { return joint_; }
  bool isLinear() const { return config_.isLinear(); }
  double dx() const { return dx_; }
  double dy() const { return dy_; }
  /// Std of the field-neighbour difference, sigma_p_per_m * dx^2.
  double fieldNeighborSigma() const;
  Eigen::Index nodeIndex(int i, int j) const;
  Point2 nodePosition(int i, int j) const;
  bool isNodeActive(int i, int j) const;
  /// Edge between (i, j) and (i + 1, j); i in [0, nx-2].
  bool isHorizontalEdgeCut(int i, int j) const;
  /// Edge between (i, j) and (i, j + 1); j in [0, ny-2].
  bool isVerticalEdgeCut(int i, int j) const;
  /// Boundary-wall flags (set by walls lying on the domain boundary).
  bool hasXBoundaryWall(int i, int j) const;
  bool hasYBoundaryWall(int i, int j) const;

  /// Full posterior state mean (size stateSize()).
  const Eigen::VectorXd& stateMean() const;
  /// Nodal field mean (size nodeCount()).
  Eigen::VectorXd fieldNodeMean() const;
  /// Nodal flow mean (joint state only, else std::logic_error).
  Eigen::VectorXd flowNodeMean(FlowComponent component) const;
  /// Posterior precision Q (stateSize() x stateSize()).
  const Eigen::SparseMatrix<double>& precisionMatrix() const;
  /// Time passed to the last update() (0 after construction/reset).
  double lastUpdateTime() const;

 private:
  struct Posterior;  // immutable once built; shared between copies
  enum class Kind : std::uint8_t { Field = 0, FlowX = 1, FlowY = 2 };

  struct Interp {
    Eigen::Index idx[4] = {0, 0, 0, 0};
    double w[4] = {0.0, 0.0, 0.0, 0.0};
    int count = 0;
  };

  struct Measurement {
    Interp interp;
    double value = 0.0;
    double t = 0.0;
    Kind kind = Kind::Field;
  };

  struct System;

  void buildGeometry(const std::vector<Point2>& seeds);
  bool tryInterpolate(double x, double y, Interp& out,
                      const char** reason) const;
  Interp interpolateOrThrow(double x, double y, const char* what) const;
  void validateQueryState() const;
  void addMeasurementImpl(double x, double y, double value, double t,
                          Kind kind);
  void assemble(const Eigen::VectorXd& x, const std::vector<Measurement>& meas,
                double t, System& sys) const;
  std::shared_ptr<const Posterior> solve(const std::vector<Measurement>& meas,
                                         double t, const Eigen::VectorXd& x0,
                                         UpdateReport& report) const;
  Eigen::SparseMatrix<double, Eigen::RowMajor> buildH(
      const Eigen::Ref<const Eigen::MatrixX2d>& xy, Eigen::Index cols) const;
  bool visible(double px, double py, double qx, double qy) const;
  Eigen::Index nid(int i, int j) const {
    return static_cast<Eigen::Index>(j) * config_.nx + i;
  }
  std::size_t hEdge(int i, int j) const {
    return static_cast<std::size_t>(j) * (config_.nx - 1) + i;
  }
  std::size_t vEdge(int i, int j) const {
    return static_cast<std::size_t>(j) * config_.nx + i;
  }

  GmrfConfig config_;
  std::vector<WallSegment> walls_;  // as given (validated)
  std::vector<WallSegment>
      cutting_;  // extended + tie-break-shifted, boundary walls removed
  double dx_ = 0.0;
  double dy_ = 0.0;
  Eigen::Index num_nodes_ = 0;
  Eigen::Index state_size_ = 0;
  bool joint_ = false;

  std::vector<std::uint8_t> cut_h_;   // (nx-1)*ny horizontal edges
  std::vector<std::uint8_t> cut_v_;   // nx*(ny-1) vertical edges
  std::vector<std::uint8_t> x_wall_;  // per node
  std::vector<std::uint8_t> y_wall_;  // per node
  std::vector<std::uint8_t> active_;  // per node

  std::vector<Measurement> measurements_;  // assimilated
  std::vector<Measurement> pending_;
  std::shared_ptr<const Posterior> prior_;
  std::shared_ptr<const Posterior> posterior_;
};

}  // namespace scalar_field_belief::gmrf

#endif  // SCALAR_FIELD_BELIEF_GMRF_GMRF_BELIEF_HPP
