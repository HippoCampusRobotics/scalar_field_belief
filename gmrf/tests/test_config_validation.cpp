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

// Configuration, wall, and seed validation (all before any state is built).
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "gmrf_test_utils.hpp"

using gmrf_test::smallConfig;
namespace gm = scalar_field_belief::gmrf;

TEST(ConfigValidation, DefaultAndSmallConfigsAreValid) {
  EXPECT_NO_THROW(gm::GmrfConfig{}.validate());
  EXPECT_NO_THROW(smallConfig().validate());
  const gm::GmrfConfig def;
  EXPECT_TRUE(def.field_neighbors);
  EXPECT_FALSE(def.flow_neighbors || def.flow_conservation || def.flow_field ||
               def.obstacle_flow || def.force_joint_state ||
               def.enable_measurement_aging);
  EXPECT_FALSE(def.usesJointState());
  EXPECT_TRUE(def.isLinear());
}

TEST(ConfigValidation, RejectsInvalidValues) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::function<void(gm::GmrfConfig&)> mutations[] = {
      [](gm::GmrfConfig& c) { c.nx = 1; },
      [](gm::GmrfConfig& c) { c.ny = 0; },
      [](gm::GmrfConfig& c) { c.x_max = c.x_min; },
      [](gm::GmrfConfig& c) { c.y_max = c.y_min - 1.0; },
      [nan](gm::GmrfConfig& c) { c.x_min = nan; },
      [inf](gm::GmrfConfig& c) { c.y_max = inf; },
      [](gm::GmrfConfig& c) { c.prior_sigma = 0.0; },
      [](gm::GmrfConfig& c) { c.prior_sigma = -1.0; },
      [](gm::GmrfConfig& c) { c.sigma_s = 0.0; },
      [](gm::GmrfConfig& c) { c.sigma_p_per_m = 0.0; },
      [nan](gm::GmrfConfig& c) { c.prior_mean = nan; },
      [](gm::GmrfConfig& c) { c.sigma_zeta = -0.1; },
      [](gm::GmrfConfig& c) { c.wall_endpoint_extension = -0.01; },
      [](gm::GmrfConfig& c) { c.max_gauss_newton_iterations = 0; },
      [](gm::GmrfConfig& c) { c.gn_min_step_fraction = 0.0; },
      [](gm::GmrfConfig& c) {
        c.flow_field = true;
        c.sigma_flow = 0.0;
      },
      [](gm::GmrfConfig& c) {
        c.flow_neighbors = true;
        c.ff_sigma_p_per_m = -1.0;
      },
      [](gm::GmrfConfig& c) {
        c.flow_conservation = true;
        c.ff_sigma_c = 0.0;
      },
      [](gm::GmrfConfig& c) {
        c.obstacle_flow = true;
        c.ff_sigma_o = 0.0;
      },
      [](gm::GmrfConfig& c) {
        c.force_joint_state = true;
        c.ff_prior_sigma = 0.0;
      },
  };
  int k = 0;
  for (const auto& mutate : mutations) {
    gm::GmrfConfig c = smallConfig();
    mutate(c);
    EXPECT_THROW(c.validate(), std::invalid_argument) << "mutation " << k;
    EXPECT_THROW(gm::GmrfBelief{c}, std::invalid_argument) << "mutation " << k;
    ++k;
  }
}

TEST(ConfigValidation, UnusedFlowSigmasMayBeZeroInScalarMode) {
  gm::GmrfConfig c = smallConfig();
  c.sigma_flow = 0.0;
  c.ff_sigma_c = 0.0;
  c.ff_sigma_o = 0.0;
  EXPECT_NO_THROW(c.validate());
}

TEST(ConfigValidation, RejectsInvalidWalls) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
      gm::GmrfBelief(smallConfig(), {gm::WallSegment{1.0, 1.0, 1.0, 1.0}}),
      std::invalid_argument);
  EXPECT_THROW(
      gm::GmrfBelief(smallConfig(), {gm::WallSegment{1.0, nan, 1.0, 2.0}}),
      std::invalid_argument);
  // Walls partly or fully outside the domain are allowed.
  EXPECT_NO_THROW(
      gm::GmrfBelief(smallConfig(), {gm::WallSegment{-5.0, 1.0, 5.0, 1.0}}));
  EXPECT_NO_THROW(
      gm::GmrfBelief(smallConfig(), {gm::WallSegment{10.0, 10.0, 11.0, 12.0}}));
}

TEST(ConfigValidation, RejectsInvalidSeeds) {
  EXPECT_THROW(gm::GmrfBelief(smallConfig(), {}, {gm::Point2{-0.1, 1.0}}),
               std::invalid_argument);
  EXPECT_THROW(
      gm::GmrfBelief(smallConfig(), {}, {gm::Point2{1.0, std::nan("")}}),
      std::invalid_argument);
  EXPECT_NO_THROW(gm::GmrfBelief(smallConfig(), {}, {gm::Point2{2.0, 4.0}}));
}

TEST(ConfigValidation, GridDefinition) {
  gm::GmrfConfig c = smallConfig();
  c.x_min = -1.0;
  c.x_max = 3.0;
  c.y_min = -2.5;
  c.y_max = 0.5;
  c.nx = 5;
  c.ny = 4;
  const gm::GmrfBelief b(c);
  EXPECT_EQ(b.nodeCount(), 20);
  EXPECT_EQ(b.stateSize(), 20);
  EXPECT_DOUBLE_EQ(b.dx(), 1.0);
  EXPECT_DOUBLE_EQ(b.dy(), 1.0);
  EXPECT_EQ(b.nodeIndex(3, 2), 2 * 5 + 3);
  EXPECT_DOUBLE_EQ(b.nodePosition(0, 0).x, -1.0);
  EXPECT_DOUBLE_EQ(b.nodePosition(0, 0).y, -2.5);
  EXPECT_DOUBLE_EQ(b.nodePosition(4, 3).x, 3.0);
  EXPECT_DOUBLE_EQ(b.nodePosition(4, 3).y, 0.5);
  EXPECT_DOUBLE_EQ(b.nodePosition(2, 1).x, 1.0);
  EXPECT_DOUBLE_EQ(b.nodePosition(2, 1).y, -1.5);
  EXPECT_THROW(b.nodeIndex(5, 0), std::out_of_range);
  EXPECT_THROW(b.nodeIndex(0, -1), std::out_of_range);
}
