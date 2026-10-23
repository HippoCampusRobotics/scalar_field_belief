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

// Basic wall handling: edge cuts, tie-breaks, separation, boundary walls,
// endpoint order/splitting, gaps, diagonal walls, and the free-space mask.
#include <gtest/gtest.h>

#include <queue>
#include <stdexcept>
#include <vector>

#include "gmrf_test_utils.hpp"

using gmrf_test::nodePoints;
using gmrf_test::points;
using gmrf_test::smallConfig;
namespace gm = scalar_field_belief::gmrf;

namespace {

// smallConfig(): 9 x 17 nodes, spacing 0.25 m on [0, 2] x [0, 4].
std::vector<bool> cutPattern(const gm::GmrfBelief& b) {
  std::vector<bool> cuts;
  for (int j = 0; j < b.ny(); ++j) {
    for (int i = 0; i + 1 < b.nx(); ++i) {
      cuts.push_back(b.isHorizontalEdgeCut(i, j));
    }
  }
  for (int j = 0; j + 1 < b.ny(); ++j) {
    for (int i = 0; i < b.nx(); ++i) {
      cuts.push_back(b.isVerticalEdgeCut(i, j));
    }
  }
  return cuts;
}

int numCuts(const gm::GmrfBelief& b) {
  int n = 0;
  for (bool c : cutPattern(b)) {
    n += c ? 1 : 0;
  }
  return n;
}

// Nodes reachable from (i0, j0) through uncut edges.
std::vector<bool> component(const gm::GmrfBelief& b, int i0, int j0) {
  std::vector<bool> seen(static_cast<std::size_t>(b.nodeCount()), false);
  std::queue<std::pair<int, int>> q;
  q.push({i0, j0});
  seen[static_cast<std::size_t>(b.nodeIndex(i0, j0))] = true;
  while (!q.empty()) {
    const auto [i, j] = q.front();
    q.pop();
    auto go = [&](int ii, int jj) {
      const auto k = static_cast<std::size_t>(b.nodeIndex(ii, jj));
      if (!seen[k]) {
        seen[k] = true;
        q.push({ii, jj});
      }
    };
    if (i + 1 < b.nx() && !b.isHorizontalEdgeCut(i, j)) {
      go(i + 1, j);
    }
    if (i > 0 && !b.isHorizontalEdgeCut(i - 1, j)) {
      go(i - 1, j);
    }
    if (j + 1 < b.ny() && !b.isVerticalEdgeCut(i, j)) {
      go(i, j + 1);
    }
    if (j > 0 && !b.isVerticalEdgeCut(i, j - 1)) {
      go(i, j - 1);
    }
  }
  return seen;
}

void nodeMeanVar(const gm::GmrfBelief& b, Eigen::VectorXd& m,
                 Eigen::VectorXd& v) {
  b.queryMeanVariance(nodePoints(b), m, v);
}

}  // namespace

TEST(WallsBasic, NoWallNoCuts) {
  const gm::GmrfBelief b(smallConfig());
  EXPECT_EQ(numCuts(b), 0);
}

TEST(WallsBasic, FullSeparatorOnNodeColumnUsesFloorRule) {
  // x = 1.0 is node column i = 4; the wall extends beyond the domain.
  gm::GmrfBelief b(smallConfig(), {gm::WallSegment{1.0, -1.0, 1.0, 5.0}});
  for (int j = 0; j < b.ny(); ++j) {
    EXPECT_FALSE(b.isHorizontalEdgeCut(3, j)) << j;
    EXPECT_TRUE(b.isHorizontalEdgeCut(4, j))
        << j;  // toward +x of the node on the wall
  }
  EXPECT_EQ(numCuts(b), b.ny());

  // A point on the wall belongs to the -x side, a point just east of it to +x.
  const auto H = b.interpolationMatrix(points({{1.0, 2.1}, {1.0 + 1e-6, 2.1}}));
  for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(H, 0); it;
       ++it) {
    EXPECT_EQ(it.col() % b.nx(), 4);
  }
  for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(H, 1); it;
       ++it) {
    EXPECT_EQ(it.col() % b.nx(), 5);
  }
}

TEST(WallsBasic, FullSeparatorBlocksInformation) {
  const std::vector<gm::WallSegment> wall{{1.1, -1.0, 1.1, 5.0}};
  gm::GmrfBelief b(smallConfig(), wall);
  Eigen::VectorXd m0;
  Eigen::VectorXd v0;
  nodeMeanVar(b, m0, v0);
  b.addMeasurement(0.9, 2.0, 5.0);   // west, cell straddles the wall
  b.addMeasurement(0.3, 0.4, -2.0);  // west
  b.update();
  Eigen::VectorXd m1;
  Eigen::VectorXd v1;
  nodeMeanVar(b, m1, v1);
  for (int j = 0; j < b.ny(); ++j) {
    for (int i = 0; i < b.nx(); ++i) {
      const Eigen::Index n = b.nodeIndex(i, j);
      if (i >= 5) {  // east of x = 1.1
        EXPECT_NEAR(m1(n), m0(n), 1e-12);
        EXPECT_NEAR(v1(n), v0(n), 1e-12);
      }
    }
  }
  EXPECT_GT(m1(b.nodeIndex(4, 8)), m0(b.nodeIndex(4, 8)) + 1.0);
  // A query just east of the wall uses east nodes only and is unaffected.
  const Eigen::VectorXd mq = b.queryMean(points({{1.12, 2.0}}));
  EXPECT_NEAR(mq(0), smallConfig().prior_mean, 1e-12);
}

TEST(WallsBasic, WallsOnTheDomainBoundaryCutNothing) {
  const gm::GmrfConfig c = smallConfig();
  const std::vector<gm::WallSegment> walls{{0.0, 0.0, 0.0, 4.0},
                                           {2.0, 0.0, 2.0, 4.0},
                                           {0.0, 0.0, 2.0, 0.0},
                                           {2.0, 4.0, 0.0, 4.0}};
  gm::GmrfBelief b(c, walls);
  EXPECT_EQ(numCuts(b), 0);
  for (int j = 0; j < b.ny(); ++j) {
    EXPECT_TRUE(b.hasXBoundaryWall(0, j));
    EXPECT_TRUE(b.hasXBoundaryWall(b.nx() - 1, j));
    EXPECT_FALSE(b.hasXBoundaryWall(3, j));
  }
  for (int i = 0; i < b.nx(); ++i) {
    EXPECT_TRUE(b.hasYBoundaryWall(i, 0));
    EXPECT_TRUE(b.hasYBoundaryWall(i, b.ny() - 1));
  }
  gm::GmrfBelief ref(c);
  for (auto* x : {&b, &ref}) {
    x->addMeasurement(0.0, 2.0, 1.0);
    x->addMeasurement(2.0, 4.0, 2.0);
    x->update();
  }
  EXPECT_TRUE(b.fieldNodeMean().isApprox(ref.fieldNodeMean(), 1e-14));
}

TEST(WallsBasic, ReversedAndSplitSegmentsGiveIdenticalCuts) {
  const gm::GmrfConfig c = smallConfig();
  const auto full = cutPattern(gm::GmrfBelief(c, {{0.6, 0.3, 1.7, 3.2}}));
  const auto reversed = cutPattern(gm::GmrfBelief(c, {{1.7, 3.2, 0.6, 0.3}}));
  // Split at an interior point of the same line.
  const double t = 0.4;
  const double xm = 0.6 + t * (1.7 - 0.6);
  const double ym = 0.3 + t * (3.2 - 0.3);
  const auto split =
      cutPattern(gm::GmrfBelief(c, {{0.6, 0.3, xm, ym}, {xm, ym, 1.7, 3.2}}));
  EXPECT_EQ(full, reversed);
  EXPECT_EQ(full, split);
  const auto vfull = cutPattern(gm::GmrfBelief(c, {{1.0, 0.5, 1.0, 3.5}}));
  const auto vsplit = cutPattern(
      gm::GmrfBelief(c, {{1.0, 0.5, 1.0, 2.0}, {1.0, 2.0, 1.0, 3.5}}));
  EXPECT_EQ(vfull, vsplit);
}

TEST(WallsBasic, EndpointExtensionCoversNodeRowsAtWallEnds) {
  // Wall ends exactly on node rows y = 0.5 and 3.5 (j = 2 and 14).
  gm::GmrfBelief b(smallConfig(), {{1.1, 0.5, 1.1, 3.5}});
  for (int j = 0; j < b.ny(); ++j) {
    EXPECT_EQ(b.isHorizontalEdgeCut(4, j), j >= 2 && j <= 14) << j;
  }
  gm::GmrfConfig c = smallConfig();
  c.wall_endpoint_extension = 0.3;  // reaches rows 0.25 and 3.75 too
  gm::GmrfBelief e(c, {{1.1, 0.5, 1.1, 3.5}});
  for (int j = 0; j < e.ny(); ++j) {
    EXPECT_EQ(e.isHorizontalEdgeCut(4, j), j >= 1 && j <= 15) << j;
  }
}

TEST(WallsBasic, GapKeepsCoupling) {
  const std::vector<gm::WallSegment> walls{{1.1, -1.0, 1.1, 1.9},
                                           {1.1, 2.6, 1.1, 5.0}};
  gm::GmrfBelief b(smallConfig(), walls);
  for (int j = 0; j < b.ny(); ++j) {
    EXPECT_EQ(b.isHorizontalEdgeCut(4, j), !(j >= 8 && j <= 10)) << j;
  }
  const auto west = component(b, 0, 0);
  EXPECT_TRUE(west[static_cast<std::size_t>(
      b.nodeIndex(8, 0))]);  // connected via the gap
  Eigen::VectorXd m0;
  Eigen::VectorXd v0;
  nodeMeanVar(b, m0, v0);
  b.addMeasurement(0.9, 2.25, 5.0);
  b.update();
  Eigen::VectorXd m1;
  Eigen::VectorXd v1;
  nodeMeanVar(b, m1, v1);
  EXPECT_GT(m1(b.nodeIndex(5, 9)), m0(b.nodeIndex(5, 9)) + 0.1);
  EXPECT_LT(v1(b.nodeIndex(5, 9)), v0(b.nodeIndex(5, 9)));
}

TEST(WallsBasic, DiagonalFullSeparator) {
  // From the left boundary to the right boundary: upper-left vs lower-right.
  gm::GmrfBelief b(smallConfig(), {{-0.1, 0.2, 2.1, 3.8}});
  const auto upper = component(b, 0, b.ny() - 1);
  EXPECT_FALSE(upper[static_cast<std::size_t>(b.nodeIndex(b.nx() - 1, 0))]);
  Eigen::VectorXd m0;
  Eigen::VectorXd v0;
  nodeMeanVar(b, m0, v0);
  b.addMeasurement(0.3, 3.0, 4.0);  // upper-left side
  b.update();
  Eigen::VectorXd m1;
  Eigen::VectorXd v1;
  nodeMeanVar(b, m1, v1);
  for (Eigen::Index n = 0; n < b.nodeCount(); ++n) {
    if (!upper[static_cast<std::size_t>(n)]) {
      EXPECT_NEAR(m1(n), m0(n), 1e-12) << n;
      EXPECT_NEAR(v1(n), v0(n), 1e-12) << n;
    }
  }
}

TEST(WallsBasic, NodeOnRisingDiagonalBelongsToMinusXSide) {
  // 45-degree wall y = x through the nodes (k, k) of a square 0.25 m grid.
  gm::GmrfConfig c = smallConfig();
  c.y_max = 2.0;
  c.ny = 9;
  gm::GmrfBelief b(c, {{-1.0, -1.0, 3.0, 3.0}});
  for (int k = 1; k + 1 < b.nx(); ++k) {
    // node (k, k) lies on the wall: connected to the -x (upper-left) side only
    EXPECT_FALSE(b.isHorizontalEdgeCut(k - 1, k)) << k;
    EXPECT_TRUE(b.isHorizontalEdgeCut(k, k)) << k;
    EXPECT_TRUE(b.isVerticalEdgeCut(k, k - 1)) << k;
    EXPECT_FALSE(b.isVerticalEdgeCut(k, k)) << k;
  }
  const auto ul = component(b, 0, b.ny() - 1);
  EXPECT_FALSE(ul[static_cast<std::size_t>(b.nodeIndex(b.nx() - 1, 0))]);
}

TEST(WallsBasic, DisconnectedCompartmentStaysValidWithoutSeeds) {
  // Closed box around [0.6, 1.4] x [1.6, 2.4].
  const std::vector<gm::WallSegment> box{{0.6, 1.6, 1.4, 1.6},
                                         {1.4, 1.6, 1.4, 2.4},
                                         {1.4, 2.4, 0.6, 2.4},
                                         {0.6, 2.4, 0.6, 1.6}};
  gm::GmrfBelief b(smallConfig(), box);
  EXPECT_TRUE(b.isNodeActive(4, 8));
  EXPECT_TRUE(b.isRepresentable(1.0, 2.0));
  const auto inside = component(b, 4, 8);
  EXPECT_FALSE(inside[0]);
  Eigen::VectorXd m0;
  Eigen::VectorXd v0;
  nodeMeanVar(b, m0, v0);
  b.addMeasurement(1.0, 2.0, 3.0);
  b.update();
  Eigen::VectorXd m1;
  Eigen::VectorXd v1;
  nodeMeanVar(b, m1, v1);
  EXPECT_GT(m1(b.nodeIndex(4, 8)), m0(b.nodeIndex(4, 8)) + 1.0);
  for (Eigen::Index n = 0; n < b.nodeCount(); ++n) {
    if (!inside[static_cast<std::size_t>(n)]) {
      EXPECT_NEAR(m1(n), m0(n), 1e-12) << n;
    }
  }
}

TEST(WallsBasic, FreeSpaceSeedsDeactivateUnreachableNodes) {
  const std::vector<gm::WallSegment> box{{0.6, 1.6, 1.4, 1.6},
                                         {1.4, 1.6, 1.4, 2.4},
                                         {1.4, 2.4, 0.6, 2.4},
                                         {0.6, 2.4, 0.6, 1.6}};
  gm::GmrfBelief b(smallConfig(), box, {gm::Point2{0.1, 0.1}});
  EXPECT_FALSE(b.isNodeActive(4, 8));
  EXPECT_TRUE(b.isNodeActive(0, 0));
  EXPECT_TRUE(b.isNodeActive(8, 16));
  EXPECT_FALSE(b.isRepresentable(1.0, 2.0));
  EXPECT_THROW(b.addMeasurement(1.0, 2.0, 3.0), std::invalid_argument);
  EXPECT_THROW(b.queryMean(points({{1.0, 2.0}})), std::invalid_argument);
  // Inactive nodes are decoupled: all of their edges are cut.
  EXPECT_TRUE(b.isHorizontalEdgeCut(3, 8));
  EXPECT_TRUE(b.isHorizontalEdgeCut(4, 8));
  EXPECT_TRUE(b.isVerticalEdgeCut(4, 7));
  EXPECT_TRUE(b.isVerticalEdgeCut(4, 8));
  // Points near the box on the outside are still representable.
  EXPECT_TRUE(b.isRepresentable(0.55, 2.0));
}

TEST(WallsBasic, SeparatorsWithoutEndpointExtensionReachTheBoundary) {
  // wall_endpoint_extension = 0: walls ending exactly on the domain boundary
  // must still cut the boundary row / column (no leak along the edge).
  gm::GmrfConfig c = smallConfig();
  c.wall_endpoint_extension = 0.0;
  {
    gm::GmrfBelief b(c, {{1.1, 0.0, 1.1, 4.0}});  // vertical, y_min to y_max
    for (int j = 0; j < b.ny(); ++j) {
      EXPECT_TRUE(b.isHorizontalEdgeCut(4, j)) << j;
    }
    EXPECT_FALSE(
        component(b, 0, 0)[static_cast<std::size_t>(b.nodeIndex(8, 0))]);
    Eigen::VectorXd m0;
    Eigen::VectorXd v0;
    nodeMeanVar(b, m0, v0);
    b.addMeasurement(0.2, 0.0, 5.0);  // on the bottom row, west side
    b.update();
    Eigen::VectorXd m1;
    Eigen::VectorXd v1;
    nodeMeanVar(b, m1, v1);
    for (int j = 0; j < b.ny(); ++j) {
      for (int i = 5; i < b.nx(); ++i) {
        EXPECT_NEAR(m1(b.nodeIndex(i, j)), m0(b.nodeIndex(i, j)), 1e-12);
        EXPECT_NEAR(v1(b.nodeIndex(i, j)), v0(b.nodeIndex(i, j)), 1e-12);
      }
    }
  }
  {
    gm::GmrfBelief b(c, {{0.0, 2.1, 2.0, 2.1}});  // horizontal, x_min to x_max
    for (int i = 0; i < b.nx(); ++i) {
      EXPECT_TRUE(b.isVerticalEdgeCut(i, 8)) << i;
    }
    EXPECT_FALSE(
        component(b, 0, 0)[static_cast<std::size_t>(b.nodeIndex(0, 16))]);
    Eigen::VectorXd m0;
    Eigen::VectorXd v0;
    nodeMeanVar(b, m0, v0);
    b.addMeasurement(0.0, 1.0, 5.0);  // on the left column, south side
    b.update();
    Eigen::VectorXd m1;
    Eigen::VectorXd v1;
    nodeMeanVar(b, m1, v1);
    for (int j = 9; j < b.ny(); ++j) {
      for (int i = 0; i < b.nx(); ++i) {
        EXPECT_NEAR(m1(b.nodeIndex(i, j)), m0(b.nodeIndex(i, j)), 1e-12);
        EXPECT_NEAR(v1(b.nodeIndex(i, j)), v0(b.nodeIndex(i, j)), 1e-12);
      }
    }
    // A point on the left column just north of the wall sees only north nodes.
    EXPECT_NEAR(b.queryMean(points({{0.0, 2.15}}))(0), c.prior_mean, 1e-12);
  }
}
