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

// TOML config / wall loaders (optional target scalar_field_belief::gmrf_toml).
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "scalar_field_belief/gmrf/toml_loader.hpp"

namespace gm = scalar_field_belief::gmrf;

TEST(TomlLoader, ReadsSourceConfigLayout) {
  const gm::GmrfConfig c = gm::loadGmrfConfigToml(
      std::string(GMRF_TEST_DATA_DIR) + "/gmrf_conf_source_layout.toml");
  EXPECT_EQ(c.nx, 50);
  EXPECT_EQ(c.ny, 25);
  EXPECT_DOUBLE_EQ(c.x_min, 0.0);
  EXPECT_DOUBLE_EQ(c.x_max, 15.0);
  EXPECT_DOUBLE_EQ(c.y_min, 0.0);
  EXPECT_DOUBLE_EQ(c.y_max, 10.0);
  EXPECT_DOUBLE_EQ(c.prior_sigma, 100.0);
  EXPECT_DOUBLE_EQ(c.sigma_p_per_m, 0.8);
  EXPECT_DOUBLE_EQ(c.sigma_s, 0.15);
  EXPECT_DOUBLE_EQ(c.sigma_flow, 0.03);
  EXPECT_DOUBLE_EQ(c.ff_prior_sigma, 10.0);
  EXPECT_DOUBLE_EQ(c.ff_sigma_p_per_m, 0.15);
  EXPECT_DOUBLE_EQ(c.ff_sigma_s, 0.05);
  EXPECT_DOUBLE_EQ(c.ff_sigma_c, 0.04);
  EXPECT_DOUBLE_EQ(c.ff_sigma_o, 0.1);
  EXPECT_TRUE(c.field_neighbors && c.flow_neighbors && c.flow_conservation &&
              c.obstacle_flow && c.flow_field);
}

TEST(TomlLoader, MissingKeysUseStationaryScalarDefaults) {
  const gm::GmrfConfig c = gm::parseGmrfConfigToml(
      "[general]\nN_X = 5\nN_Y = 6\nx_min = -1\nsize_x = 2.0\ny_max = 3\n");
  const gm::GmrfConfig d;
  EXPECT_EQ(c.nx, 5);
  EXPECT_EQ(c.ny, 6);
  EXPECT_DOUBLE_EQ(c.x_min, -1.0);
  EXPECT_DOUBLE_EQ(c.x_max, 1.0);
  EXPECT_DOUBLE_EQ(c.y_min, 0.0);
  EXPECT_DOUBLE_EQ(c.y_max, 3.0);
  EXPECT_DOUBLE_EQ(c.prior_sigma, d.prior_sigma);
  EXPECT_TRUE(c.field_neighbors);
  EXPECT_FALSE(c.usesJointState());
  EXPECT_FALSE(c.enable_measurement_aging);
}

TEST(TomlLoader, RejectsBadInput) {
  const char* bad[] = {
      "[general]\nN_X = 5.5\n",                  // integer expected
      "[general]\nN_X = 1\n",                    // fails validation
      "[field_params]\nsigma_s = \"x\"\n",       // number expected
      "[field_params]\nsigma_s = 0.0\n",         // fails validation
      "[energies]\nflow_field = 1\n",            // boolean expected
      "[field_params]\nsigma_sss = 1.0\n",       // unknown key
      "[fieldparams]\nsigma_s = 1.0\n",          // unknown table
      "[general]\nsize_x = 2.0\nx_max = 2.0\n",  // ambiguous extent
  };
  for (const char* text : bad) {
    EXPECT_THROW(gm::parseGmrfConfigToml(text), std::invalid_argument) << text;
  }
  EXPECT_THROW(gm::parseGmrfConfigToml("[general\nN_X = 5\n"),
               std::runtime_error);
  EXPECT_THROW(gm::loadGmrfConfigToml("/nonexistent/gmrf.toml"),
               std::runtime_error);
}

TEST(TomlLoader, ReadsWallFile) {
  const gm::WallFile f =
      gm::loadWallsToml(std::string(GMRF_TEST_DATA_DIR) + "/walls_simple.toml");
  EXPECT_EQ(f.name, "simple_wall_case");
  ASSERT_TRUE(f.has_bounds);
  EXPECT_DOUBLE_EQ(f.bounds.x_max, 2.0);
  EXPECT_DOUBLE_EQ(f.bounds.y_max, 4.0);
  ASSERT_EQ(f.walls.size(), 2u);
  EXPECT_DOUBLE_EQ(f.walls[1].x0, 1.0);
  EXPECT_DOUBLE_EQ(f.walls[1].y0, 2.1);
  EXPECT_DOUBLE_EQ(f.walls[1].x1, 1.5);
  EXPECT_DOUBLE_EQ(f.walls[1].y1, 2.1);
}

TEST(TomlLoader, WallFileVariantsAndErrors) {
  // Source map style with num_walls and unrelated tables; no bounds.
  const gm::WallFile f = gm::parseWallsToml(
      "[general]\nhas_obstacles = true\n[value_field]\nvalues = [[1.0]]\n"
      "[objects.walls]\nnum_walls = 1\nwall_vertices = [[0, 0, 1, 1]]\n");
  EXPECT_FALSE(f.has_bounds);
  ASSERT_EQ(f.walls.size(), 1u);
  EXPECT_TRUE(
      gm::parseWallsToml("[general]\nname = \"empty\"\n").walls.empty());

  const char* bad[] = {
      "[objects.walls]\nwall_vertices = [[0.0, 0.0, 1.0]]\n",
      "[objects.walls]\nwall_vertices = [[0.0, 0.0, 1.0, \"a\"]]\n",
      "[objects.walls]\nwall_vertices = [[1.0, 1.0, 1.0, 1.0]]\n",
      "[objects.walls]\nnum_walls = 2\nwall_vertices = [[0.0, 0.0, 1.0, "
      "1.0]]\n",
      "[bounds]\nx_min = 0.0\nx_max = 1.0\ny_min = 0.0\n",
      "[bounds]\nx_min = 1.0\nx_max = 0.0\ny_min = 0.0\ny_max = 1.0\n",
  };
  for (const char* text : bad) {
    EXPECT_THROW(gm::parseWallsToml(text), std::invalid_argument) << text;
  }
}
