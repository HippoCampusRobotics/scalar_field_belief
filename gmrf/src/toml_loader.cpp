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

#include "scalar_field_belief/gmrf/toml_loader.hpp"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <toml++/toml.hpp>

namespace scalar_field_belief::gmrf {
namespace {

std::string where(const std::string& src, std::string_view table,
                  std::string_view key) {
  return src + ": [" + std::string(table) + "] " + std::string(key);
}

const toml::table* subTable(const toml::table& root, std::string_view name,
                            const std::string& src) {
  const toml::node* node = root.get(name);
  if (!node) {
    return nullptr;
  }
  const toml::table* t = node->as_table();
  if (!t) {
    throw std::invalid_argument(src + ": '" + std::string(name) +
                                "' must be a table.");
  }
  return t;
}

void checkKeys(const toml::table& t, std::string_view table,
               const std::string& src,
               std::initializer_list<std::string_view> allowed) {
  for (auto&& [key, value] : t) {
    (void)value;
    bool ok = false;
    for (std::string_view a : allowed) {
      if (key.str() == a) {
        ok = true;
        break;
      }
    }
    if (!ok) {
      throw std::invalid_argument(where(src, table, key.str()) +
                                  " is not a known key.");
    }
  }
}

double readNumber(const toml::table* t, std::string_view table,
                  std::string_view key, double def, const std::string& src,
                  bool* present = nullptr) {
  if (present) {
    *present = false;
  }
  if (!t) {
    return def;
  }
  const toml::node* node = t->get(key);
  if (!node) {
    return def;
  }
  double v = 0.0;
  if (const auto* f = node->as_floating_point()) {
    v = f->get();
  } else if (const auto* i = node->as_integer()) {
    v = static_cast<double>(i->get());
  } else {
    throw std::invalid_argument(where(src, table, key) + " must be a number.");
  }
  if (!std::isfinite(v)) {
    throw std::invalid_argument(where(src, table, key) + " must be finite.");
  }
  if (present) {
    *present = true;
  }
  return v;
}

int readInt(const toml::table* t, std::string_view table, std::string_view key,
            int def, const std::string& src) {
  if (!t) {
    return def;
  }
  const toml::node* node = t->get(key);
  if (!node) {
    return def;
  }
  const auto* i = node->as_integer();
  if (!i) {
    throw std::invalid_argument(where(src, table, key) +
                                " must be an integer.");
  }
  const auto v = i->get();
  if (v < std::numeric_limits<int>::min() ||
      v > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(where(src, table, key) + " is out of range.");
  }
  return static_cast<int>(v);
}

bool readBool(const toml::table* t, std::string_view table,
              std::string_view key, bool def, const std::string& src) {
  if (!t) {
    return def;
  }
  const toml::node* node = t->get(key);
  if (!node) {
    return def;
  }
  const auto* b = node->as_boolean();
  if (!b) {
    throw std::invalid_argument(where(src, table, key) + " must be a boolean.");
  }
  return b->get();
}

toml::table parseText(std::string_view text, const std::string& src) {
  try {
    return toml::parse(text, src);
  } catch (const toml::parse_error& err) {
    throw std::runtime_error(
        src + ": TOML syntax error: " + std::string(err.description()) +
        " (line " + std::to_string(err.source().begin.line) + ")");
  }
}

toml::table parseFile(const std::string& path) {
  try {
    return toml::parse_file(path);
  } catch (const toml::parse_error& err) {
    throw std::runtime_error(
        path + ": cannot read/parse TOML: " + std::string(err.description()) +
        " (line " + std::to_string(err.source().begin.line) + ")");
  }
}

GmrfConfig configFromTable(const toml::table& root, const std::string& src) {
  for (auto&& [key, value] : root) {
    const std::string_view k = key.str();
    if (k != "general" && k != "field_params" && k != "flow_field_params" &&
        k != "energies" && k != "solver" && k != "source_loc") {
      throw std::invalid_argument(src + ": unknown table or key '" +
                                  std::string(k) + "'.");
    }
    if (!value.is_table()) {
      throw std::invalid_argument(src + ": '" + std::string(k) +
                                  "' must be a table.");
    }
  }
  GmrfConfig c;
  const toml::table* g = subTable(root, "general", src);
  const toml::table* fp = subTable(root, "field_params", src);
  const toml::table* ff = subTable(root, "flow_field_params", src);
  const toml::table* en = subTable(root, "energies", src);
  const toml::table* so = subTable(root, "solver", src);

  if (g) {
    checkKeys(*g, "general", src,
              {"N_X", "N_Y", "size_x", "size_y", "x_min", "y_min", "x_max",
               "y_max", "wall_endpoint_extension"});
  }
  c.nx = readInt(g, "general", "N_X", c.nx, src);
  c.ny = readInt(g, "general", "N_Y", c.ny, src);
  const double def_span_x = c.x_max - c.x_min;
  const double def_span_y = c.y_max - c.y_min;
  c.x_min = readNumber(g, "general", "x_min", c.x_min, src);
  c.y_min = readNumber(g, "general", "y_min", c.y_min, src);
  bool has_size = false;
  bool has_max = false;
  const double size_x =
      readNumber(g, "general", "size_x", def_span_x, src, &has_size);
  const double x_max = readNumber(g, "general", "x_max", 0.0, src, &has_max);
  if (has_size && has_max) {
    throw std::invalid_argument(
        src + ": [general] give either size_x or x_max, not both.");
  }
  c.x_max = has_max ? x_max : c.x_min + size_x;
  const double size_y =
      readNumber(g, "general", "size_y", def_span_y, src, &has_size);
  const double y_max = readNumber(g, "general", "y_max", 0.0, src, &has_max);
  if (has_size && has_max) {
    throw std::invalid_argument(
        src + ": [general] give either size_y or y_max, not both.");
  }
  c.y_max = has_max ? y_max : c.y_min + size_y;
  c.wall_endpoint_extension = readNumber(
      g, "general", "wall_endpoint_extension", c.wall_endpoint_extension, src);

  if (fp) {
    checkKeys(*fp, "field_params", src,
              {"prior_mean", "prior_sigma", "sigma_zeta", "sigma_p", "sigma_s",
               "sigma_flow", "enable_measurement_aging", "edge_prior_var"});
    readNumber(fp, "field_params", "edge_prior_var", 0.0,
               src);  // type-checked, unused
  }
  c.prior_mean =
      readNumber(fp, "field_params", "prior_mean", c.prior_mean, src);
  c.prior_sigma =
      readNumber(fp, "field_params", "prior_sigma", c.prior_sigma, src);
  c.sigma_zeta =
      readNumber(fp, "field_params", "sigma_zeta", c.sigma_zeta, src);
  c.sigma_p_per_m =
      readNumber(fp, "field_params", "sigma_p", c.sigma_p_per_m, src);
  c.sigma_s = readNumber(fp, "field_params", "sigma_s", c.sigma_s, src);
  c.sigma_flow =
      readNumber(fp, "field_params", "sigma_flow", c.sigma_flow, src);
  c.enable_measurement_aging =
      readBool(fp, "field_params", "enable_measurement_aging",
               c.enable_measurement_aging, src);

  if (ff) {
    checkKeys(*ff, "flow_field_params", src,
              {"prior_mean_x", "prior_mean_y", "prior_sigma", "sigma_zeta",
               "sigma_p", "sigma_s", "sigma_c", "sigma_o", "edge_prior_var"});
    readNumber(ff, "flow_field_params", "edge_prior_var", 0.0, src);  // unused
  }
  c.ff_prior_mean_x = readNumber(ff, "flow_field_params", "prior_mean_x",
                                 c.ff_prior_mean_x, src);
  c.ff_prior_mean_y = readNumber(ff, "flow_field_params", "prior_mean_y",
                                 c.ff_prior_mean_y, src);
  c.ff_prior_sigma =
      readNumber(ff, "flow_field_params", "prior_sigma", c.ff_prior_sigma, src);
  c.ff_sigma_zeta =
      readNumber(ff, "flow_field_params", "sigma_zeta", c.ff_sigma_zeta, src);
  c.ff_sigma_p_per_m =
      readNumber(ff, "flow_field_params", "sigma_p", c.ff_sigma_p_per_m, src);
  c.ff_sigma_s =
      readNumber(ff, "flow_field_params", "sigma_s", c.ff_sigma_s, src);
  c.ff_sigma_c =
      readNumber(ff, "flow_field_params", "sigma_c", c.ff_sigma_c, src);
  c.ff_sigma_o =
      readNumber(ff, "flow_field_params", "sigma_o", c.ff_sigma_o, src);

  if (en) {
    checkKeys(*en, "energies", src,
              {"field_neighbors", "flow_neighbors", "flow_conservation",
               "obstacle_flow", "flow_field", "force_joint_state"});
  }
  c.field_neighbors =
      readBool(en, "energies", "field_neighbors", c.field_neighbors, src);
  c.flow_neighbors =
      readBool(en, "energies", "flow_neighbors", c.flow_neighbors, src);
  c.flow_conservation =
      readBool(en, "energies", "flow_conservation", c.flow_conservation, src);
  c.obstacle_flow =
      readBool(en, "energies", "obstacle_flow", c.obstacle_flow, src);
  c.flow_field = readBool(en, "energies", "flow_field", c.flow_field, src);
  c.force_joint_state =
      readBool(en, "energies", "force_joint_state", c.force_joint_state, src);

  if (so) {
    checkKeys(*so, "solver", src,
              {"max_gauss_newton_iterations", "step_tolerance",
               "relative_cost_tolerance", "min_step_fraction"});
  }
  c.max_gauss_newton_iterations =
      readInt(so, "solver", "max_gauss_newton_iterations",
              c.max_gauss_newton_iterations, src);
  c.gn_step_tolerance =
      readNumber(so, "solver", "step_tolerance", c.gn_step_tolerance, src);
  c.gn_relative_cost_tolerance =
      readNumber(so, "solver", "relative_cost_tolerance",
                 c.gn_relative_cost_tolerance, src);
  c.gn_min_step_fraction = readNumber(so, "solver", "min_step_fraction",
                                      c.gn_min_step_fraction, src);

  try {
    c.validate();
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(src + ": " + e.what());
  }
  return c;
}

WallFile wallsFromTable(const toml::table& root, const std::string& src) {
  WallFile out;
  if (const toml::table* g = subTable(root, "general", src)) {
    if (const toml::node* name = g->get("name")) {
      const auto* s = name->as_string();
      if (!s) {
        throw std::invalid_argument(where(src, "general", "name") +
                                    " must be a string.");
      }
      out.name = s->get();
    }
  }
  if (const toml::table* b = subTable(root, "bounds", src)) {
    for (std::string_view k : {"x_min", "x_max", "y_min", "y_max"}) {
      if (!b->get(k)) {
        throw std::invalid_argument(where(src, "bounds", k) + " is missing.");
      }
    }
    out.bounds.x_min = readNumber(b, "bounds", "x_min", 0.0, src);
    out.bounds.x_max = readNumber(b, "bounds", "x_max", 0.0, src);
    out.bounds.y_min = readNumber(b, "bounds", "y_min", 0.0, src);
    out.bounds.y_max = readNumber(b, "bounds", "y_max", 0.0, src);
    if (!(out.bounds.x_max > out.bounds.x_min) ||
        !(out.bounds.y_max > out.bounds.y_min)) {
      throw std::invalid_argument(
          src + ": [bounds] must satisfy x_max > x_min, y_max > y_min.");
    }
    out.has_bounds = true;
  }
  const toml::table* objects = subTable(root, "objects", src);
  const toml::table* walls =
      objects ? subTable(*objects, "walls", src) : nullptr;
  if (!walls) {
    return out;
  }
  const toml::node* verts = walls->get("wall_vertices");
  if (verts) {
    const toml::array* arr = verts->as_array();
    if (!arr) {
      throw std::invalid_argument(where(src, "objects.walls", "wall_vertices") +
                                  " must be an array.");
    }
    for (std::size_t k = 0; k < arr->size(); ++k) {
      const std::string item = where(src, "objects.walls", "wall_vertices") +
                               "[" + std::to_string(k) + "]";
      const toml::array* row = (*arr)[k].as_array();
      if (!row || row->size() != 4) {
        throw std::invalid_argument(
            item + " must be an array of 4 numbers [x0, y0, x1, y1].");
      }
      double v[4];
      for (std::size_t c = 0; c < 4; ++c) {
        const toml::node& n = (*row)[c];
        if (const auto* f = n.as_floating_point()) {
          v[c] = f->get();
        } else if (const auto* i = n.as_integer()) {
          v[c] = static_cast<double>(i->get());
        } else {
          throw std::invalid_argument(item + " must contain only numbers.");
        }
        if (!std::isfinite(v[c])) {
          throw std::invalid_argument(item +
                                      " must contain only finite numbers.");
        }
      }
      if (v[0] == v[2] && v[1] == v[3]) {
        throw std::invalid_argument(item + " has zero length.");
      }
      out.walls.push_back(WallSegment{v[0], v[1], v[2], v[3]});
    }
  }
  if (const toml::node* nw = walls->get("num_walls")) {
    const auto* i = nw->as_integer();
    if (!i) {
      throw std::invalid_argument(where(src, "objects.walls", "num_walls") +
                                  " must be an integer.");
    }
    if (i->get() != static_cast<std::int64_t>(out.walls.size())) {
      throw std::invalid_argument(
          where(src, "objects.walls", "num_walls") + " = " +
          std::to_string(i->get()) +
          " does not match the number of wall_vertices rows (" +
          std::to_string(out.walls.size()) + ").");
    }
  }
  return out;
}

}  // namespace

GmrfConfig loadGmrfConfigToml(const std::string& path) {
  return configFromTable(parseFile(path), path);
}

GmrfConfig parseGmrfConfigToml(std::string_view text,
                               const std::string& source_name) {
  return configFromTable(parseText(text, source_name), source_name);
}

WallFile loadWallsToml(const std::string& path) {
  return wallsFromTable(parseFile(path), path);
}

WallFile parseWallsToml(std::string_view text, const std::string& source_name) {
  return wallsFromTable(parseText(text, source_name), source_name);
}

}  // namespace scalar_field_belief::gmrf
