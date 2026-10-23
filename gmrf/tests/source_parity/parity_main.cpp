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

// Source-parity harness: student's original GMRF (unmodified gmrf.cpp) vs the
// port, on small deterministic scalar-only cases with domain origin (0, 0).
// Compares nodal means and nodal marginal variances (student:
// covariances[field]; port: fieldNodeMean() and queryMeanVariance() at the
// node positions). Exit code 0 iff every comparison is within tolerance.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "scalar_field_belief/gmrf/gmrf_belief.hpp"
#include "student_driver.hpp"

namespace gm = scalar_field_belief::gmrf;

namespace {

constexpr double kTol = 1e-8;  // relative

struct PortResult {
  std::vector<double> mean, var;
};

PortResult runPort(const parity::Case& c,
                   const std::vector<std::array<double, 4>>& walls,
                   const std::vector<gm::Point2>& seeds) {
  gm::GmrfConfig cfg;
  cfg.x_min = 0.0;
  cfg.y_min = 0.0;
  cfg.x_max = c.size_x;
  cfg.y_max = c.size_y;
  cfg.nx = c.nx;
  cfg.ny = c.ny;
  cfg.prior_mean = c.prior_mean;
  cfg.prior_sigma = c.prior_sigma;
  cfg.sigma_p_per_m = c.sigma_p_per_m;
  cfg.sigma_s = c.sigma_s;
  std::vector<gm::WallSegment> ws;
  for (const auto& w : walls) {
    ws.push_back({w[0], w[1], w[2], w[3]});
  }
  gm::GmrfBelief b(cfg, ws, seeds);
  for (const auto& batch : c.batches) {
    for (const auto& m : batch) {
      b.addMeasurement(m[0], m[1], m[2]);
    }
    b.update();
  }
  // Inactive nodes (free-space mask) are not representable query points; their
  // variance is reported as NaN and checked separately.
  std::vector<Eigen::Index> rows;
  std::vector<std::array<double, 2>> pts;
  for (int j = 0; j < c.ny; ++j) {
    for (int i = 0; i < c.nx; ++i) {
      const auto p = b.nodePosition(i, j);
      if (b.isRepresentable(p.x, p.y)) {
        rows.push_back(b.nodeIndex(i, j));
        pts.push_back({p.x, p.y});
      }
    }
  }
  Eigen::MatrixX2d xy(static_cast<Eigen::Index>(pts.size()), 2);
  for (std::size_t k = 0; k < pts.size(); ++k) {
    xy(static_cast<Eigen::Index>(k), 0) = pts[k][0];
    xy(static_cast<Eigen::Index>(k), 1) = pts[k][1];
  }
  Eigen::VectorXd qm, qv;
  b.queryMeanVariance(xy, qm, qv);
  const Eigen::VectorXd nm = b.fieldNodeMean();
  PortResult r;
  r.mean.assign(nm.data(), nm.data() + nm.size());
  r.var.assign(static_cast<std::size_t>(nm.size()), std::nan(""));
  for (std::size_t k = 0; k < rows.size(); ++k) {
    const auto n = rows[k];
    r.var[static_cast<std::size_t>(n)] = qv(static_cast<Eigen::Index>(k));
    // Mean at node positions must equal the nodal mean (consistency of H at
    // nodes).
    if (std::abs(qm(static_cast<Eigen::Index>(k)) - nm(n)) >
        1e-12 * std::max(1.0, std::abs(nm(n)))) {
      std::printf("  WARNING: query mean at node %ld differs from nodal mean\n",
                  (long)n);
    }
  }
  return r;
}

struct Stats {
  double abs_mean = 0, rel_mean = 0, abs_var = 0, rel_var = 0;
  int n = 0;
  int n_inactive = 0;
  bool inactive_ok = true;
};

const parity::Case* g_case = nullptr;

Stats compare(const parity::NodalResult& s, const PortResult& p,
              const std::function<bool(int)>& use) {
  Stats st;
  double mscale = 0.0;
  for (std::size_t k = 0; k < s.mean.size(); ++k) {
    if (use(static_cast<int>(k))) {
      mscale = std::max(mscale, std::abs(s.mean[k]));
    }
  }
  mscale = std::max(mscale, 1e-12);
  for (std::size_t k = 0; k < s.mean.size(); ++k) {
    if (!use(static_cast<int>(k))) {
      continue;
    }
    if (std::isnan(p.var[k])) {
      // Inactive in the port (not queryable): student must hold the isolated
      // prior there and the port's nodal mean must equal it.
      const auto& c = *g_case;
      const bool ok = std::abs(s.var[k] - c.prior_sigma * c.prior_sigma) <=
                          kTol * c.prior_sigma * c.prior_sigma &&
                      std::abs(s.mean[k] - c.prior_mean) <=
                          kTol * std::max(1.0, std::abs(c.prior_mean)) &&
                      std::abs(p.mean[k] - c.prior_mean) <=
                          kTol * std::max(1.0, std::abs(c.prior_mean));
      ++st.n_inactive;
      st.inactive_ok &= ok;
      continue;
    }
    ++st.n;
    const double dm = std::abs(s.mean[k] - p.mean[k]);
    const double dv = std::abs(s.var[k] - p.var[k]);
    st.abs_mean = std::max(st.abs_mean, dm);
    st.rel_mean = std::max(st.rel_mean, dm / mscale);
    st.abs_var = std::max(st.abs_var, dv);
    st.rel_var = std::max(st.rel_var, dv / std::abs(s.var[k]));
  }
  return st;
}

bool report(const std::string& name, const Stats& st) {
  const bool ok =
      st.n > 0 && st.rel_mean < kTol && st.rel_var < kTol && st.inactive_ok;
  std::printf(
      "%-52s nodes=%3d  mean: max_abs=%.3e max_rel=%.3e  var: max_abs=%.3e "
      "max_rel=%.3e  %s\n",
      name.c_str(), st.n, st.abs_mean, st.rel_mean, st.abs_var, st.rel_var,
      ok ? "PASS" : "FAIL");
  if (st.n_inactive > 0) {
    std::printf(
        "    + %d port-inactive nodes: student var == prior_sigma^2 and both "
        "means == "
        "prior_mean: %s\n",
        st.n_inactive, st.inactive_ok ? "yes" : "NO");
  }
  return ok;
}

parity::Case baseCase() {
  parity::Case c;
  c.size_x = 2.0;
  c.size_y = 1.8;
  c.nx = 5;  // dx = 0.5
  c.ny = 4;  // dy = 0.6 (non-square; sigma_p uses dx for both directions)
  c.prior_mean = 0.1;
  c.prior_sigma = 1.5;
  c.sigma_p_per_m = 2.0;
  c.sigma_s = 0.2;
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const char* toml_path = argc > 1 ? argv[1] : GMRF_PARITY_TOML;
  bool all_ok = true;
  auto all = [](int) { return true; };

  // --- Scalar-only, no wall -----------------------------------------------
  {
    parity::Case c = baseCase();
    g_case = &c;
    const auto s = parity::runStudent(c, toml_path);
    all_ok &= report("S0 prior only (5x4, dx=0.5, dy=0.6)",
                     compare(s, runPort(c, {}, {}), all));
  }
  {
    parity::Case c = baseCase();
    c.batches = {{{0.3, 0.4, 1.0},
                  {1.7, 1.1, -0.5},
                  {0.9, 0.9, 0.7},
                  {2.0, 0.9, 0.2},
                  {1.3, 1.8, -0.3}},
                 {{2.0, 1.8, 0.5},
                  {0.3, 0.4, 1.2},
                  {0.05, 1.75, 0.4},
                  {0.0, 0.0, -0.2}}};
    g_case = &c;
    const auto s = parity::runStudent(c, toml_path);
    all_ok &= report("S1 2 batches, edges/corners, repeat (5x4)",
                     compare(s, runPort(c, {}, {}), all));
  }
  {
    parity::Case c;
    c.size_x = 3.0;
    c.size_y = 2.5;
    c.nx = 7;
    c.ny = 6;
    c.prior_mean = -0.3;
    c.prior_sigma = 3.0;
    c.sigma_p_per_m = 0.8;
    c.sigma_s = 0.05;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> ux(0.0, 3.0), uy(0.0, 2.5),
        uv(-2.0, 2.0);
    for (int b = 0; b < 3; ++b) {
      std::vector<std::array<double, 3>> batch;
      for (int k = 0; k < 6; ++k) {
        batch.push_back({ux(rng), uy(rng), uv(rng)});
      }
      c.batches.push_back(batch);
    }
    g_case = &c;
    const auto s = parity::runStudent(c, toml_path);
    all_ok &= report("S2 random 18 obs in 3 batches (7x6)",
                     compare(s, runPort(c, {}, {}), all));
  }

  // --- Axis-aligned full walls not on a grid line ---------------------------
  // The student needs flow_neighbors on: its cut handler also writes rows
  // L+num_edges and L+2*num_edges, which only exist with the FlowN block.
  // With flow_field off, the flow block is decoupled from the field, so field
  // marginals are unaffected. The student's reachability is seeded at node 0;
  // the port reproduces that only with free_space_seeds = {(0, 0)}.
  struct WallCase {
    const char* name;
    std::array<double, 4> wall;
    std::vector<std::array<double, 3>> obs;
    std::function<bool(int)>
        seed_side;  // node index -> in node 0's compartment
  };
  const int nx = baseCase().nx;
  const std::vector<WallCase> wall_cases{
      {"W1 vertical wall x=0.9 (full height)",
       {0.9, 0.0, 0.9, 1.8},
       {{0.3, 0.4, 1.0}, {0.8, 1.1, -0.5}, {0.85, 0.2, 0.7}, {0.0, 1.8, 0.3}},
       [nx](int n) { return n % nx <= 1; }},
      {"W2 horizontal wall y=0.9 (full width)",
       {0.0, 0.9, 2.0, 0.9},
       {{0.3, 0.4, 1.0}, {1.6, 0.85, -0.5}, {1.0, 0.2, 0.3}, {2.0, 0.0, 0.6}},
       [nx](int n) { return n / nx <= 1; }},
  };
  for (const auto& wc : wall_cases) {
    parity::Case c = baseCase();
    c.flow_neighbors = true;
    c.walls = {wc.wall};
    c.batches = {wc.obs};
    g_case = &c;
    const auto s = parity::runStudent(c, toml_path);
    const auto& ws = s.walls_as_stored;  // float-rounded by sf::Vertex
    std::printf("%s: student stores wall as (%.17g, %.17g)-(%.17g, %.17g)\n",
                wc.name, ws[0][0], ws[0][1], ws[0][2], ws[0][3]);
    all_ok &= report(std::string(wc.name) + " seeded (0,0), all nodes",
                     compare(s, runPort(c, ws, {{0.0, 0.0}}), all));
    all_ok &= report(std::string(wc.name) + " unseeded, node-0 side",
                     compare(s, runPort(c, ws, {}), wc.seed_side));
    // Show the documented difference on the other side (not a pass criterion).
    const Stats other =
        compare(s, runPort(c, ws, {}), [&](int n) { return !wc.seed_side(n); });
    std::printf(
        "    (info) unseeded port vs student on the far side: var max_rel=%.3e "
        "(student isolates unreachable nodes)\n",
        other.rel_var);
  }

  std::printf("%s\n",
              all_ok ? "SOURCE PARITY: ALL PASS" : "SOURCE PARITY: FAILURES");
  return all_ok ? 0 : 1;
}
