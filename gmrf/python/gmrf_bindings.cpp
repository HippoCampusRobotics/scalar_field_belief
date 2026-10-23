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

// Minimal pybind11 bindings for the scalar GMRF belief
// (scalar_field_belief::gmrf). Only the existing public C++ API is exposed;
// no numerics live here.
//
// Threading: every GmrfBelief method releases the GIL and then takes a
// per-object mutex, so a long update()/query running in one Python thread can
// neither race another call on the same belief nor block unrelated Python
// threads. The GIL is always released *before* the mutex is taken (never the
// other way round), which rules out a GIL/mutex deadlock.
//
// Exceptions use pybind11's default translation:
//   std::invalid_argument -> ValueError, std::out_of_range -> IndexError,
//   std::logic_error / std::runtime_error -> RuntimeError.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "scalar_field_belief/gmrf/config.hpp"
#include "scalar_field_belief/gmrf/gmrf_belief.hpp"

namespace py = pybind11;
namespace gm = scalar_field_belief::gmrf;

namespace {

using XyArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

/// Copies an (N, 2) float64 array (N = 0 allowed) into an owned Eigen matrix.
/// Called with the GIL held.
Eigen::MatrixX2d toXy(const XyArray& xy) {
  if (xy.ndim() != 2 || xy.shape(1) != 2) {
    std::ostringstream msg;
    msg << "xy must have shape (N, 2), got ndim=" << xy.ndim();
    if (xy.ndim() >= 1) {
      msg << " shape=(";
      for (py::ssize_t k = 0; k < xy.ndim(); ++k) {
        msg << (k ? ", " : "") << xy.shape(k);
      }
      msg << ")";
    }
    throw std::invalid_argument(msg.str());
  }
  const auto n = static_cast<Eigen::Index>(xy.shape(0));
  Eigen::MatrixX2d out(n, 2);
  auto r = xy.unchecked<2>();
  for (Eigen::Index i = 0; i < n; ++i) {
    out(i, 0) = r(i, 0);
    out(i, 1) = r(i, 1);
  }
  return out;
}

/// New owned 1-D numpy array (N,). Called with the GIL held.
py::array_t<double> toArray(const Eigen::VectorXd& v) {
  py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
  if (v.size() > 0) {
    std::memcpy(out.mutable_data(), v.data(),
                static_cast<std::size_t>(v.size()) * sizeof(double));
  }
  return out;
}

/// New owned 2-D numpy array (rows, cols), C order. Called with the GIL held.
py::array_t<double> toArray(const Eigen::MatrixXd& m) {
  py::array_t<double> out(
      {static_cast<py::ssize_t>(m.rows()), static_cast<py::ssize_t>(m.cols())});
  auto w = out.mutable_unchecked<2>();
  for (Eigen::Index i = 0; i < m.rows(); ++i) {
    for (Eigen::Index j = 0; j < m.cols(); ++j) {
      w(i, j) = m(i, j);
    }
  }
  return out;
}

/// GmrfBelief plus the mutex that serialises all access from Python.
struct PyGmrfBelief {
  PyGmrfBelief(const gm::GmrfConfig& cfg, std::vector<gm::WallSegment> walls,
               std::vector<gm::Point2> seeds)
      : belief(cfg, std::move(walls), std::move(seeds)) {}

  /// Runs fn(belief) with the GIL released and the mutex held.
  template <class Fn>
  auto locked(Fn&& fn) {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex);
    return fn(belief);
  }

  gm::GmrfBelief belief;
  std::mutex mutex;
};

std::string wallRepr(const gm::WallSegment& w) {
  std::ostringstream s;
  s << "WallSegment(" << w.x0 << ", " << w.y0 << ", " << w.x1 << ", " << w.y1
    << ")";
  return s.str();
}

}  // namespace

PYBIND11_MODULE(scalar_field_belief_gmrf_py, m) {
  m.doc() =
      "Python bindings for the scalar GMRF belief "
      "(scalar_field_belief::gmrf). Positions in m, values in "
      "measurement units, variances in measurement units^2.";

  py::class_<gm::Point2>(m, "Point2")
      .def(py::init<>())
      .def(py::init([](double x, double y) { return gm::Point2{x, y}; }),
           py::arg("x"), py::arg("y"))
      .def_readwrite("x", &gm::Point2::x)
      .def_readwrite("y", &gm::Point2::y)
      .def("__repr__", [](const gm::Point2& p) {
        std::ostringstream s;
        s << "Point2(" << p.x << ", " << p.y << ")";
        return s.str();
      });

  py::class_<gm::WallSegment>(m, "WallSegment")
      .def(py::init<>())
      .def(py::init([](double x0, double y0, double x1, double y1) {
             return gm::WallSegment{x0, y0, x1, y1};
           }),
           py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"))
      .def_readwrite("x0", &gm::WallSegment::x0)
      .def_readwrite("y0", &gm::WallSegment::y0)
      .def_readwrite("x1", &gm::WallSegment::x1)
      .def_readwrite("y1", &gm::WallSegment::y1)
      .def("__repr__", &wallRepr);

  py::class_<gm::GmrfConfig> cfg(m, "GmrfConfig");
  cfg.def(py::init<>())
      .def("validate", &gm::GmrfConfig::validate,
           "Raise ValueError if the configuration is invalid.")
      .def("uses_joint_state", &gm::GmrfConfig::usesJointState)
      .def("is_linear", &gm::GmrfConfig::isLinear);
#define GMRF_PY_FIELD(name) cfg.def_readwrite(#name, &gm::GmrfConfig::name)
  GMRF_PY_FIELD(x_min);
  GMRF_PY_FIELD(x_max);
  GMRF_PY_FIELD(y_min);
  GMRF_PY_FIELD(y_max);
  GMRF_PY_FIELD(nx);
  GMRF_PY_FIELD(ny);
  GMRF_PY_FIELD(prior_mean);
  GMRF_PY_FIELD(prior_sigma);
  GMRF_PY_FIELD(sigma_p_per_m);
  GMRF_PY_FIELD(sigma_s);
  GMRF_PY_FIELD(sigma_zeta);
  GMRF_PY_FIELD(enable_measurement_aging);
  GMRF_PY_FIELD(sigma_flow);
  GMRF_PY_FIELD(ff_prior_mean_x);
  GMRF_PY_FIELD(ff_prior_mean_y);
  GMRF_PY_FIELD(ff_prior_sigma);
  GMRF_PY_FIELD(ff_sigma_zeta);
  GMRF_PY_FIELD(ff_sigma_p_per_m);
  GMRF_PY_FIELD(ff_sigma_s);
  GMRF_PY_FIELD(ff_sigma_c);
  GMRF_PY_FIELD(ff_sigma_o);
  GMRF_PY_FIELD(field_neighbors);
  GMRF_PY_FIELD(flow_neighbors);
  GMRF_PY_FIELD(flow_conservation);
  GMRF_PY_FIELD(flow_field);
  GMRF_PY_FIELD(obstacle_flow);
  GMRF_PY_FIELD(force_joint_state);
  GMRF_PY_FIELD(wall_endpoint_extension);
  GMRF_PY_FIELD(max_gauss_newton_iterations);
  GMRF_PY_FIELD(gn_step_tolerance);
  GMRF_PY_FIELD(gn_relative_cost_tolerance);
  GMRF_PY_FIELD(gn_min_step_fraction);
#undef GMRF_PY_FIELD

  py::class_<gm::UpdateReport>(m, "UpdateReport")
      .def_readonly("solved", &gm::UpdateReport::solved)
      .def_readonly("linear", &gm::UpdateReport::linear)
      .def_readonly("iterations", &gm::UpdateReport::iterations)
      .def_readonly("converged", &gm::UpdateReport::converged)
      .def_readonly("final_cost", &gm::UpdateReport::final_cost)
      .def_readonly("num_assimilated", &gm::UpdateReport::num_assimilated);

  using B = gm::GmrfBelief;
  py::class_<PyGmrfBelief>(m, "GmrfBelief")
      .def(py::init([](const gm::GmrfConfig& config,
                       std::vector<gm::WallSegment> walls,
                       std::vector<gm::Point2> free_space_seeds) {
             py::gil_scoped_release release;
             return std::make_unique<PyGmrfBelief>(config, std::move(walls),
                                                   std::move(free_space_seeds));
           }),
           py::arg("config"), py::arg("walls") = std::vector<gm::WallSegment>{},
           py::arg("free_space_seeds") = std::vector<gm::Point2>{})
      .def("reset",
           [](PyGmrfBelief& self) { self.locked([](B& b) { b.reset(); }); })
      .def(
          "add_measurement",
          [](PyGmrfBelief& self, double x, double y, double value, double t) {
            self.locked([&](B& b) { b.addMeasurement(x, y, value, t); });
          },
          py::arg("x"), py::arg("y"), py::arg("value"), py::arg("t") = 0.0,
          "Validate and queue one scalar measurement (call update() next).")
      .def("discard_pending_measurements",
           [](PyGmrfBelief& self) {
             self.locked([](B& b) { b.discardPendingMeasurements(); });
           })
      .def(
          "update",
          [](PyGmrfBelief& self, double t) {
            return self.locked([&](B& b) { return b.update(t); });
          },
          py::arg("t") = 0.0, "Assimilate all pending measurements.")
      .def(
          "has_pending_measurements",
          [](PyGmrfBelief& self) {
            return self.locked([](B& b) { return b.hasPendingMeasurements(); });
          })
      .def(
          "num_pending_measurements",
          [](PyGmrfBelief& self) {
            return self.locked([](B& b) { return b.numPendingMeasurements(); });
          })
      .def("num_measurements",
           [](PyGmrfBelief& self) {
             return self.locked([](B& b) { return b.numMeasurements(); });
           })
      .def(
          "query_mean",
          [](PyGmrfBelief& self, const XyArray& xy) {
            const Eigen::MatrixX2d pts = toXy(xy);
            Eigen::VectorXd mean =
                self.locked([&](B& b) { return b.queryMean(pts); });
            return toArray(mean);
          },
          py::arg("xy"))
      .def(
          "query_mean_variance",
          [](PyGmrfBelief& self, const XyArray& xy) {
            const Eigen::MatrixX2d pts = toXy(xy);
            Eigen::VectorXd mean, var;
            self.locked([&](B& b) { b.queryMeanVariance(pts, mean, var); });
            return py::make_tuple(toArray(mean), toArray(var));
          },
          py::arg("xy"),
          "Latent mean (N,) and marginal variance (N,) at (N, 2) positions.")
      .def(
          "query_mean_covariance",
          [](PyGmrfBelief& self, const XyArray& xy) {
            const Eigen::MatrixX2d pts = toXy(xy);
            Eigen::VectorXd mean;
            Eigen::MatrixXd cov;
            self.locked([&](B& b) { b.queryMeanCovariance(pts, mean, cov); });
            return py::make_tuple(toArray(mean), toArray(cov));
          },
          py::arg("xy"),
          "Latent mean (N,) and full covariance (N, N) at (N, 2) positions.")
      .def(
          "is_representable",
          [](PyGmrfBelief& self, double x, double y) {
            return self.locked([&](B& b) { return b.isRepresentable(x, y); });
          },
          py::arg("x"), py::arg("y"))
      .def(
          "representable_mask",
          [](PyGmrfBelief& self, const XyArray& xy) {
            const Eigen::MatrixX2d pts = toXy(xy);
            std::vector<char> ok(static_cast<std::size_t>(pts.rows()));
            self.locked([&](B& b) {
              for (Eigen::Index i = 0; i < pts.rows(); ++i) {
                ok[static_cast<std::size_t>(i)] =
                    b.isRepresentable(pts(i, 0), pts(i, 1));
              }
            });
            py::array_t<bool> out(static_cast<py::ssize_t>(ok.size()));
            auto w = out.mutable_unchecked<1>();
            for (std::size_t i = 0; i < ok.size(); ++i) {
              w(static_cast<py::ssize_t>(i)) = ok[i] != 0;
            }
            return out;
          },
          py::arg("xy"),
          "Boolean (N,) array: is_representable() for each row of xy.")
      .def("measurement_noise_variance",
           [](PyGmrfBelief& self) {
             return self.locked(
                 [](B& b) { return b.measurementNoiseVariance(); });
           })
      .def_property_readonly(
          "config",
          [](PyGmrfBelief& self) {
            return self.locked([](B& b) { return b.config(); });
          })
      .def_property_readonly(
          "walls",
          [](PyGmrfBelief& self) {
            return self.locked([](B& b) { return b.walls(); });
          })
      .def_property_readonly("nx",
                             [](PyGmrfBelief& self) {
                               return self.locked([](B& b) { return b.nx(); });
                             })
      .def_property_readonly("ny",
                             [](PyGmrfBelief& self) {
                               return self.locked([](B& b) { return b.ny(); });
                             })
      .def_property_readonly("dx",
                             [](PyGmrfBelief& self) {
                               return self.locked([](B& b) { return b.dx(); });
                             })
      .def_property_readonly("dy",
                             [](PyGmrfBelief& self) {
                               return self.locked([](B& b) { return b.dy(); });
                             })
      .def_property_readonly(
          "node_count",
          [](PyGmrfBelief& self) {
            return self.locked([](B& b) { return b.nodeCount(); });
          })
      .def_property_readonly(
          "is_joint_state",
          [](PyGmrfBelief& self) {
            return self.locked([](B& b) { return b.isJointState(); });
          })
      .def(
          "model_grid",
          [](PyGmrfBelief& self) {
            // Read-only bulk view of the model graph for display, built only
            // from the public accessors nodePosition / isNodeActive /
            // isHorizontalEdgeCut / isVerticalEdgeCut.
            std::vector<double> xy;
            std::vector<char> active;
            std::vector<std::int64_t> edges;
            int nx = 0, ny = 0;
            self.locked([&](B& b) {
              nx = b.nx();
              ny = b.ny();
              const auto n = static_cast<std::size_t>(nx) * ny;
              xy.reserve(2 * n);
              active.reserve(n);
              for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                  const gm::Point2 p = b.nodePosition(i, j);
                  xy.push_back(p.x);
                  xy.push_back(p.y);
                  active.push_back(b.isNodeActive(i, j) ? 1 : 0);
                }
              }
              auto on = [&](int i, int j) {
                return active[static_cast<std::size_t>(j) * nx + i] != 0;
              };
              auto add = [&](int i0, int j0, int i1, int j1) {
                edges.push_back(static_cast<std::int64_t>(j0) * nx + i0);
                edges.push_back(static_cast<std::int64_t>(j1) * nx + i1);
              };
              for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                  if (i + 1 < nx && !b.isHorizontalEdgeCut(i, j) && on(i, j) &&
                      on(i + 1, j)) {
                    add(i, j, i + 1, j);
                  }
                  if (j + 1 < ny && !b.isVerticalEdgeCut(i, j) && on(i, j) &&
                      on(i, j + 1)) {
                    add(i, j, i, j + 1);
                  }
                }
              }
            });
            const auto n = static_cast<py::ssize_t>(active.size());
            py::array_t<double> node_xy({n, static_cast<py::ssize_t>(2)});
            if (n > 0) {
              std::memcpy(node_xy.mutable_data(), xy.data(),
                          xy.size() * sizeof(double));
            }
            py::array_t<bool> node_active(n);
            auto wa = node_active.mutable_unchecked<1>();
            for (py::ssize_t k = 0; k < n; ++k) {
              wa(k) = active[static_cast<std::size_t>(k)] != 0;
            }
            const auto ne = static_cast<py::ssize_t>(edges.size() / 2);
            py::array_t<std::int64_t> edge_arr(
                {ne, static_cast<py::ssize_t>(2)});
            if (ne > 0) {
              std::memcpy(edge_arr.mutable_data(), edges.data(),
                          edges.size() * sizeof(std::int64_t));
            }
            py::dict out;
            out["nx"] = nx;
            out["ny"] = ny;
            out["node_xy"] = node_xy;
            out["active"] = node_active;
            out["edges"] = edge_arr;
            return out;
          },
          "Model graph for display: dict with nx, ny, node_xy (nx*ny, 2) in "
          "node order n = j*nx + i, active (nx*ny,) bool, and edges (E, 2): "
          "node-index pairs of neighbour edges that are not cut by a wall "
          "and join two active nodes.");
}
