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

// ROS 2 wrapper for the GMRF scalar-field belief (scalar_field_belief::gmrf).
//
// Same public interface as the GP node (nodes/scalar_field_belief_node.py):
//   sub  ir_measurement scalar_field_interfaces/msg/ScalarMeasurement srv
//   query_scalar_field_belief
//   scalar_field_interfaces/srv/QueryScalarFieldBelief srv
//   query_scalar_field_belief_covariance .../QueryScalarFieldBeliefCovariance
//   srv  reset_scalar_field_belief               std_srvs/srv/Trigger
//   pub  belief/mean_cloud, belief/variance_cloud sensor_msgs/msg/PointCloud2
// The estimator itself lives in the ROS-independent core library; this file
// only handles parameters, messages, frames, locking, and visualization.
#include <Eigen/Core>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <scalar_field_interfaces/msg/scalar_measurement.hpp>
#include <scalar_field_interfaces/srv/query_scalar_field_belief.hpp>
#include <scalar_field_interfaces/srv/query_scalar_field_belief_covariance.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sstream>
#include <std_srvs/srv/trigger.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "scalar_field_belief/gmrf/gmrf_belief.hpp"
#include "scalar_field_belief/gmrf/toml_loader.hpp"

namespace gm = scalar_field_belief::gmrf;
using scalar_field_interfaces::msg::ScalarMeasurement;
using scalar_field_interfaces::srv::QueryScalarFieldBelief;
using scalar_field_interfaces::srv::QueryScalarFieldBeliefCovariance;
using sensor_msgs::msg::PointCloud2;
using std_srvs::srv::Trigger;

namespace {

rcl_interfaces::msg::ParameterDescriptor staticDescriptor(
    const std::string& description) {
  rcl_interfaces::msg::ParameterDescriptor d;
  d.description = description;
  d.additional_constraints =
      "Static parameter. Change in YAML and restart node.";
  d.read_only = true;
  return d;
}

std::invalid_argument paramError(const std::string& name,
                                 const std::string& what) {
  return std::invalid_argument("Invalid parameter '" + name + "': " + what);
}

/// Same axis as scalar_field_belief.visualization.make_axis, clamped to [start,
/// stop].
std::vector<double> makeAxis(double start, double stop, double step) {
  std::vector<double> axis;
  const auto n_full = static_cast<long long>(std::floor((stop - start) / step));
  for (long long k = 0; k <= n_full; ++k) {
    axis.push_back(std::min(start + step * static_cast<double>(k), stop));
  }
  if (std::abs(axis.back() - stop) > 1e-8 + 1e-5 * std::abs(stop)) {
    axis.push_back(stop);
  }
  return axis;
}

}  // namespace

class GmrfBeliefNode : public rclcpp::Node {
 public:
  explicit GmrfBeliefNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : rclcpp::Node("gmrf_belief", options) {
    declareParameters();
    gm::GmrfConfig config;
    std::vector<gm::WallSegment> walls;
    std::vector<gm::Point2> seeds;
    readParameters(config, walls, seeds);
    belief_ = std::make_unique<gm::GmrfBelief>(config, walls, seeds);
    buildVisualizationGrid();

    measurement_sub_ = create_subscription<ScalarMeasurement>(
        "ir_measurement", 10,
        [this](const ScalarMeasurement::ConstSharedPtr msg) {
          onMeasurement(*msg);
        });
    query_srv_ = create_service<QueryScalarFieldBelief>(
        "query_scalar_field_belief",
        [this](const std::shared_ptr<QueryScalarFieldBelief::Request> req,
               std::shared_ptr<QueryScalarFieldBelief::Response> res) {
          handleQuery(*req, *res);
        });
    query_cov_srv_ = create_service<QueryScalarFieldBeliefCovariance>(
        "query_scalar_field_belief_covariance",
        [this](
            const std::shared_ptr<QueryScalarFieldBeliefCovariance::Request>
                req,
            std::shared_ptr<QueryScalarFieldBeliefCovariance::Response> res) {
          handleQueryCovariance(*req, *res);
        });
    reset_srv_ = create_service<Trigger>(
        "reset_scalar_field_belief",
        [this](const std::shared_ptr<Trigger::Request>,
               std::shared_ptr<Trigger::Response> res) { handleReset(*res); });

    const auto qos =
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    mean_pub_ = create_publisher<PointCloud2>("belief/mean_cloud", qos);
    var_pub_ = create_publisher<PointCloud2>("belief/variance_cloud", qos);

    RCLCPP_INFO(get_logger(),
                "gmrf_belief started: grid %d x %d nodes on [%.3f, %.3f] x "
                "[%.3f, %.3f] (dx=%.4f, "
                "dy=%.4f), %zu walls, %zu free-space seeds, frame '%s', "
                "scalar-only stationary GMRF.",
                belief_->nx(), belief_->ny(), config.x_min, config.x_max,
                config.y_min, config.y_max, belief_->dx(), belief_->dy(),
                walls.size(), seeds.size(), frame_id_.c_str());
    if (publish_visualization_) {
      publishVisualization();
    }
  }

 private:
  void declareParameters() {
    declare_parameter<std::string>("frame_id", "map",
                                   staticDescriptor("Belief frame."));
    declare_parameter<double>("x_min", 0.0,
                              staticDescriptor("Domain x minimum [m]."));
    declare_parameter<double>("x_max", 2.0,
                              staticDescriptor("Domain x maximum [m]."));
    declare_parameter<double>("y_min", 0.0,
                              staticDescriptor("Domain y minimum [m]."));
    declare_parameter<double>("y_max", 4.0,
                              staticDescriptor("Domain y maximum [m]."));
    declare_parameter<int64_t>("nx", 41,
                               staticDescriptor("Grid nodes in x (>= 2)."));
    declare_parameter<int64_t>("ny", 81,
                               staticDescriptor("Grid nodes in y (>= 2)."));
    declare_parameter<double>(
        "prior_mean", 0.0, staticDescriptor("Prior mean [measurement units]."));
    declare_parameter<double>(
        "prior_sigma", 0.01,
        staticDescriptor("Per-node prior std [measurement units]."));
    declare_parameter<double>(
        "sigma_p_per_m", 0.8,
        staticDescriptor(
            "Neighbour std = sigma_p_per_m * dx^2 [measurement units / m^2]."));
    declare_parameter<double>(
        "sigma_s", 0.0005,
        staticDescriptor("Measurement noise std [measurement units]."));
    declare_parameter<double>(
        "wall_endpoint_extension", 0.01,
        staticDescriptor(
            "Wall extension beyond both endpoints along the wall [m]."));
    declare_parameter<std::vector<double>>(
        "walls", std::vector<double>{},
        staticDescriptor(
            "Flat wall list [x0, y0, x1, y1, ...] in frame_id [m]."));
    declare_parameter<std::string>(
        "wall_geometry_file", "",
        staticDescriptor(
            "Wall TOML file ([objects.walls] wall_vertices); absolute or "
            "relative "
            "to the scalar_field_belief share directory; '' for none."));
    declare_parameter<std::vector<double>>(
        "free_space_seeds", std::vector<double>{},
        staticDescriptor("Flat seed list [x, y, ...]; if non-empty, grid nodes "
                         "not connected to "
                         "a seed are inactive."));
    declare_parameter<bool>("publish_visualization", true,
                            staticDescriptor("Publish mean/variance clouds."));
    declare_parameter<double>("visualization_grid_step", 0.1,
                              staticDescriptor("Visualization grid step [m]."));
    declare_parameter<double>("z_offset", -0.8,
                              staticDescriptor("Cloud plane height [m]."));
  }

  void readParameters(gm::GmrfConfig& c, std::vector<gm::WallSegment>& walls,
                      std::vector<gm::Point2>& seeds) {
    frame_id_ = get_parameter("frame_id").as_string();
    if (frame_id_.empty()) {
      throw paramError("frame_id", "must not be empty.");
    }
    c.x_min = get_parameter("x_min").as_double();
    c.x_max = get_parameter("x_max").as_double();
    c.y_min = get_parameter("y_min").as_double();
    c.y_max = get_parameter("y_max").as_double();
    for (const char* name : {"nx", "ny"}) {
      const int64_t v = get_parameter(name).as_int();
      if (v < 2 || v > 100000) {
        throw paramError(
            name, "must be in [2, 100000], got " + std::to_string(v) + ".");
      }
    }
    c.nx = static_cast<int>(get_parameter("nx").as_int());
    c.ny = static_cast<int>(get_parameter("ny").as_int());
    c.prior_mean = get_parameter("prior_mean").as_double();
    c.prior_sigma = get_parameter("prior_sigma").as_double();
    c.sigma_p_per_m = get_parameter("sigma_p_per_m").as_double();
    c.sigma_s = get_parameter("sigma_s").as_double();
    c.wall_endpoint_extension =
        get_parameter("wall_endpoint_extension").as_double();
    // Scalar-only stationary model: flow energies and aging are not exposed.
    c.validate();

    const std::vector<double> flat = get_parameter("walls").as_double_array();
    if (flat.size() % 4 != 0) {
      throw paramError(
          "walls", "length must be a multiple of 4 ([x0, y0, x1, y1, ...]).");
    }
    for (std::size_t k = 0; k < flat.size(); k += 4) {
      walls.push_back(
          gm::WallSegment{flat[k], flat[k + 1], flat[k + 2], flat[k + 3]});
    }
    const std::string file = get_parameter("wall_geometry_file").as_string();
    if (!file.empty()) {
      std::filesystem::path path(file);
      if (path.is_relative()) {
        path =
            std::filesystem::path(ament_index_cpp::get_package_share_directory(
                "scalar_field_belief")) /
            path;
      }
      if (!std::filesystem::is_regular_file(path)) {
        throw paramError("wall_geometry_file",
                         "file not found: " + path.string());
      }
      const gm::WallFile wf = gm::loadWallsToml(path.string());
      if (wf.has_bounds &&
          (wf.bounds.x_min != c.x_min || wf.bounds.x_max != c.x_max ||
           wf.bounds.y_min != c.y_min || wf.bounds.y_max != c.y_max)) {
        RCLCPP_WARN(
            get_logger(),
            "Wall file %s declares bounds [%g, %g] x [%g, %g] that differ from "
            "the configured domain; walls are used as given in frame '%s'.",
            path.c_str(), wf.bounds.x_min, wf.bounds.x_max, wf.bounds.y_min,
            wf.bounds.y_max, frame_id_.c_str());
      }
      RCLCPP_INFO(get_logger(), "Loaded %zu walls from %s.", wf.walls.size(),
                  path.c_str());
      walls.insert(walls.end(), wf.walls.begin(), wf.walls.end());
    }
    const std::vector<double> sflat =
        get_parameter("free_space_seeds").as_double_array();
    if (sflat.size() % 2 != 0) {
      throw paramError("free_space_seeds",
                       "length must be even ([x, y, ...]).");
    }
    for (std::size_t k = 0; k < sflat.size(); k += 2) {
      seeds.push_back(gm::Point2{sflat[k], sflat[k + 1]});
    }

    publish_visualization_ = get_parameter("publish_visualization").as_bool();
    viz_step_ = get_parameter("visualization_grid_step").as_double();
    z_offset_ = get_parameter("z_offset").as_double();
    if (!std::isfinite(viz_step_) || !(viz_step_ > 0.0)) {
      throw paramError("visualization_grid_step", "must be finite and > 0.");
    }
    const double nvx = std::floor((c.x_max - c.x_min) / viz_step_) + 2.0;
    const double nvy = std::floor((c.y_max - c.y_min) / viz_step_) + 2.0;
    if (nvx * nvy > 2e6) {
      throw paramError("visualization_grid_step",
                       "too small for the domain (> 2e6 points).");
    }
    if (!std::isfinite(z_offset_)) {
      throw paramError("z_offset", "must be finite.");
    }
  }

  void buildVisualizationGrid() {
    const gm::GmrfConfig& c = belief_->config();
    const std::vector<double> xs = makeAxis(c.x_min, c.x_max, viz_step_);
    const std::vector<double> ys = makeAxis(c.y_min, c.y_max, viz_step_);
    std::vector<std::pair<double, double>> pts;
    pts.reserve(xs.size() * ys.size());
    for (double y : ys) {
      for (double x : xs) {
        // Geometry is fixed, so representability is decided once.
        if (belief_->isRepresentable(x, y)) {
          pts.emplace_back(x, y);
        }
      }
    }
    viz_xy_.resize(static_cast<Eigen::Index>(pts.size()), 2);
    for (std::size_t k = 0; k < pts.size(); ++k) {
      viz_xy_(static_cast<Eigen::Index>(k), 0) = pts[k].first;
      viz_xy_(static_cast<Eigen::Index>(k), 1) = pts[k].second;
    }
    const std::size_t skipped = xs.size() * ys.size() - pts.size();
    if (skipped > 0) {
      RCLCPP_INFO(get_logger(),
                  "Visualization: %zu of %zu grid points cannot be represented "
                  "(hidden by walls / inactive) and are omitted.",
                  skipped, xs.size() * ys.size());
    }
  }

  void onMeasurement(const ScalarMeasurement& msg) {
    const std::string frame =
        msg.header.frame_id.empty() ? frame_id_ : msg.header.frame_id;
    if (frame != frame_id_) {
      RCLCPP_WARN(get_logger(),
                  "Ignoring measurement in frame '%s', expected '%s'.",
                  frame.c_str(), frame_id_.c_str());
      return;
    }
    const double x = msg.pose.position.x;
    const double y = msg.pose.position.y;
    const double value = msg.value;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(value)) {
      RCLCPP_WARN(get_logger(),
                  "Ignoring measurement with non-finite position or value.");
      return;
    }
    std::size_t count = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      try {
        belief_->addMeasurement(x, y, value);
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "Rejected measurement: %s", e.what());
        return;
      }
      try {
        belief_->update();  // assimilate before any later query
      } catch (const std::exception& e) {
        belief_->discardPendingMeasurements();
        RCLCPP_ERROR(get_logger(),
                     "Belief update failed, measurement discarded: %s",
                     e.what());
        return;
      }
      count = belief_->numMeasurements();
    }
    RCLCPP_INFO(get_logger(),
                "Received measurement at (%.3f, %.3f) = %.6f; N=%zu", x, y,
                value, count);
    if (publish_visualization_) {
      publishVisualization();
    }
  }

  /// Validates frames and collects query positions. Returns an error message
  /// (empty on success), matching the GP node's status messages.
  std::string collectQueries(
      const std::vector<geometry_msgs::msg::PoseStamped>& queries,
      Eigen::MatrixX2d& xy) const {
    if (queries.empty()) {
      return "No query poses provided.";
    }
    xy.resize(static_cast<Eigen::Index>(queries.size()), 2);
    for (std::size_t k = 0; k < queries.size(); ++k) {
      const auto& q = queries[k];
      const std::string frame =
          q.header.frame_id.empty() ? frame_id_ : q.header.frame_id;
      if (frame != frame_id_) {
        return "Expected frame '" + frame_id_ + "', got '" + frame + "'.";
      }
      xy(static_cast<Eigen::Index>(k), 0) = q.pose.position.x;
      xy(static_cast<Eigen::Index>(k), 1) = q.pose.position.y;
    }
    return "";
  }

  void handleQuery(const QueryScalarFieldBelief::Request& req,
                   QueryScalarFieldBelief::Response& res) {
    Eigen::MatrixX2d xy;
    const std::string err = collectQueries(req.queries, xy);
    if (!err.empty()) {
      res.success = false;
      res.status_message = err;
      return;
    }
    try {
      Eigen::VectorXd mean;
      Eigen::VectorXd var;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        belief_->queryMeanVariance(xy, mean, var);
      }
      res.mean.assign(mean.data(), mean.data() + mean.size());
      res.variance.assign(var.data(), var.data() + var.size());
    } catch (const std::invalid_argument& e) {
      failQuery(res, "Belief query failed: ", e, false);
      return;
    } catch (const std::exception& e) {
      failQuery(res, "Belief query failed: ", e, true);
      return;
    }
    res.success = true;
    res.status_message = "ok";
  }

  /// Rejected client input (std::invalid_argument) is logged at WARN, genuine
  /// failures (solver, allocation, ...) at ERROR.
  template <typename Response>
  void failQuery(Response& res, const std::string& prefix,
                 const std::exception& e, bool is_error) {
    if (is_error) {
      RCLCPP_ERROR(get_logger(), "%s%s", prefix.c_str(), e.what());
    } else {
      RCLCPP_WARN(get_logger(), "%s%s", prefix.c_str(), e.what());
    }
    res = Response();
    res.success = false;
    res.status_message = prefix + e.what();
  }

  void handleQueryCovariance(
      const QueryScalarFieldBeliefCovariance::Request& req,
      QueryScalarFieldBeliefCovariance::Response& res) {
    Eigen::MatrixX2d xy;
    const std::string err = collectQueries(req.queries, xy);
    if (!err.empty()) {
      res.success = false;
      res.status_message = err;
      return;
    }
    double noise_var = 0.0;
    try {
      Eigen::VectorXd mean;
      Eigen::MatrixXd cov;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        belief_->queryMeanCovariance(xy, mean, cov);
        noise_var = belief_->measurementNoiseVariance();
      }
      const Eigen::Index n = cov.rows();
      res.mean.assign(mean.data(), mean.data() + mean.size());
      // Explicit row-major flattening (independent of Eigen's storage order).
      res.covariance_matrix.resize(static_cast<std::size_t>(n * n));
      for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
          res.covariance_matrix[static_cast<std::size_t>(i * n + j)] =
              cov(i, j);
        }
      }
    } catch (const std::invalid_argument& e) {
      failQuery(res, "Belief covariance query failed: ", e, false);
      return;
    } catch (const std::exception& e) {
      failQuery(res, "Belief covariance query failed: ", e, true);
      return;
    }
    res.measurement_noise_variance = noise_var;
    res.success = true;
    res.status_message = "ok";
  }

  void handleReset(Trigger::Response& res) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      belief_->reset();
    }
    res.success = true;
    res.message = "Belief reset.";
    RCLCPP_INFO(get_logger(), "Belief reset to the GMRF prior.");
    if (publish_visualization_) {
      publishVisualization();
    }
  }

  PointCloud2 makeCloud(const Eigen::VectorXd& values,
                        const rclcpp::Time& stamp) const {
    PointCloud2 cloud;
    cloud.header.frame_id = frame_id_;
    cloud.header.stamp = stamp;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(
        4, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
        sensor_msgs::msg::PointField::FLOAT32, "z", 1,
        sensor_msgs::msg::PointField::FLOAT32, "intensity", 1,
        sensor_msgs::msg::PointField::FLOAT32);
    modifier.resize(static_cast<std::size_t>(viz_xy_.rows()));
    sensor_msgs::PointCloud2Iterator<float> it_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> it_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> it_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> it_i(cloud, "intensity");
    for (Eigen::Index k = 0; k < viz_xy_.rows();
         ++k, ++it_x, ++it_y, ++it_z, ++it_i) {
      *it_x = static_cast<float>(viz_xy_(k, 0));
      *it_y = static_cast<float>(viz_xy_(k, 1));
      *it_z = static_cast<float>(z_offset_);
      *it_i = static_cast<float>(values(k));
    }
    cloud.is_dense = true;
    return cloud;
  }

  void publishVisualization() {
    Eigen::VectorXd mean;
    Eigen::VectorXd var;
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      belief_->queryMeanVariance(viz_xy_, mean, var);  // marginals only
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Visualization query failed: %s", e.what());
      return;
    }
    const rclcpp::Time stamp = now();
    mean_pub_->publish(makeCloud(mean, stamp));
    var_pub_->publish(makeCloud(var, stamp));
  }

  std::mutex mutex_;
  std::unique_ptr<gm::GmrfBelief> belief_;
  std::string frame_id_;
  bool publish_visualization_ = true;
  double viz_step_ = 0.1;
  double z_offset_ = -0.8;
  Eigen::MatrixX2d viz_xy_;

  rclcpp::Subscription<ScalarMeasurement>::SharedPtr measurement_sub_;
  rclcpp::Service<QueryScalarFieldBelief>::SharedPtr query_srv_;
  rclcpp::Service<QueryScalarFieldBeliefCovariance>::SharedPtr query_cov_srv_;
  rclcpp::Service<Trigger>::SharedPtr reset_srv_;
  rclcpp::Publisher<PointCloud2>::SharedPtr mean_pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr var_pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  std::shared_ptr<GmrfBeliefNode> node;
  try {
    node = std::make_shared<GmrfBeliefNode>();
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("gmrf_belief"),
                 "Failed to start gmrf_belief: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::spin(node);
  node.reset();
  rclcpp::shutdown();
  return 0;
}
