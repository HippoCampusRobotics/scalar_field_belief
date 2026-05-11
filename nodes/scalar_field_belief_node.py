#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2

from scalar_field_interfaces.msg import ScalarMeasurement
from scalar_field_interfaces.srv import QueryScalarFieldBelief
from scalar_field_belief.belief import ScalarFieldBelief
from scalar_field_belief.config import BeliefConfig
from scalar_field_belief.visualization import (
    make_grid_positions,
    make_field_pointcloud2,
)


class ScalarFieldBeliefNode(Node):
    def __init__(self):
        super().__init__("scalar_field_belief")
        self._declare_parameters()
        config = self._read_config()
        self.belief = ScalarFieldBelief(config)
        self.config = config

        self.measurement_sub = self.create_subscription(
            ScalarMeasurement,
            "ir_measurement",
            self._on_measurement,
            10,
        )
        self.query_srv = self.create_service(
            QueryScalarFieldBelief,
            "query_scalar_field_belief",
            self._handle_query,
        )
        self.reset_srv = self.create_service(
            Trigger,
            "reset_scalar_field_belief",
            self._handle_reset,
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.mean_pub = self.create_publisher(PointCloud2, "belief/mean_cloud", qos)
        self.var_pub = self.create_publisher(PointCloud2, "belief/variance_cloud", qos)

        self.get_logger().info("scalar_field_belief node started.")

    def _declare_parameters(self) -> None:
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("x_min", 0.0)
        self.declare_parameter("x_max", 2.0)
        self.declare_parameter("y_min", 0.0)
        self.declare_parameter("y_max", 4.0)
        self.declare_parameter("kernel_type", "rbf")
        self.declare_parameter("training_iter", 50)
        self.declare_parameter("learning_rate", 0.1)
        self.declare_parameter("init_lengthscale_x", 0.2)
        self.declare_parameter("init_lengthscale_y", 0.2)
        self.declare_parameter("init_outputscale", 1.0)
        self.declare_parameter("init_noise", 0.05)
        self.declare_parameter("refit_policy", "every_measurement")
        self.declare_parameter("refit_every_k", 1)
        self.declare_parameter("publish_visualization", True)
        self.declare_parameter("visualization_grid_step", 0.1)
        self.declare_parameter("visualization_z_mode", "flat")
        self.declare_parameter("visualization_height_scale", 0.5)

    def _read_config(self) -> BeliefConfig:
        cfg = BeliefConfig(
            frame_id=self.get_parameter("frame_id").value,
            x_min=float(self.get_parameter("x_min").value),
            x_max=float(self.get_parameter("x_max").value),
            y_min=float(self.get_parameter("y_min").value),
            y_max=float(self.get_parameter("y_max").value),
            kernel_type=self.get_parameter("kernel_type").value,
            training_iter=int(self.get_parameter("training_iter").value),
            learning_rate=float(self.get_parameter("learning_rate").value),
            init_lengthscale_x=float(self.get_parameter("init_lengthscale_x").value),
            init_lengthscale_y=float(self.get_parameter("init_lengthscale_y").value),
            init_outputscale=float(self.get_parameter("init_outputscale").value),
            init_noise=float(self.get_parameter("init_noise").value),
            refit_policy=self.get_parameter("refit_policy").value,
            refit_every_k=int(self.get_parameter("refit_every_k").value),
            publish_visualization=bool(
                self.get_parameter("publish_visualization").value
            ),
            visualization_grid_step=float(
                self.get_parameter("visualization_grid_step").value
            ),
            visualization_z_mode=self.get_parameter("visualization_z_mode").value,
            visualization_height_scale=float(
                self.get_parameter("visualization_height_scale").value
            ),
        )
        cfg.validate()
        return cfg

    def _on_measurement(self, msg: ScalarMeasurement) -> None:
        frame_id = msg.header.frame_id or self.config.frame_id
        if frame_id != self.config.frame_id:
            self.get_logger().warning(
                f"Ignoring measurement in frame '{frame_id}', expected '{self.config.frame_id}'."
            )
            return

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        value = float(msg.value)

        try:
            result = self.belief.add_measurement(x=x, y=y, value=value)
            self.get_logger().info(
                f"Received measurement at ({x:.3f}, {y:.3f}) = {value:.6f}; "
                f"N={result.num_measurements}, refit={result.did_refit}"
            )
            if result.did_refit and self.config.publish_visualization:
                self._publish_visualization()
        except Exception as exc:
            self.get_logger().error(f"Failed to process measurement: {exc}")

    def _handle_query(self, request, response):
        if len(request.queries) == 0:
            response.success = False
            response.status_message = "No query poses provided."
            return response
        if not self.belief.has_model():
            response.success = False
            response.status_message = "Belief has no fitted model yet."
            return response

        xy = []
        for pose_stamped in request.queries:
            frame_id = pose_stamped.header.frame_id or self.config.frame_id
            if frame_id != self.config.frame_id:
                response.success = False
                response.status_message = (
                    f"Expected frame '{self.config.frame_id}', got '{frame_id}'."
                )
                return response
            xy.append([pose_stamped.pose.position.x, pose_stamped.pose.position.y])

        try:
            mean, variance = self.belief.query(np.asarray(xy, dtype=float))
        except Exception as exc:
            self.get_logger().error(f"Belief query failed: {exc}")
            response.success = False
            response.status_message = f"Belief query failed: {exc}"
            return response

        response.success = True
        response.mean = mean.tolist()
        response.variance = variance.tolist()
        response.status_message = "ok"
        return response

    def _handle_reset(self, _request, response):
        self.belief.reset()
        response.success = True
        response.message = "Belief reset."
        self.get_logger().info("Belief reset.")
        return response

    def _publish_visualization(self) -> None:
        if not self.belief.has_model():
            return
        grid_xy = make_grid_positions(
            x_min=self.config.x_min,
            x_max=self.config.x_max,
            y_min=self.config.y_min,
            y_max=self.config.y_max,
            grid_step=self.config.visualization_grid_step,
        )
        mean, variance = self.belief.query(grid_xy)
        stamp = self.get_clock().now().to_msg()
        mean_cloud = make_field_pointcloud2(
            positions_xy=grid_xy,
            values=mean,
            frame_id=self.config.frame_id,
            stamp=stamp,
            z_mode=self.config.visualization_z_mode,
            height_scale=self.config.visualization_height_scale,
        )
        var_cloud = make_field_pointcloud2(
            positions_xy=grid_xy,
            values=variance,
            frame_id=self.config.frame_id,
            stamp=stamp,
            z_mode=self.config.visualization_z_mode,
            height_scale=self.config.visualization_height_scale,
        )
        self.mean_pub.publish(mean_cloud)
        self.var_pub.publish(var_cloud)


def main(args=None):
    rclpy.init(args=args)
    node = ScalarFieldBeliefNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
