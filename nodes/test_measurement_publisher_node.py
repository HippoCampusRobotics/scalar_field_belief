#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from scalar_field_interfaces.msg import ScalarMeasurement


class TestMeasurementPublisherNode(Node):
    def __init__(self):
        super().__init__("test_measurement_publisher")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("period_sec", 1.0)
        self.declare_parameter("x_min", 0.0)
        self.declare_parameter("x_max", 2.0)
        self.declare_parameter("y_min", 0.0)
        self.declare_parameter("y_max", 4.0)
        self.declare_parameter("seed", 1)
        self.declare_parameter("noise_std", 0.01)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.period_sec = float(self.get_parameter("period_sec").value)
        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.y_max = float(self.get_parameter("y_max").value)
        self.noise_std = float(self.get_parameter("noise_std").value)
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))

        self.pub = self.create_publisher(ScalarMeasurement, "ir_measurement", 10)
        self.timer = self.create_timer(self.period_sec, self._publish_measurement)

    def _latent_value(self, x: float, y: float) -> float:
        center_1 = np.array([0.5, 1.0])
        center_2 = np.array([1.4, 3.0])
        p = np.array([x, y])
        d1 = p - center_1
        d2 = p - center_2
        v1 = 1.0 * np.exp(-0.5 * ((d1[0] / 0.25) ** 2 + (d1[1] / 0.35) ** 2))
        v2 = 0.6 * np.exp(-0.5 * ((d2[0] / 0.30) ** 2 + (d2[1] / 0.45) ** 2))
        return float(v1 + v2)

    def _publish_measurement(self) -> None:
        x = self.rng.uniform(self.x_min, self.x_max)
        y = self.rng.uniform(self.y_min, self.y_max)
        value = self._latent_value(x, y) + self.rng.normal(0.0, self.noise_std)

        msg = ScalarMeasurement()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        msg.value = float(value)
        msg.clipped = False
        self.pub.publish(msg)
        self.get_logger().info(
            f"Published test measurement at ({x:.3f}, {y:.3f}) = {value:.6f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TestMeasurementPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
