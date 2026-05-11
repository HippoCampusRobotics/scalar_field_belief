#!/usr/bin/env python3
from __future__ import annotations

import sys
import rclpy
from rclpy.node import Node
from scalar_field_interfaces.msg import ScalarMeasurement


class MeasurementPublisher(Node):
    def __init__(self):
        super().__init__("manual_measurement_publisher")
        self.pub = self.create_publisher(ScalarMeasurement, "ir_measurement", 10)

    def publish_measurement(self, x: float, y: float, value: float, frame_id: str = "map"):
        msg = ScalarMeasurement()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.w = 1.0
        msg.value = value
        msg.clipped = False
        self.pub.publish(msg)


def main():
    if len(sys.argv) < 4:
        print("usage: publish_measurement.py X Y VALUE [FRAME_ID]")
        sys.exit(1)
    x = float(sys.argv[1])
    y = float(sys.argv[2])
    value = float(sys.argv[3])
    frame_id = sys.argv[4] if len(sys.argv) > 4 else "map"
    rclpy.init()
    node = MeasurementPublisher()
    node.publish_measurement(x, y, value, frame_id)
    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
