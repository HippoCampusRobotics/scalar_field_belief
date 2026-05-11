#!/usr/bin/env python3
from __future__ import annotations

import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scalar_field_interfaces.srv import QueryScalarFieldBelief


class QueryBeliefClient(Node):
    def __init__(self):
        super().__init__("query_belief_client")
        self.cli = self.create_client(QueryScalarFieldBelief, "query_scalar_field_belief")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for query_scalar_field_belief ...")

    def call(self, x: float, y: float, frame_id: str = "map"):
        req = QueryScalarFieldBelief.Request()
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        req.queries.append(pose)
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        return future.result()


def main():
    if len(sys.argv) < 3:
        print("usage: query_belief.py X Y [FRAME_ID]")
        sys.exit(1)
    x = float(sys.argv[1])
    y = float(sys.argv[2])
    frame_id = sys.argv[3] if len(sys.argv) > 3 else "map"
    rclpy.init()
    node = QueryBeliefClient()
    resp = node.call(x, y, frame_id)
    print(resp)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
