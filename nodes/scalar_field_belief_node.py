#!/usr/bin/env python3
"""ROS 2 node for the scalar field GP belief.

This node wraps the ROS-independent `ScalarFieldBelief` class and exposes it
through ROS topics, services, and visualization point clouds.

Subscriptions
-------------
`ir_measurement`
    Scalar measurements of type `scalar_field_interfaces/ScalarMeasurement`.

Services
--------
`query_scalar_field_belief`
    Query posterior mean and variance at requested poses.

`reset_scalar_field_belief`
    Clear all stored measurements and fitted GP state.

Publications
------------
`belief/mean_cloud`
    PointCloud2 visualization of the posterior mean.

`belief/variance_cloud`
    PointCloud2 visualization of the posterior variance.
"""

from __future__ import annotations

import traceback
from math import isfinite

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from scalar_field_interfaces.msg import ScalarMeasurement
from scalar_field_interfaces.srv import QueryScalarFieldBelief
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

from scalar_field_belief.belief import ScalarFieldBelief
from scalar_field_belief.config import BeliefConfig
from scalar_field_belief.visualization import (
    make_field_pointcloud2,
    make_grid_positions,
)


class ScalarFieldBeliefNode(Node):
    """ROS 2 interface for the scalar field GP belief.

    The node keeps ROS-specific concerns separate from the core belief model:
    message parsing, service responses, parameters, logging, and visualization
    are handled here, while GP fitting and querying are handled by
    `ScalarFieldBelief`.

    Notes
    -----
    Most parameters are static and should be changed in the YAML file before
    starting the node. Only selected visualization parameters are dynamic.
    """

    DYNAMIC_PARAMETERS = {
        'field_color_min',
        'field_color_max',
        'use_fixed_field_color_max',
        'z_offset',
    }
    PASSTHROUGH_PARAMETERS = {
        'use_sim_time',
    }

    def __init__(self):
        super().__init__('scalar_field_belief')
        self._declare_parameters()
        config = self._read_config()
        self.belief = ScalarFieldBelief(config)
        self.config = config

        self._field_color_min = (
            self.get_parameter('field_color_min')
            .get_parameter_value()
            .double_value
        )
        self._field_color_max = (
            self.get_parameter('field_color_max')
            .get_parameter_value()
            .double_value
        )
        self._use_fixed_field_color_max = (
            self.get_parameter('use_fixed_field_color_max')
            .get_parameter_value()
            .bool_value
        )
        self._z_offset = (
            self.get_parameter('z_offset').get_parameter_value().double_value
        )

        self.measurement_sub = self.create_subscription(
            ScalarMeasurement,
            'ir_measurement',
            self._on_measurement,
            10,
        )
        self.query_srv = self.create_service(
            QueryScalarFieldBelief,
            'query_scalar_field_belief',
            self._handle_query,
        )
        self.reset_srv = self.create_service(
            Trigger,
            'reset_scalar_field_belief',
            self._handle_reset,
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.mean_pub = self.create_publisher(
            PointCloud2, 'belief/mean_cloud', qos
        )
        self.var_pub = self.create_publisher(
            PointCloud2, 'belief/variance_cloud', qos
        )

        self._param_cb_handle = self.add_on_set_parameters_callback(
            self._on_set_parameters
        )

        self.get_logger().info('scalar_field_belief node started.')

    def _declare_parameters(self) -> None:
        """Declare ROS parameters used by this node.

        Static parameters are marked as read-only. They can still be initialized
        from launch/YAML files, but they cannot be changed while the node is
        running.

        Only visualization convenience parameters are declared as dynamic.
        """
        self.declare_parameter('frame_id', 'map', self._static_descriptor())
        self.declare_parameter('x_min', 0.0, self._static_descriptor())
        self.declare_parameter('x_max', 2.0, self._static_descriptor())
        self.declare_parameter('y_min', 0.0, self._static_descriptor())
        self.declare_parameter('y_max', 4.0, self._static_descriptor())
        self.declare_parameter('kernel_type', 'rbf', self._static_descriptor())
        self.declare_parameter('training_iter', 50, self._static_descriptor())
        self.declare_parameter(
            'learning_rate', (0.1), self._static_descriptor()
        )
        self.declare_parameter(
            'init_lengthscale_x', 0.2, self._static_descriptor()
        )
        self.declare_parameter(
            'init_lengthscale_y', 0.2, self._static_descriptor()
        )
        self.declare_parameter(
            'init_outputscale', 1.0, self._static_descriptor()
        )
        self.declare_parameter('init_noise', 0.05, self._static_descriptor())
        self.declare_parameter(
            'refit_policy', 'every_measurement', self._static_descriptor()
        )
        self.declare_parameter('refit_every_k', 1, self._static_descriptor())
        self.declare_parameter(
            'publish_visualization', True, self._static_descriptor()
        )
        self.declare_parameter(
            'visualization_grid_step', 0.1, self._static_descriptor()
        )
        self.declare_parameter(
            'visualization_z_mode', 'flat', self._static_descriptor()
        )
        self.declare_parameter(
            'visualization_height_scale', 0.5, self._static_descriptor()
        )
        self.declare_parameter(
            'field_color_min', 0.0, self._dynamic_descriptor()
        )
        self.declare_parameter(
            'field_color_max', 0.015, self._dynamic_descriptor()
        )
        self.declare_parameter(
            'use_fixed_field_color_max', False, self._dynamic_descriptor()
        )
        self.declare_parameter('z_offset', -0.8, self._dynamic_descriptor())

    def _static_descriptor(self, description: str = '') -> ParameterDescriptor:
        """Return a descriptor for parameters that require a node restart."""
        return ParameterDescriptor(
            description=description,
            additional_constraints='Static parameter. '
            + 'Change in YAML and restart node.',
            read_only=True,
        )

    def _dynamic_descriptor(self, description: str = '') -> ParameterDescriptor:
        """Return a descriptor for parameters that may be changed at runtime."""
        return ParameterDescriptor(
            description=description,
            additional_constraints='Dynamic parameter. '
            + 'May be changed at runtime.',
            read_only=False,
        )

    def _on_set_parameters(
        self, params: list[Parameter]
    ) -> SetParametersResult:
        """Update selected visualization parameters at runtime."""
        unsupported = [
            param.name
            for param in params
            if param.name not in self.DYNAMIC_PARAMETERS
            and param.name not in self.PASSTHROUGH_PARAMETERS
        ]
        if unsupported:
            return SetParametersResult(
                successful=False,
                reason=(
                    'These parameters are not dynamic: '
                    f'{", ".join(unsupported)}. '
                    'Change them in the YAML file and restart the node.'
                ),
            )

        values = {
            'field_color_min': self._field_color_min,
            'field_color_max': self._field_color_max,
            'use_fixed_field_color_max': self._use_fixed_field_color_max,
            'z_offset': self._z_offset,
        }

        for param in params:
            try:
                values[param.name] = self._parameter_value(param)
            except ValueError as exc:
                return SetParametersResult(successful=False, reason=str(exc))

        try:
            self._validate_visualization_params(values)
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        self._field_color_min = float(values['field_color_min'])
        self._field_color_max = float(values['field_color_max'])
        self._use_fixed_field_color_max = bool(
            values['use_fixed_field_color_max']
        )
        self._z_offset = float(values['z_offset'])

        self._log_dynamic_parameter_update({param.name for param in params})

        if self.config.publish_visualization:
            self._publish_visualization()

        return SetParametersResult(successful=True)

    @staticmethod
    def _parameter_value(param: Parameter) -> float | bool:
        """Convert a supported dynamic ROS parameter to a Python value."""
        if param.name == 'use_fixed_field_color_max':
            if param.type_ != Parameter.Type.BOOL:
                raise ValueError('use_fixed_field_color_max must be a boolean.')
            return bool(param.value)

        if param.type_ not in {Parameter.Type.DOUBLE, Parameter.Type.INTEGER}:
            raise ValueError(f'{param.name} must be a number.')

        value = float(param.value)
        if not isfinite(value):
            raise ValueError(f'{param.name} must be finite.')

        return value

    @staticmethod
    def _validate_visualization_params(values: dict[str, float | bool]) -> None:
        """Validate dynamic visualization parameter values."""
        field_color_min = float(values['field_color_min'])
        field_color_max = float(values['field_color_max'])

        if field_color_min >= field_color_max:
            raise ValueError(
                'field_color_min must be smaller than field_color_max.'
            )

    def _log_dynamic_parameter_update(self, changed_names: set[str]) -> None:
        """Log successful dynamic parameter updates."""
        if {
            'field_color_min',
            'field_color_max',
            'use_fixed_field_color_max',
        } & changed_names:
            if self._use_fixed_field_color_max:
                self.get_logger().info(
                    'Using fixed field color range '
                    f'[{self._field_color_min:.6f}, {self._field_color_max:.6f}].'
                )
            else:
                self.get_logger().info(
                    'Using dynamic field color scaling. Stored fixed range is '
                    f'[{self._field_color_min:.6f}, {self._field_color_max:.6f}].'
                )

        if 'z_offset' in changed_names:
            self.get_logger().info(f'Using z_offset {self._z_offset:.6f}.')

    def _read_config(self) -> BeliefConfig:
        cfg = BeliefConfig(
            frame_id=self.get_parameter('frame_id').value,
            x_min=float(self.get_parameter('x_min').value),
            x_max=float(self.get_parameter('x_max').value),
            y_min=float(self.get_parameter('y_min').value),
            y_max=float(self.get_parameter('y_max').value),
            kernel_type=self.get_parameter('kernel_type').value,
            training_iter=int(self.get_parameter('training_iter').value),
            learning_rate=float(self.get_parameter('learning_rate').value),
            init_lengthscale_x=float(
                self.get_parameter('init_lengthscale_x').value
            ),
            init_lengthscale_y=float(
                self.get_parameter('init_lengthscale_y').value
            ),
            init_outputscale=float(
                self.get_parameter('init_outputscale').value
            ),
            init_noise=float(self.get_parameter('init_noise').value),
            refit_policy=self.get_parameter('refit_policy').value,
            refit_every_k=int(self.get_parameter('refit_every_k').value),
            publish_visualization=bool(
                self.get_parameter('publish_visualization').value
            ),
            visualization_grid_step=float(
                self.get_parameter('visualization_grid_step').value
            ),
            visualization_z_mode=self.get_parameter(
                'visualization_z_mode'
            ).value,
            visualization_height_scale=float(
                self.get_parameter('visualization_height_scale').value
            ),
        )
        cfg.validate()
        return cfg

    def _on_measurement(self, msg: ScalarMeasurement) -> None:
        """Store one incoming scalar measurement and refit if required.

        Only the x and y position components are used by the current 2D belief.
        The z coordinate and orientation in the measurement pose are ignored.

        Measurements with a frame different from the configured belief frame are
        rejected. This node does not transform measurements between frames.
        """
        frame_id = msg.header.frame_id or self.config.frame_id
        if frame_id != self.config.frame_id:
            self.get_logger().warning(
                f"Ignoring measurement in frame '{frame_id}'"
                + f", expected '{self.config.frame_id}'."
            )
            return

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        value = float(msg.value)

        try:
            result = self.belief.add_measurement(x=x, y=y, value=value)
            self.get_logger().info(
                f'Received measurement at ({x:.3f}, {y:.3f}) = {value:.6f}; '
                f'N={result.num_measurements}, refit={result.did_refit}'
            )
            if result.did_refit and self.config.publish_visualization:
                self._publish_visualization()
        except Exception as exc:
            self.get_logger().error(
                f'Failed to process measurement: {exc}\n{traceback.format_exc()}'
            )

    def _handle_query(self, request, response):
        """Handle a posterior belief query service request.

        The service expects query poses in the configured belief frame. The
        current model returns latent posterior mean and variance in original
        measurement units.
        """
        if len(request.queries) == 0:
            response.success = False
            response.status_message = 'No query poses provided.'
            return response
        if not self.belief.has_model():
            response.success = False
            response.status_message = 'Belief has no fitted model yet.'
            return response

        xy = []
        for pose_stamped in request.queries:
            frame_id = pose_stamped.header.frame_id or self.config.frame_id
            if frame_id != self.config.frame_id:
                response.success = False
                response.status_message = (
                    f"Expected frame '{self.config.frame_id}'"
                    + f", got '{frame_id}'."
                )
                return response
            xy.append(
                [pose_stamped.pose.position.x, pose_stamped.pose.position.y]
            )

        try:
            mean, variance = self.belief.query(np.asarray(xy, dtype=float))
        except Exception as exc:
            self.get_logger().error(f'Belief query failed: {exc}')
            response.success = False
            response.status_message = f'Belief query failed: {exc}'
            return response

        response.success = True
        response.mean = mean.tolist()
        response.variance = variance.tolist()
        response.status_message = 'ok'
        return response

    def _handle_reset(self, _request, response):
        self.belief.reset()
        response.success = True
        response.message = 'Belief reset.'
        self.get_logger().info('Belief reset.')
        return response

    def _publish_visualization(self) -> None:
        """Publish mean and variance point clouds for the current belief.

        The visualization grid is generated in physical coordinates over the
        configured domain. The belief is queried at all grid points, and the
        resulting mean and variance are published as separate clouds.

        The mean cloud uses the configurable field color range. If fixed color
        scaling is disabled, the upper color limit is chosen from the current
        posterior mean cloud.

        The variance cloud currently uses the default color range of
        `make_field_pointcloud2`.
        """
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

        if self._use_fixed_field_color_max:
            effective_color_max = self._field_color_max
        else:
            effective_color_max = max(
                float(np.max(mean)),
                self._field_color_min + 1e-6,
            )
        mean_cloud = make_field_pointcloud2(
            positions_xy=grid_xy,
            values=mean,
            frame_id=self.config.frame_id,
            stamp=stamp,
            z_mode=self.config.visualization_z_mode,
            z_offset=self._z_offset,
            colormap_min=self._field_color_min,
            colormap_max=effective_color_max,
            height_scale=self.config.visualization_height_scale,
        )
        var_cloud = make_field_pointcloud2(
            positions_xy=grid_xy,
            values=variance,
            frame_id=self.config.frame_id,
            stamp=stamp,
            z_mode=self.config.visualization_z_mode,
            z_offset=self._z_offset,
            height_scale=self.config.visualization_height_scale,
        )
        self.mean_pub.publish(mean_cloud)
        self.var_pub.publish(var_cloud)


def main(args=None):
    rclpy.init(args=args)
    node = ScalarFieldBeliefNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
