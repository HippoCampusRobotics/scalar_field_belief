# Copyright (C) 2026 Nathalie Bauschmann
# SPDX-License-Identifier: GPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""ROS side of the live viewer (rclpy, no Qt).

`LiveViewerNode` subscribes to one estimator's belief clouds, the measurement
topic and a pose topic, decodes them and writes them into a `LiveState`.
It also finds the node that publishes the mean cloud and names its method
(`method_name`) from its parameters, without blocking. `RosSpinner` runs the
node in a daemon thread. Callbacks never touch Qt; the
GUI polls the `LiveState`. `discover_parameters` reads the parameters that
define the domain from the running field server / belief node before the
viewer node is created.

Both clouds are read as positions `x`, `y` and the numeric field `intensity`
(posterior mean / variance), with any numeric PointField type, either byte
order and padded points. A mean cloud without that field (an RViz-only cloud
with packed `rgb` colours) is reported as an error, not decoded from colours.
Frames are compared with `LiveState.frame_id`; there is no TF.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
import rclpy.client
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from scalar_field_interfaces.msg import ScalarMeasurement
from sensor_msgs.msg import PointCloud2, PointField

from belief_playground.live_data import (
    DEFAULT_METHOD,
    LiveState,
    MethodInfo,
    method_name,
)

NODE_NAME = 'belief_live_viewer'
VALUE_FIELD = 'intensity'

# Matches the estimators' latched clouds (reliable, transient local, 1).
CLOUD_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
MEASUREMENT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.RELIABLE,
)
# Best effort is compatible with reliable and best-effort publishers.
POSE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)

_POINT_FIELD_TYPES = {
    PointField.INT8: 'i1',
    PointField.UINT8: 'u1',
    PointField.INT16: 'i2',
    PointField.UINT16: 'u2',
    PointField.INT32: 'i4',
    PointField.UINT32: 'u4',
    PointField.FLOAT32: 'f4',
    PointField.FLOAT64: 'f8',
}

# --pose-type: message type and how to get the position from it.
POSE_TYPES = {
    'odometry': (Odometry, lambda m: m.pose.pose.position),
    'pose': (PoseStamped, lambda m: m.pose.position),
    'pose_cov': (PoseWithCovarianceStamped, lambda m: m.pose.pose.position),
}


def decode_cloud(
    msg: PointCloud2, value_field: str = VALUE_FIELD
) -> tuple[np.ndarray, np.ndarray]:
    """Positions (N, 2) and values (N,) of a PointCloud2 as float64.

    Reads copies; the message is not modified. Raises ValueError if `x`,
    `y` or `value_field` is missing, has a non-numeric type or count != 1,
    or if the data does not match the declared layout.
    """
    fields = {f.name: f for f in msg.fields}
    wanted = ('x', 'y', value_field)
    missing = [n for n in wanted if n not in fields]
    if missing:
        raise ValueError(
            f'no {", ".join(repr(n) for n in missing)} field '
            f'(fields: {", ".join(fields) or "none"})'
        )
    order = '>' if msg.is_bigendian else '<'
    step, width, height = msg.point_step, msg.width, msg.height
    formats = []
    for name in wanted:
        f = fields[name]
        code = _POINT_FIELD_TYPES.get(f.datatype)
        if code is None or f.count != 1:
            raise ValueError(
                f'field {name!r} has datatype {f.datatype} and count '
                f'{f.count}; expected one numeric value'
            )
        if f.offset + int(code[1]) > step:
            raise ValueError(
                f'field {name!r} ends after point_step {step} (offset '
                f'{f.offset})'
            )
        formats.append(order + code)
    if width * height == 0:
        return np.empty((0, 2)), np.empty(0)
    if msg.row_step < width * step:
        raise ValueError(
            f'row_step {msg.row_step} < width {width} * point_step {step}'
        )
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if len(raw) < height * msg.row_step:
        raise ValueError(
            f'{len(raw)} data bytes, expected {height * msg.row_step}'
        )
    rows = raw[: height * msg.row_step].reshape(height, msg.row_step)
    packed = np.ascontiguousarray(rows[:, : width * step]).reshape(-1)
    dtype = np.dtype(
        {
            'names': list(wanted),
            'formats': formats,
            'offsets': [fields[n].offset for n in wanted],
            'itemsize': step,
        }
    )
    points = packed.view(dtype)
    xy = np.column_stack([points['x'], points['y']]).astype(float)
    return xy, points[value_field].astype(float)


class LiveViewerNode(Node):
    """Subscriptions of the live viewer; writes into `state`.

    Topics are relative by default and therefore resolve in the node
    namespace. An empty measurement or pose topic disables that stream.
    With `method_lookup`, a timer finds the method of the node publishing
    the mean cloud (`belief_node` while there is no publisher yet);
    `known_params` maps fully qualified node names to parameters that are
    already known.
    """

    def __init__(
        self,
        state: LiveState,
        mean_topic: str = 'belief/mean_cloud',
        variance_topic: str = 'belief/variance_cloud',
        measurement_topic: str = 'ir_measurement',
        pose_topic: str = 'odometry',
        pose_type: str = 'odometry',
        context: Context | None = None,
        method_lookup: bool = True,
        belief_node: str = 'scalar_field_belief',
        known_params: dict | None = None,
        lookup_period: float = 0.25,
        retry_s: float = 2.0,
    ):
        super().__init__(NODE_NAME, context=context)
        if pose_type not in POSE_TYPES:
            raise ValueError(
                f'pose_type must be one of {", ".join(POSE_TYPES)}, '
                f'got {pose_type!r}'
            )
        self.state = state
        self._subs = {}
        self._subscribe(
            'mean',
            PointCloud2,
            mean_topic,
            lambda m: self._on_cloud('mean', m),
            CLOUD_QOS,
        )
        self._subscribe(
            'variance',
            PointCloud2,
            variance_topic,
            lambda m: self._on_cloud('variance', m),
            CLOUD_QOS,
        )
        self._subscribe(
            'measurements',
            ScalarMeasurement,
            measurement_topic,
            self._on_measurement,
            MEASUREMENT_QOS,
        )
        msg_type, self._position = POSE_TYPES[pose_type]
        self._subscribe('pose', msg_type, pose_topic, self._on_pose, POSE_QOS)
        self.method_lookup = None
        if method_lookup:
            self.method_lookup = MethodLookup(
                self,
                self._subs['mean'].topic_name,
                belief_node,
                known_params or {},
                retry_s,
            )
            self.create_timer(lookup_period, self.method_lookup.tick)

    def _subscribe(self, stream, msg_type, topic, callback, qos) -> None:
        if not topic:
            self.state.set_topic(stream, None)
            return
        sub = self.create_subscription(msg_type, topic, callback, qos)
        self._subs[stream] = sub
        self.state.set_topic(stream, sub.topic_name)

    def _on_cloud(self, stream: str, msg: PointCloud2) -> None:
        try:
            xy, values = decode_cloud(msg)
        except Exception as exc:  # noqa: BLE001 - any malformed message
            self.state.report_error(stream, f'{stream} cloud: {exc}')
            return
        self.state.set_cloud(stream, xy, values, msg.header.frame_id)

    def _on_measurement(self, msg: ScalarMeasurement) -> None:
        p = msg.pose.position
        self.state.add_measurement(
            p.x, p.y, msg.value, msg.clipped, msg.header.frame_id
        )

    def _on_pose(self, msg) -> None:
        p = self._position(msg)
        self.state.set_pose(p.x, p.y, msg.header.frame_id)


FIELD_SERVER_PARAMS = ('scenario_path',)
BELIEF_PARAMS = (
    'frame_id',
    'x_min',
    'x_max',
    'y_min',
    'y_max',
    'model_variant',
    'belief_geometry',
    'scenario_path',
    'walls',
    'kernel_type',
    'nx',
    'ny',
)
METHOD_PARAMS = ('model_variant', 'kernel_type', 'nx', 'ny')
METHOD_REQUEST_TIMEOUT = 5.0  # [s] then the request is sent again
DISCOVERY_GRACE = 2.0  # [s] extra wait for the other node once one answered


@dataclass(frozen=True)
class Discovery:
    """Parameters read from the field server and the belief node (None:
    not reached); undeclared parameters are missing from the dicts."""

    field_server: str  # resolved node names
    belief_node: str
    field_server_params: dict | None
    belief_params: dict | None


class _Request:
    """Parameter request to one node: list the declared names, then get
    the wanted ones (rclpy nodes answer a request with any undeclared name
    with an empty list). Advanced by `step()` without blocking."""

    def __init__(self, client: AsyncParameterClient, names):
        self.client = client
        self.names = tuple(names)
        self.future = None
        self.getting: list[str] | None = None
        self.result: dict | None = None

    def step(self) -> None:
        if self.result is not None:
            return
        if self.future is None:
            if self.client.services_are_ready():
                self.future = self.client.list_parameters()
            return
        if not self.future.done():
            return
        future, self.future = self.future, None
        if future.exception() is not None:
            self.getting = None  # retry from the start
            return
        if self.getting is None:
            declared = set(future.result().result.names)
            self.getting = [n for n in self.names if n in declared]
            if self.getting:
                self.future = self.client.get_parameters(self.getting)
            else:
                self.result = {}
            return
        self.result = _to_python(self.getting, future.result().values)


def _to_python(names, values) -> dict:
    """{name: value} without parameters that are not set."""
    out = {}
    for name, value in zip(names, values, strict=False):
        try:
            v = parameter_value_to_python(value)
        except RuntimeError:
            v = None
        if v is not None:
            out[name] = v
    return out


def discover_parameters(
    context: Context,
    field_server: str,
    belief_node: str,
    *,
    need_belief: bool,
    timeout: float,
    grace: float = DISCOVERY_GRACE,
    log=None,
    poll_s: float = 0.1,
) -> Discovery:
    """Read `FIELD_SERVER_PARAMS` of `field_server` and `BELIEF_PARAMS` of
    `belief_node` (relative names resolve in the context's namespace).

    Both nodes are queried concurrently. Stops when the field server has a
    non-empty `scenario_path` (and, if `need_belief`, the belief node
    answered too), or when both answered; once one of them gave something
    usable, the other one gets at most `grace` more seconds; never later
    than `timeout`. Requests still pending then are abandoned. Spins in
    steps of `poll_s`, so a KeyboardInterrupt is raised promptly. Raises
    ValueError for an invalid node name.
    """
    node = rclpy.create_node(f'{NODE_NAME}_discovery', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        names, requests = {}, {}
        for key, target, wanted in (
            ('fs', field_server, FIELD_SERVER_PARAMS),
            ('bn', belief_node, BELIEF_PARAMS),
        ):
            try:
                names[key] = node.resolve_service_name(target)
                client = AsyncParameterClient(node, target)
            except Exception as exc:  # noqa: BLE001 - rclpy name errors
                raise ValueError(
                    f'invalid node name {target!r}: {exc}'
                ) from exc
            requests[key] = _Request(client, wanted)
        if log is not None:
            log(
                f'waiting up to {timeout:g} s for the parameters of '
                f'{names["fs"]} or {names["bn"]}'
            )
        deadline = time.monotonic() + timeout
        stop_at = deadline
        while True:
            for request in requests.values():
                request.step()
            fs, bn = requests['fs'].result, requests['bn'].result
            path = str((fs or {}).get('scenario_path', '')).strip()
            if path and (bn is not None or not need_belief):
                break
            if fs is not None and bn is not None:
                break
            now = time.monotonic()
            if path or bn is not None:
                stop_at = min(stop_at, now + grace)
            if now >= stop_at:
                break
            executor.spin_once(timeout_sec=min(poll_s, stop_at - now))
        return Discovery(names['fs'], names['bn'], fs, bn)
    finally:
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()


def _full_name(info) -> str:
    return info.node_namespace.rstrip('/') + '/' + info.node_name


def _destroy_clients(node: Node, client: AsyncParameterClient) -> None:
    for value in vars(client).values():
        if isinstance(value, rclpy.client.Client):
            node.destroy_client(value)


class MethodLookup:
    """Names the method of the node that publishes the mean cloud.

    `tick()` runs on the ROS thread (timer) and never blocks: every
    `retry_s` it checks the publishers of `topic` (without one, the last
    target, else `fallback`); a new target is queried with a non-blocking
    parameter request, resent after METHOD_REQUEST_TIMEOUT. The result goes
    to `node.state.set_method`.
    """

    def __init__(self, node, topic, fallback, known, retry_s, clock=None):
        self.node, self.topic, self.known = node, topic, known
        self.retry_s = retry_s
        self.clock = clock or time.monotonic
        try:
            self.fallback = node.resolve_service_name(fallback)
        except Exception:  # noqa: BLE001 - invalid name: no fallback
            self.fallback = None
        self.target = None
        self.client = None
        self.request = None
        self.started = 0.0
        self.next_check = 0.0
        self.publishers = 0
        self.name, self.details = DEFAULT_METHOD, ''
        self.note = None

    def tick(self) -> None:
        now = self.clock()
        if now >= self.next_check:
            self.next_check = now + self.retry_s
            self._check_target(now)
        if self.request is None:
            return
        self.request.step()
        if self.request.result is not None:
            self.name, self.details = method_name(
                self.request.result, self.target
            )
            self.note = None
            self._finish_request()
        elif now - self.started > METHOD_REQUEST_TIMEOUT:
            self.note = (
                f'method unknown ({self.target} does not answer; retrying)'
            )
            self.request = _Request(self.client, METHOD_PARAMS)
            self.started = now
        self._publish()

    def _check_target(self, now: float) -> None:
        infos = self.node.get_publishers_info_by_topic(self.topic)
        self.publishers = len(infos)
        target = _full_name(infos[0]) if infos else self.target
        target = target or self.fallback
        if target is not None and target != self.target:
            self._finish_request()
            self.target = target
            if target in self.known:
                self.name, self.details = method_name(
                    self.known[target], target
                )
                self.note = None
            else:
                self.name, self.details = DEFAULT_METHOD, ''
                self.note = f'method: querying {target}…'
                self.client = AsyncParameterClient(self.node, target)
                self.request = _Request(self.client, METHOD_PARAMS)
                self.started = now
        self._publish()

    def _finish_request(self) -> None:
        self.request = None
        if self.client is not None:
            _destroy_clients(self.node, self.client)
            self.client = None

    def _publish(self) -> None:
        notes = [self.note] if self.note else []
        if self.publishers > 1:
            notes.append(
                f'{self.publishers} publishers on {self.topic}, '
                'showing the first'
            )
        elif self.publishers == 0 and self.target is not None:
            notes.append(f'no publisher on {self.topic} yet')
        self.node.state.set_method(
            MethodInfo(
                self.name,
                self.details,
                self.target,
                '; '.join(notes) or None,
                self.publishers,
            )
        )


def init_context(args: list[str] | None = None) -> Context:
    """A new initialised rclpy context. ROS arguments in `args` (after
    `--ros-args`) apply to nodes created in it. rclpy installs no signal
    handlers, so SIGINT stays with Python/Qt."""
    context = Context()
    rclpy.init(
        args=args,
        context=context,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    return context


class RosSpinner:
    """Spins one node in a daemon thread; `stop()` is idempotent.

    Exceptions while spinning are reported to `state` (default: the node's
    `state`) and spinning continues. If spinning ends without `stop()`, for
    example because the context was shut down, the state is marked as not
    alive.
    """

    def __init__(
        self, node: Node, poll_s: float = 0.1, state: LiveState | None = None
    ):
        self.node = node
        self.state = state if state is not None else node.state
        self.context = node.context
        self._poll_s = poll_s
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(node)
        self._stop = threading.Event()
        self._stopped = False
        self._lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._spin, name=f'{NODE_NAME}_ros', daemon=True
        )

    def start(self) -> RosSpinner:
        self.thread.start()
        return self

    def _spin(self) -> None:
        reason = 'ROS spin loop ended'
        try:
            while not self._stop.is_set():
                if not self.context.ok():
                    reason = 'ROS context was shut down'
                    break
                try:
                    self.executor.spin_once(timeout_sec=self._poll_s)
                except ExternalShutdownException:
                    reason = 'ROS context was shut down'
                    break
                except Exception as exc:  # noqa: BLE001 - keep receiving
                    self.state.report_ros_error(f'{type(exc).__name__}: {exc}')
                    self._stop.wait(0.01)
        except BaseException as exc:
            reason = f'ROS thread failed: {type(exc).__name__}: {exc}'
            raise
        finally:
            if not self._stop.is_set():
                self.state.set_ros_stopped(reason)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop spinning, destroy the node and shut the context down."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout)
        if self.thread.is_alive():
            # Still inside a callback: leave the node to the daemon thread.
            self.node.get_logger().warning(
                f'ROS thread did not stop within {timeout} s; shutting the '
                'context down without destroying the node'
            )
        else:
            self.executor.shutdown(timeout_sec=timeout)
            self.node.destroy_node()
        rclpy.try_shutdown(context=self.context)
