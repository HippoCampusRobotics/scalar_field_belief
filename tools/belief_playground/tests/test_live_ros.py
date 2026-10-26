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

"""Live viewer ROS side, in-process: real publishers and the viewer node in
one rclpy context under a random namespace. Also the belief clouds'
`intensity` field. Needs a sourced ROS workspace."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('nav_msgs')
pytest.importorskip('scalar_field_interfaces')
visualization = pytest.importorskip('scalar_field_belief.visualization')

from belief_playground.live_data import LiveState, MethodInfo  # noqa: E402
from belief_playground.live_main import resolve_auto_domain  # noqa: E402
from belief_playground.live_ros import (  # noqa: E402
    CLOUD_QOS,
    LiveViewerNode,
    RosSpinner,
    decode_cloud,
    discover_parameters,
    init_context,
)
from geometry_msgs.msg import PoseWithCovarianceStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from scalar_field_interfaces.msg import ScalarMeasurement  # noqa: E402
from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402

TIMEOUT = 10.0


def wait_for(cond, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def stamp():
    from builtin_interfaces.msg import Time

    return Time()


def belief_clouds(frame='map'):
    xy = visualization.make_grid_positions(-1.0, 1.0, 2.0, 2.95, 0.1)
    rng = np.random.default_rng(3)
    mean = rng.normal(size=len(xy))
    var = rng.uniform(0.0, 2.0, size=len(xy))
    mean_cloud = visualization.make_field_pointcloud2(
        xy, mean, frame, stamp(), include_intensity=True
    )
    var_cloud = visualization.make_intensity_pointcloud2(
        xy, var, frame, stamp()
    )
    return xy, mean, var, mean_cloud, var_cloud


def odometry(x, y, frame='map'):
    msg = Odometry()
    msg.header.frame_id = frame
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    return msg


def measurement(x, y, value, frame='map', clipped=False):
    msg = ScalarMeasurement()
    msg.header.frame_id = frame
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.value = value
    msg.clipped = clipped
    return msg


class Harness:
    """Own context, a publisher node and (on demand) the viewer node."""

    def __init__(self):
        self.ns = f'/live_test_{uuid.uuid4().hex[:12]}'
        self.context = init_context(['test'])
        self.pub_node = rclpy.create_node(
            'live_test_pub', namespace=self.ns, context=self.context
        )
        self.state = LiveState(frame_id='map', trajectory_min_step=0.0)
        self.spinner = None

    def start_viewer(self, **kw):
        node = LiveViewerNode(self.state, context=self.context, **kw)
        # The node uses relative topics; give it the test namespace.
        assert node.get_namespace() == '/'
        self.spinner = RosSpinner(node).start()
        return node

    def publisher(self, msg_type, topic, qos=10):
        return self.pub_node.create_publisher(
            msg_type, f'{self.ns}/{topic}', qos
        )

    def close(self):
        self.pub_node.destroy_node()
        if self.spinner is not None:
            self.spinner.stop()
        rclpy.try_shutdown(context=self.context)


@pytest.fixture
def ros():
    h = Harness()
    yield h
    h.close()


def ns_topics(ns):
    return {
        'mean_topic': f'{ns}/belief/mean_cloud',
        'variance_topic': f'{ns}/belief/variance_cloud',
        'measurement_topic': f'{ns}/ir_measurement',
        'pose_topic': f'{ns}/odometry',
    }


def test_clouds_poses_and_measurements_are_decoded(ros):
    xy, mean, var, mean_cloud, var_cloud = belief_clouds()
    mean_pub = ros.publisher(PointCloud2, 'belief/mean_cloud', CLOUD_QOS)
    var_pub = ros.publisher(PointCloud2, 'belief/variance_cloud', CLOUD_QOS)
    # Latched: published before the viewer subscribes.
    mean_pub.publish(mean_cloud)
    var_pub.publish(var_cloud)
    ros.start_viewer(**ns_topics(ros.ns))
    odom_pub = ros.publisher(Odometry, 'odometry')
    meas_pub = ros.publisher(ScalarMeasurement, 'ir_measurement')
    assert wait_for(
        lambda: (
            odom_pub.get_subscription_count() >= 1
            and meas_pub.get_subscription_count() >= 1
        )
    )
    for k in range(5):
        odom_pub.publish(odometry(0.1 * k, 2.5))
    meas_pub.publish(measurement(0.2, 2.5, 7.25, clipped=True))
    meas_pub.publish(measurement(0.3, 2.6, -1.0, frame=''))

    def done():
        s = ros.state.snapshot()
        return (
            s.mean is not None
            and s.variance is not None
            and s.streams['pose'].count == 5
            and s.streams['measurements'].count == 2
        )

    assert wait_for(done), ros.state.snapshot().streams
    snap = ros.state.snapshot()
    np.testing.assert_array_equal(snap.mean.xy, xy.astype(np.float32))
    np.testing.assert_array_equal(snap.mean.values, mean.astype(np.float32))
    np.testing.assert_array_equal(snap.variance.values, var.astype(np.float32))
    assert snap.mean.frame_id == 'map'
    assert snap.pose.x == pytest.approx(0.4) and snap.pose.y == 2.5
    np.testing.assert_allclose(snap.trajectory[:, 0], 0.1 * np.arange(5))
    np.testing.assert_allclose(
        snap.measurements, [[0.2, 2.5, 7.25, 1.0], [0.3, 2.6, -1.0, 0.0]]
    )
    assert snap.streams['mean'].topic == f'{ros.ns}/belief/mean_cloud'
    assert all(st.errors == 0 for st in snap.streams.values())


def test_wrong_frame_is_ignored_and_counted(ros):
    ros.start_viewer(**ns_topics(ros.ns))
    odom_pub = ros.publisher(Odometry, 'odometry')
    meas_pub = ros.publisher(ScalarMeasurement, 'ir_measurement')
    mean_pub = ros.publisher(PointCloud2, 'belief/mean_cloud', CLOUD_QOS)
    assert wait_for(lambda: odom_pub.get_subscription_count() >= 1)
    assert wait_for(lambda: meas_pub.get_subscription_count() >= 1)
    odom_pub.publish(odometry(1.0, 1.0, frame='odom'))
    meas_pub.publish(measurement(1.0, 1.0, 3.0, frame='base_link'))
    mean_pub.publish(belief_clouds(frame='odom')[3])
    odom_pub.publish(odometry(2.0, 2.0))

    def done():
        st = ros.state.stream_status()
        return (
            st['pose'].frame_mismatch == 1
            and st['pose'].count == 1
            and st['measurements'].frame_mismatch == 1
            and st['mean'].frame_mismatch == 1
        )

    assert wait_for(done), ros.state.stream_status()
    snap = ros.state.snapshot()
    assert (snap.pose.x, snap.pose.y) == (2.0, 2.0)
    assert len(snap.trajectory) == 1
    assert len(snap.measurements) == 0
    assert snap.mean is None
    assert snap.streams['pose'].last_bad_frame == 'odom'


def test_rgb_only_mean_cloud_is_an_error_not_data(ros):
    ros.start_viewer(**ns_topics(ros.ns))
    xy = visualization.make_grid_positions(0.0, 1.0, 0.0, 1.0, 0.25)
    rgb_only = visualization.make_field_pointcloud2(
        xy, np.linspace(0, 1, len(xy)), 'map', stamp()
    )
    mean_pub = ros.publisher(PointCloud2, 'belief/mean_cloud', CLOUD_QOS)
    mean_pub.publish(rgb_only)
    assert wait_for(lambda: ros.state.stream_status()['mean'].errors == 1)
    snap = ros.state.snapshot()
    assert snap.mean is None and snap.streams['mean'].count == 0
    assert 'intensity' in snap.streams['mean'].last_error


def test_pose_with_covariance_and_disabled_streams(ros):
    ros.start_viewer(
        mean_topic=f'{ros.ns}/m',
        variance_topic=f'{ros.ns}/v',
        measurement_topic='',
        pose_topic=f'{ros.ns}/vision_pose_cov',
        pose_type='pose_cov',
    )
    pub = ros.publisher(PoseWithCovarianceStamped, 'vision_pose_cov')
    assert wait_for(lambda: pub.get_subscription_count() >= 1)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.pose.pose.position.x, msg.pose.pose.position.y = 0.5, -0.25
    pub.publish(msg)
    assert wait_for(lambda: ros.state.snapshot().pose is not None)
    snap = ros.state.snapshot()
    assert (snap.pose.x, snap.pose.y) == (0.5, -0.25)
    assert snap.streams['measurements'].topic is None


def test_relative_topics_resolve_in_the_ros_args_namespace():
    ns = f'/live_test_{uuid.uuid4().hex[:12]}'
    context = init_context(['viewer', '--ros-args', '-r', f'__ns:={ns}'])
    try:
        node = LiveViewerNode(LiveState(), context=context)
        spinner = RosSpinner(node).start()
        st = node.state.stream_status()
        assert st['mean'].topic == f'{ns}/belief/mean_cloud'
        assert st['variance'].topic == f'{ns}/belief/variance_cloud'
        assert st['measurements'].topic == f'{ns}/ir_measurement'
        assert st['pose'].topic == f'{ns}/odometry'
        spinner.stop()
        spinner.stop()  # idempotent
        assert not spinner.thread.is_alive()
        assert not context.ok()
    finally:
        rclpy.try_shutdown(context=context)


def test_field_cloud_default_layout_is_unchanged_and_intensity_is_added():
    xy = visualization.make_grid_positions(0.0, 0.3, 0.0, 0.2, 0.1)
    values = np.linspace(-1.0, 2.0, len(xy)) + 1e-9
    plain = visualization.make_field_pointcloud2(
        xy, values, 'map', stamp(), colormap_min=-1.0, colormap_max=2.0
    )
    with_i = visualization.make_field_pointcloud2(
        xy,
        values,
        'map',
        stamp(),
        colormap_min=-1.0,
        colormap_max=2.0,
        include_intensity=True,
    )
    f32 = PointField.FLOAT32
    layout = [(f.name, f.offset, f.datatype, f.count) for f in plain.fields]
    assert layout == [
        ('x', 0, f32, 1),
        ('y', 4, f32, 1),
        ('z', 8, f32, 1),
        ('rgb', 12, f32, 1),
    ]
    assert plain.point_step == 16 and plain.row_step == 16 * len(xy)
    assert [
        (f.name, f.offset, f.datatype, f.count) for f in with_i.fields
    ] == layout + [('intensity', 16, f32, 1)]
    assert with_i.point_step == 20
    a = np.frombuffer(bytes(plain.data), dtype=np.uint8).reshape(-1, 16)
    b = np.frombuffer(bytes(with_i.data), dtype=np.uint8).reshape(-1, 20)
    np.testing.assert_array_equal(b[:, :16], a)
    stored = b[:, 16:].copy().view(np.float32).ravel()
    np.testing.assert_array_equal(stored, values.astype(np.float32))
    xy_dec, v_dec = decode_cloud(with_i)
    np.testing.assert_array_equal(v_dec, values.astype(np.float32))
    np.testing.assert_array_equal(xy_dec, xy.astype(np.float32))
    with pytest.raises(ValueError, match='intensity'):
        decode_cloud(plain)


def raw_cloud(xy, values, *, big_endian=False, step=16, value_type='f4'):
    """PointCloud2 with x, y at 0/4 and the value at offset 8 (padded to
    `step` bytes per point)."""
    order = '>' if big_endian else '<'
    datatype = {'f4': PointField.FLOAT32, 'f8': PointField.FLOAT64}
    dtype = np.dtype(
        {
            'names': ['x', 'y', 'intensity'],
            'formats': [order + 'f4', order + 'f4', order + value_type],
            'offsets': [0, 4, 8],
            'itemsize': step,
        }
    )
    pts = np.zeros(len(xy), dtype=dtype)
    pts['x'], pts['y'], pts['intensity'] = xy[:, 0], xy[:, 1], values
    msg = PointCloud2()
    msg.height, msg.width = 1, len(xy)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(
            name='intensity', offset=8, datatype=datatype[value_type], count=1
        ),
    ]
    msg.is_bigendian = big_endian
    msg.point_step, msg.row_step = step, step * len(xy)
    msg.data = pts.tobytes()
    return msg


@pytest.mark.parametrize(
    'kw',
    [
        {'big_endian': True},
        {'step': 32},  # padded points
        {'value_type': 'f8'},  # FLOAT64 values are decoded, not rejected
        {'value_type': 'f8', 'big_endian': True, 'step': 24},
    ],
)
def test_decode_cloud_layouts(kw):
    xy = visualization.make_grid_positions(0.0, 0.5, 0.0, 0.3, 0.1)
    values = np.linspace(-2.0, 3.0, len(xy)) + 1e-12
    msg = raw_cloud(xy, values, **kw)
    before = bytes(msg.data)
    xy_dec, v_dec = decode_cloud(msg)
    np.testing.assert_array_equal(xy_dec, xy.astype(np.float32))
    np.testing.assert_array_equal(
        v_dec, values.astype(np.dtype(kw.get('value_type', 'f4')))
    )
    assert bytes(msg.data) == before  # the message is not modified


def test_decode_cloud_rejects_bad_layouts():
    xy = visualization.make_grid_positions(0.0, 0.5, 0.0, 0.3, 0.1)
    msg = raw_cloud(xy, np.ones(len(xy)))
    msg.data = bytes(msg.data)[:-4]
    with pytest.raises(ValueError, match='data bytes'):
        decode_cloud(msg)
    msg = raw_cloud(xy, np.ones(len(xy)))
    msg.fields[2].offset = 14  # ends after point_step 16
    with pytest.raises(ValueError, match='point_step'):
        decode_cloud(msg)
    msg = raw_cloud(xy, np.ones(len(xy)))
    msg.fields[2].count = 3
    with pytest.raises(ValueError, match='one numeric value'):
        decode_cloud(msg)


def test_callback_exceptions_are_reported_and_spinning_continues(ros):
    ros.start_viewer(**ns_topics(ros.ns))
    pub = ros.publisher(Odometry, 'odometry')
    assert wait_for(lambda: pub.get_subscription_count() >= 1)
    original = ros.state.set_pose

    def fail_once(*a, **k):
        ros.state.set_pose = original
        raise RuntimeError('boom')

    ros.state.set_pose = fail_once
    pub.publish(odometry(1.0, 1.0))
    assert wait_for(lambda: ros.state.snapshot().ros_errors == 1)
    pub.publish(odometry(2.0, 1.0))
    assert wait_for(lambda: ros.state.snapshot().pose is not None)
    snap = ros.state.snapshot()
    assert snap.ros_alive and snap.ros_last_error == 'RuntimeError: boom'
    assert snap.pose.x == 2.0


def test_external_context_shutdown_marks_ros_stopped(ros):
    ros.start_viewer(**ns_topics(ros.ns))
    rclpy.try_shutdown(context=ros.context)
    assert wait_for(lambda: not ros.state.snapshot().ros_alive)
    assert 'shut down' in ros.state.snapshot().ros_stopped_reason
    ros.spinner.thread.join(2.0)
    assert not ros.spinner.thread.is_alive()


def test_stop_does_not_destroy_a_node_busy_in_a_callback(ros):
    node = ros.start_viewer(**ns_topics(ros.ns))
    pub = ros.publisher(Odometry, 'odometry')
    assert wait_for(lambda: pub.get_subscription_count() >= 1)
    entered = []
    original = ros.state.set_pose

    def slow(*a, **k):
        entered.append(1)
        time.sleep(1.0)
        return original(*a, **k)

    destroyed = []
    destroy = node.destroy_node
    node.destroy_node = lambda: destroyed.append(1)
    ros.state.set_pose = slow
    pub.publish(odometry(1.0, 1.0))
    assert wait_for(lambda: entered)
    ros.spinner.stop(timeout=0.1)
    assert ros.spinner.thread.is_alive()  # still in the callback
    assert not ros.context.ok()
    assert destroyed == []
    ros.spinner.thread.join(3.0)
    assert not ros.spinner.thread.is_alive()
    destroy()


class FakeSystem:
    """A field server and a belief node in their own context, spun in a
    thread, standing in for the simulation processes."""

    def __init__(self):
        from rclpy.executors import SingleThreadedExecutor

        tag = uuid.uuid4().hex[:12]
        self.ns = f'/live_test_{tag}'
        self.field_server = f'/field_server_{tag}'
        self.context = init_context(['fake'])
        self.executor = SingleThreadedExecutor(context=self.context)
        self.nodes = []
        self._stop = False
        self.thread = None

    def add_field_server(self, scenario_path):
        node = rclpy.create_node(
            self.field_server[1:], namespace='/', context=self.context
        )
        node.declare_parameter('scenario_path', scenario_path)
        self._add(node)

    def add_belief_node(self, **params):
        node = rclpy.create_node(
            'scalar_field_belief', namespace=self.ns, context=self.context
        )
        for name, value in params.items():
            node.declare_parameter(name, value)
        self._add(node)

    def add_hung_belief_node(self):
        """Parameter services exist but are never served (not spun)."""
        node = rclpy.create_node(
            'scalar_field_belief', namespace=self.ns, context=self.context
        )
        node.declare_parameter('x_min', 0.0)
        self.nodes.append(node)

    def add_estimator(self, name, namespace, topic, **params):
        """A node that publishes the mean cloud and declares `params`."""
        node = rclpy.create_node(
            name, namespace=namespace, context=self.context
        )
        for key, value in params.items():
            node.declare_parameter(key, value)
        node.create_publisher(PointCloud2, topic, CLOUD_QOS)
        self._add(node)
        return node

    def remove(self, node):
        self.executor.remove_node(node)
        self.nodes.remove(node)
        node.destroy_node()

    def _add(self, node):
        self.nodes.append(node)
        self.executor.add_node(node)
        if self.thread is None:
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()

    def _spin(self):
        while not self._stop and self.context.ok():
            self.executor.spin_once(timeout_sec=0.05)

    def close(self):
        self._stop = True
        if self.thread is not None:
            self.thread.join(2.0)
        self.executor.shutdown()
        for node in self.nodes:
            node.destroy_node()
        rclpy.try_shutdown(context=self.context)


@pytest.fixture
def fake():
    system = FakeSystem()
    yield system
    system.close()


def discover(fake, *, need_belief=True, timeout=10.0, grace=2.0):
    context = init_context(['viewer', '--ros-args', '-r', f'__ns:={fake.ns}'])
    try:
        t0 = time.monotonic()
        found = discover_parameters(
            context,
            fake.field_server,
            'scalar_field_belief',
            need_belief=need_belief,
            timeout=timeout,
            grace=grace,
        )
        return found, time.monotonic() - t0
    finally:
        rclpy.try_shutdown(context=context)


def test_discovery_prefers_the_field_server_scenario(
    fake, offset_scenario_path
):
    fake.add_field_server(str(offset_scenario_path))
    fake.add_belief_node(
        frame_id='world', x_min=0.0, x_max=2.0, y_min=0.0, y_max=4.0
    )
    found, elapsed = discover(fake)
    assert found.field_server == fake.field_server
    assert found.belief_node == f'{fake.ns}/scalar_field_belief'
    assert found.field_server_params == {
        'scenario_path': str(offset_scenario_path)
    }
    assert found.belief_params == {
        'frame_id': 'world',
        'x_min': 0.0,
        'x_max': 2.0,
        'y_min': 0.0,
        'y_max': 4.0,
    }  # undeclared parameters are left out
    assert elapsed < 8.0
    choice = resolve_auto_domain(
        found.field_server_params,
        found.belief_params,
        field_server=found.field_server,
        belief_node=found.belief_node,
    )
    assert choice.domain.x_range == (-3.0, 1.0) and choice.scenario is not None
    assert choice.frame_id == 'world'


def test_discovery_falls_back_to_the_belief_node(fake):
    fake.add_belief_node(
        frame_id='map', x_min=-1.0, x_max=1.0, y_min=0.0, y_max=2.0
    )
    found, elapsed = discover(fake, timeout=10.0, grace=0.5)
    assert found.field_server_params is None
    assert found.belief_params['x_min'] == -1.0
    assert elapsed < 5.0  # grace, not the full timeout
    choice = resolve_auto_domain(None, found.belief_params)
    assert choice.domain.x_range == (-1.0, 1.0) and choice.scenario is None


def test_field_server_alone_is_enough_with_an_explicit_frame(
    fake, offset_scenario_path
):
    fake.add_field_server(str(offset_scenario_path))
    found, elapsed = discover(fake, need_belief=False, timeout=10.0)
    assert found.field_server_params is not None
    assert found.belief_params is None
    assert elapsed < 5.0


def test_discovery_times_out(fake):
    found, elapsed = discover(fake, timeout=0.5)
    assert found.field_server_params is None and found.belief_params is None
    assert 0.4 < elapsed < 3.0


def test_a_hung_node_cannot_extend_the_timeout(fake):
    fake.add_hung_belief_node()
    found, elapsed = discover(fake, timeout=1.0)
    assert found.field_server_params is None and found.belief_params is None
    assert elapsed < 1.5


def test_grace_also_caps_pending_requests(fake, offset_scenario_path):
    fake.add_field_server(str(offset_scenario_path))
    fake.add_hung_belief_node()
    found, elapsed = discover(fake, timeout=10.0, grace=0.5)
    assert found.field_server_params is not None
    assert found.belief_params is None
    assert elapsed < 3.0


HUNG_NODE = """
import sys, time, rclpy
rclpy.init(args=sys.argv)
node = rclpy.create_node('scalar_field_belief')
node.declare_parameter('x_min', 0.0)
try:
    time.sleep(60)
except KeyboardInterrupt:
    pass
"""


def test_ctrl_c_during_a_pending_request_exits_130_promptly():
    ns = f'/live_test_{uuid.uuid4().hex[:12]}'
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ('DISPLAY', 'WAYLAND_DISPLAY')
    }
    env.setdefault('ROS_DOMAIN_ID', '98')
    ros = ['--ros-args', '-r', f'__ns:={ns}']
    hung = subprocess.Popen([sys.executable, '-c', HUNG_NODE, *ros], env=env)
    try:
        time.sleep(2.0)
        viewer = subprocess.Popen(
            [
                sys.executable,
                '-m',
                'belief_playground.live_main',
                '--field-server-node',
                f'{ns}/no_field_server',
                *ros,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(4.0)  # discovery is waiting for the hung node
        assert viewer.poll() is None
        t0 = time.monotonic()
        viewer.send_signal(signal.SIGINT)
        _, err = viewer.communicate(timeout=30)
        elapsed = time.monotonic() - t0
    finally:
        hung.send_signal(signal.SIGINT)
        hung.wait(10)
    assert viewer.returncode == 130, err
    assert 'waiting up to 15 s' in err
    assert 'belief_live_viewer: interrupted' in err
    assert 'Traceback' not in err
    assert elapsed < 1.5


def method_is(ros, name, details=None, timeout=TIMEOUT):
    def check():
        m = ros.state.snapshot().method
        return m.name == name and (details is None or m.details == details)

    return wait_for(check, timeout)


def test_method_of_the_mean_cloud_publisher(ros, fake):
    topic = f'{ros.ns}/belief/mean_cloud'
    fake.add_estimator(
        'scalar_field_belief',
        ros.ns,
        topic,
        model_variant='wall_aware',
        kernel_type='matern',
        x_min=0.0,
    )
    ros.start_viewer(**ns_topics(ros.ns), retry_s=0.3)
    assert method_is(ros, 'GP wall-aware', 'kernel matern'), (
        ros.state.snapshot().method
    )
    m = ros.state.snapshot().method
    assert m.node == f'{ros.ns}/scalar_field_belief'
    assert (m.publishers, m.note) == (1, None)


def test_publisher_that_appears_later_or_changes_is_picked_up(ros, fake):
    topic = f'{ros.ns}/belief/mean_cloud'
    ros.start_viewer(**ns_topics(ros.ns), retry_s=0.3)
    # No publisher yet: the --belief-node fallback is queried.
    assert wait_for(
        lambda: (
            'querying /scalar_field_belief'
            in (ros.state.snapshot().method.note or '')
        )
    ), ros.state.snapshot().method
    assert ros.state.snapshot().method.name == 'Belief'
    gp = fake.add_estimator(
        'scalar_field_belief',
        ros.ns,
        topic,
        model_variant='baseline',
        kernel_type='rbf',
    )
    assert method_is(ros, 'GP baseline', 'kernel rbf')
    fake.remove(gp)
    fake.add_estimator('gmrf_belief', ros.ns, topic, nx=41, ny=81)
    assert method_is(ros, 'GMRF', '41x81 nodes'), ros.state.snapshot().method
    assert ros.state.snapshot().method.node == f'{ros.ns}/gmrf_belief'


def test_known_parameters_are_reused(ros, fake):
    topic = f'{ros.ns}/belief/mean_cloud'
    fake.add_estimator('scalar_field_belief', ros.ns, topic)  # no params
    known = {
        f'{ros.ns}/scalar_field_belief': {
            'model_variant': 'baseline',
            'kernel_type': 'rbf',
        }
    }
    ros.start_viewer(**ns_topics(ros.ns), retry_s=0.3, known_params=known)
    assert method_is(ros, 'GP baseline', 'kernel rbf')


def test_named_after_a_node_without_known_parameters(ros, fake):
    topic = f'{ros.ns}/belief/mean_cloud'
    fake.add_estimator('my_estimator', ros.ns, topic, x_min=0.0)
    ros.start_viewer(**ns_topics(ros.ns), retry_s=0.3)
    assert method_is(ros, 'my_estimator', '')


def test_no_lookup_with_a_method_label(ros, fake):
    topic = f'{ros.ns}/belief/mean_cloud'
    fake.add_estimator('scalar_field_belief', ros.ns, topic, nx=3, ny=3)
    node = ros.start_viewer(
        **ns_topics(ros.ns), retry_s=0.3, method_lookup=False
    )
    time.sleep(1.0)
    assert node.method_lookup is None
    assert ros.state.snapshot().method == MethodInfo()
