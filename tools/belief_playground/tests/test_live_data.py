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

"""Live viewer data side: display resampling and the shared state (no Qt,
no ROS)."""

from __future__ import annotations

import numpy as np
import pytest
from belief_playground.live_data import (
    LiveState,
    MethodInfo,
    PointBuffer,
    TrajectoryBuffer,
    cloud_to_image,
)
from belief_playground.scenario import make_display_grid


def axis(start, stop, step):
    """Same rule as scalar_field_belief.visualization.make_axis."""
    n = int(np.floor((stop - start) / step))
    a = start + step * np.arange(n + 1)
    return a if np.isclose(a[-1], stop) else np.append(a, stop)


def lattice(xs, ys, dtype=float):
    """Row-major cloud over meshgrid(xs, ys), like make_grid_positions."""
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()]).astype(dtype)


def field(xy):
    return xy[:, 0] + 10.0 * xy[:, 1]  # orientation-revealing


def test_rows_are_y_and_node_values_are_hit_exactly():
    # Display cell centres at 0.05, 0.15, ... coincide with the nodes.
    grid = make_display_grid((0.0, 1.0), (2.0, 2.6), 0.1)
    xy = lattice(grid.xs, grid.ys)
    img = cloud_to_image(xy, field(xy), grid)
    assert img.shape == grid.shape == (6, 10)
    np.testing.assert_array_equal(img, grid.to_image(field(grid.xy)))
    assert img[0, 9] == pytest.approx(grid.xs[9] + 10 * grid.ys[0])
    assert img[5, 0] == pytest.approx(grid.xs[0] + 10 * grid.ys[5])


def test_nearest_node_per_axis_without_interpolation():
    xs, ys = axis(0.0, 1.0, 0.1), axis(0.0, 0.5, 0.1)
    xy = lattice(xs, ys)
    values = field(xy)
    grid = make_display_grid((0.0, 1.0), (0.0, 0.5), 0.02)
    img = cloud_to_image(xy, values, grid)
    assert np.all(np.isfinite(img))
    assert set(np.unique(img)) <= set(values)
    for j in (0, 7, 24):
        for i in (0, 13, 49):
            ix = np.argmin(np.abs(xs - grid.xs[i]))
            iy = np.argmin(np.abs(ys - grid.ys[j]))
            assert img[j, i] == xs[ix] + 10 * ys[iy]


def test_shorter_last_spacing_is_covered():
    xs = axis(0.0, 0.95, 0.1)  # ..., 0.8, 0.9, 0.95
    ys = axis(0.0, 0.4, 0.1)
    assert np.isclose(xs[-1] - xs[-2], 0.05)
    xy = lattice(xs, ys)
    grid = make_display_grid((0.0, 0.95), (0.0, 0.4), 0.01)
    img = cloud_to_image(xy, field(xy), grid)
    assert np.all(np.isfinite(img))
    last = grid.xs > 0.925 + 1e-9  # nearest node is 0.95
    np.testing.assert_allclose(img[0, last], 0.95)


def test_float32_coordinates_form_one_lattice():
    xs, ys = axis(-3.0, 1.0, 0.07), axis(2.0, 3.5, 0.07)
    xy = lattice(xs, ys, dtype=np.float32)
    grid = make_display_grid((-3.0, 1.0), (2.0, 3.5), 0.05)
    img = cloud_to_image(xy, field(xy.astype(float)), grid)
    assert np.all(np.isfinite(img))


def test_holes_stay_nan():
    xs, ys = axis(0.0, 1.0, 0.1), axis(0.0, 1.0, 0.1)
    xy = lattice(xs, ys)
    hole = np.isclose(xy[:, 0], 0.5) & np.isclose(xy[:, 1], 0.3)
    grid = make_display_grid((0.0, 1.0), (0.0, 1.0), 0.01)
    img = cloud_to_image(xy[~hole], field(xy[~hole]), grid)
    near_x = np.abs(grid.xs - 0.5) < 0.05 - 1e-9
    near_y = np.abs(grid.ys - 0.3) < 0.05 - 1e-9
    assert np.all(np.isnan(img[np.ix_(near_y, near_x)]))
    assert np.isfinite(img).sum() == img.size - near_x.sum() * near_y.sum()


def test_outside_the_cloud_is_nan_not_extrapolated():
    xs, ys = axis(0.0, 0.5, 0.1), axis(0.0, 1.0, 0.1)
    xy = lattice(xs, ys)
    grid = make_display_grid((0.0, 1.0), (0.0, 1.0), 0.02)
    img = cloud_to_image(xy, field(xy), grid)
    inside = grid.xs <= 0.55
    assert np.all(np.isfinite(img[:, inside]))
    assert np.all(np.isnan(img[:, ~inside]))
    # A cloud that misses the display domain entirely: all grey.
    far = lattice(xs + 5.0, ys)
    assert np.all(np.isnan(cloud_to_image(far, field(far), grid)))


def test_single_node_axes_do_not_crash():
    grid = make_display_grid((0.0, 1.0), (0.0, 1.0), 0.02)
    row = lattice(axis(0.0, 1.0, 0.1), [0.5])
    img = cloud_to_image(row, field(row), grid)
    finite_rows = np.where(np.isfinite(img).any(axis=1))[0]
    assert np.all(np.abs(grid.ys[finite_rows] - 0.5) <= 0.05 + 1e-9)
    assert len(finite_rows) > 0
    one = cloud_to_image([[0.31, 0.71]], [4.0], grid)
    assert np.nanmax(one) == 4.0
    assert np.isfinite(one).sum() == 1
    empty = cloud_to_image(np.empty((0, 2)), [], grid)
    assert np.all(np.isnan(empty))


def test_non_finite_values_and_positions():
    xs = ys = axis(0.0, 1.0, 0.5)
    xy = lattice(xs, ys)
    values = field(xy)
    values[0] = np.nan
    xy_bad = np.vstack([xy, [[np.nan, 0.0]]])
    grid = make_display_grid((0.0, 1.0), (0.0, 1.0), 0.5)
    img = cloud_to_image(xy_bad, np.append(values, 1.0), grid)
    assert np.isnan(img).sum() == 1
    with pytest.raises(ValueError):
        cloud_to_image(xy, values[:-1], grid)


def test_point_buffer_keeps_newest_in_order():
    buf = PointBuffer(2, max_points=5)
    for k in range(12):
        buf.append((k, -k))
    assert len(buf) == 5 and buf.dropped == 7
    np.testing.assert_array_equal(buf.array()[:, 0], [7, 8, 9, 10, 11])
    big = PointBuffer(1, max_points=3000)  # grows past the initial size
    for k in range(2500):
        big.append((k,))
    np.testing.assert_array_equal(big.array()[:, 0], np.arange(2500))


def test_trajectory_min_step_and_cap():
    traj = TrajectoryBuffer(min_step=0.1, max_points=3)
    assert traj.add(0.0, 0.0)
    assert not traj.add(0.05, 0.05)  # 0.07 m < 0.1 m
    assert traj.add(0.125, 0.0)
    assert not traj.add(np.nan, 1.0)
    assert traj.add(0.25, 0.0)
    assert traj.add(0.375, 0.0)
    np.testing.assert_allclose(traj.array()[:, 0], [0.125, 0.25, 0.375])
    traj.clear()
    assert len(traj) == 0 and traj.add(0.375, 0.0)
    with pytest.raises(ValueError):
        TrajectoryBuffer(min_step=-1.0)


def test_state_frames_counts_and_versions():
    now = [100.0]
    state = LiveState(frame_id='map', clock=lambda: now[0])
    v0 = state.version
    assert state.set_pose(1.0, 2.0, 'map')
    assert state.set_pose(1.5, 2.0, '')  # empty frame = expected frame
    assert not state.set_pose(9.0, 9.0, 'odom')
    assert not state.add_measurement(1.0, 1.0, 3.0, False, 'base_link')
    assert state.add_measurement(1.0, 1.5, 3.0, True, 'map')
    now[0] = 101.0
    assert state.set_cloud('mean', [[0, 0], [1, 0]], [1.0, 2.0], 'map')
    assert not state.set_cloud('variance', [[0, 0]], [1.0], 'odom')
    assert state.version > v0
    snap = state.snapshot()
    st = snap.streams
    assert (st['pose'].count, st['pose'].frame_mismatch) == (2, 1)
    assert st['pose'].last_bad_frame == 'odom'
    assert (st['measurements'].count, st['measurements'].frame_mismatch) == (
        1,
        1,
    )
    assert (st['variance'].count, st['variance'].frame_mismatch) == (0, 1)
    assert st['mean'].last_received == 101.0
    assert st['pose'].last_received == 100.0
    assert snap.pose.x == 1.5 and snap.pose.frame_id == 'map'
    np.testing.assert_allclose(snap.trajectory, [[1.0, 2.0], [1.5, 2.0]])
    np.testing.assert_allclose(snap.measurements, [[1.0, 1.5, 3.0, 1.0]])
    np.testing.assert_allclose(snap.mean.values, [1.0, 2.0])
    assert snap.variance is None


def test_state_part_versions_and_errors():
    state = LiveState(trajectory_min_step=0.5)
    s0 = state.snapshot()
    state.set_pose(0.0, 0.0)
    state.set_pose(0.1, 0.0)  # pose changes, trajectory does not
    s1 = state.snapshot()
    assert s1.pose_version == s0.pose_version + 2
    assert s1.trajectory_version == s0.trajectory_version + 1
    assert s1.mean_version == s0.mean_version
    assert not state.set_cloud('mean', [[0, 0], [1, 1]], [1.0])
    state.report_error('variance', 'variance cloud: broken')
    s2 = state.snapshot()
    assert s2.streams['mean'].errors == 1
    assert 'positions' in s2.streams['mean'].last_error
    assert s2.streams['variance'].last_error == 'variance cloud: broken'
    assert s2.mean is None and s2.mean_version == s1.mean_version
    state.clear_trajectory()
    s3 = state.snapshot()
    assert len(s3.trajectory) == 0 and s3.pose is not None
    assert s3.trajectory_version == s2.trajectory_version + 1
    with pytest.raises(ValueError):
        state.set_cloud('pose', [[0, 0]], [1.0])


def test_uneven_three_node_axis_has_no_false_grey_stripe():
    # Gaps 0.1 and 0.05: node 0.0 covers up to 0.05 (its own gap), not
    # half the median gap.
    xs = np.array([0.0, 0.1, 0.15])
    ys = axis(0.0, 0.2, 0.1)
    xy = lattice(xs, ys)
    grid = make_display_grid((0.0, 0.15), (0.0, 0.2), 0.005)
    img = cloud_to_image(xy, field(xy), grid)
    assert np.all(np.isfinite(img))
    np.testing.assert_allclose(img[0, grid.xs < 0.05 - 1e-9], 0.0)


def test_display_step_equal_to_cloud_step_uses_each_node_once():
    # Cell centres 0.05, 0.15, ... lie exactly between two nodes; ties go
    # to the lower node, so cells use nodes 0..9 in order and the last node
    # (x = 1.0, on the domain edge) is never shown: an unavoidable end
    # effect of nodes at the edges and cells inside.
    xs = axis(0.0, 1.0, 0.1)
    ys = axis(0.0, 0.5, 0.1)
    xy = lattice(xs, ys)
    index = np.argmin(np.abs(xy[:, 0][:, None] - xs[None, :]), axis=1)
    grid = make_display_grid((0.0, 1.0), (0.0, 0.5), 0.1)
    img = cloud_to_image(xy, index.astype(float), grid)
    assert np.all(np.isfinite(img))
    row = img[0].astype(int)
    np.testing.assert_array_equal(row, np.arange(10))
    assert np.all(img == img[0])


def test_scattered_points_are_rejected_as_non_lattice():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.0, 1.0, size=(500, 2))
    with pytest.raises(ValueError, match='not a regular lattice'):
        cloud_to_image(xy, np.ones(500), make_display_grid((0, 1), (0, 1), 0.1))
    state = LiveState()
    assert not state.set_cloud('mean', xy, np.ones(500), 'map')
    st = state.stream_status()['mean']
    assert st.errors == 1 and st.count == 0
    assert 'not a regular lattice' in st.last_error
    assert state.snapshot().mean is None
    # A lattice with many holes is still accepted.
    full = lattice(axis(0.0, 1.0, 0.1), axis(0.0, 1.0, 0.1))
    assert state.set_cloud('mean', full[::3], np.ones(len(full[::3])), 'map')


def test_snapshot_reuses_unchanged_arrays():
    state = LiveState(trajectory_min_step=0.5)
    state.set_pose(0.0, 0.0)
    state.add_measurement(0.0, 0.0, 1.0)
    s1 = state.snapshot()
    state.set_pose(0.1, 0.0)  # pose only
    s2 = state.snapshot(since=s1)
    assert s2.trajectory is s1.trajectory
    assert s2.measurements is s1.measurements
    assert s2.version > s1.version and s2.pose.x == 0.1
    state.set_pose(1.0, 0.0)  # stored
    state.clear_measurements()
    s3 = state.snapshot(since=s2)
    assert s3.trajectory is not s2.trajectory and len(s3.trajectory) == 2
    assert len(s3.measurements) == 0
    assert s3.streams['measurements'].count == 1


def test_ros_errors_and_stop_are_recorded():
    state = LiveState()
    snap = state.snapshot()
    assert snap.ros_alive and snap.ros_errors == 0
    state.report_ros_error('RuntimeError: x')
    state.set_ros_stopped('ROS context was shut down')
    snap = state.snapshot()
    assert not snap.ros_alive
    assert snap.ros_stopped_reason == 'ROS context was shut down'
    assert (snap.ros_errors, snap.ros_last_error) == (1, 'RuntimeError: x')


@pytest.mark.parametrize(
    'params, node, expected',
    [
        (
            {'model_variant': 'baseline', 'kernel_type': 'rbf'},
            '/uuv00/scalar_field_belief',
            ('GP baseline', 'kernel rbf'),
        ),
        (
            {'model_variant': 'wall_aware', 'kernel_type': 'matern'},
            None,
            ('GP wall-aware', 'kernel matern'),
        ),
        ({'model_variant': 'baseline'}, None, ('GP baseline', '')),
        ({'nx': 41, 'ny': 81}, '/gmrf_belief', ('GMRF', '41x81 nodes')),
        ({'nx': 41}, '/uuv00/my_estimator', ('my_estimator', '')),
        ({}, '/uuv00/my_estimator', ('my_estimator', '')),
        (None, None, ('Belief', '')),
    ],
)
def test_method_names(params, node, expected):
    from belief_playground.live_data import method_name

    assert method_name(params, node) == expected


def test_set_method_bumps_only_on_change():
    state = LiveState()
    s0 = state.snapshot()
    assert s0.method == MethodInfo() and s0.method.name == 'Belief'
    state.set_method(MethodInfo())
    assert state.snapshot().method_version == s0.method_version
    info = MethodInfo('GMRF', '41x81 nodes', '/gmrf_belief', None, 1)
    state.set_method(info)
    s1 = state.snapshot()
    assert s1.method == info and s1.method_version == s0.method_version + 1
