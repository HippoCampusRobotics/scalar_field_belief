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

"""Live viewer window on Xvfb, fed directly through a `LiveState` (no ROS).

Run under `env -u WAYLAND_DISPLAY QT_QPA_PLATFORM=xcb xvfb-run -a -s
"-screen 0 1920x1080x24" python -m pytest ...`; skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from belief_playground.live_data import LiveState, MethodInfo
from belief_playground.scenario import Domain, make_display_grid
from conftest import process_events

gui = pytest.importorskip('belief_playground.gui')
live_gui = pytest.importorskip('belief_playground.live_gui')
from PySide6 import QtCore  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

DOMAIN = Domain(
    x_range=(-3.0, 1.0),
    y_range=(2.0, 3.5),
    walls=(((0.0, 2.2), (0.0, 3.3)),),
    unit='mV',
)
GRID = make_display_grid(DOMAIN.x_range, DOMAIN.y_range, 0.25)  # 16 x 6


def field(xy):
    return xy[:, 0] + 10.0 * xy[:, 1]


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def state(clock):
    return LiveState(frame_id='map', trajectory_min_step=0.0, clock=clock)


@pytest.fixture
def make_window(qapp):
    windows = []

    def make(state, truth=None, **kw):
        w = live_gui.LiveViewerWindow(state, DOMAIN, GRID, truth, **kw)
        w.resize(1200, 800)
        w.show()
        process_events(qapp, 0.2)
        windows.append(w)
        return w

    yield make
    for w in windows:
        w.close()
    process_events(qapp, 0.05)


def feed_clouds(state, mean=None, var=None):
    xy = GRID.xy  # nodes at the display cell centres: exact hits
    mean = field(xy) if mean is None else mean
    var = (0.1 + 0.01 * np.arange(len(xy))) if var is None else var
    state.set_cloud('mean', xy, mean, 'map')
    state.set_cloud('variance', xy, var, 'map')
    return mean, var


def test_images_levels_and_std(qapp, state, make_window):
    truth = field(GRID.xy) - 5.0
    w = make_window(state, truth=truth, method_label='GP')
    mean, var = feed_clouds(state)
    w.refresh()
    np.testing.assert_array_equal(w._mean_img, GRID.to_image(mean))
    np.testing.assert_allclose(w._std_img, GRID.to_image(np.sqrt(var)))
    assert w.mean_clim == pytest.approx(
        (min(truth.min(), mean.min()), max(truth.max(), mean.max()))
    )
    assert w.std_clim == pytest.approx((0.0, np.sqrt(var).max()))
    for key, img, clim in (
        ('truth', GRID.to_image(truth), w.mean_clim),
        ('mean', GRID.to_image(mean), w.mean_clim),
        ('std', GRID.to_image(np.sqrt(var)), w.std_clim),
    ):
        p = w.panels[key]
        np.testing.assert_array_equal(
            p.image.image, gui.colorize(img, clim, p.lut)
        )
    assert w.panels['mean'].base_title == 'GP — estimate'
    assert w.panels['std'].base_title == 'GP — standard deviation'


def test_fixed_ranges_and_grey_cells(qapp, state, make_window):
    w = make_window(state, mean_range=(0.0, 1.0), std_max=0.5)
    keep = np.ones(len(GRID.xy), dtype=bool)
    keep[7] = False  # a hole
    var = np.full(len(GRID.xy), 0.04)
    var[3] = -1.0  # clearly negative
    state.set_cloud('mean', GRID.xy[keep], field(GRID.xy[keep]), 'map')
    state.set_cloud('variance', GRID.xy, var, 'map')
    w.refresh()
    assert w.mean_clim == (0.0, 1.0) and w.std_clim == (0.0, 0.5)
    grey = np.array(gui.INVALID_RGBA, dtype=np.uint8)
    mean_rgba = w.panels['mean'].image.image.reshape(-1, 4)
    np.testing.assert_array_equal(mean_rgba[7], grey)
    assert (mean_rgba == grey).all(axis=1).sum() == 1
    assert w._std_img.ravel()[3] == 0.0
    assert w._neg_var == 1
    assert 'negative' in w.status_label.toolTip()
    assert 'different points' in w.status_label.text()
    # Truth not available: the whole truth panel is grey.
    truth_rgba = w.panels['truth'].image.image.reshape(-1, 4)
    assert (truth_rgba == grey).all()
    assert 'not available' in w.panels['truth'].title_label.text()


def test_overlays_and_toggles(qapp, state, make_window):
    w = make_window(state, show_trajectory=False)
    assert not w.trajectory_checkbox.isChecked()
    assert w.measurements_checkbox.isChecked()
    for x, y in ((-2.0, 2.5), (-1.5, 2.5), (-1.0, 3.0)):
        state.set_pose(x, y, 'map')
    for k in range(300):  # many measurements: no per-point labels
        state.add_measurement(-2.5 + 0.01 * k, 2.2, float(k), False, 'map')
    w.refresh()
    for p, traj, tail, pos in zip(
        w.panels.values(),
        w.trajectory_items,
        w.tail_items,
        w.position_items,
        strict=True,
    ):
        x, y = traj.getData()
        np.testing.assert_allclose(x, [-2.0, -1.5, -1.0])
        np.testing.assert_allclose(y, [2.5, 2.5, 3.0])
        tx, ty = tail.getData()
        np.testing.assert_allclose(tx, [-1.0, -1.0])
        np.testing.assert_allclose(ty, [3.0, 3.0])
        px, py = pos.getData()
        assert (list(px), list(py)) == ([-1.0], [3.0])
        mx, _ = p.markers.getData()
        assert len(mx) == 300
        assert p._labels == []
        assert not traj.isVisible() and not tail.isVisible()
        assert pos.isVisible()
        assert traj.zValue() > gui.Z_WALLS + 1 and pos.zValue() > gui.Z_WALLS
    w.trajectory_checkbox.setChecked(True)
    w.measurements_checkbox.setChecked(False)
    assert all(t.isVisible() for t in w.trajectory_items + w.tail_items)
    assert not any(p.markers.isVisible() for p in w.panels.values())
    assert w.show_trajectory and not w.show_measurements
    w.actions_by_name['Clear trajectory'].trigger()
    x, _ = w.trajectory_items[0].getData()
    assert len(x) == 0 and len(state.snapshot().trajectory) == 0
    assert w.position_items[0].getData()[0][0] == -1.0
    w.actions_by_name['Clear measurements'].trigger()
    assert all(len(p.markers.getData()[0]) == 0 for p in w.panels.values())
    assert state.snapshot().streams['measurements'].count == 300


def test_pose_only_updates_do_not_redraw_the_trajectory(
    qapp, make_window, monkeypatch
):
    state = LiveState(trajectory_min_step=1.0)
    w = make_window(state)
    state.set_pose(-2.0, 2.5)
    w.refresh()
    calls = []
    item = w.trajectory_items[0]
    original = item.setData
    monkeypatch.setattr(
        item, 'setData', lambda *a, **k: (calls.append(1), original(*a, **k))
    )
    state.set_pose(-1.8, 2.6)  # 0.22 m < 1 m: not stored
    w.refresh()
    assert calls == []
    tx, ty = w.tail_items[0].getData()
    np.testing.assert_allclose(tx, [-2.0, -1.8])
    np.testing.assert_allclose(ty, [2.5, 2.6])
    assert w.position_items[0].getData()[0][0] == -1.8
    state.set_pose(-0.5, 2.6)  # stored
    w.refresh()
    assert calls == [1]


def test_clicks_do_not_add_observations(qapp, state, make_window):
    w = make_window(state)
    feed_clouds(state)
    w.refresh()
    version = state.version
    p = w.panels['mean']
    QTest.mouseClick(
        p.view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        p.map_to_view(-1.0, 2.75),
    )
    process_events(qapp, 0.1)
    assert state.version == version
    assert len(state.snapshot().measurements) == 0


def test_status_waiting_counts_stale_and_errors(
    qapp, state, clock, make_window
):
    state.set_topic('mean', '/uuv00/belief/mean_cloud')
    state.set_topic('variance', '/uuv00/belief/variance_cloud')
    state.set_topic('pose', '/uuv00/odometry')
    state.set_topic('measurements', None)
    w = make_window(state, stale_after=5.0)
    text = w.status_label.text()
    assert 'waiting for /uuv00/belief/mean_cloud' in text
    assert 'waiting for /uuv00/odometry' in text
    assert 'waiting' in w.panels['mean'].title_label.text()
    feed_clouds(state)
    state.set_pose(0.5, 3.0, 'map')
    state.set_pose(0.6, 3.0, 'odom')
    state.report_error('mean', "mean cloud: no 'intensity' field")
    clock.t += 2.0
    w.refresh()
    text = w.status_label.text()
    assert '<b>mean</b>: 1, 2.0 s ago' in text
    assert '1 ignored' in text and 'odom' in text
    assert 'no &#x27;intensity&#x27; field' in text
    assert 'stale' not in text
    clock.t += 10.0
    w.refresh()
    text = w.status_label.text()
    # Only the pose stream becomes stale; old clouds are normal.
    assert '<b>variance</b>: 1, 12 s ago &nbsp;' in text
    assert (
        '<b>pose</b>: 1, 12 s ago <span style="color:#ffca28">stale</span>'
        in text
    )
    assert text.count('stale') == 1
    assert 'stale' not in w.panels['mean'].title_label.text()
    assert 'ROS STOPPED' not in text
    state.report_ros_error('RuntimeError: boom')
    state.set_ros_stopped('ROS context was shut down')
    w.refresh()
    text = w.status_label.text()
    assert text.startswith(
        '<b>Belief</b> &nbsp;|&nbsp; <b><span style="color:#ff7043">ROS STOPPED'
    )
    assert 'ROS context was shut down' in text
    assert '1 ROS errors, last: RuntimeError: boom' in text


def test_redraw_only_on_new_data(qapp, state, make_window, monkeypatch):
    w = make_window(state)
    calls = []
    original = w._update_maps
    monkeypatch.setattr(
        w, '_update_maps', lambda *a: (calls.append(1), original(*a))
    )
    w.refresh()
    w.refresh()
    assert calls == []
    feed_clouds(state)
    w.refresh()
    assert len(calls) == 1


def test_export_and_clean_close(qapp, state, make_window, tmp_path):
    hook_calls = []
    w = make_window(state, shutdown_hook=lambda: hook_calls.append(1))
    feed_clouds(state)
    out = w.export_image(tmp_path / 'img' / 'live.png')
    assert out is not None and out.is_file()
    assert 'exported' in w.message
    w.close()
    process_events(qapp, 0.05)
    assert not w.timer.isActive()
    assert hook_calls == [1]
    w.shutdown()
    w.refresh()
    assert hook_calls == [1]


def test_domain_and_frame_source_in_tooltip(qapp, state, make_window):
    w = make_window(
        state,
        domain_source='/field_server scenario_path: /s/a.toml',
        frame_source='/uuv00/scalar_field_belief frame_id',
        domain_warnings=('belief bounds differ',),
    )
    tip = w.status_label.toolTip().splitlines()
    assert tip[0] == 'method: Belief'
    assert tip[1] == 'domain: /field_server scenario_path: /s/a.toml'
    assert tip[2] == 'frame: map (/uuv00/scalar_field_belief frame_id)'
    assert tip[3] == 'WARNING: belief bounds differ'


def test_method_name_in_titles_status_and_window_title(
    qapp, state, make_window
):
    w = make_window(state, title='offset_rect')
    assert w.panels['mean'].base_title == 'Belief — estimate'
    assert w.windowTitle() == 'belief live viewer - offset_rect - Belief'
    state.set_method(
        MethodInfo(note='method: querying /uuv00/scalar_field_belief…')
    )
    w.refresh()
    assert 'querying /uuv00/scalar_field_belief' in w.status_label.text()
    state.set_method(
        MethodInfo(
            'GP wall-aware',
            'kernel rbf',
            '/uuv00/scalar_field_belief',
            publishers=1,
        )
    )
    w.refresh()
    assert w.panels['mean'].base_title == 'GP wall-aware — estimate'
    assert w.panels['std'].base_title == 'GP wall-aware — standard deviation'
    assert 'GP wall-aware — estimate' in w.panels['mean'].title_label.text()
    assert w.windowTitle() == (
        'belief live viewer - offset_rect - GP wall-aware'
    )
    assert w.status_label.text().startswith(
        '<b>GP wall-aware</b> (kernel rbf) · '
        '<span style="color:#9e9e9e">/uuv00/scalar_field_belief</span>'
    )
    assert w.status_label.toolTip().splitlines()[0] == (
        'method: GP wall-aware (kernel rbf) from /uuv00/scalar_field_belief'
    )
    assert 'querying' not in w.status_label.text()


def test_method_label_overrides_the_detected_method(qapp, state, make_window):
    w = make_window(state, method_label='My GMRF')
    state.set_method(MethodInfo('GP baseline', 'kernel rbf', '/a/b'))
    w.refresh()
    assert w.panels['mean'].base_title == 'My GMRF — estimate'
    assert w.windowTitle() == 'belief live viewer - My GMRF'
    text = w.status_label.text()
    assert text.startswith('<b>My GMRF</b> &nbsp;|&nbsp; ')
    assert 'GP baseline' not in text and '/a/b' not in text
