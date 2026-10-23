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

"""Qt GUI tests (PySide6 + PyQtGraph) on Xvfb with real Qt mouse events.

Run under `env -u WAYLAND_DISPLAY QT_QPA_PLATFORM=xcb xvfb-run -a -s
"-screen 0 1920x1080x24" python -m pytest ...`; skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from belief_playground.backends import ModelGridOverlay
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import Session
from conftest import FakeBackend, process_events

gui = pytest.importorskip('belief_playground.gui')
pg = pytest.importorskip('pyqtgraph')
from PySide6 import QtCore  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

LEFT = QtCore.Qt.MouseButton.LeftButton
RIGHT = QtCore.Qt.MouseButton.RightButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier
OFFSET_ASPECT = 4.0 / 1.5  # offset_rect: x [-3, 1], y [2, 3.5]


def make_window(
    qapp, path, backends, tmp_path, step=0.25, size=(1400, 900), **kw
):
    sc = Scenario(path)
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, step)
    session = Session(sc, backends, grid)
    w = gui.PlaygroundWindow(session, out_dir=tmp_path / 'out', **kw)
    w.resize(*size)
    w.show()
    process_events(qapp, 0.3)
    settle(qapp, w)
    return w


def settle(qapp, w):
    assert w.session.wait_idle(30)
    w.refresh(force=True)
    process_events(qapp, 0.1)


def px_size(panel):
    """World length of one screen pixel on this panel [m]."""
    (x0, x1) = panel.domain.x_range
    return (x1 - x0) / panel.view.width()


def click_world(qapp, panel, x, y, button=LEFT):
    QTest.mouseClick(
        panel.view.viewport(), button, NOMOD, panel.map_to_view(x, y)
    )
    process_events(qapp, 0.05)


def rendered_rgb(panel, x, y):
    img = panel.view.grab().toImage()
    p = panel.map_to_view(x, y)
    c = img.pixelColor(p.x(), p.y())
    return np.array([c.red(), c.green(), c.blue()])


@pytest.fixture
def win(qapp, offset_scenario_path, tmp_path):
    backends = {'a': FakeBackend('a'), 'b': FakeBackend('b', ready_after=2)}
    w = make_window(qapp, offset_scenario_path, backends, tmp_path)
    yield w
    w.close()


# ------------------------------------------------------------------ layout
def test_grid_is_cell_centred_and_row_major_in_y(offset_scenario_path):
    sc = Scenario(offset_scenario_path)
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, 0.25)
    assert grid.shape == (6, 16)  # (ny, nx) for a 4 m x 1.5 m domain
    np.testing.assert_allclose(grid.xs[[0, -1]], [-2.875, 0.875])
    np.testing.assert_allclose(grid.ys[[0, -1]], [2.125, 3.375])
    img = grid.to_image(grid.xy[:, 0] + 10 * grid.xy[:, 1])
    for j in (0, 3, 5):
        for i in (0, 7, 15):
            assert img[j, i] == pytest.approx(grid.xs[i] + 10 * grid.ys[j])


@pytest.mark.parametrize('names', [['a'], ['a', 'b'], ['a', 'b', 'c']])
def test_layout_columns_titles_and_reserved_cell(
    qapp, offset_scenario_path, tmp_path, names
):
    backends = {n: FakeBackend(n) for n in names}
    w = make_window(qapp, offset_scenario_path, backends, tmp_path)
    try:
        grid = w.panel_grid
        assert grid.ncols == 1 + len(names)
        expected = {(0, 0): 'truth', (1, 0): None}
        for k, n in enumerate(names):
            expected[(0, 1 + k)] = (n, 'mean')
            expected[(1, 1 + k)] = (n, 'std')
        assert set(grid.cells) == set(expected)
        assert set(w.panels) == {v for v in expected.values() if v is not None}
        for cell, key in expected.items():
            if key is not None:
                assert grid.cells[cell] is w.panels[key]

        # Titles.
        assert w.panels['truth'].title_label.text() == 'Ground truth'
        for n in names:
            assert (
                w.panels[(n, 'mean')].title_label.text()
                == f'{n.upper()} — estimate'
            )
            assert (
                w.panels[(n, 'std')].title_label.text()
                == f'{n.upper()} — standard deviation'
            )

        # Equal, aspect-correct maps packed with GAP between columns/rows.
        lay = grid.layout_info
        truth = w.panels['truth']
        for (row, col), widget in grid.cells.items():
            g = widget.geometry()
            if (row, col) == (1, 0):
                continue
            assert g.x() == truth.geometry().x() + col * (lay.map_w + gui.GAP)
            assert g.y() == truth.geometry().y() + row * (
                gui.TITLE_H + lay.map_h + gui.GAP
            )
            assert widget.view.width() == lay.map_w
            assert widget.view.height() == lay.map_h
        assert lay.map_w / lay.map_h == pytest.approx(
            OFFSET_ASPECT, abs=2.0 / lay.map_h * OFFSET_ASPECT
        )
        assert (
            lay.map_w >= gui.MIN_PANEL_W
            or lay.map_h * OFFSET_ASPECT < gui.MIN_PANEL_W
        )

        # Reserved bottom-left cell: exactly under the truth map, empty, black.
        res = grid.reserved
        rg, tg = res.geometry(), truth.geometry()
        assert (
            rg.x() == tg.x()
            and rg.width() == lay.map_w
            and rg.height() == lay.map_h
        )
        assert (
            rg.y() == tg.y() + gui.TITLE_H + lay.map_h + gui.GAP + gui.TITLE_H
        )
        assert res.findChildren(QtCore.QObject) == []
        img = res.grab().toImage()
        colors = {
            img.pixelColor(x, y).name()
            for x in range(0, img.width(), 7)
            for y in range(0, img.height(), 7)
        }
        assert colors == {'#000000'}
    finally:
        w.close()


def test_maps_have_no_axes_colorbars_or_controls(win):
    for p in win.panels.values():
        items = p.view.scene().items()
        bad = (pg.AxisItem, pg.HistogramLUTItem, pg.ColorBarItem, pg.PlotItem)
        assert not [i for i in items if isinstance(i, bad)]
        assert p.vb.state['mouseEnabled'] == [False, False]
        assert not p.vb.menuEnabled()
        assert p.vb.state['aspectLocked'] is not False
        # Whole domain visible, nothing more than rounding outside it.
        (vx0, vx1), (vy0, vy1) = p.vb.viewRange()
        (x0, x1), (y0, y1) = win.domain.x_range, win.domain.y_range
        tol = 2 * px_size(p)
        assert (
            vx0 <= x0 + 1e-9
            and vx1 >= x1 - 1e-9
            and vy0 <= y0 + 1e-9
            and vy1 >= y1 - 1e-9
        )
        assert (
            vx0 > x0 - tol
            and vx1 < x1 + tol
            and vy0 > y0 - tol
            and vy1 < y1 + tol
        )
    # Only the compact toolbar actions (+ Quit shortcut).
    assert set(win.actions_by_name) == {
        'Undo',
        'Reset',
        'Save session',
        'Load session',
        'Export image',
        'Quit',
    }


def test_compute_panel_layout_packs_and_scrolls():
    # Tall domain in a wide window: maps limited by height, block centred,
    # columns exactly GAP apart (no spreading).
    lay = gui.compute_panel_layout(1900, 1000, 3, 0.5)
    assert lay.map_h <= (1000 - 2 * gui.MARGIN - gui.GAP - 2 * gui.TITLE_H) / 2
    assert lay.content_w == 2 * gui.MARGIN + 3 * lay.map_w + 2 * gui.GAP
    assert lay.x0 - gui.MARGIN == (1900 - lay.content_w) // 2
    # Many methods in a narrow window: minimum size, wider than available.
    lay = gui.compute_panel_layout(800, 1000, 8, 2.0)
    assert lay.map_w == gui.MIN_PANEL_W and lay.content_w > 800
    assert lay.min_content_w == lay.content_w
    # Wide domain: limited by width.
    lay = gui.compute_panel_layout(1000, 1000, 3, 3.0)
    assert lay.content_w <= 1000


def test_many_methods_scroll_horizontally(qapp, offset_scenario_path, tmp_path):
    backends = {n: FakeBackend(n) for n in 'abcdef'}
    w = make_window(
        qapp, offset_scenario_path, backends, tmp_path, size=(900, 700)
    )
    try:
        assert w.panel_grid.layout_info.map_w >= min(
            gui.MIN_PANEL_W, w.panel_grid.layout_info.map_h * OFFSET_ASPECT - 1
        )
        assert w.panel_grid.width() > w.scroll.viewport().width()
        assert w.scroll.horizontalScrollBar().maximum() > 0
    finally:
        w.close()


def test_window_fits_the_screen(win):
    avail = win.screen().availableGeometry()
    w2 = gui.PlaygroundWindow(win.session, out_dir=win.out_dir)  # default size
    try:
        assert w2.height() <= avail.height() and w2.width() <= avail.width()
        # Large: the packed maps fill (almost) the height or the width.
        assert (
            w2.height() >= 0.8 * avail.height()
            or w2.width() >= 0.8 * avail.width()
        )
        # Not much more than the packed maps need.
        lay = w2.panel_grid.layout_info
        assert lay is None or w2.height() <= lay.content_h + 200
    finally:
        w2.timer.stop()  # shares the session with `win`; win closes it


# ------------------------------------------------------------------ clicks
def test_click_maps_to_the_same_physical_point_on_every_panel(qapp, win):
    targets = [(-2.9, 2.05), (0.95, 3.45), (-1.3, 2.7), (0.2, 2.3), (-0.6, 3.1)]
    for k, p in enumerate(win.panels.values()):
        x, y = targets[k % len(targets)]
        before = len(win.session.observations)
        click_world(qapp, p, x, y)
        obs = win.session.observations
        assert len(obs) == before + 1, p.key
        tol = px_size(p)
        assert obs[-1].x == pytest.approx(x, abs=tol)
        assert obs[-1].y == pytest.approx(y, abs=tol)


def test_ignored_clicks(qapp, win):
    p = win.panels['truth']
    click_world(qapp, p, -1.0, 3.0, button=RIGHT)
    # Drag: press, move far, release -> not a click.
    vp = p.view.viewport()
    start = p.map_to_view(-1.0, 3.0)
    end = p.map_to_view(0.0, 2.5)
    QTest.mousePress(vp, LEFT, NOMOD, start)
    for t in np.linspace(0, 1, 8):
        pt = QtCore.QPoint(
            int(start.x() + t * (end.x() - start.x())),
            int(start.y() + t * (end.y() - start.y())),
        )
        QTest.mouseMove(vp, pt)
        process_events(qapp, 0.01)
    QTest.mouseRelease(vp, LEFT, NOMOD, end)
    process_events(qapp, 0.05)
    # Title strip, reserved cell, a gap between panels.
    QTest.mouseClick(p.title_label, LEFT, NOMOD, QtCore.QPoint(20, 5))
    QTest.mouseClick(
        win.panel_grid.reserved, LEFT, NOMOD, QtCore.QPoint(30, 30)
    )
    gap = p.geometry().right() + 1 + gui.GAP // 2
    QTest.mouseClick(
        win.panel_grid,
        LEFT,
        NOMOD,
        QtCore.QPoint(gap, p.geometry().center().y()),
    )
    # Outside the domain (direct, as from a pixel at the very border).
    win.on_map_click(1.5, 3.0)
    process_events(qapp, 0.05)
    assert win.session.observations == ()
    click_world(qapp, p, 0.5, 3.0)
    assert len(win.session.observations) == 1


# ------------------------------------------------------- images and levels
def test_rendered_orientation_and_values(qapp, win):
    """The colour under a world point is the colour of the value there."""
    click_world(qapp, win.panels['truth'], -0.6, 3.1)
    settle(qapp, win)
    win.show_measurements = False  # markers must not cover sampled pixels
    process_events(qapp, 0.1)
    g = win.grid
    dx, dy = g.xs[1] - g.xs[0], g.ys[1] - g.ys[0]
    # Points well inside a cell but off its centre, so a half-cell shift,
    # a flip or a transpose puts them into a neighbouring cell.
    offsets = [(-0.3, -0.3), (0.3, 0.3), (-0.3, 0.3), (0.3, -0.3)]
    for i, j in [(0, 0), (15, 5), (3, 4), (12, 1)]:
        cx, cy = g.xs[i], g.ys[j]
        for ox, oy in offsets:
            x, y = cx + ox * dx, cy + oy * dy
            truth = win.scenario.truth(np.array([[cx, cy]]))[0]
            lut = win.panels['truth'].lut
            exp = gui.colorize(np.array([truth]), win.mean_clim, lut)[0, :3]
            np.testing.assert_array_equal(
                rendered_rgb(win.panels['truth'], x, y), exp
            )
            mean = cx + 10 * cy + 1  # FakeBackend pattern + 1 observation
            exp = gui.colorize(np.array([mean]), win.mean_clim, lut)[0, :3]
            np.testing.assert_array_equal(
                rendered_rgb(win.panels[('a', 'mean')], x, y), exp
            )
            std = abs(cx + 10 * cy)  # sqrt of the variance (cx + 10 cy)^2
            p = win.panels[('a', 'std')]
            exp = gui.colorize(np.array([std]), win.std_clim, p.lut)[0, :3]
            np.testing.assert_array_equal(rendered_rgb(p, x, y), exp)


def test_waiting_backend_shows_no_fabricated_maps(qapp, win):
    p = win.panels[('b', 'mean')]
    assert 'waiting' in p.title_label.text()
    assert 'waiting for fitted model' in win.status_label.text()
    np.testing.assert_array_equal(
        rendered_rgb(p, -1.0, 3.0), gui.INVALID_RGBA[:3]
    )
    click_world(qapp, win.panels['truth'], -1.0, 3.0)
    click_world(qapp, win.panels['truth'], -0.5, 3.0)
    settle(qapp, win)
    assert 'waiting' not in p.title_label.text()
    assert not np.array_equal(rendered_rgb(p, -2.0, 2.2), gui.INVALID_RGBA[:3])


class NanBackend(FakeBackend):
    """Not representable for x > 0 (like GMRF cells hidden by a wall)."""

    def query(self, xy):
        mean, var = super().query(xy)
        hidden = np.asarray(xy)[:, 0] > 0.0
        mean[hidden] = np.nan
        var[hidden] = np.nan
        return mean, var


def test_unrepresentable_cells_are_not_shown_as_zero(
    qapp, offset_scenario_path, tmp_path
):
    w = make_window(
        qapp, offset_scenario_path, {'n': NanBackend('n')}, tmp_path
    )
    try:
        for kind in ('mean', 'std'):
            p = w.panels[('n', kind)]
            np.testing.assert_array_equal(
                rendered_rgb(p, 0.6, 3.0), gui.INVALID_RGBA[:3]
            )
            low = gui.colorize(np.array([0.0]), (0.0, 1.0), p.lut)[0, :3]
            assert not np.array_equal(rendered_rgb(p, 0.6, 3.0), low)
            assert not np.array_equal(
                rendered_rgb(p, -2.0, 3.0), gui.INVALID_RGBA[:3]
            )
    finally:
        w.close()


def test_shared_and_fixed_levels(qapp, win, offset_scenario_path, tmp_path):
    click_world(qapp, win.panels['truth'], -1.0, 3.0)
    click_world(qapp, win.panels['truth'], -0.5, 3.0)
    settle(qapp, win)
    lo, hi = win.mean_clim
    for arr in [win.truth] + [
        win.session.poll()[1][n].mean for n in ('a', 'b')
    ]:
        assert lo <= np.nanmin(arr) and np.nanmax(arr) <= hi
    assert win.std_clim[0] == 0.0
    assert win.std_clim[1] == pytest.approx(
        max(np.nanmax(s) for s in win._std.values())
    )
    w = make_window(
        qapp,
        offset_scenario_path,
        {'a': FakeBackend('a')},
        tmp_path,
        mean_range=(0.0, 2.0),
        std_max=0.5,
    )
    try:
        assert w.mean_clim == (0.0, 2.0) and w.std_clim == (0.0, 0.5)
        x, y = -2.875, 2.125  # truth cell centre
        truth = w.scenario.truth(np.array([[x, y]]))[0]
        exp = gui.colorize(
            np.array([truth]), (0.0, 2.0), w.panels['truth'].lut
        )[0, :3]
        np.testing.assert_array_equal(
            rendered_rgb(w.panels['truth'], x, y), exp
        )
    finally:
        w.close()


def test_variance_to_std_display():
    std, bad = gui.variance_to_std(np.array([4.0, 0.0, -1e-12, np.nan]))
    np.testing.assert_allclose(std[:3], [2.0, 0.0, 0.0])
    assert np.isnan(std[3]) and bad == 0
    _, bad = gui.variance_to_std(np.array([4.0, -0.1]))
    assert bad == 1


# ------------------------------------------------------------ overlays
class GridBackend(FakeBackend):
    def __init__(self, name):
        super().__init__(name)
        self.calls_grid = 0

    def model_grid(self):
        self.calls_grid += 1
        seg = np.array([[[-3.0, 2.0], [-2.0, 2.0]], [[-2.0, 2.0], [-2.0, 3.0]]])
        return ModelGridOverlay(
            segments=seg, nodes=seg.reshape(-1, 2), nx=2, ny=2, dx=1.0, dy=1.0
        )


def test_model_grid_overlay_is_an_optional_capability(
    qapp, offset_scenario_path, tmp_path
):
    backends = {'g': GridBackend('g'), 'a': FakeBackend('a')}
    w = make_window(qapp, offset_scenario_path, backends, tmp_path)
    try:
        item = w.panels[('g', 'mean')].model_grid_item
        assert item is not None
        xs, ys = item.getData()
        np.testing.assert_allclose(xs, [-3, -2, -2, -2])
        np.testing.assert_allclose(ys, [2, 2, 2, 3])
        assert backends['g'].calls_grid == 1  # built once, not per refresh
        for key, p in w.panels.items():
            if key != ('g', 'mean'):
                assert p.model_grid_item is None
        # Walls above the grid, grid above the image.
        walls = [
            i
            for i in w.panels[('g', 'mean')].vb.addedItems
            if isinstance(i, pg.PlotCurveItem) and i is not item
        ]
        assert walls and min(i.zValue() for i in walls) > item.zValue()
        assert item.zValue() > w.panels[('g', 'mean')].image.zValue()
        w.refresh(force=True)
        assert backends['g'].calls_grid == 1
    finally:
        w.close()


def test_walls_on_every_panel(win):
    ((a, b),) = win.domain.walls
    for p in win.panels.values():
        curves = [i for i in p.vb.addedItems if isinstance(i, pg.PlotCurveItem)]
        assert len(curves) == 2  # outline + line
        for c in curves:
            xs, ys = c.getData()
            np.testing.assert_allclose(xs, [a[0], b[0]])
            np.testing.assert_allclose(ys, [a[1], b[1]])


# ------------------------------------------------ measurements and actions
def test_hide_measurements_is_presentation_only(
    qapp, offset_scenario_path, tmp_path
):
    w = make_window(
        qapp,
        offset_scenario_path,
        {'a': FakeBackend('a')},
        tmp_path,
        show_measurements=False,
    )
    try:
        click_world(qapp, w.panels['truth'], -1.0, 3.0)
        click_world(qapp, w.panels[('a', 'mean')], -0.5, 2.5)
        settle(qapp, w)
        assert len(w.session.observations) == 2
        assert w.session.backends['a'].state == [
            (o.x, o.y, o.value) for o in w.session.observations
        ]
        for p in w.panels.values():
            assert not p.markers.isVisible()
            assert len(p._labels) == 2 and not any(
                t.isVisible() for t in p._labels
            )
        w.show_measurements = True
        for p in w.panels.values():
            assert p.markers.isVisible() and all(
                t.isVisible() for t in p._labels
            )
            assert len(p.markers.data) == 2
    finally:
        w.close()


def test_actions_undo_reset_save_load_export(qapp, win, tmp_path, monkeypatch):
    for x in (-2.0, -1.0, 0.5):
        click_world(qapp, win.panels['truth'], x, 3.0)
    win.actions_by_name['Undo'].trigger()
    settle(qapp, win)
    assert len(win.session.observations) == 2
    asked = []

    def ask(title, suggested, filters):
        asked.append((title, suggested))
        name = 's1' if title == 'Save session' else 'fig'
        return str(tmp_path / 'out' / name), filters.split(';;')[0]

    monkeypatch.setattr(win, '_ask_save_path', ask)
    win.actions_by_name['Save session'].trigger()
    csv_path = win.last_session_path
    assert csv_path == tmp_path / 'out' / 's1.csv' and csv_path.is_file()
    # --out-dir did not exist yet: dialog started in its existing parent.
    assert asked[0][1].parent == tmp_path and asked[0][1].name.startswith(
        'session_'
    )
    saved = win.session.observations
    win.actions_by_name['Reset'].trigger()
    settle(qapp, win)
    assert win.session.observations == ()
    assert all(len(p.markers.data) == 0 for p in win.panels.values())
    opened = []
    monkeypatch.setattr(
        win,
        '_ask_open_path',
        lambda title, start, filters: opened.append(start) or str(csv_path),
    )
    win.actions_by_name['Load session'].trigger()
    assert opened == [tmp_path / 'out']  # last save directory
    settle(qapp, win)
    assert win.session.observations == saved
    for p in win.panels.values():
        assert [t.toPlainText() for t in p._labels] == ['1', '2']
    win.actions_by_name['Export image'].trigger()
    fig = tmp_path / 'out' / 'fig.png'  # .png appended
    assert fig.is_file() and fig.stat().st_size > 1000
    assert (
        asked[1][1].name.startswith('figure_') and asked[1][1].suffix == '.png'
    )


def test_marker_labels_follow_a_loaded_log(qapp, win, tmp_path):
    other = Session(
        win.scenario, {'a': FakeBackend('a'), 'b': FakeBackend('b')}, win.grid
    )
    pts5 = [(-2.8, 2.1), (-2.0, 3.3), (-1.0, 2.6), (0.2, 3.0), (0.9, 2.2)]
    for x, y in pts5:
        other.add_manual(x, y, 1.0)
    other.wait_idle(10)
    csv5, _ = other.save(tmp_path / 's5.csv')
    other.close()
    for x, y in [(-1.5, 2.5), (0.5, 3.2), (-0.3, 2.9)]:
        click_world(qapp, win.panels['truth'], x, y)
    assert win.load_session(csv5)
    settle(qapp, win)
    for p in win.panels.values():
        assert [t.toPlainText() for t in p._labels] == ['1', '2', '3', '4', '5']
        for t, (x, y) in zip(p._labels, pts5):
            assert (t.pos().x(), t.pos().y()) == pytest.approx((x, y))
        np.testing.assert_allclose(p.markers.data['x'], [q[0] for q in pts5])
    win.undo()
    click_world(qapp, win.panels['truth'], -0.5, 2.3)
    settle(qapp, win)
    for p in win.panels.values():
        assert len(p._labels) == 5 and p._labels[-1].toPlainText() == '5'
        tol = px_size(p)
        assert p._labels[-1].pos().x() == pytest.approx(-0.5, abs=tol)


def test_stopped_backend_is_flagged(qapp, offset_scenario_path, tmp_path):
    backends = {'a': FakeBackend('a'), 'b': FakeBackend('b', fail_value=7.0)}
    w = make_window(qapp, offset_scenario_path, backends, tmp_path)
    try:
        w.session.add_manual(-1.0, 3.0, 1.0)
        w.session.add_manual(-0.5, 3.0, 7.0)  # rejected by b
        w.session.add_manual(-0.2, 3.0, 2.0)
        settle(qapp, w)
        assert 'STOPPED n=1' in w.panels[('b', 'mean')].title_label.text()
        assert 'STOPPED n=1/3' in w.status_label.text()
        assert '#2 failed' in w.status_label.text()
        assert 'STOPPED' not in w.panels[('a', 'mean')].title_label.text()
        assert 'observation #2 failed' in w.status_label.toolTip()
    finally:
        w.close()


def test_reset_never_shows_stale_results(qapp, offset_scenario_path, tmp_path):
    slow = FakeBackend('a', delay=0.15)
    w = make_window(qapp, offset_scenario_path, {'a': slow}, tmp_path)
    try:
        for x in (-2.0, -1.0, 0.0):
            w.session.add_sample(x, 3.0)
        w.refresh(force=True)
        w.reset()
        settle(qapp, w)
        _, st, _, _ = w.session.poll()
        assert st['a'].result_count == 0 and slow.state == []
        g = w.grid
        np.testing.assert_allclose(
            w._mean_img['a'], g.to_image(g.xy[:, 0] + 10 * g.xy[:, 1])
        )
        assert all(len(p.markers.data) == 0 for p in w.panels.values())
    finally:
        w.close()


def test_saved_metadata_includes_display_state(qapp, win):
    click_world(qapp, win.panels['truth'], -1.0, 3.0)
    settle(qapp, win)
    path = win.save_session(win.out_dir / 'meta.csv')
    meta = json.loads(Path(path).with_suffix('.json').read_text())
    d = meta['display']
    assert d['backend_status_at_save']['a']['n_assimilated'] == 1
    assert d['backend_status_at_save']['b']['state'] == 'waiting'
    assert d['show_measurements'] is True and d['auto_color_limits'] is True
    assert d['mean_clim'] == list(win.mean_clim)


def test_close_stops_timer_and_worker(qapp, offset_scenario_path, tmp_path):
    w = make_window(
        qapp, offset_scenario_path, {'a': FakeBackend('a')}, tmp_path
    )
    w.close()
    process_events(qapp, 0.05)
    assert not w.timer.isActive()
    assert not w.session._thread.is_alive()
    w.refresh(force=True)  # no-op after close


def test_status_refreshes_when_only_the_job_count_changes(qapp, win):
    """The worker publishes its result before it decrements the job count;
    a change of only `pending` must still update the status text."""
    version, status, _, gen = win.session.poll()
    real_poll = win.session.poll
    win.session.poll = lambda: (version, status, 3, gen)
    try:
        win.refresh()
        assert 'worker: 3 job(s)' in win.status_label.text()
        win.session.poll = lambda: (version, status, 0, gen)
        win.refresh()
        assert 'worker:' not in win.status_label.text()
    finally:
        win.session.poll = real_poll


def test_narrow_title_keeps_the_state_suffix(qapp, win):
    from PySide6 import QtGui

    p = win.panels[('b', 'std')]
    p.resize(150, p.height())
    process_events(qapp, 0.05)
    plain = QtGui.QTextDocumentFragment.fromHtml(
        p.title_label.text()
    ).toPlainText()
    assert plain.endswith('· waiting')
    assert '…' in plain  # base title elided instead of the suffix
    fm = p.title_label.fontMetrics()
    assert fm.horizontalAdvance(plain) <= p.title_label.width()
    assert p.title_label.toolTip() == 'B — standard deviation'


# ------------------------------------------------ save/export dialogs
class DialogStub:
    """Replaces the window's dialog hooks; records what was asked."""

    def __init__(self, win, monkeypatch, answer, confirm=True, selected=None):
        self.answer, self.confirm, self.selected = answer, confirm, selected
        self.asked, self.confirmed = [], []
        monkeypatch.setattr(win, '_ask_save_path', self._ask)
        monkeypatch.setattr(win, '_confirm_overwrite', self._confirm)

    def _ask(self, title, suggested, filters):
        self.asked.append((title, suggested, filters))
        if self.answer is None:
            return None
        return str(self.answer), self.selected or filters.split(';;')[0]

    def _confirm(self, existing):
        self.confirmed.append(list(existing))
        return self.confirm


def _files(root):
    return sorted(p.relative_to(root) for p in Path(root).rglob('*'))


def test_save_dialog_cancel_writes_nothing(qapp, win, tmp_path, monkeypatch):
    before = _files(tmp_path)
    stub = DialogStub(win, monkeypatch, answer=None)
    assert win.save_session() is None
    assert win.export_image() is None
    assert _files(tmp_path) == before and not win.out_dir.exists()
    assert len(stub.asked) == 2 and stub.confirmed == []
    assert 'export cancelled' in win.message
    assert (
        stub.asked[0][1].parent == tmp_path
    )  # nearest existing parent of out_dir


def test_save_dialog_destination_and_suffix(qapp, win, tmp_path, monkeypatch):
    click_world(qapp, win.panels['truth'], -1.0, 3.0)
    settle(qapp, win)
    other = tmp_path / 'elsewhere'
    other.mkdir()
    DialogStub(win, monkeypatch, answer=other / 'my_run')
    path = win.save_session()
    assert path == other / 'my_run.csv'
    meta = json.loads((other / 'my_run.json').read_text())
    assert meta['csv_file'] == 'my_run.csv' and meta['num_observations'] == 1
    assert sorted(p.name for p in other.iterdir()) == [
        'my_run.csv',
        'my_run.json',
    ]
    assert not win.out_dir.exists()


@pytest.mark.parametrize('existing', ['csv', 'json', 'both'])
def test_save_overwrite_needs_confirmation(
    qapp, win, tmp_path, monkeypatch, existing
):
    target = tmp_path / 'dup.csv'
    old = {}
    if existing in ('csv', 'both'):
        target.write_text('old csv\n')
        old[target] = 'old csv\n'
    if existing in ('json', 'both'):
        target.with_suffix('.json').write_text('{"old": true}')
        old[target.with_suffix('.json')] = '{"old": true}'
    stub = DialogStub(win, monkeypatch, answer=target, confirm=False)
    assert win.save_session() is None
    assert stub.confirmed == [list(old)]  # names exactly the existing file(s)
    for p, text in old.items():
        assert p.read_text() == text
    for p in (target, target.with_suffix('.json')):
        if p not in old:
            assert not p.exists()  # declined: nothing new written
    stub.confirm = True
    assert win.save_session() == target
    assert (
        target.read_text().startswith('index,')
        and json.loads(target.with_suffix('.json').read_text())['csv_file']
        == 'dup.csv'
    )


def test_export_dialog_destination_suffix_and_overwrite(
    qapp, win, tmp_path, monkeypatch
):
    stub = DialogStub(win, monkeypatch, answer=tmp_path / 'shot')
    assert win.export_image() == tmp_path / 'shot.png'
    assert (tmp_path / 'shot.png').stat().st_size > 1000
    assert stub.asked[0][1].suffix == '.png' and 'JPEG' in stub.asked[0][2]
    stub.answer = tmp_path / 'shot.jpg'
    assert win.export_image() == tmp_path / 'shot.jpg'
    # No extension typed: the selected file type decides the suffix.
    stub.answer, stub.selected = tmp_path / 'typed', 'JPEG image (*.jpg *.jpeg)'
    assert win.export_image() == tmp_path / 'typed.jpg'
    assert (tmp_path / 'typed.jpg').read_bytes()[
        :3
    ] == b'\xff\xd8\xff'  # JPEG data
    stub.selected = None
    (tmp_path / 'keep.png').write_bytes(b'old')
    stub.answer, stub.confirm = tmp_path / 'keep.png', False
    assert win.export_image() is None
    assert (tmp_path / 'keep.png').read_bytes() == b'old'
    assert stub.confirmed == [[tmp_path / 'keep.png']]


# ------------------------------------------------------------- timings
def test_timing_labels():
    from belief_playground.session import BackendStatus

    st = BackendStatus(
        name='x',
        update_ms=12.4,
        update_count=1,
        update_replay=False,
        query_ms=153.2,
        query_cells=3200,
    )
    assert gui.timing_labels(st) == [
        'assimilate 12 ms',
        'grid query 153 ms (3200 cells)',
    ]
    st = BackendStatus(
        name='x', update_ms=40.0, update_count=5, update_replay=True
    )
    assert gui.timing_labels(st) == ['reset+replay 5 obs 40 ms']
    st = BackendStatus(
        name='x', update_ms=3.0, update_count=0, update_replay=True
    )
    assert gui.timing_labels(st) == ['reset 3 ms']
    assert gui.timing_labels(BackendStatus(name='x')) == []


def test_status_shows_what_each_timing_covers(qapp, win):
    n_cells = len(win.grid.xy)
    click_world(qapp, win.panels['truth'], -1.0, 3.0)
    settle(qapp, win)
    _, st, _, _ = win.session.poll()
    assert st['a'].update_count == 1 and st['a'].update_replay is False
    assert st['a'].query_cells == n_cells
    text = win.status_label.text()
    assert (
        'assimilate ' in text
        and 'grid query ' in text
        and f'({n_cells} cells)' in text
    )
    click_world(qapp, win.panels['truth'], -0.5, 3.0)
    win.undo()
    settle(qapp, win)
    _, st, _, _ = win.session.poll()
    assert st['a'].update_count == 1 and st['a'].update_replay is True
    assert 'reset+replay 1 obs' in win.status_label.text()
    tip = win.status_label.toolTip()
    assert 'wall-clock' in tip and 'refit_policy' in tip and 'display' in tip


# --------------------------------------------- show measurement positions
class CountingBackend(FakeBackend):
    def __init__(self, name):
        super().__init__(name)
        self.n_queries = 0

    def query(self, xy):
        self.n_queries += 1
        return super().query(xy)


def _session_work(w):
    """Everything a display toggle must leave untouched."""
    version, status, pending, gen = w.session.poll()
    return (
        version,
        gen,
        pending,
        {n: s.result_id for n, s in status.items()},
        w.session.observations,
        w.session._noise_draws,
        {n: (len(b.calls), b.n_queries) for n, b in w.session.backends.items()},
    )


def _all_markers(w, visible):
    for p in w.panels.values():
        assert p.markers.isVisible() is visible, p.key
        assert all(t.isVisible() is visible for t in p._labels), p.key


def _markers_match_log(w):
    obs = w.session.observations
    for p in w.panels.values():
        assert [t.toPlainText() for t in p._labels] == [
            str(o.index) for o in obs
        ]
        for t, o in zip(p._labels, obs):
            assert (t.pos().x(), t.pos().y()) == pytest.approx((o.x, o.y))
        np.testing.assert_allclose(p.markers.data['x'], [o.x for o in obs])
        np.testing.assert_allclose(p.markers.data['y'], [o.y for o in obs])


def test_measurement_checkbox_is_display_only(
    qapp, offset_scenario_path, tmp_path
):
    w = make_window(
        qapp, offset_scenario_path, {'c': CountingBackend('c')}, tmp_path
    )
    try:
        box = w.measurements_checkbox
        assert box.text() == 'Show measurement positions' and box.isChecked()
        for x in (-2.0, -1.0):
            click_world(qapp, w.panels['truth'], x, 3.0)
        settle(qapp, w)
        before = _session_work(w)
        box.click()  # real widget interaction
        process_events(qapp, 0.3)  # let the poll timer run
        assert w.show_measurements is False and not box.isChecked()
        _all_markers(w, False)
        assert _session_work(w) == before
        box.setChecked(True)
        process_events(qapp, 0.3)
        assert w.show_measurements is True
        _all_markers(w, True)
        assert _session_work(w) == before
    finally:
        w.close()


def test_hidden_measurements_stay_hidden_through_log_changes(
    qapp, win, tmp_path
):
    other = Session(
        win.scenario, {'a': FakeBackend('a'), 'b': FakeBackend('b')}, win.grid
    )
    for x, y in [(-2.8, 2.1), (-2.0, 3.3), (0.9, 2.2)]:
        other.add_manual(x, y, 1.0)
    other.wait_idle(10)
    csv3, _ = other.save(tmp_path / 's3.csv')
    other.close()

    win.measurements_checkbox.click()
    assert not win.show_measurements
    for x in (-1.5, 0.5):
        click_world(qapp, win.panels['truth'], x, 3.0)
    settle(qapp, win)
    _all_markers(win, False)
    _markers_match_log(win)
    win.undo()
    settle(qapp, win)
    _all_markers(win, False)
    _markers_match_log(win)
    click_world(qapp, win.panels['truth'], -0.5, 2.5)  # new label after undo
    settle(qapp, win)
    _all_markers(win, False)
    win.reset()
    settle(qapp, win)
    assert all(len(p._labels) == 0 for p in win.panels.values())
    assert win.load_session(csv3)
    settle(qapp, win)
    assert all(len(p._labels) == 3 for p in win.panels.values())
    _all_markers(win, False)
    win.measurements_checkbox.click()
    _all_markers(win, True)
    _markers_match_log(win)


def test_hide_flag_starts_unchecked_and_property_drives_checkbox(
    qapp, offset_scenario_path, tmp_path
):
    w = make_window(
        qapp,
        offset_scenario_path,
        {'a': FakeBackend('a')},
        tmp_path,
        show_measurements=False,
    )
    try:
        box = w.measurements_checkbox
        assert not box.isChecked()
        emitted = []
        box.toggled.connect(emitted.append)
        w.show_measurements = (
            True  # programmatic: checkbox follows, no signal loop
        )
        assert box.isChecked() and emitted == []
        _all_markers(w, True)
        w.show_measurements = False
        assert not box.isChecked() and emitted == []
    finally:
        w.close()


def test_saved_metadata_follows_the_checkbox(qapp, win):
    win.measurements_checkbox.click()
    p1 = win.save_session(win.out_dir / 'hidden.csv')
    win.measurements_checkbox.click()
    p2 = win.save_session(win.out_dir / 'shown.csv')
    show = [
        json.loads(p.with_suffix('.json').read_text())['display'][
            'show_measurements'
        ]
        for p in (p1, p2)
    ]
    assert show == [False, True]


def test_checkbox_is_right_aligned_in_the_toolbar(qapp, win):
    tb = win.toolbar
    box = win.measurements_checkbox
    assert box.parent() is tb and box.isVisible()
    rights = [
        tb.widgetForAction(a).geometry().right()
        for n, a in win.actions_by_name.items()
        if n != 'Quit'
    ]
    assert box.geometry().left() > max(rights) + 50  # spacer in between
    assert box.geometry().right() >= tb.width() - 30  # at the right edge
