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

"""Independent orientation check of the Qt frontend.

Puts a known, strongly asymmetric truth on a non-square domain whose origin is
negative in both x and y, renders the real window (Xvfb) and reads the
*rendered pixels* (QWidget.grab) at world positions. The colour under a world
point must be the colour of the value at that same physical point, for the
truth panel and a backend mean panel; a real mouse click at that point must
create an observation at that point whose latent equals the truth there.
"""

from __future__ import annotations

import numpy as np
import pytest
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import Session
from conftest import FakeBackend, process_events

gui = pytest.importorskip('belief_playground.gui')
from PySide6 import QtCore  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

# x in [-5, -1] (4 m), y in [-2, -0.5] (1.5 m): non-square, negative origin.
# One narrow source near the upper-left corner, no background, no walls,
# no clipping -> the field is far from symmetric under x or y flips.
NEG_ORIGIN_SCENARIO = """
[general]
name = "neg_origin_asym"

[units]
field_value = "mV"

[bounds]
x_min = -5.0
x_max = -1.0
y_min = -2.0
y_max = -0.5

[objects.sources]
source_positions = [[-4.4, -0.8]]

[simulation.source_defaults]
amplitude = 5.0
sigma_x = 1.2
sigma_y = 0.5

[simulation]
background_level = 0.0
"""

STEP = 0.25  # -> nx = 16, ny = 6
COLOR_TOL = 1  # 8-bit rendering of an RGBA image: exact up to rounding
MIN_COLOR_SEP = 20  # flipped-location colour must differ at least this much


class OrientBackend(FakeBackend):
    """Fake mean (x + 5) + 3 (y + 2): asymmetric, independent of the data,
    and in the same range as the truth so the shared colour scale stays
    informative and does not drift with clicks."""

    def query(self, xy):
        xy = np.asarray(xy, dtype=float)
        f = (xy[:, 0] + 5.0) + 3.0 * (xy[:, 1] + 2.0)
        return f, np.ones(len(xy))


def _fake_mean(x, y):
    return (x + 5.0) + 3.0 * (y + 2.0)


@pytest.fixture
def neg_win(qapp, tmp_path):
    path = tmp_path / 'neg_origin.toml'
    path.write_text(NEG_ORIGIN_SCENARIO)
    sc = Scenario(path)
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, STEP)
    session = Session(sc, {'a': OrientBackend('a')}, grid)
    w = gui.PlaygroundWindow(
        session, out_dir=tmp_path / 'out', show_measurements=False
    )
    w.resize(1200, 700)
    w.show()
    process_events(qapp, 0.3)
    assert session.wait_idle(20)
    w.refresh(force=True)
    process_events(qapp, 0.1)
    yield w
    w.close()


def _pixel(panel, x, y):
    img = panel.view.grab().toImage()
    p = panel.map_to_view(x, y)
    c = img.pixelColor(p.x(), p.y())
    return np.array([c.red(), c.green(), c.blue()], dtype=float)


def _color(panel, value, clim):
    return gui.colorize(np.array([value]), clim, panel.lut)[0, :3].astype(float)


def _cells(grid):
    nx, ny = len(grid.xs), len(grid.ys)
    # Corners and interior cells, all away from the symmetry axes.
    return [
        (1, ny - 2),
        (nx - 2, 1),
        (2, 0),
        (nx - 3, ny - 1),
        (5, ny - 1),
        (0, 1),
    ]


def test_domain_and_cells_are_as_intended(neg_win):
    w = neg_win
    assert w.grid.shape == (6, 16)
    p = w.panels['truth']
    a = p.map_to_view(-5.0, -2.0)
    b = p.map_to_view(-5.0 + STEP, -2.0 + STEP)
    cw, ch = abs(b.x() - a.x()), abs(b.y() - a.y())
    assert cw >= 4 and ch >= 4, (cw, ch)  # several pixels per cell
    assert cw == pytest.approx(ch, abs=1)  # equal aspect
    assert b.y() < a.y()  # +y is up on screen


def test_rendered_truth_pixel_matches_value_at_that_location(neg_win):
    w = neg_win
    p = w.panels['truth']
    nx, ny = len(w.grid.xs), len(w.grid.ys)
    sep = [0.0, 0.0]
    for i, j in _cells(w.grid):
        x, y = w.grid.xs[i], w.grid.ys[j]
        expected = _color(
            p, w.scenario.truth(np.array([[x, y]]))[0], w.mean_clim
        )
        np.testing.assert_allclose(
            _pixel(p, x, y),
            expected,
            atol=COLOR_TOL,
            err_msg=f'cell ({i},{j}) at ({x},{y})',
        )
        # Sensitivity: colour difference to the x- and y-mirrored cells.
        for k, (fi, fj) in enumerate(((nx - 1 - i, j), (i, ny - 1 - j))):
            fx, fy = w.grid.xs[fi], w.grid.ys[fj]
            flipped = _color(
                p, w.scenario.truth(np.array([[fx, fy]]))[0], w.mean_clim
            )
            sep[k] = max(sep[k], float(np.max(np.abs(flipped - expected))))
    # Each kind of flip would change at least one checked pixel clearly, so
    # this test cannot pass with a flipped/transposed image.
    assert min(sep) > MIN_COLOR_SEP, sep


def test_rendered_backend_mean_pixel_matches_query_at_that_location(neg_win):
    w = neg_win
    p = w.panels[('a', 'mean')]
    for i, j in _cells(w.grid):
        x, y = w.grid.xs[i], w.grid.ys[j]
        np.testing.assert_allclose(
            _pixel(p, x, y),
            _color(p, _fake_mean(x, y), w.mean_clim),
            atol=COLOR_TOL,
            err_msg=f'cell ({i},{j})',
        )


@pytest.mark.parametrize('panel', ['truth', 'mean'])
def test_click_location_equals_displayed_location(qapp, neg_win, panel):
    w = neg_win
    p = w.panels['truth'] if panel == 'truth' else w.panels[('a', 'mean')]
    truth_panel = w.panels['truth']
    for k, (i, j) in enumerate(_cells(w.grid)[:4]):
        x, y = w.grid.xs[i], w.grid.ys[j]
        seen = _pixel(truth_panel, x, y)
        seen_expected = _color(
            truth_panel, w.scenario.truth(np.array([[x, y]]))[0], w.mean_clim
        )
        QTest.mouseClick(
            p.view.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            p.map_to_view(x, y),
        )
        process_events(qapp, 0.05)
        obs = w.session.observations
        assert len(obs) == k + 1
        o = obs[-1]
        tol = (
            w.domain.x_range[1] - w.domain.x_range[0]
        ) / p.view.width()  # 1 px
        assert o.x == pytest.approx(x, abs=tol) and o.y == pytest.approx(
            y, abs=tol
        )
        assert o.latent == pytest.approx(
            w.scenario.truth(np.array([[o.x, o.y]]))[0]
        )
        np.testing.assert_allclose(seen, seen_expected, atol=COLOR_TOL)
