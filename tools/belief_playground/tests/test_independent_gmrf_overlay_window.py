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

"""Independent check: the real GMRF adapter's model grid as
drawn in the Qt window on a non-square, negative/nonzero-origin domain.

The display grid (0.25 m) and the GMRF model grid (nx=41, ny=81 from the
default GMRF YAML -> dx = 4/40 = 0.1 m, dy = 1.5/80 = 0.01875 m) differ on
purpose, so the overlay must follow the model, not the display grid.
Run on Xvfb like the other GUI tests (skipped otherwise).
"""

from __future__ import annotations

import numpy as np
import pytest
from belief_playground.backends import GmrfAdapter
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import Session
from conftest import process_events

pytest.importorskip('scalar_field_belief_gmrf_py')
gui = pytest.importorskip('belief_playground.gui')
pg = pytest.importorskip('pyqtgraph')


def _segments_cross(seg, wall):
    (ax, ay), (bx, by) = wall
    p, q = seg[:, 0], seg[:, 1]

    def orient(px, py, qx, qy, rx, ry):
        return np.sign((qx - px) * (ry - py) - (qy - py) * (rx - px))

    o1 = orient(ax, ay, bx, by, p[:, 0], p[:, 1])
    o2 = orient(ax, ay, bx, by, q[:, 0], q[:, 1])
    o3 = orient(p[:, 0], p[:, 1], q[:, 0], q[:, 1], ax, ay)
    o4 = orient(p[:, 0], p[:, 1], q[:, 0], q[:, 1], bx, by)
    return (o1 * o2 < 0) & (o3 * o4 < 0)


def test_real_gmrf_overlay_in_window_follows_model_grid(
    qapp, offset_scenario_path, tmp_path
):
    sc = Scenario(offset_scenario_path)
    (x0, x1), (y0, y1) = sc.domain.x_range, sc.domain.y_range
    assert (x0, x1, y0, y1) == (-3.0, 1.0, 2.0, 3.5)
    gm = GmrfAdapter(sc.domain)
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, 0.25)
    session = Session(sc, {'gmrf': gm}, grid)
    w = gui.PlaygroundWindow(session, out_dir=tmp_path / 'out')
    try:
        w.resize(1400, 900)
        w.show()
        process_events(qapp, 0.3)
        assert session.wait_idle(30)
        w.refresh(force=True)

        overlay = gm.model_grid()
        assert (overlay.nx, overlay.ny) == (41, 81)
        assert overlay.dx == pytest.approx((x1 - x0) / 40)
        assert overlay.dy == pytest.approx((y1 - y0) / 80)
        assert overlay.dx != pytest.approx(
            0.25
        ) and overlay.dy != pytest.approx(0.25)

        # Node positions span exactly the physical domain (no origin shift).
        nodes = overlay.nodes
        assert nodes[:, 0].min() == pytest.approx(x0) and nodes[
            :, 0
        ].max() == pytest.approx(x1)
        assert nodes[:, 1].min() == pytest.approx(y0) and nodes[
            :, 1
        ].max() == pytest.approx(y1)
        ix = (nodes[:, 0] - x0) / overlay.dx
        iy = (nodes[:, 1] - y0) / overlay.dy
        np.testing.assert_allclose(ix, np.round(ix), atol=1e-9)
        np.testing.assert_allclose(iy, np.round(iy), atol=1e-9)

        # What the panel draws is exactly the adapter's segment list.
        panel = w.panels[('gmrf', 'mean')]
        item = panel.model_grid_item
        assert item is not None
        xs, ys = item.getData()
        np.testing.assert_allclose(xs, overlay.segments[:, :, 0].ravel())
        np.testing.assert_allclose(ys, overlay.segments[:, :, 1].ravel())
        assert w.panels[('gmrf', 'std')].model_grid_item is None
        assert w.panels['truth'].model_grid_item is None

        # Edges are single model steps, and none is drawn through the wall
        # x = 0, y in [2.2, 3.3] (cut edges skipped); edges beyond the wall
        # ends across x = 0 are still drawn.
        seg = overlay.segments
        d = np.abs(seg[:, 1] - seg[:, 0])
        horiz = np.isclose(d[:, 0], overlay.dx) & np.isclose(d[:, 1], 0.0)
        vert = np.isclose(d[:, 0], 0.0) & np.isclose(d[:, 1], overlay.dy)
        assert np.all(horiz | vert)
        (wall,) = sc.domain.walls
        assert not np.any(_segments_cross(seg, wall))
        # The wall lies on the node column x = 0 here, so the model cuts
        # the horizontal edges touching it. Check that some horizontal edges
        # ending at x=0 inside the wall span are absent, while those well
        # outside the wall span are present.
        at_wall = horiz & (
            np.isclose(seg[:, 0, 0], 0.0) | np.isclose(seg[:, 1, 0], 0.0)
        )
        ys_at_wall = seg[at_wall, 0, 1]
        assert np.any(ys_at_wall < 2.15) and np.any(ys_at_wall > 3.35)
        full_h = 40 * 81
        assert horiz.sum() < full_h  # some horizontal edges were cut

        # Walls above the grid, grid above the image.
        walls = [
            i
            for i in panel.vb.addedItems
            if isinstance(i, pg.PlotCurveItem) and i is not item
        ]
        assert (
            walls
            and min(i.zValue() for i in walls)
            > item.zValue()
            > panel.image.zValue()
        )
    finally:
        w.close()
        process_events(qapp, 0.05)
