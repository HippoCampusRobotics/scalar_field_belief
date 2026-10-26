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

"""Real adapters (installed GP and GMRF bindings), no mocks."""

from __future__ import annotations

import numpy as np
import pytest
from belief_playground.backends import (
    DEFAULT_GP_CONFIG,
    REPO_ROOT,
    GmrfAdapter,
    GPAdapter,
    NotReadyError,
    make_backends,
)
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import Session

pytest.importorskip('scalar_field_belief_gmrf_py')

WALL_AWARE_GP = REPO_ROOT / 'config' / 'belief_params_wall_aware.yaml'


def test_gp_adapter_readiness_shapes_and_bounds_override(offset_scenario_path):
    sc = Scenario(offset_scenario_path)
    gp = GPAdapter(sc.domain, DEFAULT_GP_CONFIG)
    cfg = gp.belief.config
    assert (cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max) == (-3.0, 1.0, 2.0, 3.5)
    snap = gp.effective_config()
    assert snap['bounds_override']['changed'] is True
    assert snap['bounds_override']['from_file']['x_max'] == 2.0
    assert snap['effective']['x_min'] == -3.0
    assert snap['value_unit'] == 'mV'
    assert snap['config_file']['sha256']

    assert not gp.is_ready()
    with pytest.raises(NotReadyError):
        gp.query(np.zeros((1, 2)))
    gp.add_measurement(-1.0, 3.0, 2.0)
    gp.add_measurement(0.5, 2.5, 1.0)
    assert gp.is_ready() and gp.num_assimilated() == 2
    xy = np.array([[-1.0, 3.0], [0.5, 2.5], [-2.9, 2.1]])
    mean, var = gp.query(xy)
    assert mean.shape == (3,) and var.shape == (3,)
    assert np.all(np.isfinite(mean)) and np.all(var >= 0)
    # Latent mean in measurement units: near the data, not standardized.
    assert mean[0] == pytest.approx(2.0, abs=0.2)
    assert mean[1] == pytest.approx(1.0, abs=0.2)
    gp.reset()
    assert not gp.is_ready() and gp.num_assimilated() == 0


def test_wall_aware_gp_uses_scenario_walls(single_wall_path):
    sc = Scenario(single_wall_path)
    gp = GPAdapter(sc.domain, WALL_AWARE_GP)
    assert gp.belief.config.model_variant == 'wall_aware'
    walls = gp.belief.wall_geometry.walls
    assert [(w.start, w.end) for w in walls] == [((1.0, 0.4), (1.0, 3.6))]
    assert 'belief_geometry' in gp.effective_config()['unused_parameters']


def test_gp_wall_aware_backend_uses_scenario_walls(single_wall_path):
    sc = Scenario(single_wall_path)
    backends = make_backends(['gp', 'gp_wall_aware'], sc.domain)
    gp, wa = backends['gp'], backends['gp_wall_aware']
    assert gp.name == 'gp' and gp.effective_config()['backend'] == 'gp'
    assert gp.belief.config.model_variant == 'baseline'
    assert wa.name == 'gp_wall_aware' and wa.display_name == 'GP wall-aware'
    assert wa.belief.config.model_variant == 'wall_aware'
    walls = wa.belief.wall_geometry.walls
    assert [(w.start, w.end) for w in walls] == [((1.0, 0.4), (1.0, 3.6))]
    snap = wa.effective_config()
    assert snap['backend'] == 'gp_wall_aware'
    assert snap['walls_used'] == [[1.0, 0.4, 1.0, 3.6]]


def test_gp_wall_aware_backend_rejects_a_baseline_config(single_wall_path):
    sc = Scenario(single_wall_path)
    backends = make_backends(
        ['gp_wall_aware'], sc.domain, gp_wall_aware_config=DEFAULT_GP_CONFIG
    )
    reason = backends['gp_wall_aware'].unavailable_reason
    assert reason.startswith('ValueError') and 'wall_aware' in reason


def test_gmrf_adapter_prior_shapes_and_units(offset_scenario_path):
    sc = Scenario(offset_scenario_path)
    gm = GmrfAdapter(sc.domain)
    cfg = gm.belief.config
    assert (cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max) == (-3.0, 1.0, 2.0, 3.5)
    assert gm.effective_config()['walls_used'] == [[0.0, 2.2, 0.0, 3.3]]
    assert gm.is_ready()  # prior available immediately
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, 0.25)
    mean, var = gm.query(grid.xy)
    assert mean.shape == (len(grid.xy),) and var.shape == mean.shape
    assert np.all(np.isfinite(mean)) and np.all(var > 0)
    gm.add_measurement(-1.0, 3.0, 2.0)  # update() happens inside
    assert gm.num_assimilated() == 1
    mean2, var2 = gm.query(np.array([[-1.0, 3.0]]))
    assert var2[0] < np.max(var)
    # Variance stays a variance; the noise variance is sigma_s^2 [units^2].
    noise_var = gm.belief.measurement_noise_variance()
    sigma_s = gm.effective_config()['effective']['sigma_s']
    assert noise_var == pytest.approx(sigma_s**2)


def test_gmrf_hidden_cells_are_nan_not_zero(tmp_path):
    # A wall closing a corner pocket leaves no visible corner for points in
    # it if the free-space seed excludes the pocket.
    path = tmp_path / 'pocket.toml'
    path.write_text("""
[general]
name = "pocket"
[bounds]
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 4.0
[objects.sources]
source_positions = [[1.0, 2.0]]
[simulation.source_defaults]
amplitude = 1.0
sigma_x = 0.3
sigma_y = 0.3
[objects.walls]
wall_vertices = [[0.0, 1.0, 2.0, 1.0]]
""")
    cfg = tmp_path / 'gmrf.yaml'
    cfg.write_text(
        (REPO_ROOT / 'gmrf' / 'ros' / 'belief_params_gmrf.yaml').read_text()
        + '    free_space_seeds: [1.0, 3.0]\n'
    )
    sc = Scenario(path)
    gm = GmrfAdapter(sc.domain, cfg)
    xy = np.array([[1.0, 0.5], [1.0, 3.0]])
    mean, var = gm.query(xy)
    assert np.isnan(mean[0]) and np.isnan(var[0])
    assert np.isfinite(mean[1]) and np.isfinite(var[1])
    assert gm.hidden_cells == 1


def test_unavailable_backend_keeps_the_others_usable(
    offset_scenario_path, tmp_path
):
    sc = Scenario(offset_scenario_path)
    bad = tmp_path / 'bad.yaml'
    bad.write_text('/**:\n  ros__parameters:\n    nx: 41\n')
    backends = make_backends(['gp', 'gmrf'], sc.domain, gmrf_config=bad)
    assert 'missing required parameters' in backends['gmrf'].unavailable_reason
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, 0.5)
    s = Session(sc, backends, grid)
    try:
        s.add_manual(0.0, 3.0, 1.0)
        assert s.wait_idle(60)
        _, st, _, _ = s.poll()
        assert st['gmrf'].state == 'error' and st['gmrf'].mean is None
        assert st['gp'].state == 'ready' and st['gp'].n_assimilated == 1
    finally:
        s.close()


def _crosses(seg, wall):
    """Proper intersection of segment seg ((x0, y0), (x1, y1)) with wall."""
    (ax, ay), (bx, by) = seg
    (cx, cy), (dx, dy) = wall

    def orient(px, py, qx, qy, rx, ry):
        return np.sign((qx - px) * (ry - py) - (qy - py) * (rx - px))

    return (
        orient(ax, ay, bx, by, cx, cy) * orient(ax, ay, bx, by, dx, dy) < 0
        and orient(cx, cy, dx, dy, ax, ay) * orient(cx, cy, dx, dy, bx, by) < 0
    )


def _wall_scenario(tmp_path, walls):
    path = tmp_path / 'grid_wall.toml'
    path.write_text(f"""
[general]
name = "grid_wall"
[bounds]
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 4.0
[objects.sources]
source_positions = [[0.5, 2.0]]
[simulation.source_defaults]
amplitude = 1.0
sigma_x = 0.3
sigma_y = 0.3
[objects.walls]
wall_vertices = {walls}
""")
    return Scenario(path)


def test_gmrf_model_grid_uses_model_spacing_and_cut_edges(tmp_path):
    # Wall between two node columns (x = 1.0 and 1.05), away from the ends.
    wall = ((1.025, 0.8), (1.025, 3.2))
    sc = _wall_scenario(tmp_path, [[1.025, 0.8, 1.025, 3.2]])
    gm = GmrfAdapter(sc.domain)
    g = gm.model_grid()
    assert g is gm.model_grid()  # built once and cached
    assert (g.nx, g.ny) == (41, 81)  # from the GMRF YAML, not the display grid
    assert g.dx == pytest.approx(0.05) and g.dy == pytest.approx(0.05)
    seg = g.segments
    d = np.abs(seg[:, 1] - seg[:, 0])
    horiz = np.isclose(d[:, 0], g.dx) & np.isclose(d[:, 1], 0.0)
    vert = np.isclose(d[:, 0], 0.0) & np.isclose(d[:, 1], g.dy)
    assert np.all(horiz | vert)
    assert len(g.nodes) == 41 * 81  # no seeds: every node active
    full = 40 * 81 + 41 * 80
    assert not any(_crosses(s, wall) for s in seg)
    crossing = (np.minimum(seg[:, 0, 0], seg[:, 1, 0]) < 1.025) & (
        np.maximum(seg[:, 0, 0], seg[:, 1, 0]) > 1.025
    )
    ys = seg[crossing, 0, 1]
    assert not np.any((ys >= 0.8) & (ys <= 3.2))
    assert np.any(ys < 0.7) and np.any(ys > 3.3)  # intact beyond the wall ends
    # Exactly the 49 edges at y = 0.8 .. 3.2 (0.05 m steps) are cut.
    assert len(seg) == full - 49
    # Nodes next to the wall keep their vertical edges on both sides.
    for x in (1.0, 1.05):
        assert np.any(
            vert
            & np.isclose(seg[:, 0, 0], x)
            & (seg[:, 0, 1] > 1.0)
            & (seg[:, 0, 1] < 3.0)
        )


def test_gmrf_model_grid_respects_the_active_mask(tmp_path):
    # Full-width wall at y = 1.02; the seed keeps only the upper part active.
    sc = _wall_scenario(tmp_path, [[0.0, 1.02, 2.0, 1.02]])
    cfg = tmp_path / 'gmrf.yaml'
    cfg.write_text(
        (REPO_ROOT / 'gmrf' / 'ros' / 'belief_params_gmrf.yaml').read_text()
        + '    free_space_seeds: [1.0, 3.0]\n'
    )
    g = GmrfAdapter(sc.domain, cfg).model_grid()
    assert np.all(g.nodes[:, 1] > 1.02)
    assert len(g.nodes) == 41 * (81 - 21)  # rows y = 0 .. 1.0 are inactive
    assert np.all(g.segments[:, :, 1] > 1.02)
    assert len(g.segments) == 40 * 60 + 41 * 59
