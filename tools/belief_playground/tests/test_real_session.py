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

"""Small real GP + GMRF comparison session (installed backends, no mocks)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from belief_playground.backends import make_backends
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import Session, rmse_on_common_grid

pytest.importorskip('scalar_field_belief_gmrf_py')


def test_real_gp_gmrf_session(single_wall_path, tmp_path):
    sc = Scenario(single_wall_path)
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, 0.1)
    backends = make_backends(['gp', 'gmrf'], sc.domain)
    assert not any(hasattr(b, 'unavailable_reason') for b in backends.values())
    s = Session(sc, backends, grid, noise_std=5e-4, seed=11)
    try:
        assert s.wait_idle(60)
        _, st, _, _ = s.poll()
        assert (
            st['gp'].note == 'waiting for fitted model'
            and st['gp'].mean is None
        )
        assert st['gmrf'].state == 'ready' and st['gmrf'].mean is not None

        points = [(0.5, 1.0), (0.7, 2.0), (1.5, 3.0), (0.4, 3.3), (1.6, 0.8)]
        obs = [s.add_sample(x, y) for x, y in points]
        assert len({o.noise for o in obs}) == len(obs)  # noise actually drawn
        assert s.wait_idle(120)

        values = np.array([o.value for o in obs])
        xy = np.array([[o.x, o.y] for o in obs])
        gp = backends['gp'].belief
        np.testing.assert_array_equal(gp.y_train, values)  # exact same values
        np.testing.assert_array_equal(gp.xy_train_phys, xy)
        assert backends['gmrf'].num_assimilated() == len(obs)

        _, st, _, _ = s.poll()
        n = len(grid.xy)
        for name in ('gp', 'gmrf'):
            r = st[name]
            assert r.error is None and r.query_error is None, (name, r)
            assert r.state == 'ready' and r.result_count == len(obs)
            assert r.mean.shape == (n,) and r.var.shape == (n,)
            assert np.all(np.isfinite(r.mean)) and np.all(np.isfinite(r.var))
            assert np.all(r.var > -1e-12)
            assert r.update_ms is not None and r.query_ms is not None

        truth = sc.truth(grid.xy)
        rmse, n_valid = rmse_on_common_grid(
            truth, {k: st[k].mean for k in ('gp', 'gmrf')}
        )
        assert n_valid == n
        assert all(np.isfinite(v) for v in rmse.values())

        csv_path, json_path = s.save(tmp_path / 'real.csv')
        meta = json.loads(json_path.read_text())
        assert meta['backends']['gp']['effective']['refit_policy']
        assert meta['backends']['gmrf']['effective']['nx'] == 41
        assert meta['scenario']['field_file']['text']
        s.undo()
        assert s.wait_idle(120)
        assert len(gp.y_train) == len(obs) - 1
        assert backends['gmrf'].num_assimilated() == len(obs) - 1
    finally:
        s.close()
