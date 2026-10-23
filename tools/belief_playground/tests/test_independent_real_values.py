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

"""Independent real-backend check: the GMRF posterior shown by
a real session equals a separately constructed GMRF fed the *stored noisy*
values of the log, and differs from one fed the noiseless latents. (The GP
side is covered by test_real_session: y_train == stored values.)"""

from __future__ import annotations

import numpy as np
import pytest
from belief_playground.backends import GmrfAdapter, make_backends
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import Session

pytest.importorskip('scalar_field_belief_gmrf_py')


def _direct(sc, xy_obs, values, grid_xy):
    ref = GmrfAdapter(sc.domain)
    for (x, y), v in zip(xy_obs, values):
        ref.add_measurement(x, y, v)
    return ref.query(grid_xy)


def test_gmrf_session_result_equals_direct_belief_on_stored_noisy_values(
    single_wall_path,
):
    sc = Scenario(single_wall_path)
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, 0.1)
    backends = make_backends(['gmrf'], sc.domain)
    assert not hasattr(backends['gmrf'], 'unavailable_reason')
    s = Session(sc, backends, grid, noise_std=0.002, seed=3)
    try:
        for x, y in [(0.3, 0.5), (0.6, 2.4), (1.4, 1.1), (1.7, 3.6)]:
            s.add_sample(x, y)
        s.add_manual(0.5, 3.5, 0.0123)
        assert s.wait_idle(60)
        _, st, _, _ = s.poll()
        r = st['gmrf']
        assert r.error is None and r.result_count == 5
        obs = s.observations
        xy_obs = [(o.x, o.y) for o in obs]
        noisy = [o.value for o in obs]
        assert all(o.noise != 0.0 for o in obs[:4])

        mean, var = _direct(sc, xy_obs, noisy, grid.xy)
        np.testing.assert_allclose(r.mean, mean, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(r.var, var, rtol=1e-12, atol=1e-15)

        latents = [o.latent if o.latent is not None else o.value for o in obs]
        mean_latent, _ = _direct(sc, xy_obs, latents, grid.xy)
        assert np.nanmax(np.abs(mean_latent - r.mean)) > 1e-4
    finally:
        s.close()
