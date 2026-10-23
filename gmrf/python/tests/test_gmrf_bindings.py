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

"""Binding tests: the Python module must reproduce direct C++ calls.

The C++ side is `gmrf_py_reference` (python/reference_case.cpp), built next
to the extension module. Run with the build directory on PYTHONPATH:

    PYTHONPATH=<build>/python python -m pytest gmrf/python/tests
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

gmrf_py = pytest.importorskip('scalar_field_belief_gmrf_py')

# Same case as reference_case.cpp.
QUERY = np.array(
    [[-0.9, 0.6], [-0.6, 1.0], [0.1, 1.2], [0.35, 1.2], [1.45, 1.95]]
)
MEASUREMENTS = [(-0.6, 1.0, 0.3), (0.7, 1.2, -0.2), (0.0, 1.8, 0.5)]
# Same compiled core on both sides; allow only rounding-level differences.
RTOL, ATOL = 1e-10, 1e-14


def _reference_config():
    cfg = gmrf_py.GmrfConfig()
    cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max = -1.0, 1.5, 0.5, 2.0
    cfg.nx, cfg.ny = 11, 7
    cfg.prior_mean = 0.1
    cfg.prior_sigma = 0.5
    cfg.sigma_p_per_m = 2.0
    cfg.sigma_s = 0.05
    return cfg


def _reference_belief():
    return gmrf_py.GmrfBelief(
        _reference_config(), [gmrf_py.WallSegment(0.2, 0.4, 0.2, 1.6)]
    )


@pytest.fixture(scope='module')
def cpp_reference() -> dict[str, np.ndarray]:
    exe = Path(gmrf_py.__file__).resolve().parent / 'gmrf_py_reference'
    if not exe.is_file():
        pytest.fail(f'C++ reference executable not found next to module: {exe}')
    out = subprocess.run(
        [str(exe)], check=True, capture_output=True, text=True, timeout=60
    ).stdout
    ref = {}
    for line in out.splitlines():
        key, *values = line.split()
        ref[key] = np.array([float(v) for v in values])
    return ref


def test_matches_direct_cpp_calls(cpp_reference):
    ref = cpp_reference
    belief = _reference_belief()

    mean, var = belief.query_mean_variance(QUERY)
    np.testing.assert_allclose(mean, ref['prior_mean'], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(var, ref['prior_var'], rtol=RTOL, atol=ATOL)

    for x, y, v in MEASUREMENTS:
        belief.add_measurement(x, y, v)
    assert belief.has_pending_measurements()
    report = belief.update()
    assert report.num_assimilated == int(ref['num_assimilated'][0])
    assert belief.num_measurements() == int(ref['num_measurements'][0])

    mean, var = belief.query_mean_variance(QUERY)
    np.testing.assert_allclose(mean, ref['post_mean'], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(var, ref['post_var'], rtol=RTOL, atol=ATOL)
    cmean, cov = belief.query_mean_covariance(QUERY)
    np.testing.assert_allclose(
        cmean, ref['post_cov_mean'], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        cov.ravel(), ref['post_cov'], rtol=RTOL, atol=ATOL
    )

    assert belief.measurement_noise_variance() == pytest.approx(
        ref['noise_var'][0], rel=RTOL
    )
    nx, ny, dx, dy = ref['grid']
    assert (belief.nx, belief.ny) == (int(nx), int(ny))
    assert belief.dx == pytest.approx(dx, rel=RTOL)
    assert belief.dy == pytest.approx(dy, rel=RTOL)

    belief.reset()
    mean, var = belief.query_mean_variance(QUERY)
    np.testing.assert_allclose(mean, ref['reset_mean'], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(var, ref['reset_var'], rtol=RTOL, atol=ATOL)


def test_shapes_and_ownership():
    belief = _reference_belief()
    mean, var = belief.query_mean_variance(QUERY)
    assert mean.shape == (5,) and var.shape == (5,)
    assert mean.dtype == np.float64 and var.dtype == np.float64
    assert mean.flags.owndata and mean.flags.writeable
    _, cov = belief.query_mean_covariance(QUERY)
    assert cov.shape == (5, 5)
    np.testing.assert_allclose(np.diag(cov), var, rtol=1e-9)

    empty_mean, empty_var = belief.query_mean_variance(np.empty((0, 2)))
    assert empty_mean.shape == (0,) and empty_var.shape == (0,)

    # Non-contiguous / integer input is converted.
    q_f = np.asfortranarray(QUERY)
    np.testing.assert_allclose(belief.query_mean_variance(q_f)[0], mean)
    m_int, _ = belief.query_mean_variance(np.array([[0, 1]]))
    assert m_int.shape == (1,)

    mask = belief.representable_mask(QUERY)
    assert mask.dtype == np.bool_ and mask.shape == (5,) and mask.all()


def test_errors_map_to_python_exceptions():
    belief = _reference_belief()
    with pytest.raises(ValueError):
        belief.query_mean_variance(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        belief.query_mean_variance(np.zeros(2))
    with pytest.raises(ValueError):  # outside the domain
        belief.add_measurement(5.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        belief.add_measurement(0.0, 1.0, float('nan'))
    belief.add_measurement(0.0, 1.0, 0.2)
    with pytest.raises(RuntimeError):  # query while pending (logic_error)
        belief.query_mean_variance(QUERY)
    belief.update()
    assert belief.num_measurements() == 1

    bad = _reference_config()
    bad.nx = 1
    with pytest.raises(ValueError):
        bad.validate()
    with pytest.raises(ValueError):
        gmrf_py.GmrfBelief(bad)


def test_concurrent_calls_are_serialised():
    """Updates and queries from several threads must not corrupt a belief."""
    belief = _reference_belief()
    errors = []

    def writer(k):
        try:
            for i in range(20):
                belief.add_measurement(-0.8 + 0.01 * i, 0.6 + 0.05 * k, 0.1)
                belief.update()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        for _ in range(40):
            try:
                belief.query_mean_variance(QUERY)
            except RuntimeError:
                pass  # pending data from a writer is a legal outcome
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    belief.update()
    assert belief.num_measurements() == 60
    mean, var = belief.query_mean_variance(QUERY)
    assert np.all(np.isfinite(mean)) and np.all(var > 0)


def test_model_grid_nodes_and_cut_edges():
    belief = _reference_belief()
    g = belief.model_grid()
    nx, ny = g['nx'], g['ny']
    assert (nx, ny) == (11, 7)
    xy, active, edges = g['node_xy'], g['active'], g['edges']
    assert xy.shape == (nx * ny, 2) and active.shape == (nx * ny,)
    assert active.all()  # no free-space seeds -> every node active
    # Node order n = j*nx + i, spacing = model dx/dy, corners on the bounds.
    grid = xy.reshape(ny, nx, 2)
    np.testing.assert_allclose(grid[0, 0], [-1.0, 0.5])
    np.testing.assert_allclose(grid[-1, -1], [1.5, 2.0])
    np.testing.assert_allclose(np.diff(grid[..., 0], axis=1), belief.dx)
    np.testing.assert_allclose(np.diff(grid[..., 1], axis=0), belief.dy)
    assert edges.shape[1] == 2 and edges.dtype == np.int64
    a, b = xy[edges[:, 0]], xy[edges[:, 1]]
    step = np.abs(b - a)
    # Only nearest-neighbour edges, axis-aligned, one model cell long.
    assert np.all(
        np.isclose(step[:, 0], belief.dx) & np.isclose(step[:, 1], 0.0)
        | np.isclose(step[:, 0], 0.0) & np.isclose(step[:, 1], belief.dy)
    )
    full = (nx - 1) * ny + nx * (ny - 1)
    assert len(edges) < full  # the wall cuts some edges
    # The wall x = 0.2, y in [0.4, 1.6] separates node columns x = 0 and 0.25.
    crossing = (np.minimum(a[:, 0], b[:, 0]) < 0.2) & (
        np.maximum(a[:, 0], b[:, 0]) > 0.2
    )
    ys = a[crossing, 1]
    assert not np.any(ys <= 1.5 + 1e-9)  # cut along the wall
    assert np.any(np.isclose(ys, 2.0))  # intact beyond the wall end
