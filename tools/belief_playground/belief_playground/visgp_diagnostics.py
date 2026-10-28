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

"""Headless numerical diagnostic of the visGP belief.

    python -m belief_playground.visgp_diagnostics [--out-dir DIR]
        [--scenarios 00_open,01_single_wall,02_wall_with_gap]
        [--visgp-config YAML] [--export-image]

Model invariants are checked on three small hand-made geometries
(open_tank, single_wall_coarse_grid, closed_box_components) and on the
scenario reference grids. Reference values (parent kernel, rows b(x) and
F(x), covariance selection residuals, the dense Gaussian posterior) are
recomputed independently of the belief's own code paths. Each scenario then
runs through the playground Session with the baseline GP and visGP after the
same measurements left of the wall; its reconstruction quality is reported,
not judged. Values are in the scenario unit.

Every check ends as passed, failed or not run (with the reason, e.g. a
scenario that was not selected). Exit code 1 if a check failed, 3 if none
failed but some did not run, else 0; 2 is a usage error (argparse), including
--export-image without a display.

Output (default `./belief_playground_sessions/visgp_diagnostics/<UTC time>/`):
report.json, report.md, arrays.npz and the sessions (CSV + JSON as saved by
the playground), all written before any Qt use. --export-image then adds a
PNG of the playground window per scenario; it is refused without a display
(DISPLAY or WAYLAND_DISPLAY, e.g. under xvfb-run, or QT_QPA_PLATFORM=offscreen).
Qt is imported only for --export-image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from belief_playground.backends import make_backends
from belief_playground.launcher import (
    ResourceError,
    default_visgp_config,
    resolve_scenario,
)
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import (
    Session,
    read_observations_csv,
    rmse_on_common_grid,
)
from scalar_field_belief.geometry import WallGeometry, WallSegment
from scalar_field_belief.visgp import (
    VisGPBelief,
    VisGPConfig,
    covariance_selection,
)
from scalar_field_belief.visgp.visibility import (
    contact_tolerance,
    point_wall_distance,
    visible_pairs,
    wall_array,
)

S2 = 1.0e-4  # signal variance of the small geometries [V^2]
ELL = 0.5  # lengthscale [m]
NOISE = 2.5e-7  # measurement noise variance [V^2]
MU = 0.0
LEFT_SIDE = ((0.5, 1.0), (0.7, 2.0), (0.4, 3.0), (0.75, 3.4))
DEFAULT_SCENARIOS = ('00_open', '01_single_wall', '02_wall_with_gap')
OPEN_SCENARIO, SESSION_SCENARIO = '00_open', '01_single_wall'
WALL_01 = ((1.0, 0.4), (1.0, 3.6))
DIAGONAL_WALL = ((0.2, 0.2), (1.8, 1.8))
PASSED, FAILED, NOT_RUN = 'passed', 'failed', 'not run'
DISPLAYLESS_QT_PLATFORMS = ('offscreen', 'minimal', 'vnc')
MODEL_ARRAYS = ('reference_locations', 'parent_covariance', 'L', 'adjacency')


@dataclass
class Check:
    name: str
    status: str = PASSED
    reason: str = ''
    details: dict = field(default_factory=dict)

    def expect(self, label: str, ok, value=None) -> None:
        entry = {'ok': bool(ok)}
        if value is not None:
            entry['value'] = value
        self.details[label] = entry
        if not entry['ok']:
            self.status = FAILED


def run_check(label: str, fn, *args, status=PASSED, reason='') -> Check:
    """`fn(check, *args)` unless `status` says it cannot run; exceptions
    become a failed check with their message (no traceback). The check
    title is the first line of `fn`'s docstring."""
    title = fn.__doc__.strip().splitlines()[0].rstrip('.')
    c = Check(f'{title} ({label})', status, reason)
    if status == PASSED:
        try:
            fn(c, *args)
        except Exception as exc:  # noqa: BLE001 - report, run the others
            c.status, c.reason = FAILED, f'{type(exc).__name__}: {exc}'
    return c


def _geometry(name, x_range, y_range, walls=()) -> WallGeometry:
    return WallGeometry(
        name=name,
        x_range=x_range,
        y_range=y_range,
        walls=tuple(WallSegment(start=a, end=b) for a, b in walls),
    )


def _box(x0, y0, x1, y1):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(corners[k], corners[(k + 1) % 4]) for k in range(4)]


def small_geometry_beliefs() -> dict:
    """open_tank: 2 x 4 m, no walls, 0.25 m grid. single_wall_coarse_grid:
    the 01 wall on a 0.5 m grid (32 nodes). closed_box_components: 2 x 1 m,
    full-height wall at x = 1, a closed box around the node (1.75, 0.75) and
    an empty box around (0.5, 0.5); 0.5 m grid."""

    def config(spacing):
        return VisGPConfig(
            signal_variance=S2,
            lengthscale=ELL,
            measurement_noise_variance=NOISE,
            reference_spacing=spacing,
            prior_mean=MU,
        )

    walls = [((1.0, 0.0), (1.0, 1.0))]
    walls += _box(1.65, 0.65, 1.85, 0.85) + _box(0.4, 0.4, 0.6, 0.6)
    return {
        'open_tank': VisGPBelief(
            config(0.25), _geometry('open', (0, 2), (0, 4))
        ),
        'single_wall_coarse_grid': VisGPBelief(
            config(0.5), _geometry('wall', (0, 2), (0, 4), [WALL_01])
        ),
        'closed_box_components': VisGPBelief(
            config(0.5), _geometry('boxes', (0, 2), (0, 1), walls)
        ),
    }


def _rows(text: str) -> np.ndarray:
    """Float table from 'a b c; d e f; ...'."""
    return np.array([row.split() for row in text.split(';')], dtype=float)


def _rel(a, b, scale) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / scale


# ------------------------------------------------- independent references
def reference_matern32(a, b, s2, ell):
    """Matern-3/2 coded independently of covariance.matern32."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = np.sqrt(3.0) * np.linalg.norm(a[:, None] - b[None, :], axis=-1) / ell
    return s2 * np.exp(-r) * (1.0 + r)


def nearest_clique(b: VisGPBelief, x) -> list[int]:
    """Algorithm 2 with plain Python sorting (stop at the first candidate
    that is not adjacent to all chosen ones)."""
    V = b.reference_locations
    vis = visible_pairs(
        np.repeat([x], len(V), 0), V, b.walls, b.contact_tolerance
    )
    cands = sorted(
        ((x[0] - v[0]) * (x[0] - v[0]) + (x[1] - v[1]) * (x[1] - v[1]), j)
        for j, v in enumerate(V)
        if vis[j]
    )[: b.config.max_neighbors]
    chosen = []
    for _, j in cands:
        if not all(b.adjacency[j, k] for k in chosen):
            break
        chosen.append(j)
    return chosen


def reference_rows(b: VisGPBelief, xy):
    """b(x) and F(x) from reference_matern32, nearest_clique and
    np.linalg.solve (independent of extension.rows)."""
    V = b.reference_locations
    s2, ell = b.config.signal_variance, b.config.lengthscale
    index = {(float(v[0]) + 0.0, float(v[1]) + 0.0): i for i, v in enumerate(V)}
    xy = np.asarray(xy, float).reshape(-1, 2)
    B, F = np.zeros((len(xy), len(V))), np.full(len(xy), s2)
    for r, x in enumerate(xy):
        i = index.get((float(x[0]) + 0.0, float(x[1]) + 0.0))
        if i is not None:
            B[r, i], F[r] = 1.0, 0.0
        elif nb := nearest_clique(b, x):
            c = reference_matern32(x[None], V[nb], s2, ell)[0]
            w = np.linalg.solve(reference_matern32(V[nb], V[nb], s2, ell), c)
            B[r, nb], F[r] = w, s2 - c @ w
    return B, F


def reference_prior_covariance(b: VisGPBelief, xy) -> np.ndarray:
    """C_vis(xy, xy) from reference_rows; same location = equal float tuples."""
    B, F = reference_rows(b, xy)
    keys = [(float(x) + 0.0, float(y) + 0.0) for x, y in xy]
    same = np.array([[k == q for q in keys] for k in keys])
    return B @ b.L @ B.T + np.where(same, F[:, None], 0.0)


def covsel_residuals(b: VisGPBelief) -> tuple[float, float]:
    """(max |L - K| on diagonal and edges / s2, max normalised non-edge
    precision), recomputed from L, K and the adjacency."""
    edge = b.adjacency | np.eye(len(b.L), dtype=bool)
    P = np.linalg.inv(b.L)
    d = np.sqrt(np.diag(P))
    partial = np.abs(P) / np.outer(d, d)
    r_edge = float(np.max(np.abs(b.L - b.parent_covariance)[edge]))
    r_prec = float(partial[~edge].max()) if (~edge).any() else 0.0
    return r_edge / b.config.signal_variance, r_prec


def column_coordinate_ascent(K, A, precision_tol=1e-11, max_sweeps=20000):
    """Independent covariance selection (Hastie, Tibshirani and Friedman,
    Algorithm 17.1 with known structure)."""
    W = K.copy()
    nonedge = ~A & ~np.eye(len(K), dtype=bool)
    sweeps, r = 0, np.inf
    while r > precision_tol and sweeps < max_sweeps:
        for j in range(len(K)):
            nb, non = np.flatnonzero(A[j]), np.flatnonzero(nonedge[j])
            new = np.zeros(non.size)
            if nb.size and non.size:
                beta = np.linalg.solve(W[np.ix_(nb, nb)], K[nb, j])
                new = W[np.ix_(non, nb)] @ beta
            W[non, j] = W[j, non] = new
        P = np.linalg.inv(W)
        d = np.sqrt(np.diag(P))
        r = float(np.max(np.abs(P[nonedge]) / np.outer(d, d)[nonedge]))
        sweeps += 1
    return W, sweeps, r


# ------------------------------------------------------------ invariants
def check_visibility_predicate(c: Check) -> None:
    """Visibility predicate."""
    g01 = _geometry('g01', (0, 2), (0, 4), [WALL_01])
    gd = _geometry('gd', (0, 2), (0, 4), [DIAGONAL_WALL])
    eps = 3.6 + 1e-9
    cases = [
        ('crossing', g01, (0.5, 2.0), (1.5, 2.0), False),
        ('through wall endpoint', g01, (0.5, 3.1), (1.5, 4.1), False),
        ('collinear overlap', g01, (1.0, 0.0), (1.0, 1.0), False),
        ('collinear endpoint contact', g01, (1.0, 0.0), (1.0, 0.4), False),
        ('collinear, 0.1 m gap', g01, (1.0, 0.0), (1.0, 0.3), True),
        ('1e-6 m beyond wall end', g01, (0.5, 3.600001), (1.5, 3.600001), True),
        (
            '1e-9 m beyond wall end (within tolerance)',
            g01,
            (0.5, eps),
            (1.5, eps),
            False,
        ),
        ('parallel, same side', g01, (0.9, 1.0), (0.9, 3.0), True),
        ('diagonal wall crossing', gd, (0.2, 1.0), (1.0, 0.2), False),
        ('diagonal wall parallel', gd, (0.2, 0.4), (1.6, 1.8), True),
    ]
    for label, g, p, q, expected in cases:
        args = (wall_array(g), contact_tolerance(g))
        got = bool(visible_pairs(np.array([p]), np.array([q]), *args)[0])
        c.expect(label, got == expected, {'visible': got})
    walls = [WALL_01, DIAGONAL_WALL, ((0.3, 3.0), (1.6, 3.2))]
    pts = np.random.default_rng(1).uniform((0, 0), (2, 4), size=(150, 2))
    i, j = np.triu_indices(len(pts), 1)

    def visible(ws, p, q):
        g = _geometry('sym', (0, 2), (0, 4), ws)
        return visible_pairs(p, q, wall_array(g), contact_tolerance(g))

    ab = visible(walls, pts[i], pts[j])
    c.expect(
        'symmetric in (p, q)',
        np.array_equal(ab, visible(walls, pts[j], pts[i])),
    )
    rev = visible([(b, a) for a, b in walls], pts[i], pts[j])
    c.expect('wall endpoint order invariant', np.array_equal(ab, rev))
    c.expect(
        'wall order invariant',
        np.array_equal(ab, visible(walls[::-1], pts[i], pts[j])),
    )
    c.details['random_pairs'] = int(len(i))


def check_covariance_selection(c: Check, beliefs: dict) -> None:
    """Covariance selection."""
    for name, b in beliefs.items():
        r_edge, r_prec = covsel_residuals(b)
        eig = np.linalg.eigvalsh(b.L)
        cs = b.covariance_selection
        c.expect(
            f'{name}: converged, symmetric, positive definite, residuals '
            'recomputed from L, K and G within the tolerances',
            cs.converged
            and np.array_equal(b.L, b.L.T)
            and eig[0] > 0
            and r_edge <= b.config.covsel_tolerance
            and r_prec <= b.config.covsel_precision_tolerance,
            {
                'diag_edge_residual_rel_s2': r_edge,
                'nonedge_precision': r_prec,
                'sweeps': cs.sweeps,
                'cliques': cs.num_cliques,
                'components': cs.num_components,
                'min_eigenvalue': float(eig[0]),
                'condition_number': float(eig[-1] / eig[0]),
                'solver_time_s': cs.elapsed_s,
            },
        )
        s2, ell = b.config.signal_variance, b.config.lengthscale
        V = b.reference_locations
        d = _rel(reference_matern32(V, V, s2, ell), b.parent_covariance, s2)
        c.expect(
            f'{name}: K matches an independent Matern-3/2 (<= 1e-14 s2)',
            d <= 1e-14,
            {'max_abs_diff_rel_s2': d},
        )


def check_small_covariance_selection(c: Check, beliefs: dict) -> None:
    """Covariance selection with an independent solver."""
    check_covariance_selection(c, beliefs)
    b = beliefs['single_wall_coarse_grid']
    W, sweeps, r = column_coordinate_ascent(b.parent_covariance, b.adjacency)
    diff = _rel(W, b.L, S2)
    c.expect(
        'single_wall_coarse_grid: L agrees with an independent column solver '
        '(<= 1e-7 s2)',
        diff <= 1e-7,
        {
            'max_abs_diff_rel_s2': diff,
            'reference_sweeps': sweeps,
            'reference_precision': r,
        },
    )


def check_no_wall_reduction(c: Check, b: VisGPBelief) -> None:
    """No-wall reduction on V."""
    c.expect(
        'L == K exactly (complete visibility graph)',
        np.array_equal(b.L, b.parent_covariance),
    )
    prior = b if b.num_measurements == 0 else VisGPBelief(b.config, b.geometry)
    _, cov = prior.query_with_covariance(prior.reference_locations)
    d = _rel(cov, prior.parent_covariance, prior.config.signal_variance)
    c.expect(
        'prior covariance on V == K (<= 1e-12 s2)',
        d <= 1e-12,
        {'max_abs_diff_rel_s2': d},
    )


def check_disconnected_components(c: Check, b: VisGPBelief) -> None:
    """Disconnected components."""
    cs, V = b.covariance_selection, b.reference_locations
    left = V[:, 0] < 1.0
    iso = b.extension.reference_index(1.75, 0.75)
    others = np.arange(len(V)) != iso
    c.expect(
        'three components, one isolated node',
        cs.num_components == 3 and cs.num_isolated == 1,
    )
    c.expect(
        'L exactly 0 across the wall', np.all(b.L[np.ix_(left, ~left)] == 0)
    )
    c.expect(
        'isolated node: L_ii == K_ii, row exactly 0',
        b.L[iso, iso] == b.parent_covariance[iso, iso]
        and np.all(b.L[iso, others] == 0),
    )
    _, prior_cov = b.query_with_covariance(V)
    c.expect(
        'prior covariance exactly 0 across components',
        np.all(prior_cov[np.ix_(left, ~left)] == 0)
        and np.all(prior_cov[iso, others] == 0),
    )
    rows = b.extension.rows(np.array([[0.5, 0.5]]))
    c.expect(
        'no-neighbour point (0.5, 0.5): b = 0, F = s2',
        np.all(rows.weights == 0) and rows.residual[0] == S2,
    )
    queries = _rows('1.25 0.25; 1.5 0.5; 1.7 0.8; 1.75 0.75; 0.5 0.5')
    prior_mean, prior_var = b.query(queries)
    left = '0.3 0.6 0.010; 0.7 0.3 0.012; 0.65 0.8 0.009; 0.25 0.25 0.011'
    for obs in _rows(left):
        b.add_measurement(*obs)
    mean, var = b.query(queries)
    b.reset()
    dv = _rel(var, prior_var, S2)
    c.expect(
        'left data only: right side, isolated node and no-neighbour '
        'point keep the prior (mean exactly, variance <= 1e-12 s2)',
        np.all(mean == MU) and np.all(prior_mean == MU) and dv <= 1e-12,
        {'max_var_change_rel_s2': dv},
    )


def check_mixed_joint_covariance(c: Check, b: VisGPBelief) -> None:
    """Mixed reference/off-reference joint covariance."""
    pts = _rows('0.75 1.75; 1.25 1.75; 0.5 2.0; 1.05 3.65; 1.5 2.0; 0.5 2.0')
    ref = [b.extension.reference_index(*p) for p in pts[:2]]
    obs = _rows(
        '0.5 1.0 0.010; 0.75 2.25 0.008; 0.4 3.0 0.012; 1.5 1.0 0.003; '
        '1.3 3.2 0.006'
    )
    B, _ = reference_rows(b, pts)
    for n_obs in (0, 5):
        b.reset()
        for o in obs[:n_obs]:
            b.add_measurement(*o)
        mean, cov = b.query_with_covariance(pts)
        _, var = b.query(pts)
        tag = f'{n_obs} obs'
        c.expect(f'{tag}: symmetric', np.array_equal(cov, cov.T))
        dup = max(_rel(cov[2], cov[5], S2), _rel(mean[2], mean[5], 0.012))
        c.expect(
            f'{tag}: duplicate position rows equal (<= 1e-12)',
            dup <= 1e-12,
            {'max_rel_diff': dup},
        )
        eig = float(np.linalg.eigvalsh(cov)[0]) / S2
        bound = -1e-14 if n_obs == 0 else -1e-12
        c.expect(
            f'{tag}: min eigenvalue >= {bound} s2',
            eig >= bound,
            {'min_eig_rel_s2': eig},
        )
        d = _rel(np.diag(cov), var, S2)
        c.expect(
            f'{tag}: joint diagonal == marginal variance (<= 1e-12 s2)',
            d <= 1e-12,
            {'max_abs_diff_rel_s2': d},
        )
    b.reset()
    _, cov = b.query_with_covariance(pts)
    d = _rel(cov[:2, :2], b.L[np.ix_(ref, ref)], S2)
    c.expect('prior: reference block == L (<= 1e-12 s2)', d <= 1e-12)
    d = _rel(cov, reference_prior_covariance(b, pts), S2)
    c.expect(
        'prior: equals C_vis from independent rows (<= 1e-12 s2)',
        d <= 1e-12,
        {'max_abs_diff_rel_s2': d},
    )
    dev = np.abs(np.diag(cov) - S2) / S2
    bound = np.abs(B).sum(axis=1) ** 2 * b.config.covsel_tolerance + 1e-12
    c.expect(
        'prior: diagonal == s2 within ||b||_1^2 * covsel_tolerance '
        '+ 1e-12 (relative)',
        np.all(dev <= bound),
        {'max_rel_dev': float(dev.max()), 'max_rel_bound': float(bound.max())},
    )


def check_neighbour_sets(c: Check, beliefs: dict) -> None:
    """Nearest-clique neighbour sets and rows."""
    rng = np.random.default_rng(3)
    for name, b in beliefs.items():
        g, V, s2 = b.geometry, b.reference_locations, b.config.signal_variance
        lo, hi = (g.x_range[0], g.y_range[0]), (g.x_range[1], g.y_range[1])
        pts = np.vstack(
            [
                rng.uniform(lo, hi, size=(300, 2)),
                V[:5],
                [[1.05, 3.65], [0.95, 2.0], [1.05, 2.0]],
            ]
        )
        pts = pts[b.valid_mask(pts)]
        nbs = b.extension.neighbours(pts)
        rows = b.extension.rows(pts)
        bad = {'visible': 0, 'clique': 0, 'size': 0, 'algorithm2': 0}
        for x, nb in zip(pts, nbs, strict=True):
            vis = visible_pairs(
                np.repeat([x], len(nb), 0), V[nb], b.walls, b.contact_tolerance
            )
            sub = b.adjacency[np.ix_(nb, nb)] | np.eye(len(nb), dtype=bool)
            bad['visible'] += not vis.all()
            bad['clique'] += not sub.all()
            bad['size'] += len(nb) > b.config.max_neighbors
            if b.extension.reference_index(*x) is None:
                bad['algorithm2'] += list(nb) != nearest_clique(b, x)
        B, F = reference_rows(b, pts)
        dw = float(np.max(np.abs(rows.weights - B)))
        dF = _rel(rows.residual, F, s2)
        perm = rng.permutation(len(pts))
        prow = b.extension.rows(pts[perm])
        dperm = max(
            _rel(prow.weights, rows.weights[perm], 1.0),
            _rel(prow.residual, rows.residual[perm], s2),
        )
        sizes = [len(nb) for nb in nbs]
        c.expect(
            f'{name}: visible, clique, <= k, equal to Algorithm 2; rows '
            'match independent rows (b <= 1e-10, F <= 1e-12 s2); 0 <= F <= '
            's2; permutation-independent (<= 1e-14)',
            not any(bad.values())
            and dw <= 1e-10
            and dF <= 1e-12
            and np.all((rows.residual >= 0) & (rows.residual <= s2))
            and dperm <= 1e-14,
            {
                'points': int(len(pts)),
                'violations': bad,
                'max_abs_diff_b': dw,
                'max_abs_diff_F_rel_s2': dF,
                'permutation_diff': dperm,
                'neighbour_set_size_min_max': [min(sizes), max(sizes)],
            },
        )


def check_repeated_measurements(c: Check, b: VisGPBelief) -> None:
    """Repeated off-reference measurements."""
    x0 = np.array([0.6, 1.1])
    B, F = reference_rows(b, x0[None])
    f0 = float(F[0])
    c00 = float(B[0] @ b.L @ B[0]) + f0
    values = [0.010, 0.012, 0.011]
    for n in (1, 2, 3):
        b.reset()
        for v in values[:n]:
            b.add_measurement(*x0, v)
        mean, var = b.query(x0[None])
        exp_mean = MU + c00 * sum(v - MU for v in values[:n]) / (
            NOISE + n * c00
        )
        exp_var = c00 * NOISE / (NOISE + n * c00)
        em = abs(mean[0] - exp_mean) / abs(exp_mean - MU)
        ev = abs(var[0] - exp_var) / exp_var
        c.expect(
            f'n={n}: closed form from independent rows (shared F, '
            'independent noise; <= 1e-10 relative)',
            em <= 1e-10 and ev <= 1e-10,
            {'rel_err_mean': em, 'rel_err_var': ev, 'var': float(var[0])},
        )
    c.expect('latent variance: n=3 is below the noise variance', var[0] < NOISE)
    b.reset()
    _, cov = b.query_with_covariance(np.vstack([x0, x0 + [1e-9, 0.0]]))
    jump = float(cov[0, 0] - cov[0, 1])
    c.expect(
        'x0 + (1e-9, 0) is another location: K00 - K01 == F(x0) '
        '(<= 1e-3 relative)',
        f0 > 0 and abs(jump - f0) <= 1e-3 * f0,
        {'K00_minus_K01_rel_s2': jump / S2, 'F_x0_rel_s2': f0 / S2},
    )


def check_dense_oracle(c: Check, b: VisGPBelief) -> None:
    """Dense Gaussian oracle."""
    b.reset()
    train = _rows(
        '0.75 1.25 0.009; 0.25 2.75 0.011; 0.6 1.1 0.010; 0.6 1.1 0.0102; '
        '1.5 1.0 0.004; 1.4 3.0 0.005'
    )
    queries = _rows(
        '0.75 1.25; 0.6 1.1; 0.6 1.2; 0.5 2.0; 1.05 3.65; 0.95 3.7; 1.5 2.0; '
        '1.75 3.75; 1.2 0.3; 0.05 3.95'
    )
    for o in train:
        b.add_measurement(*o)
    n, y = len(train), train[:, 2]
    K = reference_prior_covariance(b, np.vstack([train[:, :2], queries]))
    S = K[:n, :n] + NOISE * np.eye(n)
    oracle_mean = MU + K[n:, :n] @ np.linalg.solve(S, y - MU)
    oracle_cov = K[n:, n:] - K[n:, :n] @ np.linalg.solve(S, K[:n, n:])
    mean, cov = b.query_with_covariance(queries)
    _, var = b.query(queries)
    b.reset()
    dm = _rel(mean, oracle_mean, float(np.max(np.abs(y - MU))))
    dc, dv = _rel(cov, oracle_cov, S2), _rel(var, np.diag(oracle_cov), S2)
    c.expect(
        'mean, covariance, variance vs C_vis from independent rows (<= 1e-9)',
        dm <= 1e-9 and dc <= 1e-9 and dv <= 1e-9,
        {'mean_rel': dm, 'cov_rel_s2': dc, 'var_rel_s2': dv},
    )


def check_noise_order(c: Check) -> None:
    """Noise-order example."""
    K = np.full((3, 3), 0.5) + 0.5 * np.eye(3)
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    after = covariance_selection(K, A, 1.0, 1e-12, 1e-12, 500, 10).L[0, 2]
    inside = covariance_selection(
        K + np.eye(3), A, 1.0, 1e-12, 1e-12, 500, 10
    ).L[0, 2]
    c.expect(
        'CovSel(C) + tau^2 I: Cov(Y1, Y3) = 0.25',
        abs(after - 0.25) <= 1e-10,
        float(after),
    )
    c.expect(
        'CovSel(C + tau^2 I): Cov(Y1, Y3) = 0.125',
        abs(inside - 0.125) <= 1e-10,
        float(inside),
    )


def _wait(s: Session) -> None:
    if not s.wait_idle(600):
        raise RuntimeError('session worker did not finish within 600 s')


def _session(scenario, backends: dict, step: float, samples=LEFT_SIDE):
    """Session with the given backends after `samples`; raises with the
    message if a backend is unavailable or has no map after the samples."""
    for name, backend in backends.items():
        if getattr(backend, 'unavailable_reason', None):
            raise RuntimeError(
                f'backend {name} unavailable: {backend.unavailable_reason}'
            )
    d = scenario.domain
    s = Session(
        scenario, backends, make_display_grid(d.x_range, d.y_range, step)
    )
    for x, y in samples:
        s.add_sample(x, y)
    _wait(s)
    _, status, _, _ = s.poll()
    for name, st in status.items():
        if st.error or (samples and (st.mean is None or st.var is None)):
            s.close()
            raise RuntimeError(
                f'backend {name}: {st.error or st.note or st.state}'
            )
    return s


def check_batching(c: Check, scenario: Scenario, config_path) -> None:
    """Batching and display-grid independence."""

    def visgp_session(step):
        b = make_backends(['visgp'], scenario.domain, visgp_config=config_path)
        return _session(scenario, b, step)

    s = visgp_session(0.05)
    try:
        b = s.backends['visgp'].belief
        s2, xy = b.config.signal_variance, s.grid.xy
        mean, var = b.query(xy)
        scale = float(np.max(np.abs(b.y_train - b.config.prior_mean)))
        rng = np.random.default_rng(5)
        perm = rng.permutation(len(xy))
        single = rng.choice(len(xy), 200, replace=False)
        chunks = [b.query(xy[k : k + 333]) for k in range(0, len(xy), 333)]
        singles = [b.query(xy[[k]]) for k in single]
        variants = {
            'permuted': (*b.query(xy[perm]), perm),
            'chunked (333)': (
                *map(np.concatenate, zip(*chunks, strict=True)),
                ...,
            ),
            'singletons (200)': (
                *map(np.concatenate, zip(*singles, strict=True)),
                single,
            ),
        }
        for label, (m2, v2, ref) in variants.items():
            dm, dv = _rel(m2, mean[ref], scale), _rel(v2, var[ref], s2)
            c.expect(
                f'{label} (<= 1e-12)',
                dm <= 1e-12 and dv <= 1e-12,
                {'mean_rel': dm, 'var_rel_s2': dv},
            )
        L_before = b.L.copy()
        probe = np.vstack([xy[::7], rng.uniform((0, 0), (2, 4), (100, 2))])
        for step in (0.1, 0.1 / 3):
            o = visgp_session(step)
            try:
                ob = o.backends['visgp'].belief
                same_vg = np.array_equal(
                    ob.reference_locations, b.reference_locations
                ) and np.array_equal(ob.adjacency, b.adjacency)
                dL = _rel(ob.L, b.L, s2)
                (m1, v1), (m2, v2) = b.query(probe), ob.query(probe)
                dm, dv = _rel(m1, m2, scale), _rel(v1, v2, s2)
            finally:
                o.close()
            c.expect(
                f'display step {step:.4g} m: same V and G (bitwise, built '
                'elementwise), L and predictions (<= 1e-12)',
                same_vg and dL <= 1e-12 and dm <= 1e-12 and dv <= 1e-12,
                {
                    'display_cells': len(o.grid.xy),
                    'L_rel_s2': dL,
                    'mean_rel': dm,
                    'var_rel_s2': dv,
                },
            )
        c.expect('L unchanged by measurements', np.array_equal(L_before, b.L))
        tracemalloc.start()
        b.query(xy)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        dense = len(xy) ** 2 * 8
        c.expect(
            f'{len(xy)}-cell map query: peak memory < 25% of an (N, N) matrix',
            peak < 0.25 * dense,
            {'peak_bytes': int(peak), 'dense_nxn_bytes': dense},
        )
        try:
            b.query_with_covariance(
                np.repeat(xy[:1], b.config.max_joint_query_size + 1, 0)
            )
            raised = False
        except ValueError:
            raised = True
        c.expect('oversized joint query raises ValueError', raised)
    finally:
        s.close()


def check_session_replay(
    c: Check, scenario: Scenario, config_path, out: Path
) -> None:
    """Reset/undo/save/load through the session."""
    backends = make_backends(
        ['gp', 'visgp'], scenario.domain, visgp_config=config_path
    )
    fresh = make_backends(['visgp'], scenario.domain, visgp_config=config_path)[
        'visgp'
    ]
    s = _session(scenario, backends, 0.1, ())
    try:
        s2 = backends['visgp'].belief.config.signal_variance
        prior_mean, prior_var = fresh.query(s.grid.xy)

        def maps():
            _wait(s)
            _, st, _, _ = s.poll()
            return st['visgp'], st['gp']

        def close(st, m, v, scale):
            kw = {'rtol': 0.0, 'equal_nan': True}
            return (
                st.mean is not None
                and np.allclose(st.mean, m, atol=1e-12 * scale, **kw)
                and np.allclose(st.var, v, atol=1e-12 * s2, **kw)
            )

        st0, _ = maps()
        m0, v0 = st0.mean, st0.var
        c.expect(
            'prior before any observation equals a fresh instance (<= 1e-12)',
            st0.state == 'ready' and close(st0, prior_mean, prior_var, s2),
        )
        for x, y in LEFT_SIDE:
            s.add_sample(x, y)
        st4, _ = maps()
        m4, v4 = st4.mean, st4.var
        scale = float(np.nanmax(np.abs(m4)))
        s.add_sample(1.3, 2.0)
        maps()
        s.undo()
        c.expect(
            'add -> undo reproduces the 4-observation maps (<= 1e-12)',
            close(maps()[0], m4, v4, scale),
        )
        csv_path, _ = s.save(out / 'replay_01_gp_visgp.csv')
        s.reset()
        st, _ = maps()
        c.expect(
            'reset -> prior, bitwise equal to the same instance before any '
            'observation',
            st.n_assimilated == 0
            and np.array_equal(st.mean, m0, equal_nan=True)
            and np.array_equal(st.var, v0, equal_nan=True),
        )
        warnings = s.load(csv_path)
        c.expect(
            'save -> reset -> load reproduces the maps (<= 1e-12)',
            close(maps()[0], m4, v4, scale),
            {'load_warnings': warnings},
        )
        c.expect(
            'load: no effective-configuration warning for visgp',
            not any(w.startswith('visgp') for w in warnings),
        )
        saved = read_observations_csv(csv_path)
        c.expect(
            'CSV coordinates equal the requested coordinates bitwise',
            [(o.x, o.y) for o in saved] == list(LEFT_SIDE),
        )
        s.add_sample(1.0, 2.0)
        st, gp = maps()
        c.expect(
            'observation on the wall stops visgp only',
            st.state == 'error'
            and st.failed_at == 5
            and gp.error is None
            and gp.n_assimilated == 5,
            {'visgp_error': st.error},
        )
        s.undo()
        st, _ = maps()
        c.expect(
            'undo recovers visgp (<= 1e-12)',
            st.error is None and close(st, m4, v4, scale),
        )
    finally:
        s.close()


# ------------------------------------------------------- reconstruction
def _adjacent_jumps(grid, mean, walls, tol) -> dict:
    """Largest |mean difference| between adjacent display cells: over all
    pairs, and over pairs whose connecting segment does not touch a wall."""
    img = grid.to_image(mean)
    centres = np.stack(np.meshgrid(grid.xs, grid.ys), -1)
    out = {'all': 0.0, 'visible': 0.0}
    for p, q, d in (
        (centres[:-1], centres[1:], img[1:] - img[:-1]),
        (centres[:, :-1], centres[:, 1:], img[:, 1:] - img[:, :-1]),
    ):
        d = np.abs(d).reshape(-1)
        vis = visible_pairs(p.reshape(-1, 2), q.reshape(-1, 2), walls, tol)
        out['all'] = max(out['all'], float(np.nanmax(d)))
        out['visible'] = max(
            out['visible'], float(np.nanmax(np.where(vis, d, np.nan)))
        )
    return out


def _residual_statistics(b: VisGPBelief, xy) -> dict:
    """F / signal_variance on the display grid (valid cells)."""
    s2, g = b.config.signal_variance, b.geometry
    valid = b.valid_mask(xy)
    v = xy[valid]
    f = b.extension.rows(v).residual / s2
    h = b.config.reference_spacing
    away = (
        (v[:, 0] > g.x_range[0] + h)
        & (v[:, 0] < g.x_range[1] - h)
        & (v[:, 1] > g.y_range[0] + h)
        & (v[:, 1] < g.y_range[1] - h)
        & (point_wall_distance(v, b.walls) > h)
    )
    corner = np.array([[g.x_range[0], g.y_range[0]]])
    return {
        'mean': float(f.mean()),
        'max': float(f.max()),
        'corner_display_cells': [float(f[0]), float(f[-1])],
        'exact_domain_corner': float(b.extension.rows(corner).residual[0] / s2),
        'cells_away_from_boundary_and_walls_max': float(f[away].max()),
        'cells_away_from_boundary_and_walls_mean': float(f[away].mean()),
        'display_cells_in_reference_set': int(
            sum(b.extension.reference_index(x, y) is not None for x, y in v)
        ),
        'display_cells_on_walls': int((~valid).sum()),
    }


def reconstruction_run(
    c: Check, scenario, adapter, out: Path, artefacts
) -> None:
    """Reconstruction run.

    Report-only comparison of the baseline GP and visGP (not judged).
    """
    name = scenario.path.stem
    gp = make_backends(['gp'], scenario.domain)['gp']
    s = _session(scenario, {'gp': gp, 'visgp': adapter}, 0.05)
    artefacts['sessions'][name] = s  # closed by main()
    counts = adapter.last_query_counts  # of the grid query
    c.details['session_csv'] = str(s.save(out / f'{name}_gp_visgp.csv')[0])
    _, st, _, _ = s.poll()
    grid, b = s.grid, adapter.belief
    truth = scenario.truth(grid.xy)
    rmse, cells = rmse_on_common_grid(truth, {n: st[n].mean for n in st})
    x = grid.xy[:, 0]
    right, left = (x > 1.0) & (x < 1.3), (x > 0.7) & (x < 1.0)
    probes = {
        'upper wall end (1.0, 3.65)': (1.0, 3.65),
        'lower wall end (1.0, 0.35)': (1.0, 0.35),
        'gap centre / wall (1.0, 2.0)': (1.0, 2.0),
        'behind wall (1.3, 2.0)': (1.3, 2.0),
    }
    bands, probe_std = {}, {}
    for n, backend in s.backends.items():
        std = np.sqrt(np.clip(st[n].var, 0.0, None))
        bands[n] = {
            'mean_right': float(np.nanmean(st[n].mean[right])),
            'mean_left': float(np.nanmean(st[n].mean[left])),
            'std_right': float(np.nanmean(std[right])),
            'std_left': float(np.nanmean(std[left])),
        }
        pts = np.array(list(probes.values()))
        # visGP through the core belief: keeps the adapter's grid counts.
        pvar = (
            b.predict_marginal(pts).variance
            if backend is adapter
            else backend.query(pts)[1]
        )
        pstd = np.sqrt(np.clip(pvar, 0, None))
        probe_std[n] = {
            k: float(v) if np.isfinite(v) else None  # None: on a wall
            for k, v in zip(probes, pstd, strict=True)
        }
    c.details.update(
        {
            'observations': [[o.x, o.y, o.value] for o in s.observations],
            'rmse_vs_truth': rmse,
            'rmse_cells': cells,
            'bands_x_1_to_1.3_vs_0.7_to_1': bands,
            'std_at_probes': probe_std,
            'largest_adjacent_cell_mean_jump': {
                n: _adjacent_jumps(
                    grid, st[n].mean, b.walls, b.contact_tolerance
                )
                for n in st
            },
            'F_over_signal_variance': _residual_statistics(b, grid.xy),
            'last_grid_query_counts_visgp': counts,
            'covariance_selection': adapter.effective_config()[
                'covariance_selection'
            ],
            'timings_ms': {
                n: {
                    'last_assimilation': st[n].update_ms,
                    'grid_query': st[n].query_ms,
                    'grid_cells': st[n].query_cells,
                }
                for n in st
            },
        }
    )
    arrays = {'xy': grid.xy, 'truth': truth}
    arrays.update({k: getattr(b, k) for k in MODEL_ARRAYS})
    for n in st:
        arrays[f'{n}_mean'], arrays[f'{n}_var'] = st[n].mean, st[n].var
    artefacts['arrays'].update({f'{name}_{k}': v for k, v in arrays.items()})


# ------------------------------------------------------------------ main
def _display_available() -> bool:
    platform = os.environ.get('QT_QPA_PLATFORM', '').split(':')[0]
    return bool(
        os.environ.get('DISPLAY')
        or os.environ.get('WAYLAND_DISPLAY')
        or platform in DISPLAYLESS_QT_PLATFORMS
    )


def _export_images(runs, sessions: dict, out: Path) -> None:
    """PNG of the playground window for each scenario session (Qt)."""
    from belief_playground.gui import PlaygroundWindow, ensure_app

    app = ensure_app()
    for name, run in runs:
        if name not in sessions:
            continue
        win = PlaygroundWindow(sessions[name], out_dir=out)
        win.show()
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        image = win.export_image(out / f'{name}_gp_visgp.png')
        win.close()
        app.processEvents()
        run.details['image'] = str(image) if image else None
        if not image:
            run.status, run.reason = FAILED, 'image export failed'


def _scenario_checks(name, state, config_path, out, artefacts):
    """Invariant checks and the reconstruction run of one scenario; checks
    that cannot run carry the scenario's status and reason."""
    status, payload = state
    sc, adapter = payload if status == PASSED else (None, None)
    gate = {'status': status, 'reason': '' if status == PASSED else payload}
    b = adapter.belief if adapter is not None else None
    table = [
        (check_covariance_selection, {name: b}),
        (check_neighbour_sets, {name: b}),
    ]
    if name == OPEN_SCENARIO or (b is not None and len(b.walls) == 0):
        table.append((check_no_wall_reduction, b))
    if name == SESSION_SCENARIO:
        table.append((check_batching, sc, config_path))
        table.append((check_session_replay, sc, config_path, out))
    table.append((reconstruction_run, sc, adapter, out, artefacts))
    checks = [run_check(name, *t, **gate) for t in table]
    return checks[:-1], checks[-1]


def small_geometry_checks() -> tuple[dict, list]:
    """The small geometries and their checks. If building them fails, the
    checks that need them fail with the message; the others still run."""
    names = ('open_tank', 'single_wall_coarse_grid', 'closed_box_components')
    try:
        small, gate = small_geometry_beliefs(), {}
    except Exception as exc:  # noqa: BLE001 - report, run the others
        small = dict.fromkeys(names)
        reason = f'small geometries: {type(exc).__name__}: {exc}'
        gate = {'status': FAILED, 'reason': reason}
    coarse, boxes = (
        small['single_wall_coarse_grid'],
        small['closed_box_components'],
    )
    pair = {'single_wall_coarse_grid': coarse, 'closed_box_components': boxes}
    table = [
        ('small geometries', check_small_covariance_selection, pair),
        ('open_tank', check_no_wall_reduction, small['open_tank']),
        ('closed_box_components', check_disconnected_components, boxes),
        ('single_wall_coarse_grid', check_mixed_joint_covariance, coarse),
        ('small geometries', check_neighbour_sets, pair),
        ('single_wall_coarse_grid', check_repeated_measurements, coarse),
        ('single_wall_coarse_grid', check_dense_oracle, coarse),
    ]
    checks = [run_check('synthetic segments', check_visibility_predicate)]
    checks += [run_check(*t, **gate) for t in table]
    checks.append(run_check('path 1-2-3, rho = 0.5', check_noise_order))
    return small, checks


def _summary(checks) -> dict:
    return {
        'passed': [c.name for c in checks if c.status == PASSED],
        'failed': [
            {'name': c.name, 'reason': c.reason}
            for c in checks
            if c.status == FAILED
        ],
        'not_run': [
            {'name': c.name, 'reason': c.reason}
            for c in checks
            if c.status == NOT_RUN
        ],
    }


def _markdown(report: dict) -> str:
    s = report['summary']
    lines = [
        '# visGP diagnostic',
        '',
        f'Created {report["created_utc"]}; '
        f'settings: {json.dumps(report["settings"])}',
        '',
        '## Summary',
        '',
        f'- passed ({len(s["passed"])}): ' + ', '.join(s['passed']),
    ]
    for key in ('failed', 'not_run'):
        lines.append(
            f'- {key.replace("_", " ")} ({len(s[key])})'
            + ''.join(f'\n  - {e["name"]}: {e["reason"]}' for e in s[key])
        )
    for title, key in (
        ('Model invariants', 'invariants'),
        (
            'Reconstruction quality (reported only, not judged)',
            'reconstruction',
        ),
    ):
        lines += ['', f'## {title}', '']
        for chk in report[key]:
            reason = f' ({chk["reason"]})' if chk['reason'] else ''
            lines.append(f'- **{chk["status"].upper()}** {chk["name"]}{reason}')
            for label, d in chk['details'].items():
                if isinstance(d, dict) and 'ok' in d:
                    v = f': {json.dumps(d["value"])}' if 'value' in d else ''
                    lines.append(
                        f'  - {"ok" if d["ok"] else "FAILED"} {label}{v}'
                    )
                else:
                    lines.append(f'  - {label}: {json.dumps(d)}')
    return '\n'.join(lines) + '\n'


def _scenario_states(spec_list: str, config_path, config_error: str) -> dict:
    """name -> (PASSED, (scenario, visgp adapter)) or (status, reason)."""
    states = {
        name: (NOT_RUN, f'scenario {name} not selected (--scenarios)')
        for name in DEFAULT_SCENARIOS
    }
    for spec in [s.strip() for s in spec_list.split(',') if s.strip()]:
        try:
            sc = Scenario(resolve_scenario(spec))
        except Exception as exc:  # noqa: BLE001
            states[spec] = (FAILED, f'cannot load: {type(exc).__name__}: {exc}')
            continue
        name = sc.path.stem
        if config_error:
            states[name] = (NOT_RUN, config_error)
            continue
        print(f'[visgp_diagnostics] building visGP: {name}', file=sys.stderr)
        adapter = make_backends(['visgp'], sc.domain, visgp_config=config_path)
        reason = getattr(adapter['visgp'], 'unavailable_reason', None)
        states[name] = (
            (FAILED, f'backend visgp unavailable: {reason}')
            if reason
            else (PASSED, (sc, adapter['visgp']))
        )
    return states


def _report(stamp, config_path, invariants, runs) -> dict:
    summary = _summary(invariants + [c for _, c in runs])
    config = str(Path(config_path).resolve()) if config_path else None
    return {
        'created_utc': stamp,
        'settings': {
            'small_geometries': {
                'signal_variance': S2,
                'lengthscale_m': ELL,
                'measurement_noise_variance': NOISE,
                'prior_mean': MU,
            },
            'scenario_visgp_config': config,
            'unit': 'scenario unit (V for the test scenarios)',
        },
        'exit_code': 1 if summary['failed'] else 3 if summary['not_run'] else 0,
        'summary': summary,
        'invariants': [vars(c) for c in invariants],
        'reconstruction': [vars(c) for _, c in runs],
    }


def _write_report(out: Path, report: dict) -> None:
    (out / 'report.json').write_text(json.dumps(report, indent=2, default=str))
    (out / 'report.md').write_text(_markdown(report))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog='python -m belief_playground.visgp_diagnostics'
    )
    p.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help='output directory (default: ./belief_playground_sessions/'
        'visgp_diagnostics/<UTC time>)',
    )
    p.add_argument(
        '--scenarios',
        default=','.join(DEFAULT_SCENARIOS),
        help='comma-separated scenario names or TOML paths (default: '
        '%(default)s); checks of unselected default scenarios are reported '
        'as not run',
    )
    p.add_argument(
        '--visgp-config',
        type=Path,
        default=None,
        help='visGP parameter YAML for the scenario runs (default: '
        'config/belief_params_visgp.yaml of the checkout, else of the '
        'installed share)',
    )
    p.add_argument(
        '--export-image',
        action='store_true',
        help='after writing the reports, export a PNG of the playground '
        'window per scenario (Qt; needs a display)',
    )
    args = p.parse_args(argv)
    if args.export_image and not _display_available():
        p.error(
            '--export-image needs a display: DISPLAY and WAYLAND_DISPLAY are '
            'not set and QT_QPA_PLATFORM is not one of '
            f'{", ".join(DISPLAYLESS_QT_PLATFORMS)}; run it under xvfb-run '
            '(Qt was not started)'
        )
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.out_dir or (
        Path('belief_playground_sessions') / 'visgp_diagnostics' / stamp
    )
    out.mkdir(parents=True, exist_ok=True)
    config_path, config_error = args.visgp_config, ''
    if config_path is None:
        try:
            config_path = default_visgp_config(installed=False)
        except ResourceError as exc:
            config_error = f'no visGP config: {exc}'

    states = _scenario_states(args.scenarios, config_path, config_error)
    small, invariants = small_geometry_checks()
    artefacts = {'arrays': {}, 'sessions': {}}
    runs = []
    try:
        for name, state in states.items():
            print(f'[visgp_diagnostics] checks: {name}', file=sys.stderr)
            checks, run = _scenario_checks(
                name, state, config_path, out, artefacts
            )
            invariants += checks
            runs.append((name, run))
        for name, b in small.items():
            for key in MODEL_ARRAYS if b is not None else ():
                artefacts['arrays'][f'{name}_{key}'] = getattr(b, key)
        np.savez_compressed(out / 'arrays.npz', **artefacts['arrays'])
        report = _report(stamp, config_path, invariants, runs)
        _write_report(out, report)  # before any Qt use
        if args.export_image:
            _export_images(runs, artefacts['sessions'], out)
            report = _report(stamp, config_path, invariants, runs)
            _write_report(out, report)
    finally:
        for s in artefacts['sessions'].values():
            s.close()
    for chk in invariants + [c for _, c in runs]:
        reason = f': {chk.reason}' if chk.reason else ''
        print(f'{chk.status.upper():8s} {chk.name}{reason}')
    print(f'output: {out.resolve()}')
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
