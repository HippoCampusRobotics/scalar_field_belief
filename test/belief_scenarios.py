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

"""Deterministic domain, base configuration, measurements and query points
shared by the target transform tests."""

from __future__ import annotations

import numpy as np

# Physical domain (deliberately not the unit square and not origin-centered).
DOMAIN = dict(x_min=-1.0, x_max=2.0, y_min=0.5, y_max=2.0)

# Base configuration with the original fields only (no target_* fields).
BASE_CONFIG = dict(
    frame_id='map',
    **DOMAIN,
    kernel_type='rbf',
    training_iter=20,
    learning_rate=0.1,
    init_lengthscale_x=0.3,
    init_lengthscale_y=0.25,
    init_outputscale=1.0,
    init_noise=0.04,
    refit_policy='every_k_measurements',
    refit_every_k=1,
    optimize_hyperparameters=False,
    publish_visualization=False,
    visualization_grid_step=0.1,
    visualization_z_mode='flat',
    visualization_height_scale=1.0,
)


def field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth synthetic field in physical units."""
    return 0.002 + 0.001 * (
        np.sin(2.0 * x) * np.cos(3.0 * y) + 0.5 * np.exp(-((x - 0.5) ** 2))
    )


def make_points(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xy = np.column_stack(
        [
            rng.uniform(DOMAIN['x_min'], DOMAIN['x_max'], n),
            rng.uniform(DOMAIN['y_min'], DOMAIN['y_max'], n),
        ]
    )
    vals = field(xy[:, 0], xy[:, 1]) + 5e-5 * rng.standard_normal(n)
    return xy, vals


def query_points() -> tuple[np.ndarray, np.ndarray]:
    """Fixed query grid (5x5) and a small set for full covariances."""
    gx = np.linspace(DOMAIN['x_min'], DOMAIN['x_max'], 5)
    gy = np.linspace(DOMAIN['y_min'], DOMAIN['y_max'], 5)
    grid = np.array([[a, b] for a in gx for b in gy], dtype=float)
    cov_pts = np.array(
        [
            [-0.7, 0.6],
            [0.0, 1.0],
            [0.4, 1.3],
            [1.1, 1.9],
            [1.8, 0.9],
            [3.0, 3.0],
        ],
        dtype=float,
    )
    return grid, cov_pts
