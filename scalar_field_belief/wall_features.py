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

"""Smooth wall features for the wall-aware scalar field belief.

Each wall segment contributes one dimensionless feature that is appended to
the normalized spatial GP inputs. For a segment with start `a`, end `b`,
length `L = norm(b - a)`, unit tangent `t = (b - a) / L`, and normal
`n = (-t_y, t_x)`, the feature at a physical point `x` is

    s_m(x) = dot(x - a, n)                      signed distance to the line
    u      = clip(dot(x - a, t), 0, L)          clamped projection
    p      = a + u * t                          closest point on the segment
    d_m(x) = norm(x - p)                        distance to the segment
    h_m(x) = tanh(beta * s_m(x)) * exp(-d_m(x)**2 / (2 * rho**2))

The features are computed from physical coordinates in meters. `beta` has
units 1/m and `rho` has units m; both are fixed configuration values, not
learned parameters.

This module depends only on NumPy and the geometry containers.
"""

from __future__ import annotations

import math

import numpy as np

from scalar_field_belief.geometry import WallGeometry


class WallFeatureMap:
    """Map physical positions to one smooth feature per wall segment.

    Parameters
    ----------
    geometry
        Static wall geometry. The feature columns follow the order of
        `geometry.walls`.
    beta
        Sharpness of the sign transition across the wall line in 1/m.
    rho
        Spatial decay length of the feature away from the segment in m.

    Raises
    ------
    TypeError
        If `geometry` is not a `WallGeometry`.
    ValueError
        If `beta` or `rho` is not positive and finite.

    Notes
    -----
    An empty geometry yields features of shape `(N, 0)`.
    """

    def __init__(
        self,
        geometry: WallGeometry,
        beta: float = 1.0,
        rho: float = 1.0,
    ):
        if not isinstance(geometry, WallGeometry):
            raise TypeError('geometry must be a WallGeometry.')
        if isinstance(beta, bool) or isinstance(rho, bool):
            raise ValueError('beta and rho must be numbers, not booleans.')
        beta = float(beta)
        rho = float(rho)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError('beta must be positive and finite.')
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError('rho must be positive and finite.')

        self.geometry = geometry
        self.beta = beta
        self.rho = rho

        num_walls = len(geometry.walls)
        starts = np.array(
            [wall.start for wall in geometry.walls], dtype=np.float64
        ).reshape(num_walls, 2)
        ends = np.array(
            [wall.end for wall in geometry.walls], dtype=np.float64
        ).reshape(num_walls, 2)
        deltas = ends - starts
        lengths = np.linalg.norm(deltas, axis=1)
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
            raise ValueError('All wall segments must have nonzero length.')
        tangents = deltas / lengths[:, None]
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

        # Per-wall cached quantities, shapes (M, 2) and (M,).
        self._starts = starts
        self._tangents = tangents
        self._normals = normals
        self._lengths = lengths

    @property
    def num_walls(self) -> int:
        """Number of walls `M`, equal to the number of feature columns."""
        return int(self._lengths.shape[0])

    def transform_xy_to_wall_features(self, xy_phys: np.ndarray) -> np.ndarray:
        """Evaluate the wall features at physical positions.

        Parameters
        ----------
        xy_phys
            Physical positions in meters with shape `(N, 2)`, or one point
            with shape `(2,)`.

        Returns
        -------
        np.ndarray
            Float64 features of shape `(N, M)`, one column per wall in
            geometry order. A single point returns shape `(1, M)`.

        Raises
        ------
        ValueError
            If `xy_phys` has the wrong shape or contains non-finite values.
        """
        xy = np.asarray(xy_phys, dtype=np.float64)
        if xy.shape == (2,):
            xy = xy.reshape(1, 2)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(
                'Expected xy_phys to have shape (N, 2) or (2,), '
                f'got {tuple(xy.shape)}'
            )
        if not np.all(np.isfinite(xy)):
            raise ValueError('xy_phys must contain only finite values.')

        # Offsets from every wall start, shape (N, M, 2).
        offsets = xy[:, None, :] - self._starts[None, :, :]

        signed_dist = np.einsum('nmk,mk->nm', offsets, self._normals)
        along = np.einsum('nmk,mk->nm', offsets, self._tangents)
        along = np.clip(along, 0.0, self._lengths[None, :])

        closest = (
            self._starts[None, :, :] + along[:, :, None] * self._tangents[None]
        )
        seg_dist = np.linalg.norm(xy[:, None, :] - closest, axis=2)

        return np.tanh(self.beta * signed_dist) * np.exp(
            -(seg_dist**2) / (2.0 * self.rho**2)
        )
