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

"""Scenario access for the playground (controller/evaluation side only).

Wraps scalar_field_sim's own loader. The ground-truth field is reachable only
through `Scenario.truth()`; backends receive a `Domain` (bounds, walls, unit)
and never the field.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scalar_field_sim.config import load_field_from_toml

UNKNOWN_UNIT = 'unknown unit'

Wall = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class Domain:
    """What backends may know: bounds [m], known walls [m], value unit."""

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    walls: tuple[Wall, ...]
    unit: str

    def contains(self, x: float, y: float) -> bool:
        return (
            bool(np.isfinite(x) and np.isfinite(y))
            and self.x_range[0] <= x <= self.x_range[1]
            and self.y_range[0] <= y <= self.y_range[1]
        )


@dataclass(frozen=True)
class DisplayGrid:
    """Cell-centred display/evaluation grid.

    `xs` (nx,) and `ys` (ny,) are cell centres; `xy` is (ny*nx, 2) built with
    `np.meshgrid(xs, ys)`, so `values.reshape(ny, nx)[j, i]` belongs to
    `(xs[i], ys[j])` (row = y). With `imshow(origin='lower',
    extent=(x_min, x_max, y_min, y_max))` every pixel centre is exactly the
    queried position.
    """

    xs: np.ndarray
    ys: np.ndarray
    xy: np.ndarray
    extent: tuple[float, float, float, float]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.ys), len(self.xs))

    def to_image(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(self.shape)


def make_display_grid(
    x_range: tuple[float, float], y_range: tuple[float, float], step: float
) -> DisplayGrid:
    """Grid with about `step` [m] spacing; cells tile the domain exactly."""
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f'grid step must be positive, got {step}')
    (x0, x1), (y0, y1) = x_range, y_range
    nx = max(2, int(round((x1 - x0) / step)))
    ny = max(2, int(round((y1 - y0) / step)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    gx, gy = np.meshgrid(xs, ys)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    return DisplayGrid(xs=xs, ys=ys, xy=xy, extent=(x0, x1, y0, y1))


class Scenario:
    """A scalar_field_sim scenario loaded with `load_field_from_toml`."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        # File contents are captured once; metadata and hash checks use these
        # bytes, not whatever is on disk at save/load time.
        self._file_bytes = self.path.read_bytes()
        self.config, self._field = load_field_from_toml(self.path)
        self._field_file_bytes = (
            Path(self.config.field_file).read_bytes()
            if self.config.field_file is not None
            else None
        )
        h = hashlib.sha256(self._file_bytes)
        if self._field_file_bytes is not None:
            h.update(self._field_file_bytes)
        self._content_hash = h.hexdigest()
        geom = self.config.geometry
        walls = tuple(
            (tuple(map(float, w.start)), tuple(map(float, w.end)))
            for w in geom.walls
        )
        self.domain = Domain(
            x_range=tuple(map(float, geom.x_range)),
            y_range=tuple(map(float, geom.y_range)),
            walls=walls,
            unit=self.config.field_value_unit or UNKNOWN_UNIT,
        )

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def clip_range(self) -> tuple[float, float] | None:
        return self.config.clip_range

    def truth(self, xy: np.ndarray) -> np.ndarray:
        """Noiseless latent field at (N, 2) positions (controller only)."""
        return self._field.evaluate(np.asarray(xy, dtype=float))

    def content_hash(self) -> str:
        """sha256 over the scenario file and its referenced field file, as
        read when the scenario was loaded."""
        return self._content_hash

    def record(self) -> dict:
        """Scenario reference and full file contents for session metadata."""
        rec = {
            'path': str(self.path),
            'name': self.config.name,
            'content_sha256': self.content_hash(),
            'file_sha256': hashlib.sha256(self._file_bytes).hexdigest(),
            'file_text': self._file_bytes.decode(),
            'field_file': None,
            'x_range': list(self.domain.x_range),
            'y_range': list(self.domain.y_range),
            'walls': [list(w[0]) + list(w[1]) for w in self.domain.walls],
            'field_value_unit': self.config.field_value_unit,
            'clip_range': (
                list(self.clip_range) if self.clip_range is not None else None
            ),
            'scenario_measurement_noise_std': self.config.measurement_noise_std,
            'scenario_seed': self.config.seed,
            'scenario_visualization_grid_step': (
                self.config.visualization_grid_step
            ),
        }
        if self.config.field_file is not None:
            rec['field_file'] = {
                'path': str(self.config.field_file),
                'sha256': hashlib.sha256(self._field_file_bytes).hexdigest(),
                'text': self._field_file_bytes.decode(),
            }
        return rec
