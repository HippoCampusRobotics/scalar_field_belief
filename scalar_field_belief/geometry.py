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

"""Static wall geometry for the wall-aware scalar field belief.

This module defines small immutable containers for straight wall segments and
a loader for TOML geometry files. It contains no GP logic and no ROS
dependencies.

All coordinates are physical coordinates in meters and must refer to the same
frame as the measurements and queries of the belief (the configured
`frame_id`). No frame transformation is performed here.

TOML schema
-----------
A geometry file (or a self-contained scalar_field_sim scenario) defines the
geometry inline:

.. code-block:: toml

    [general]
    name = "tank_2x4_two_walls"      # optional, defaults to the file stem

    [bounds]
    x_min = 0.0
    x_max = 2.0
    y_min = 0.0
    y_max = 4.0

    [objects.walls]
    wall_vertices = [
        [1.0, 0.6, 1.0, 3.4],        # [x1, y1, x2, y2]
    ]

Other sections (for example source descriptions) are ignored.

A scalar_field_sim scenario in reference form instead contains a top-level
`geometry_file` key; the geometry is then read from that file:

.. code-block:: toml

    geometry_file = "../geometry/tank_2x4_two_walls.toml"
    field_file = "../fields/test_four_sources.toml"   # ignored here

The reference is resolved relative to the directory of the referencing file
(absolute paths are used as given) and must not lead to another reference. A
file with `field_file` but without `geometry_file` is rejected.
"""

from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


def _as_finite_pair(value, name: str) -> tuple[float, float]:
    """Convert a two-element sequence to a tuple of two finite floats."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f'{name} must be a sequence of two numbers.')
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a sequence of two numbers.') from exc
    if len(items) != 2:
        raise ValueError(
            f'{name} must contain exactly two numbers, got {len(items)}.'
        )
    if any(isinstance(item, bool) for item in items):
        raise ValueError(f'{name} must contain only numbers, not booleans.')
    try:
        pair = (float(items[0]), float(items[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain only numbers.') from exc
    if not (math.isfinite(pair[0]) and math.isfinite(pair[1])):
        raise ValueError(f'{name} must contain only finite values.')
    return pair


@dataclass(frozen=True)
class WallSegment:
    """Straight wall segment between two physical points.

    Parameters
    ----------
    start
        Physical start point `(x, y)` in meters.
    end
        Physical end point `(x, y)` in meters.

    Raises
    ------
    ValueError
        If an endpoint is not a pair of finite numbers or if the segment has
        zero length.

    Notes
    -----
    The direction from `start` to `end` defines the sign of the wall feature:
    with unit tangent `t = (end - start) / L`, the normal is
    `n = (-t_y, t_x)`, so points to the left of the directed segment have a
    positive signed distance.
    """

    start: tuple[float, float]
    end: tuple[float, float]

    def __post_init__(self) -> None:
        start = _as_finite_pair(self.start, 'WallSegment start')
        end = _as_finite_pair(self.end, 'WallSegment end')
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(
                'WallSegment must have a finite, nonzero length; got '
                f'start={start}, end={end}.'
            )
        object.__setattr__(self, 'start', start)
        object.__setattr__(self, 'end', end)


@dataclass(frozen=True)
class WallGeometry:
    """Static set of wall segments inside a rectangular physical domain.

    Parameters
    ----------
    name
        Human-readable geometry name.
    x_range, y_range
        Physical domain bounds `(min, max)` in meters. They must be finite and
        strictly increasing.
    walls
        Wall segments. Their order is kept and defines the order of the wall
        feature columns.

    Raises
    ------
    ValueError
        If the bounds are invalid.
    TypeError
        If `name` is not a string or a wall is not a `WallSegment`.

    Notes
    -----
    An empty `walls` tuple is a valid geometry. The wall-aware belief rejects
    it explicitly.
    """

    name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    walls: tuple[WallSegment, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError('WallGeometry name must be a string.')

        x_range = _as_finite_pair(self.x_range, 'WallGeometry x_range')
        y_range = _as_finite_pair(self.y_range, 'WallGeometry y_range')
        if x_range[1] <= x_range[0]:
            raise ValueError('WallGeometry x_range must be increasing.')
        if y_range[1] <= y_range[0]:
            raise ValueError('WallGeometry y_range must be increasing.')

        walls = tuple(self.walls)
        for index, wall in enumerate(walls):
            if not isinstance(wall, WallSegment):
                raise TypeError(
                    f'WallGeometry walls[{index}] must be a WallSegment, '
                    f'got {type(wall).__name__}.'
                )

        object.__setattr__(self, 'x_range', x_range)
        object.__setattr__(self, 'y_range', y_range)
        object.__setattr__(self, 'walls', walls)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _get_table(parent: dict, key: str, path: Path) -> dict:
    table = parent.get(key, {})
    if not isinstance(table, dict):
        raise ValueError(f'{path}: [{key}] must be a table.')
    return table


def load_wall_geometry_from_toml(path: str | os.PathLike) -> WallGeometry:
    """Load the wall geometry of a geometry or scenario TOML file.

    Parameters
    ----------
    path
        Path to a geometry file, a self-contained scenario, or a scenario
        with a top-level `geometry_file` reference. It is used as given;
        relative paths are interpreted relative to the current working
        directory.

    Returns
    -------
    WallGeometry
        Geometry with walls in file order. If `[objects.walls]` or
        `wall_vertices` is absent, the geometry has no walls. With a
        `geometry_file` reference, name, bounds, and walls come from the
        referenced file; all other content of `path` is ignored.

    Raises
    ------
    OSError
        If the file cannot be opened.
    FileNotFoundError
        If a referenced geometry file does not exist.
    tomllib.TOMLDecodeError
        If a file is not valid TOML.
    ValueError
        If required bounds are missing, any value is invalid, `path` defines
        `field_file` without `geometry_file`, or a referenced geometry file
        contains `geometry_file` or `field_file`.
    """
    path = Path(path)
    with path.open('rb') as file:
        data = tomllib.load(file)

    if 'geometry_file' in data:
        scenario_path = path
        path = _referenced_geometry_path(path, data['geometry_file'])
        with path.open('rb') as file:
            data = tomllib.load(file)
        nested = sorted({'geometry_file', 'field_file'} & set(data))
        if nested:
            raise ValueError(
                f'Geometry file {path} referenced by {scenario_path} must '
                f'not contain {nested}.'
            )
    elif 'field_file' in data:
        raise ValueError(
            f'{path} defines field_file but no geometry_file; a scenario in '
            'reference form needs geometry_file for its geometry.'
        )

    return _geometry_from_data(data, path)


def _referenced_geometry_path(path: Path, ref) -> Path:
    """Resolve a `geometry_file` reference of `path` to an existing file."""
    if not isinstance(ref, str) or not ref:
        raise ValueError(f'{path}: geometry_file must be a non-empty string.')
    resolved = Path(ref)
    if not resolved.is_absolute():
        resolved = path.parent / resolved
    if not resolved.is_file():
        raise FileNotFoundError(
            f'Geometry file {resolved} referenced by {path} not found.'
        )
    return resolved


def _geometry_from_data(data: dict, path: Path) -> WallGeometry:
    """Build a `WallGeometry` from parsed TOML `data` read from `path`."""
    general = _get_table(data, 'general', path)
    name = general.get('name', path.stem)
    if not isinstance(name, str):
        raise ValueError(f'{path}: general.name must be a string.')

    bounds = _get_table(data, 'bounds', path)
    bound_values = {}
    for key in ('x_min', 'x_max', 'y_min', 'y_max'):
        if key not in bounds:
            raise ValueError(f'{path}: missing bounds.{key}.')
        value = bounds[key]
        if not _is_number(value) or not math.isfinite(value):
            raise ValueError(f'{path}: bounds.{key} must be a finite number.')
        bound_values[key] = float(value)

    objects = _get_table(data, 'objects', path)
    walls_table = objects.get('walls', {})
    if not isinstance(walls_table, dict):
        raise ValueError(f'{path}: [objects.walls] must be a table.')
    rows = walls_table.get('wall_vertices', [])
    if not isinstance(rows, list):
        raise ValueError(f'{path}: wall_vertices must be an array.')

    walls = []
    for index, row in enumerate(rows):
        if (
            not isinstance(row, list)
            or len(row) != 4
            or not all(_is_number(v) and math.isfinite(v) for v in row)
        ):
            raise ValueError(
                f'{path}: wall_vertices[{index}] must be four finite numbers '
                f'[x1, y1, x2, y2], got {row!r}.'
            )
        x1, y1, x2, y2 = (float(v) for v in row)
        try:
            walls.append(WallSegment(start=(x1, y1), end=(x2, y2)))
        except ValueError as exc:
            raise ValueError(f'{path}: wall_vertices[{index}]: {exc}') from exc

    try:
        return WallGeometry(
            name=name,
            x_range=(bound_values['x_min'], bound_values['x_max']),
            y_range=(bound_values['y_min'], bound_values['y_max']),
            walls=tuple(walls),
        )
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc
