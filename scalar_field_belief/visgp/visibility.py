# Visibility-graph GP (visGP) after Gilbert and Datta (2024), Biometrics:
# https://academic.oup.com/biometrics/article/doi/10.1093/biomtc/ujae089/7753413
# Authors' R implementation: https://github.com/bjg345/visGP

"""Visibility between positions separated by zero-width wall segments.

Walls are finite straight segments without thickness (no buffering). With
the contact tolerance `delta` (`CONTACT_TOLERANCE_FACTOR` times the larger
domain extent, in meters):

* a position is *on a wall* if its distance to a wall is at most `delta`;
* two positions `p`, `q` are *visible* to each other if, for every wall, the
  distance between the segment `[p, q]` and the wall is larger than `delta`.

The segment distance is 0 for a proper crossing (all four orientation signs
strictly opposite), otherwise the smallest of the four endpoint-to-segment
distances, which is exact for non-crossing segments in the plane. Crossing,
touching (including contact at a wall endpoint) and collinear overlap are
therefore blocked. An orientation sign that roundoff could flip belongs to a
near-degenerate configuration whose endpoint distance is far below `delta`, so
those cases are blocked as well. Pairs and wall endpoints are put into a
canonical (lexicographic) order before evaluation, so the result is exactly
symmetric in `(p, q)` and independent of the wall endpoint order; it is also
independent of the wall order. The rectangular domain boundary is not a wall.

Everything is elementwise NumPy over pairs, so a pair's result never depends
on the other pairs of the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scalar_field_belief.geometry import WallGeometry

CONTACT_TOLERANCE_FACTOR = 1e-9  # [-], times the larger domain extent [m]

# Pairs per vectorised block in visibility_matrix (bounds temporary memory).
_PAIR_BLOCK = 65536


def contact_tolerance(geometry: WallGeometry) -> float:
    """Contact tolerance `delta` [m] of `geometry` (scale-aware)."""
    extent = max(
        geometry.x_range[1] - geometry.x_range[0],
        geometry.y_range[1] - geometry.y_range[0],
    )
    return CONTACT_TOLERANCE_FACTOR * extent


def wall_array(geometry: WallGeometry) -> np.ndarray:
    """Walls as a `(W, 4)` array `[x0, y0, x1, y1]` with the endpoints of each
    wall in lexicographic order (wall order kept)."""
    rows = []
    for wall in geometry.walls:
        a, b = sorted((tuple(wall.start), tuple(wall.end)))
        rows.append((a[0], a[1], b[0], b[1]))
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def as_positions(xy, name: str = 'xy') -> np.ndarray:
    """Validate an `(N, 2)` array of finite positions [m] (float64 copy)."""
    xy = np.array(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f'{name} must have shape (N, 2), got {xy.shape}.')
    if not np.all(np.isfinite(xy)):
        raise ValueError(f'{name} must contain only finite values.')
    return xy


def _orientation(ax, ay, bx, by, cx, cy):
    """Twice the signed area of the triangle (a, b, c)."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from p to the segment [a, b] (a == b allowed)."""
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    safe = np.where(denom > 0.0, denom, 1.0)
    t = np.where(denom > 0.0, ((px - ax) * abx + (py - ay) * aby) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _segment_distance(px, py, qx, qy, wall) -> np.ndarray:
    """Distance between the segments [p, q] and `wall` (canonical inputs)."""
    ax, ay, bx, by = wall
    s1 = np.sign(_orientation(px, py, qx, qy, ax, ay))
    s2 = np.sign(_orientation(px, py, qx, qy, bx, by))
    s3 = np.sign(_orientation(ax, ay, bx, by, px, py))
    s4 = np.sign(_orientation(ax, ay, bx, by, qx, qy))
    crossing = (s1 * s2 < 0) & (s3 * s4 < 0)
    d = np.minimum(
        np.minimum(
            _point_segment_distance(px, py, ax, ay, bx, by),
            _point_segment_distance(qx, qy, ax, ay, bx, by),
        ),
        np.minimum(
            _point_segment_distance(ax, ay, px, py, qx, qy),
            _point_segment_distance(bx, by, px, py, qx, qy),
        ),
    )
    return np.where(crossing, 0.0, d)


def point_wall_distance(xy: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """Distance [m] from each position to its nearest wall (inf: no walls)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    dist = np.full(len(xy), np.inf)
    for wall in walls:
        ax, ay, bx, by = wall
        dist = np.minimum(
            dist,
            _point_segment_distance(xy[:, 0], xy[:, 1], ax, ay, bx, by),
        )
    return dist


def on_wall_mask(
    xy: np.ndarray, walls: np.ndarray, tolerance: float
) -> np.ndarray:
    """True where a position is within `tolerance` [m] of a wall."""
    return point_wall_distance(xy, walls) <= tolerance


def inside_bounds_mask(xy: np.ndarray, geometry: WallGeometry) -> np.ndarray:
    """True where a position lies in the closed domain rectangle."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    (x0, x1), (y0, y1) = geometry.x_range, geometry.y_range
    return (
        (xy[:, 0] >= x0)
        & (xy[:, 0] <= x1)
        & (xy[:, 1] >= y0)
        & (xy[:, 1] <= y1)
    )


def visible_pairs(
    p: np.ndarray, q: np.ndarray, walls: np.ndarray, tolerance: float
) -> np.ndarray:
    """Visibility of the pairs `(p[i], q[i])`, both `(N, 2)` arrays [m]."""
    p = np.asarray(p, dtype=float).reshape(-1, 2)
    q = np.asarray(q, dtype=float).reshape(-1, 2)
    swap = (p[:, 0] > q[:, 0]) | ((p[:, 0] == q[:, 0]) & (p[:, 1] > q[:, 1]))
    lo = np.where(swap[:, None], q, p)
    hi = np.where(swap[:, None], p, q)
    visible = np.ones(len(p), dtype=bool)
    for wall in walls:
        d = _segment_distance(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1], wall)
        visible &= d > tolerance
    return visible


def visibility_matrix(
    a: np.ndarray, b: np.ndarray, walls: np.ndarray, tolerance: float
) -> np.ndarray:
    """`(len(a), len(b))` visibility between every position of `a` and `b`."""
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    out = np.ones((len(a), len(b)), dtype=bool)
    if len(walls) == 0 or out.size == 0:
        return out
    rows = max(1, _PAIR_BLOCK // max(len(b), 1))
    for start in range(0, len(a), rows):
        block = a[start : start + rows]
        p = np.repeat(block, len(b), axis=0)
        q = np.tile(b, (len(block), 1))
        out[start : start + len(block)] = visible_pairs(
            p, q, walls, tolerance
        ).reshape(len(block), len(b))
    return out


def visibility_graph(
    locations: np.ndarray, walls: np.ndarray, tolerance: float
) -> np.ndarray:
    """Adjacency `(m, m)` of the undirected visibility graph (no self loops)."""
    adjacency = visibility_matrix(locations, locations, walls, tolerance)
    np.fill_diagonal(adjacency, False)
    return adjacency


def connected_components(adjacency: np.ndarray) -> list[np.ndarray]:
    """Connected components as sorted index arrays, ordered by their smallest
    node index."""
    m = len(adjacency)
    label = np.full(m, -1)
    components = []
    for seed in range(m):
        if label[seed] >= 0:
            continue
        label[seed] = len(components)
        members = [seed]
        frontier = [seed]
        while frontier:
            nxt = np.flatnonzero(adjacency[frontier].any(axis=0) & (label < 0))
            label[nxt] = len(components)
            members.extend(nxt.tolist())
            frontier = nxt.tolist()
        components.append(np.array(sorted(members), dtype=np.intp))
    return components


@dataclass(frozen=True)
class ReferenceGrid:
    """Generated reference set: cell centres of an `nx` x `ny` grid.

    `locations` (m, 2) are the kept nodes in grid order (x fastest, then y),
    `lattice_index` (m, 2) their `(i, j)` grid indices, `dropped_on_wall` the
    number of grid nodes within the contact tolerance of a wall.
    """

    locations: np.ndarray
    lattice_index: np.ndarray
    nx: int
    ny: int
    dx: float
    dy: float
    spacing: float
    dropped_on_wall: int


def reference_grid(
    geometry: WallGeometry,
    spacing: float,
    max_nodes: int,
    walls: np.ndarray,
    tolerance: float,
) -> ReferenceGrid:
    """Cell-centred reference grid from the domain bounds and walls only.

    `nx = max(1, round((x_max - x_min) / spacing))` (same for y), node `(i, j)`
    at `(x_min + (i + 0.5) dx, y_min + (j + 0.5) dy)` with
    `dx = (x_max - x_min) / nx`. Nodes on a wall are dropped. Raises
    `ValueError` if `nx * ny` exceeds `max_nodes` or no node is left.
    """
    (x0, x1), (y0, y1) = geometry.x_range, geometry.y_range
    nx = max(1, int(round((x1 - x0) / spacing)))
    ny = max(1, int(round((y1 - y0) / spacing)))
    if nx * ny > max_nodes:
        raise ValueError(
            f'reference grid {nx} x {ny} = {nx * ny} nodes '
            f'(reference_spacing {spacing} m on {x1 - x0} m x {y1 - y0} m) '
            f'exceeds max_reference_nodes = {max_nodes}; increase '
            'reference_spacing or max_reference_nodes.'
        )
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    lattice = np.column_stack([ii.ravel(), jj.ravel()])
    xy = np.column_stack(
        [x0 + (lattice[:, 0] + 0.5) * dx, y0 + (lattice[:, 1] + 0.5) * dy]
    )
    keep = ~on_wall_mask(xy, walls, tolerance)
    if not keep.any():
        raise ValueError('every reference grid node lies on a wall.')
    return ReferenceGrid(
        locations=xy[keep],
        lattice_index=lattice[keep],
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        spacing=float(spacing),
        dropped_on_wall=int((~keep).sum()),
    )


def check_reference_locations(
    locations,
    geometry: WallGeometry,
    max_nodes: int,
    walls: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Validate explicit reference locations: finite, distinct (exact),
    inside the domain, off walls, at most `max_nodes`."""
    xy = as_positions(locations, 'reference_locations') + 0.0
    if len(xy) == 0:
        raise ValueError('reference_locations must not be empty.')
    if len(xy) > max_nodes:
        raise ValueError(
            f'{len(xy)} reference locations exceed max_reference_nodes = '
            f'{max_nodes}.'
        )
    _, first, counts = np.unique(
        xy, axis=0, return_index=True, return_counts=True
    )
    if np.any(counts > 1):
        dup = sorted(first[counts > 1].tolist())
        raise ValueError(
            'reference_locations contain duplicates, e.g. '
            f'{xy[dup[0]].tolist()} (first occurrences at rows {dup}).'
        )
    outside = np.flatnonzero(~inside_bounds_mask(xy, geometry))
    if outside.size:
        raise ValueError(
            f'reference location {xy[outside[0]].tolist()} (row '
            f'{outside[0]}) is outside the domain bounds.'
        )
    blocked = np.flatnonzero(on_wall_mask(xy, walls, tolerance))
    if blocked.size:
        raise ValueError(
            f'reference location {xy[blocked[0]].tolist()} (row '
            f'{blocked[0]}) lies on a wall (contact tolerance {tolerance} m).'
        )
    return xy
