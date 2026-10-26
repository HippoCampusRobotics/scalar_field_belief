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

"""Data side of the live viewer (no Qt, no rclpy).

The ROS thread writes decoded messages into a `LiveState`; the GUI polls it on
the Qt main thread. Every write increments a version counter, so the GUI can
skip redraws when nothing changed. Receive times use a monotonic wall clock,
independent of ROS (sim) time.

An estimator's visualization grid (scattered points on a regular, possibly
incomplete lattice) is turned into a `Lattice` when it is stored (ROS thread);
`lattice_to_image` maps it onto the playground's cell-centred `DisplayGrid`
by per-axis nearest node, without interpolation.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from belief_playground.scenario import DisplayGrid

STREAMS = ('mean', 'variance', 'pose', 'measurements')

DEFAULT_TRAJECTORY_MIN_STEP = 0.01  # [m]
DEFAULT_MAX_POINTS = 200_000


@dataclass(frozen=True)
class Lattice:
    """Cloud values on the cloud's own axes: `values[j, i]` belongs to
    `(xs[i], ys[j])`; NaN where the cloud has no point. `tol_x`/`tol_y` are
    the coordinate tolerances used to merge float32 round-off."""

    xs: np.ndarray
    ys: np.ndarray
    values: np.ndarray
    tol_x: float = 0.0
    tol_y: float = 0.0


@dataclass(frozen=True)
class CloudSample:
    """One decoded cloud: positions (N, 2) [m], values (N,) and the same
    values on the cloud lattice."""

    xy: np.ndarray
    values: np.ndarray
    lattice: Lattice
    frame_id: str
    received: float  # monotonic clock [s]


@dataclass(frozen=True)
class PoseSample:
    x: float
    y: float
    frame_id: str
    received: float


@dataclass(frozen=True)
class Measurement:
    x: float
    y: float
    value: float
    clipped: bool
    frame_id: str
    received: float


DEFAULT_METHOD = 'Belief'


@dataclass(frozen=True)
class MethodInfo:
    """The running belief method, as shown in the window."""

    name: str = DEFAULT_METHOD
    details: str = ''  # e.g. 'kernel rbf' or '41x81 nodes'
    node: str | None = None  # fully qualified node name
    note: str | None = None  # lookup state or problem
    publishers: int = 0  # publishers of the mean cloud


def method_name(params: dict | None, node: str | None = None):
    """(name, details) of a belief node from its parameters, named like the
    playground adapters: a node with `model_variant` is a GP ('GP
    baseline', 'GP wall-aware'; details: the kernel), one with `nx` and
    `ny` is the GMRF (details: its grid), any other node is shown by its
    name. Without parameters and node: ('Belief', '')."""
    params = params or {}
    if 'model_variant' in params:
        name = 'GP ' + str(params['model_variant']).replace('_', '-')
        kernel = params.get('kernel_type')
        return name, f'kernel {kernel}' if kernel else ''
    if 'nx' in params and 'ny' in params:
        return 'GMRF', f'{params["nx"]}x{params["ny"]} nodes'
    if node:
        return node.rstrip('/').rsplit('/', 1)[-1], ''
    return DEFAULT_METHOD, ''


@dataclass
class StreamStatus:
    """Receive statistics of one input stream."""

    topic: str | None = None  # resolved topic name; None = disabled
    count: int = 0  # accepted messages
    last_received: float | None = None
    frame_mismatch: int = 0
    last_bad_frame: str | None = None
    errors: int = 0  # malformed / undecodable messages
    last_error: str | None = None


class PointBuffer:
    """Rows of `columns` floats; keeps the newest `max_points` rows.

    Ring buffer that grows on demand up to `max_points`, then overwrites the
    oldest row, so that appending and copying stay cheap for long runs.
    """

    def __init__(self, columns: int, max_points: int = DEFAULT_MAX_POINTS):
        if int(max_points) < 1:
            raise ValueError(f'max_points must be >= 1, got {max_points}')
        self.columns = int(columns)
        self.max_points = int(max_points)
        self.clear()

    def clear(self) -> None:
        self._buf = np.empty((min(1024, self.max_points), self.columns))
        self._start = 0
        self._n = 0
        self.dropped = 0

    def __len__(self) -> int:
        return self._n

    def append(self, row) -> None:
        cap = len(self._buf)
        if self._n == cap and cap < self.max_points:
            grown = np.empty((min(2 * cap, self.max_points), self.columns))
            grown[: self._n] = self.array()
            self._buf, self._start, cap = grown, 0, len(grown)
        if self._n < cap:
            self._buf[(self._start + self._n) % cap] = row
            self._n += 1
        else:
            self._buf[self._start] = row
            self._start = (self._start + 1) % cap
            self.dropped += 1

    def array(self) -> np.ndarray:
        """Copy of the stored rows, oldest first, shape (n, columns)."""
        end = self._start + self._n
        if end <= len(self._buf):
            return self._buf[self._start : end].copy()
        return np.concatenate(
            [self._buf[self._start :], self._buf[: end - len(self._buf)]]
        )

    def last(self) -> np.ndarray | None:
        if not self._n:
            return None
        return self._buf[(self._start + self._n - 1) % len(self._buf)]


class TrajectoryBuffer(PointBuffer):
    """Travelled (x, y) path; a position is stored only if it is at least
    `min_step` [m] away from the previously stored one."""

    def __init__(
        self,
        min_step: float = DEFAULT_TRAJECTORY_MIN_STEP,
        max_points: int = DEFAULT_MAX_POINTS,
    ):
        if not (math.isfinite(min_step) and min_step >= 0.0):
            raise ValueError(
                f'min_step must be finite and >= 0, got {min_step}'
            )
        self.min_step = float(min_step)
        super().__init__(2, max_points)

    def add(self, x: float, y: float) -> bool:
        """Store (x, y) if finite and far enough from the last point."""
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        last = self.last()
        if last is not None and (
            math.hypot(x - last[0], y - last[1]) < self.min_step
        ):
            return False
        self.append((x, y))
        return True


@dataclass(frozen=True)
class LiveSnapshot:
    """Consistent copy of a `LiveState` for the GUI thread.

    The `*_version` fields change whenever the corresponding content changes.
    """

    version: int
    mean: CloudSample | None
    variance: CloudSample | None
    pose: PoseSample | None
    trajectory: np.ndarray  # (n, 2)
    measurements: np.ndarray  # (m, 4): x, y, value, clipped
    streams: dict[str, StreamStatus]
    mean_version: int
    variance_version: int
    pose_version: int
    trajectory_version: int
    measurements_version: int
    trajectory_dropped: int
    measurements_dropped: int
    ros_alive: bool = True
    ros_stopped_reason: str | None = None
    ros_errors: int = 0
    ros_last_error: str | None = None
    method_version: int = 0
    method: MethodInfo = MethodInfo()


class LiveState:
    """Thread-safe store written by the ROS thread, polled by the GUI.

    All setters check the message frame against `frame_id` first: an empty
    frame is accepted as `frame_id`; any other frame is counted per stream
    and the message is ignored (no TF).
    """

    def __init__(
        self,
        frame_id: str = 'map',
        trajectory_min_step: float = DEFAULT_TRAJECTORY_MIN_STEP,
        max_points: int = DEFAULT_MAX_POINTS,
        clock=time.monotonic,
    ):
        self.frame_id = frame_id
        self.clock = clock
        self._lock = threading.Lock()
        self._version = 0
        self._parts = dict.fromkeys(
            (
                'mean',
                'variance',
                'pose',
                'trajectory',
                'measurements',
                'method',
            ),
            0,
        )
        self._mean: CloudSample | None = None
        self._variance: CloudSample | None = None
        self._pose: PoseSample | None = None
        self._trajectory = TrajectoryBuffer(trajectory_min_step, max_points)
        self._measurements = PointBuffer(4, max_points)
        self._streams = {name: StreamStatus() for name in STREAMS}
        self._ros_alive = True
        self._ros_stopped_reason: str | None = None
        self._ros_errors = 0
        self._ros_last_error: str | None = None
        self._method = MethodInfo()

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def _bump(self, *parts: str) -> None:
        self._version += 1
        for part in parts:
            self._parts[part] += 1

    def set_topic(self, stream: str, topic: str | None) -> None:
        """Record the resolved topic of a stream (None: not subscribed)."""
        with self._lock:
            self._streams[stream].topic = topic
            self._bump()

    def _accept(self, stream: str, frame_id: str) -> tuple[str, float] | None:
        """(frame, receive time) if the frame matches; caller holds the
        lock. Mismatches are counted."""
        now = self.clock()
        if frame_id and frame_id != self.frame_id:
            st = self._streams[stream]
            st.frame_mismatch += 1
            st.last_bad_frame = frame_id
            self._bump()
            return None
        st = self._streams[stream]
        st.count += 1
        st.last_received = now
        return frame_id or self.frame_id, now

    def set_cloud(self, stream: str, xy, values, frame_id: str = '') -> bool:
        """Store the latest 'mean' or 'variance' cloud; False if ignored."""
        if stream not in ('mean', 'variance'):
            raise ValueError(f'not a cloud stream: {stream!r}')
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        values = np.asarray(values, dtype=float).reshape(-1)
        try:
            lattice = make_lattice(xy, values)
        except ValueError as exc:
            self.report_error(stream, f'{stream} cloud: {exc}')
            return False
        with self._lock:
            accepted = self._accept(stream, frame_id)
            if accepted is None:
                return False
            sample = CloudSample(xy, values, lattice, *accepted)
            if stream == 'mean':
                self._mean = sample
            else:
                self._variance = sample
            self._bump(stream)
            return True

    def set_pose(self, x: float, y: float, frame_id: str = '') -> bool:
        """Store the current position and extend the trajectory."""
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            self.report_error('pose', 'pose: non-finite position')
            return False
        with self._lock:
            accepted = self._accept('pose', frame_id)
            if accepted is None:
                return False
            self._pose = PoseSample(x, y, *accepted)
            if self._trajectory.add(x, y):
                self._bump('pose', 'trajectory')
            else:
                self._bump('pose')
            return True

    def add_measurement(
        self,
        x: float,
        y: float,
        value: float,
        clipped: bool = False,
        frame_id: str = '',
    ) -> bool:
        x, y, value = float(x), float(y), float(value)
        if not (math.isfinite(x) and math.isfinite(y)):
            self.report_error(
                'measurements', 'measurement: non-finite position'
            )
            return False
        with self._lock:
            if self._accept('measurements', frame_id) is None:
                return False
            self._measurements.append((x, y, value, float(bool(clipped))))
            self._bump('measurements')
            return True

    def report_error(self, stream: str, message: str) -> None:
        """Count a malformed / undecodable message of `stream`."""
        with self._lock:
            st = self._streams[stream]
            st.errors += 1
            st.last_error = message
            self._bump()

    def report_ros_error(self, message: str) -> None:
        """Count an exception raised while spinning the ROS node."""
        with self._lock:
            self._ros_errors += 1
            self._ros_last_error = message
            self._bump()

    def set_ros_stopped(self, reason: str) -> None:
        """The ROS thread stopped receiving (not by a requested shutdown)."""
        with self._lock:
            self._ros_alive = False
            self._ros_stopped_reason = reason
            self._bump()

    def set_method(self, method: MethodInfo) -> None:
        """Record the running belief method (only changes bump versions)."""
        with self._lock:
            if method != self._method:
                self._method = method
                self._bump('method')

    def clear_trajectory(self) -> None:
        with self._lock:
            self._trajectory.clear()
            self._bump('trajectory')

    def clear_measurements(self) -> None:
        """Forget the stored measurement positions (display only)."""
        with self._lock:
            self._measurements.clear()
            self._bump('measurements')

    def stream_status(self) -> dict[str, StreamStatus]:
        with self._lock:
            return {k: dataclasses.replace(v) for k, v in self._streams.items()}

    def snapshot(self, since: LiveSnapshot | None = None) -> LiveSnapshot:
        """Copy of the current state. Trajectory and measurement arrays whose
        part version equals the one in `since` are taken from `since`
        instead of being copied again (treat them as read-only)."""
        with self._lock:
            parts = self._parts
            if since is not None and (
                since.trajectory_version == parts['trajectory']
            ):
                trajectory = since.trajectory
            else:
                trajectory = self._trajectory.array()
            if since is not None and (
                since.measurements_version == parts['measurements']
            ):
                measurements = since.measurements
            else:
                measurements = self._measurements.array()
            return LiveSnapshot(
                version=self._version,
                mean=self._mean,
                variance=self._variance,
                pose=self._pose,
                trajectory=trajectory,
                measurements=measurements,
                streams={
                    k: dataclasses.replace(v) for k, v in self._streams.items()
                },
                mean_version=self._parts['mean'],
                variance_version=self._parts['variance'],
                pose_version=self._parts['pose'],
                trajectory_version=self._parts['trajectory'],
                measurements_version=self._parts['measurements'],
                method_version=self._parts['method'],
                method=self._method,
                trajectory_dropped=self._trajectory.dropped,
                measurements_dropped=self._measurements.dropped,
                ros_alive=self._ros_alive,
                ros_stopped_reason=self._ros_stopped_reason,
                ros_errors=self._ros_errors,
                ros_last_error=self._ros_last_error,
            )


def _cluster_axis(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Distinct coordinates of `v` (merging float32 round-off), the index of
    each value in them, and the merge tolerance."""
    tol = 1e-5 * max(1.0, float(np.max(np.abs(v))))
    order = np.sort(np.unique(v))
    starts = order[np.concatenate([[True], np.diff(order) > tol])]
    idx = np.searchsorted(starts, v + tol, side='right') - 1
    return starts, idx, tol


def make_lattice(xy, values) -> Lattice:
    """The cloud's distinct x and y coordinates as axes, its values on them.

    Points with non-finite positions are dropped; lattice nodes without a
    point (holes) are NaN. Raises ValueError if the lengths differ or if the
    points are clearly not on a regular lattice (more than about four
    lattice nodes per point).
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(xy) != len(values):
        raise ValueError(f'{len(xy)} positions but {len(values)} values')
    keep = np.all(np.isfinite(xy), axis=1)
    xy, values = xy[keep], values[keep]
    if not len(xy):
        return Lattice(np.empty(0), np.empty(0), np.empty((0, 0)))
    ax, ix, tol_x = _cluster_axis(xy[:, 0])
    ay, iy, tol_y = _cluster_axis(xy[:, 1])
    if len(ax) * len(ay) > 4 * len(xy) + 64:
        raise ValueError(
            f'cloud is not a regular lattice ({len(xy)} points with '
            f'{len(ax)} distinct x and {len(ay)} distinct y coordinates)'
        )
    grid = np.full((len(ay), len(ax)), np.nan)
    grid[iy, ix] = values
    return Lattice(ax, ay, grid, tol_x, tol_y)


def _axis_lookup(
    axis: np.ndarray, q: np.ndarray, tol: float, single_half: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest axis node for each query coordinate, and whether the query
    lies within half the larger gap next to that node (+ `tol`).

    Near-ties (within 1e-6 of the gap) go to the lower node, so that display
    cells centred between two nodes are assigned deterministically. An axis
    with a single node accepts queries within `single_half`.
    """
    q = np.asarray(q, dtype=float)
    if len(axis) == 1:
        idx = np.zeros(len(q), dtype=np.intp)
        return idx, np.abs(q - axis[0]) <= single_half + tol
    hi = np.clip(np.searchsorted(axis, q), 1, len(axis) - 1)
    lo = hi - 1
    gap = axis[hi] - axis[lo]
    pick_hi = (axis[hi] - q) < (q - axis[lo]) - 1e-6 * gap
    idx = np.where(pick_hi, hi, lo)
    gaps = np.diff(axis)
    half = 0.5 * np.maximum(
        np.concatenate([gaps[:1], gaps]), np.concatenate([gaps, gaps[-1:]])
    )
    return idx, np.abs(axis[idx] - q) <= half[idx] + tol


def lattice_to_image(lattice: Lattice, grid: DisplayGrid) -> np.ndarray:
    """Lattice values on the display grid, shape `grid.shape`.

    Each display cell takes the value of the nearest lattice node per axis,
    so every node is drawn as a block of display cells around it. Cells
    farther than half the larger node gap next to that node (per axis) are
    NaN, so nothing is extrapolated beyond the cloud; holes stay NaN. An
    axis with a single node uses half the other axis' median gap, or half
    the display cell size if both axes have a single node.
    """
    if lattice.values.size == 0:
        return np.full(grid.shape, np.nan)
    xs, ys = lattice.xs, lattice.ys

    def median_gap(axis):
        return float(np.median(np.diff(axis))) if len(axis) > 1 else None

    sx, sy = median_gap(xs), median_gap(ys)
    if sx is None and sy is None:
        (x0, x1, y0, y1) = grid.extent
        sx, sy = (x1 - x0) / len(grid.xs), (y1 - y0) / len(grid.ys)
    jx, ok_x = _axis_lookup(xs, grid.xs, lattice.tol_x, 0.5 * (sx or sy))
    jy, ok_y = _axis_lookup(ys, grid.ys, lattice.tol_y, 0.5 * (sy or sx))
    img = lattice.values[np.ix_(jy, jx)]
    img[~ok_y, :] = np.nan
    img[:, ~ok_x] = np.nan
    return img


def cloud_to_image(xy, values, grid: DisplayGrid) -> np.ndarray:
    """`lattice_to_image(make_lattice(xy, values), grid)`."""
    return lattice_to_image(make_lattice(xy, values), grid)
