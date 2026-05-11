from __future__ import annotations

import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


def make_axis(start: float, stop: float, step: float) -> np.ndarray:
    if stop <= start:
        raise ValueError("Axis stop must be larger than start.")
    if step <= 0.0:
        raise ValueError("Axis step must be positive.")
    n_full_steps = int(np.floor((stop - start) / step))
    axis = start + step * np.arange(n_full_steps + 1, dtype=np.float64)
    if not np.isclose(axis[-1], stop):
        axis = np.append(axis, stop)
    return axis


def make_grid_positions(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_step: float,
) -> np.ndarray:
    x = make_axis(x_min, x_max, grid_step)
    y = make_axis(y_min, y_max, grid_step)
    X, Y = np.meshgrid(x, y)
    return np.column_stack([X.ravel(), Y.ravel()])


def make_field_pointcloud2(
    positions_xy: np.ndarray,
    values: np.ndarray,
    frame_id: str,
    stamp,
    z_mode: str = "flat",
    height_scale: float = 1.0,
) -> PointCloud2:
    positions_xy = np.asarray(positions_xy, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if positions_xy.ndim != 2 or positions_xy.shape[1] != 2:
        raise ValueError("positions_xy must have shape (N, 2).")
    if len(positions_xy) != len(values):
        raise ValueError("positions_xy and values must have matching length.")

    if z_mode == "flat":
        z = np.zeros(len(values), dtype=np.float32)
    elif z_mode == "height":
        vmin = float(values.min())
        vmax = float(values.max())
        denom = max(vmax - vmin, 1e-12)
        z = ((values - vmin) / denom * height_scale).astype(np.float32)
    else:
        raise ValueError("z_mode must be 'flat' or 'height'.")

    points = np.column_stack(
        [positions_xy[:, 0], positions_xy[:, 1], z, values]
    ).tolist()
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return point_cloud2.create_cloud(header, fields, points)
