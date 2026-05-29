from __future__ import annotations

import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from matplotlib import colormaps


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


def _values_to_rgb_uint32(
    values: np.ndarray,
    color_min: float,
    color_max: float,
    cmap_name: str = "viridis",
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)

    if color_max <= color_min:
        raise ValueError("color_max must be greater than color_min.")

    normalized = (values - color_min) / (color_max - color_min)
    normalized = np.clip(normalized, 0.0, 1.0)

    cmap = colormaps[cmap_name]
    rgba = cmap(normalized)  # shape (N, 4), floats in [0, 1]

    rgb_u8 = (255.0 * rgba[:, :3]).astype(np.uint8)
    rgb_u32 = (
        (rgb_u8[:, 0].astype(np.uint32) << 16)
        | (rgb_u8[:, 1].astype(np.uint32) << 8)
        | rgb_u8[:, 2].astype(np.uint32)
    )
    return rgb_u32


def make_field_pointcloud2(
    positions_xy: np.ndarray,
    values: np.ndarray,
    frame_id: str,
    stamp,
    z_mode: str = "flat",
    z_offset: float = -0.6,
    height_scale: float = 1.0,
    colormap_min: float = 0.0,
    colormap_max: float = 0.015,
    cmap_name: str = "viridis",
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

    rgb = _values_to_rgb_uint32(
        values=values,
        color_min=colormap_min,
        color_max=colormap_max,
        cmap_name=cmap_name,
    )

    points = np.column_stack(
        [
            positions_xy[:, 0],
            positions_xy[:, 1],
            z + z_offset,
            rgb.view(np.float32),
        ]
    ).astype(np.float32, copy=False)

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        # PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return point_cloud2.create_cloud(header, fields, points)
