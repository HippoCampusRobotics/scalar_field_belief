"""Grid and PointCloud2 helpers for scalar field belief visualization.

This module provides small utilities for:
- generating regular 2D query grids,
- mapping scalar values to explicit RGB colors,
- creating RViz-friendly `PointCloud2` messages.

Scalar values are converted to RGB before publishing. This makes the displayed
colors independent of RViz's intensity-color settings and allows ground-truth
and belief clouds to use the same color range.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def make_axis(start: float, stop: float, step: float) -> np.ndarray:
    if stop <= start:
        raise ValueError('Axis stop must be larger than start.')
    if step <= 0.0:
        raise ValueError('Axis step must be positive.')
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
    cmap_name: str = 'viridis',
) -> np.ndarray:
    """Map scalar values to packed 24-bit RGB colors.

    Parameters
    ----------
    values
        Scalar values of shape `(N,)`.
    color_min
        Lower bound of the fixed color range.
    color_max
        Upper bound of the fixed color range.
    cmap_name
        Name of the matplotlib colormap used for color mapping.

    Returns
    -------
    np.ndarray
        Packed RGB values as `uint32`, shape `(N,)`, with layout `0xRRGGBB`.

    Raises
    ------
    ValueError
        If `color_max <= color_min`.

    Notes
    -----
    Values below `color_min` are clipped to the low end of the colormap.
    Values above `color_max` are clipped to the high end.

    The returned values are not yet in the final RViz representation. They are
    later reinterpreted as `float32` because RViz expects the `rgb` field in a
    `PointCloud2` message to use a packed floating-point representation.
    """
    values = np.asarray(values, dtype=np.float32)

    if color_max <= color_min:
        raise ValueError('color_max must be greater than color_min.')

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
    z_mode: str = 'flat',
    z_offset: float = -0.6,
    height_scale: float = 1.0,
    colormap_min: float = 0.0,
    colormap_max: float = 0.015,
    cmap_name: str = 'viridis',
) -> PointCloud2:
    """Build a PointCloud2 message for scalar field visualization.

    Parameters
    ----------
    positions_xy
        2D point positions of shape `(N, 2)`.
    values
        Scalar values of shape `(N,)`.
    frame_id
        ROS frame of the point cloud.
    stamp
        ROS timestamp for the cloud header.
    z_mode
        Vertical visualization mode:
        - `'flat'`: all points lie on one plane.
        - `'height'`: values are additionally shown as height.
    z_offset
        Constant vertical offset added to all points.
    height_scale
        Maximum relative height used when `z_mode == 'height'`.
    colormap_min
        Lower bound of the fixed color range.
    colormap_max
        Upper bound of the fixed color range.
    cmap_name
        Name of the matplotlib colormap used for RGB mapping.

    Returns
    -------
    PointCloud2
        Point cloud with fields `x`, `y`, `z`, and `rgb`.

    Raises
    ------
    ValueError
        If `positions_xy` does not have shape `(N, 2)`.
    ValueError
        If `positions_xy` and `values` have different lengths.
    ValueError
        If `z_mode` is neither `'flat'` nor `'height'`.

    Notes
    -----
    The `'flat'` mode is the standard visualization mode.

    The `'height'` mode is mainly useful for debugging. In this mode, z values
    are normalized per cloud to the interval `[0, height_scale]` before
    `z_offset` is added. This makes local structure easier to see, but it also
    means that heights are relative within one cloud and not directly comparable
    between different clouds.

    Colors use the fixed range `[colormap_min, colormap_max]`. This is
    independent of `z_mode`.
    """
    positions_xy = np.asarray(positions_xy, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if positions_xy.ndim != 2 or positions_xy.shape[1] != 2:
        raise ValueError('positions_xy must have shape (N, 2).')
    if len(positions_xy) != len(values):
        raise ValueError('positions_xy and values must have matching length.')

    if z_mode == 'flat':
        z = np.zeros(len(values), dtype=np.float32)
    elif z_mode == 'height':
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

    # RViz expects packed RGB colors in a FLOAT32 field. The uint32 color values
    # are therefore reinterpreted as float32 without changing the underlying bit
    # pattern.
    points = np.column_stack(
        [
            positions_xy[:, 0],
            positions_xy[:, 1],
            z + z_offset,
            rgb.view(np.float32),
        ]
    ).astype(np.float32, copy=False)

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        # PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return point_cloud2.create_cloud(header, fields, points)


def make_intensity_pointcloud2(
    positions_xy: np.ndarray,
    values: np.ndarray,
    frame_id: str,
    stamp,
    z_mode: str = 'flat',
    z_offset: float = -0.6,
    height_scale: float = 1.0,
) -> PointCloud2:
    """Build a PointCloud2 message for scalar field visualization.

    Parameters
    ----------
    positions_xy
        2D point positions of shape `(N, 2)`.
    values
        Scalar values of shape `(N,)`.
    frame_id
        ROS frame of the point cloud.
    stamp
        ROS timestamp for the cloud header.
    z_mode
        Vertical visualization mode:
        - `'flat'`: all points lie on one plane.
        - `'height'`: values are additionally shown as height.
    z_offset
        Constant vertical offset added to all points.
    height_scale
        Maximum relative height used when `z_mode == 'height'`.

    Returns
    -------
    PointCloud2
        Point cloud with fields `x`, `y`, `z`, and `intensity`.

    Raises
    ------
    ValueError
        If `positions_xy` does not have shape `(N, 2)`.
    ValueError
        If `positions_xy` and `values` have different lengths.
    ValueError
        If `z_mode` is neither `'flat'` nor `'height'`.

    Notes
    -----
    The `'flat'` mode is the standard visualization mode.

    The `'height'` mode is mainly useful for debugging. In this mode, z values
    are normalized per cloud to the interval `[0, height_scale]` before
    `z_offset` is added. This makes local structure easier to see, but it also
    means that heights are relative within one cloud and not directly comparable
    between different clouds.

    """
    positions_xy = np.asarray(positions_xy, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if positions_xy.ndim != 2 or positions_xy.shape[1] != 2:
        raise ValueError('positions_xy must have shape (N, 2).')
    if len(positions_xy) != len(values):
        raise ValueError('positions_xy and values must have matching length.')

    if z_mode == 'flat':
        z = np.zeros(len(values), dtype=np.float32)
    elif z_mode == 'height':
        vmin = float(values.min())
        vmax = float(values.max())
        denom = max(vmax - vmin, 1e-12)
        z = ((values - vmin) / denom * height_scale).astype(np.float32)
    else:
        raise ValueError("z_mode must be 'flat' or 'height'.")

    points = np.column_stack(
        [
            positions_xy[:, 0],
            positions_xy[:, 1],
            z + z_offset,
            values,
        ]
    ).astype(np.float32, copy=False)

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(
            name='intensity', offset=12, datatype=PointField.FLOAT32, count=1
        ),
    ]

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return point_cloud2.create_cloud(header, fields, points)
