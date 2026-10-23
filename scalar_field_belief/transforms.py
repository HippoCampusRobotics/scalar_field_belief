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

"""Input and target transforms for the scalar field GP belief.

This module contains the small preprocessing helpers used by the belief model.

There are two separate transformations:

1. Input normalization
   Physical `(x, y)` positions are mapped to normalized GP input coordinates
   using fixed physical domain bounds from the configuration.

2. Target standardization
   Scalar measurement values are standardized using the mean and standard
   deviation of the current training targets.

This distinction is important:
- input normalization depends on known physical bounds,
- target standardization depends on observed measurement values.

`build_gp_inputs` is the single preparation path for training and query
inputs. It combines the normalized coordinates with optional wall features,
which are always evaluated on the original physical coordinates. The public
belief API stays `(N, 2)` physical positions; the model array is `(N, 2)` in
baseline mode and `(N, 2 + M)` in wall-aware mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from scalar_field_belief.wall_features import WallFeatureMap


@dataclass
class TargetStandardizer:
    """Standardize scalar target values for GP training.

    The standardizer stores the mean and standard deviation of the training
    targets. It is fitted during model fitting and then reused during querying
    to convert posterior mean and variance back to original measurement units.

    Parameters
    ----------
    mean
        Mean of the training targets.
    std
        Standard deviation of the training targets.

    Notes
    -----
    The stored tensors are detached from the computation graph because the
    standardization parameters are preprocessing state, not learnable model
    parameters.
    """

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, y: torch.Tensor, eps: float = 1e-12) -> TargetStandardizer:
        """Fit a target standardizer from one-dimensional training targets.

        Parameters
        ----------
        y
            Floating-point target tensor of shape `(N,)`.
        eps
            Minimum standard deviation threshold. If the empirical standard
            deviation is smaller than this value, a standard deviation of one is
            used instead.

        Returns
        -------
        TargetStandardizer
            Standardizer fitted to the provided target values.

        Raises
        ------
        ValueError
            If `y` does not have shape `(N,)`.
        TypeError
            If `y` is not a floating-point tensor.
        """
        if y.ndim != 1:
            raise ValueError(
                f'Expected y to have shape (N,), got {tuple(y.shape)}'
            )
        if not y.is_floating_point():
            raise TypeError('Expected floating-point tensor for y')

        mean = y.mean()
        std = y.std(correction=0)

        # If all target values are equal, the empirical standard deviation is
        # zero. Using one keeps the transform well-defined and maps all targets
        # to zero.
        std = torch.where(std < eps, torch.ones_like(std), std)
        return cls(mean=mean.detach(), std=std.detach())

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def inverse_transform_mean_var(
        self, mean_norm: torch.Tensor, var_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = mean_norm * self.std + self.mean
        var = var_norm * (self.std**2)
        return mean, var

    def inverse_transform_covar(self, covar_norm: torch.Tensor) -> torch.Tensor:
        # Same rescaling as inverse_transform_mean_var, applied to the whole
        # matrix: y_std = (y - mean) / std, so Cov(y) = Cov(y_std) * std**2
        # for every entry, not just the diagonal.
        return covar_norm * (self.std**2)


@dataclass(frozen=True)
class InputNormalizer2d:
    """Normalize physical 2D positions using fixed domain bounds.

    The normalizer maps physical coordinates to normalized coordinates according
    to

    `x_norm = (x - x_min) / (x_max - x_min)`

    and equivalently for `y`.

    Parameters
    ----------
    x_min
        Minimum physical x coordinate of the modeled domain.
    x_max
        Maximum physical x coordinate of the modeled domain.
    y_min
        Minimum physical y coordinate of the modeled domain.
    y_max
        Maximum physical y coordinate of the modeled domain.

    Notes
    -----
    This class does not estimate bounds from data. The bounds are fixed by the
    configuration and should correspond to the known physical workspace.

    Points outside the configured bounds are not rejected. They are mapped to
    normalized values outside the interval `[0, 1]`.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError('x_max must be greater than x_min.')
        if self.y_max <= self.y_min:
            raise ValueError('y_max must be greater than y_min.')

    def transform(self, xy: np.ndarray) -> np.ndarray:
        """Normalize physical `(x, y)` positions.

        Parameters
        ----------
        xy
            Physical positions with shape `(N, 2)`.

        Returns
        -------
        np.ndarray
            Normalized positions with shape `(N, 2)`.

        Raises
        ------
        ValueError
            If `xy` does not have shape `(N, 2)`.
        """
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(
                f'Expected xy to have shape (N, 2), got {tuple(xy.shape)}'
            )
        xy_norm = np.empty_like(xy, dtype=float)
        xy_norm[:, 0] = (xy[:, 0] - self.x_min) / (self.x_max - self.x_min)
        xy_norm[:, 1] = (xy[:, 1] - self.y_min) / (self.y_max - self.y_min)
        return xy_norm


def _check_xy(xy: np.ndarray, name: str) -> None:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f'Expected {name} to have shape (N, 2), got {tuple(xy.shape)}'
        )


@dataclass(frozen=True, eq=False)
class GPInputs:
    """Prepared GP inputs for a batch of physical positions.

    Parameters
    ----------
    xy_phys
        Physical positions in meters with shape `(N, 2)`.
    xy_norm
        Positions normalized by the fixed domain bounds, shape `(N, 2)`.
    wall_features
        Dimensionless wall features evaluated at `xy_phys`, shape `(N, M)`.
        `M` is zero when no wall feature map is used.

    Raises
    ------
    ValueError
        If any array has the wrong shape or the row counts differ.
    """

    xy_phys: np.ndarray
    xy_norm: np.ndarray
    wall_features: np.ndarray

    def __post_init__(self) -> None:
        # np.asarray does not copy arrays that are already float64.
        for name in ('xy_phys', 'xy_norm', 'wall_features'):
            object.__setattr__(
                self, name, np.asarray(getattr(self, name), dtype=float)
            )
        _check_xy(self.xy_phys, 'xy_phys')
        _check_xy(self.xy_norm, 'xy_norm')
        if self.wall_features.ndim != 2:
            raise ValueError(
                'Expected wall_features to have shape (N, M), got '
                f'{tuple(self.wall_features.shape)}'
            )
        num_rows = self.xy_phys.shape[0]
        if (
            self.xy_norm.shape[0] != num_rows
            or self.wall_features.shape[0] != num_rows
        ):
            raise ValueError(
                'xy_phys, xy_norm, and wall_features must have the same '
                'number of rows.'
            )

    @property
    def wall_feature_dim(self) -> int:
        """Number of wall feature columns `M`."""
        return int(self.wall_features.shape[1])

    def to_model_array(self, model_variant: str) -> np.ndarray:
        """Return the GP input array for the given model variant.

        Parameters
        ----------
        model_variant
            `'baseline'` returns `xy_norm` with shape `(N, 2)`.
            `'wall_aware'` returns `[x_norm, y_norm, h_0, ..., h_(M-1)]` with
            shape `(N, 2 + M)`.

        Raises
        ------
        ValueError
            If `model_variant` is unknown, or if it is `'wall_aware'` and
            there are no wall feature columns.
        """
        if model_variant == 'baseline':
            return self.xy_norm
        if model_variant == 'wall_aware':
            if self.wall_feature_dim <= 0:
                raise ValueError(
                    'wall_aware model inputs require at least one wall '
                    'feature column.'
                )
            return np.concatenate([self.xy_norm, self.wall_features], axis=1)
        raise ValueError(f'Unknown model_variant: {model_variant}')


def build_gp_inputs(
    xy_phys: np.ndarray,
    input_normalizer: InputNormalizer2d,
    wall_feature_map: WallFeatureMap | None = None,
) -> GPInputs:
    """Prepare GP inputs from physical positions.

    Parameters
    ----------
    xy_phys
        Physical positions in meters with shape `(N, 2)`.
    input_normalizer
        Normalizer with the fixed domain bounds.
    wall_feature_map
        Optional wall feature map. It is evaluated on the physical positions,
        not on the normalized ones. If `None`, no features are computed and
        the feature matrix has shape `(N, 0)`.

    Returns
    -------
    GPInputs
        Physical, normalized, and wall-feature representations.

    Raises
    ------
    ValueError
        If `xy_phys` does not have shape `(N, 2)` or contains non-finite
        values.
    """
    xy_phys = np.asarray(xy_phys, dtype=float)
    _check_xy(xy_phys, 'xy_phys')
    if not np.all(np.isfinite(xy_phys)):
        raise ValueError('xy_phys must contain only finite values.')

    xy_norm = input_normalizer.transform(xy_phys)
    if wall_feature_map is None:
        wall_features = np.empty((xy_phys.shape[0], 0), dtype=float)
    else:
        wall_features = wall_feature_map.transform_xy_to_wall_features(xy_phys)

    return GPInputs(
        xy_phys=xy_phys,
        xy_norm=xy_norm,
        wall_features=wall_features,
    )
