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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


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
