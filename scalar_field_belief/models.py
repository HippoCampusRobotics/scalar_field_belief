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

"""GPyTorch model and kernel helpers for the scalar field belief.

This module contains only the GPyTorch-specific model definition and small
helpers for constructing kernels and initializing hyperparameters.

The surrounding belief code is responsible for:
- converting physical positions to normalized inputs,
- standardizing scalar measurement values,
- training the model,
- converting predictions back to physical measurement units.

Conventions
-----------
The GP in this module operates on preprocessed data:
- input coordinates are normalized 2D positions, optionally followed by
  dimensionless wall features (wall-aware mode),
- target values are standardized scalar measurements.

The belief's public API always takes physical `(N, 2)` positions. The model
inputs are internal: `(N, 2)` normalized positions in baseline mode, and
`(N, 2 + M)` augmented inputs `[x_norm, y_norm, h_0, ..., h_(M-1)]` in
wall-aware mode.

Therefore:
- spatial kernel lengthscales are expressed in normalized input units,
- wall-feature lengthscales are dimensionless feature units,
- outputscale and likelihood noise are expressed in standardized target units.
"""

from __future__ import annotations

import math

import gpytorch
import torch


class ExactFieldGP(gpytorch.models.ExactGP):
    """Exact GP model for a 2D scalar field.

    The model uses a constant mean and a configurable covariance module.
    It does not perform any input or target transformations internally.

    Parameters
    ----------
    train_x
        Training model inputs: normalized positions of shape `(N, 2)` in
        baseline mode, or augmented inputs of shape `(N, 2 + M)` in
        wall-aware mode.
    train_y
        Standardized scalar training targets of shape `(N,)`.
    likelihood
        Gaussian likelihood used by the exact GP.
    covar_module
        Kernel module used to model covariance between model inputs.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood,
        covar_module,
    ):
        super().__init__(train_x, train_y, likelihood)

        # A constant mean is sufficient for the current belief model
        self.mean_module = gpytorch.means.ConstantMean()

        # The covariance module is constructed outside the model so that the
        # belief code can select different kernels from configuration.
        self.covar_module = covar_module

    def forward(self, x: torch.Tensor):
        """Evaluate the latent GP prior/posterior at model inputs.

        Parameters
        ----------
        x
            Model inputs with the same column layout as `train_x`.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Latent GP distribution at the queried positions.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _build_spatial_kernel(
    kernel_type: str,
    active_dims: tuple[int, ...] | None = None,
) -> gpytorch.kernels.Kernel:
    """Build the 2D ARD spatial kernel, optionally restricted to columns."""
    kwargs = {'ard_num_dims': 2}
    if active_dims is not None:
        kwargs['active_dims'] = active_dims
    if kernel_type == 'rbf':
        return gpytorch.kernels.RBFKernel(**kwargs)
    if kernel_type == 'matern32':
        return gpytorch.kernels.MaternKernel(nu=1.5, **kwargs)
    if kernel_type == 'matern52':
        return gpytorch.kernels.MaternKernel(nu=2.5, **kwargs)
    raise ValueError(f'Unknown kernel_type: {kernel_type}')


def build_covar_module(
    kernel_type: str,
    *,
    model_variant: str = 'baseline',
    wall_feature_dim: int = 0,
):
    """Build the covariance module configured for the scalar field GP.

    The returned kernel is always wrapped in a single `ScaleKernel`, which
    gives the GP a learnable outputscale in addition to the base kernel
    parameters.

    Parameters
    ----------
    kernel_type
        Name of the spatial base kernel. Supported values so far are:
        - `'rbf'`
        - `'matern32'`
        - `'matern52'`
    model_variant
        `'baseline'` builds `ScaleKernel(spatial)` over `(N, 2)` normalized
        inputs. `'wall_aware'` builds `ScaleKernel(spatial * wall)` over
        `(N, 2 + M)` inputs, where `spatial` acts on columns `(0, 1)` and
        `wall` is an ARD RBF kernel on columns `2 .. 2 + M - 1`.
    wall_feature_dim
        Number of wall features `M`. Must be `0` in baseline mode and
        positive in wall-aware mode.

    Returns
    -------
    gpytorch.kernels.ScaleKernel
        Scaled covariance module with Automatic Relevance Determination (ARD)
        over the two spatial input dimensions, i.e. individual lengthscales
        per dimension. In wall-aware mode, every wall feature also has its own
        lengthscale.

    Raises
    ------
    ValueError
        If `kernel_type` or `model_variant` is unknown, or if
        `wall_feature_dim` does not fit the variant.
    """
    if model_variant == 'baseline':
        if wall_feature_dim != 0:
            raise ValueError('wall_feature_dim must be 0 for baseline model.')
        return gpytorch.kernels.ScaleKernel(_build_spatial_kernel(kernel_type))

    if model_variant == 'wall_aware':
        if wall_feature_dim <= 0:
            raise ValueError(
                'wall_feature_dim must be > 0 for wall_aware model.'
            )
        spatial_kernel = _build_spatial_kernel(kernel_type, active_dims=(0, 1))
        wall_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=wall_feature_dim,
            active_dims=tuple(range(2, 2 + wall_feature_dim)),
        )
        return gpytorch.kernels.ScaleKernel(spatial_kernel * wall_kernel)

    raise ValueError(f'Unknown model_variant: {model_variant}')


def get_spatial_kernel(covar_module) -> gpytorch.kernels.Kernel:
    """Return the spatial kernel of a module from `build_covar_module`.

    Its lengthscales are in normalized input units.
    """
    base_kernel = covar_module.base_kernel
    if isinstance(base_kernel, gpytorch.kernels.ProductKernel):
        return base_kernel.kernels[0]
    return base_kernel


def get_wall_kernel(covar_module) -> gpytorch.kernels.Kernel | None:
    """Return the wall-feature kernel, or `None` for a baseline module.

    Its lengthscales are dimensionless feature units.
    """
    base_kernel = covar_module.base_kernel
    if isinstance(base_kernel, gpytorch.kernels.ProductKernel):
        return base_kernel.kernels[1]
    return None


def initialize_model_hyperparameters(
    model: ExactFieldGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: torch.Tensor,
    init_lengthscale: tuple[float, float],
    init_outputscale: float,
    init_noise: float,
    init_wall_lengthscale: float | None = None,
) -> None:
    """Set initial GP hyperparameter values before training.

    Parameters
    ----------
    model
        GP model whose kernel hyperparameters should be initialized.
    likelihood
        Gaussian likelihood whose noise value should be initialized.
    train_x
        Training model inputs. Used only to match dtype and device.
    init_lengthscale
        Initial ARD lengthscales of the spatial kernel for x and y. These are
        in normalized input units, not meters.
    init_outputscale
        Initial kernel outputscale. This is in standardized target units.
    init_noise
        Initial Gaussian likelihood noise. This is also in standardized target
        units.
    init_wall_lengthscale
        Initial lengthscale assigned to every wall feature. Dimensionless.
        Required if the model has a wall kernel, ignored otherwise.

    Raises
    ------
    ValueError
        If the model has a wall kernel and `init_wall_lengthscale` is missing,
        not finite, or not positive.

    Notes
    -----
    GPyTorch hyperparameters remain learnable after this initialization. These
    values only define the optimizer starting point.
    """
    get_spatial_kernel(model.covar_module).lengthscale = torch.tensor(
        init_lengthscale,
        dtype=train_x.dtype,
        device=train_x.device,
    ).view(1, -1)

    wall_kernel = get_wall_kernel(model.covar_module)
    if wall_kernel is not None:
        if init_wall_lengthscale is None:
            raise ValueError(
                'init_wall_lengthscale is required for a wall-aware model.'
            )
        if (
            not math.isfinite(init_wall_lengthscale)
            or init_wall_lengthscale <= 0.0
        ):
            raise ValueError(
                'init_wall_lengthscale must be positive and finite.'
            )
        wall_kernel.lengthscale = torch.full(
            (1, wall_kernel.ard_num_dims),
            fill_value=float(init_wall_lengthscale),
            dtype=train_x.dtype,
            device=train_x.device,
        )

    model.covar_module.outputscale = torch.tensor(
        init_outputscale,
        dtype=train_x.dtype,
        device=train_x.device,
    )
    likelihood.noise = torch.tensor(
        init_noise, dtype=train_x.dtype, device=train_x.device
    )
