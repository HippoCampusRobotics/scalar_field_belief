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
- input coordinates are normalized 2D positions,
- target values are standardized scalar measurements.

Therefore:
- kernel lengthscales are expressed in normalized input units,
- outputscale and likelihood noise are expressed in standardized target units.
"""

from __future__ import annotations

import gpytorch
import torch


class ExactFieldGP(gpytorch.models.ExactGP):
    """Exact GP model for a 2D scalar field.

    The model uses a constant mean and a configurable covariance module.
    It does not perform any input or target transformations internally.

    Parameters
    ----------
    train_x
        Normalized training input positions of shape `(N, 2)`.
    train_y
        Standardized scalar training targets of shape `(N,)`.
    likelihood
        Gaussian likelihood used by the exact GP.
    covar_module
        Kernel module used to model spatial covariance.
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
        """Evaluate the latent GP prior/posterior at normalized input positions.

        Parameters
        ----------
        x
            Normalized query positions of shape `(N, 2)`.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Latent GP distribution at the queried positions.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def build_covar_module(kernel_type: str):
    """Build the covariance module configured for the scalar field GP.

    The returned kernel is always wrapped in a `ScaleKernel`, which gives the GP
    a learnable outputscale in addition to the base kernel parameters.

    Parameters
    ----------
    kernel_type
        Name of the spatial base kernel. Supported values so far are:
        - `'rbf'`
        - `'matern32'`
        - `'matern52'`

    Returns
    -------
    gpytorch.kernels.ScaleKernel
        Scaled spatial covariance module with Automatic Relevance Determination
        (ARD)over the two input dimensions. This means, that the length-scales
        are assigned individually to each input dimension.

    Raises
    ------
    ValueError
        If `kernel_type` is unknown.
    """
    if kernel_type == 'rbf':
        spatial_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2)
    elif kernel_type == 'matern32':
        spatial_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2)
    elif kernel_type == 'matern52':
        spatial_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
    else:
        raise ValueError(f'Unknown kernel_type: {kernel_type}')
    return gpytorch.kernels.ScaleKernel(spatial_kernel)


def initialize_model_hyperparameters(
    model: ExactFieldGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: torch.Tensor,
    init_lengthscale: tuple[float, float],
    init_outputscale: float,
    init_noise: float,
) -> None:
    """Set initial GP hyperparameter values before training.

    Parameters
    ----------
    model
        GP model whose kernel hyperparameters should be initialized.
    likelihood
        Gaussian likelihood whose noise value should be initialized.
    train_x
        Normalized training inputs. Used only to match dtype and device.
    init_lengthscale
        Initial ARD lengthscales for x and y. These are in normalized input
        units, not meters.
    init_outputscale
        Initial kernel outputscale. This is in standardized target units.
    init_noise
        Initial Gaussian likelihood noise. This is also in standardized target
        units.

    Notes
    -----
    GPyTorch hyperparameters remain learnable after this initialization. These
    values only define the optimizer starting point.
    """
    model.covar_module.base_kernel.lengthscale = torch.tensor(
        init_lengthscale,
        dtype=train_x.dtype,
        device=train_x.device,
    ).view(1, -1)
    model.covar_module.outputscale = torch.tensor(
        init_outputscale,
        dtype=train_x.dtype,
        device=train_x.device,
    )
    likelihood.noise = torch.tensor(
        init_noise, dtype=train_x.dtype, device=train_x.device
    )
