from __future__ import annotations

import gpytorch
import torch


class ExactFieldGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def build_covar_module(kernel_type: str):
    if kernel_type == "rbf":
        spatial_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2)
    elif kernel_type == "matern32":
        spatial_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2)
    elif kernel_type == "matern52":
        spatial_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    return gpytorch.kernels.ScaleKernel(spatial_kernel)


def initialize_model_hyperparameters(
    model: ExactFieldGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: torch.Tensor,
    init_lengthscale: tuple[float, float],
    init_outputscale: float,
    init_noise: float,
) -> None:
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
    likelihood.noise = torch.tensor(init_noise, dtype=train_x.dtype, device=train_x.device)
