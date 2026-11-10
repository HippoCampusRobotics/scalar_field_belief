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

"""Independent closed-form exact-GP posterior (NumPy) used as a test oracle.

Implements the latent posterior of a constant-mean GP with an ARD
ScaleKernel(RBF / Matern) and Gaussian noise directly from the formulas, so
it does not share prediction code with GPyTorch:

    K   = s * k(X, X) + sigma2 * I
    m*  = c + k(Q, X) s K^{-1} (y_t - c)
    C*  = s * k(Q, Q) - s^2 k(Q, X) K^{-1} k(X, Q)

and then maps to physical units with `offset + scale * m*`,
`scale**2 * C*`.
"""

from __future__ import annotations

import numpy as np


def normalize_inputs(xy: np.ndarray, domain: dict) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    return np.column_stack(
        [
            (xy[:, 0] - domain['x_min']) / (domain['x_max'] - domain['x_min']),
            (xy[:, 1] - domain['y_min']) / (domain['y_max'] - domain['y_min']),
        ]
    )


def empirical_standardizer(y: np.ndarray, eps: float = 1e-12):
    """Population mean/std with the std < eps -> 1 fallback (contract)."""
    y = np.asarray(y, dtype=float)
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=0))
    if std < eps:
        std = 1.0
    return mean, std


def base_kernel(a: np.ndarray, b: np.ndarray, lengthscale, kind: str):
    ls = np.asarray(lengthscale, dtype=float).reshape(1, 1, 2)
    d = (a[:, None, :] - b[None, :, :]) / ls
    r2 = np.sum(d * d, axis=-1)
    if kind == 'rbf':
        return np.exp(-0.5 * r2)
    r = np.sqrt(np.maximum(r2, 0.0))
    if kind == 'matern32':
        s3 = np.sqrt(3.0) * r
        return (1.0 + s3) * np.exp(-s3)
    if kind == 'matern52':
        s5 = np.sqrt(5.0) * r
        return (1.0 + s5 + 5.0 / 3.0 * r2) * np.exp(-s5)
    raise ValueError(kind)


def posterior(
    x_train_norm,
    y_t,
    x_query_norm,
    *,
    lengthscale,
    outputscale,
    noise,
    constant,
    kind,
):
    """Latent posterior mean/cov in transformed units."""
    kxx = outputscale * base_kernel(
        x_train_norm, x_train_norm, lengthscale, kind
    )
    kxx = kxx + noise * np.eye(len(x_train_norm))
    kqx = outputscale * base_kernel(
        x_query_norm, x_train_norm, lengthscale, kind
    )
    kqq = outputscale * base_kernel(
        x_query_norm, x_query_norm, lengthscale, kind
    )
    chol = np.linalg.cholesky(kxx)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_t - constant))
    v = np.linalg.solve(chol, kqx.T)
    mean = constant + kqx @ alpha
    cov = kqq - v.T @ v
    return mean, cov


def physical_posterior(
    xy_train, values, xy_query, domain, *, offset, scale, **hyper
):
    """Posterior in physical units for a given target offset/scale."""
    xt = normalize_inputs(xy_train, domain)
    xq = normalize_inputs(xy_query, domain)
    y_t = (np.asarray(values, dtype=float) - offset) / scale
    mean_t, cov_t = posterior(xt, y_t, xq, **hyper)
    return offset + scale * mean_t, scale**2 * cov_t


def model_hyper(belief) -> dict:
    """Read the GP hyperparameters (transformed units) from a belief."""
    model = belief.model
    return dict(
        lengthscale=model.covar_module.base_kernel.lengthscale.detach()
        .cpu()
        .numpy()
        .reshape(-1),
        outputscale=float(model.covar_module.outputscale.detach()),
        noise=float(belief.likelihood.noise.detach().reshape(-1)[0]),
        constant=float(model.mean_module.constant.detach().reshape(-1)[0]),
        kind=belief.config.kernel_type,
    )
