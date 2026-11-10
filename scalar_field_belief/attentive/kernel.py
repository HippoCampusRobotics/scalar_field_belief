# Copyright (c) 2022 Weizhe (Wesley) Chen
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

"""Attentive Kernel (AK) of Chen, Khardon and Liu (RSS 2022) for GPyTorch.

For normalized inputs `u, v` and the L2-normalized output `r(u)` of one
shared attention network (`w = z = r`),

    k(u, v) = <r(u), r(v)> * sum_m r_m(u) r_m(v) k_m(u, v),
    k_m(u, v) = exp(-||u - v||^2 / (2 ell_m^2)),

with fixed primitive lengthscales `ell_m` (a registered buffer, not
parameters). The kernel has unit amplitude (`k(u, u) = 1` up to roundoff);
wrap it in `gpytorch.kernels.ScaleKernel` for the one positive amplitude
`alpha`.

Positive semidefiniteness for any network weights: every
`diag(r_m) K_m diag(r_m)` is PSD because `K_m` is an RBF Gram matrix, so
their sum is PSD; the membership Gram matrix `R R^T` is PSD; the elementwise
(Schur) product of two PSD matrices is PSD; and `alpha > 0`. Training changes
the parameters of a valid kernel, never its validity.
"""

from __future__ import annotations

from collections.abc import Sequence

import gpytorch
import torch


class TwoHiddenLayerTanhNN(torch.nn.Sequential):
    """Linear-tanh-Linear-tanh-Linear-softmax attention network.

    Maps `(..., dim_input)` inputs to `(..., dim_output)` softmax weights
    (strictly positive in exact arithmetic, summing to one). The module
    names match the reference implementation.
    """

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.add_module(
            'linear1', torch.nn.Linear(dim_input, dim_hidden, dtype=dtype)
        )
        self.add_module('activation1', torch.nn.Tanh())
        self.add_module(
            'linear2', torch.nn.Linear(dim_hidden, dim_hidden, dtype=dtype)
        )
        self.add_module('activation2', torch.nn.Tanh())
        self.add_module(
            'linear3', torch.nn.Linear(dim_hidden, dim_output, dtype=dtype)
        )
        self.add_module('activation3', torch.nn.Softmax(dim=-1))


def _squared_distances(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances `(..., n, m)`.

    Accumulated per input dimension: exactly zero for identical rows and
    exactly symmetric, without an `(n, m, d)` temporary.
    """
    d2 = None
    for j in range(x1.size(-1)):
        diff = x1[..., :, j].unsqueeze(-1) - x2[..., :, j].unsqueeze(-2)
        d2 = diff * diff if d2 is None else d2 + diff * diff
    return d2


class AttentiveKernel(gpytorch.kernels.Kernel):
    """Unit-amplitude Attentive Kernel with a shared attention network.

    Parameters
    ----------
    primitive_lengthscales
        The `M` fixed, positive primitive RBF lengthscales [normalized input
        units]. Stored as the buffer `primitive_lengthscales`.
    hidden_width
        Width of both hidden layers.
    input_dim
        Number of input columns (2 for planar positions).
    seed
        If given, the network is initialized from this seed inside
        `torch.random.fork_rng`, so the global torch RNG is unchanged.
        Otherwise the global RNG is used, as for any `torch.nn` module.
    dtype
        Dtype of the network and the lengthscale buffer.

    Notes
    -----
    `forward(x1, x2)` supports leading batch dimensions; `diag=True` returns
    the row-matched cross-covariance `k(x1[i], x2[i])` (equal to one only
    when the rows coincide). `last_dim_is_batch` is not supported.
    """

    has_lengthscale = False

    def __init__(
        self,
        primitive_lengthscales: Sequence[float] | torch.Tensor,
        hidden_width: int = 10,
        input_dim: int = 2,
        seed: int | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        ell = torch.as_tensor(primitive_lengthscales, dtype=dtype).detach()
        ell = ell.clone().reshape(-1)
        if ell.numel() == 0:
            raise ValueError('at least one primitive lengthscale is needed.')
        if not bool(torch.all(torch.isfinite(ell) & (ell > 0))):
            raise ValueError('primitive lengthscales must be finite and > 0.')
        if int(hidden_width) < 1 or int(input_dim) < 1:
            raise ValueError('hidden_width and input_dim must be >= 1.')
        self.input_dim = int(input_dim)
        self.register_buffer('primitive_lengthscales', ell)
        if seed is None:
            self.network = TwoHiddenLayerTanhNN(
                self.input_dim, int(hidden_width), ell.numel(), dtype
            )
        else:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(int(seed))
                self.network = TwoHiddenLayerTanhNN(
                    self.input_dim, int(hidden_width), ell.numel(), dtype
                )

    @property
    def is_stationary(self) -> bool:
        return False

    @property
    def num_components(self) -> int:
        return int(self.primitive_lengthscales.numel())

    def representations(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalized attention representations `(..., M)` of inputs
        `(..., input_dim)`. Does not change any state.

        The softmax output has a component of at least `1 / M`, so the norm
        is never zero."""
        s = self.network(x)
        return s / torch.linalg.vector_norm(s, dim=-1, keepdim=True)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise ValueError(
                'AttentiveKernel does not support last_dim_is_batch.'
            )
        if x1.size(-1) != self.input_dim or x2.size(-1) != self.input_dim:
            raise ValueError(
                f'expected inputs with {self.input_dim} columns, got '
                f'{tuple(x1.shape)} and {tuple(x2.shape)}.'
            )
        # Equal values share representations only if neither input needs
        # its own gradient.
        same = x1 is x2 or (
            not (x1.requires_grad or x2.requires_grad)
            and x1.shape == x2.shape
            and torch.equal(x1, x2)
        )
        r1 = self.representations(x1)
        r2 = r1 if same else self.representations(x2)
        inv_two_ell2 = 0.5 / self.primitive_lengthscales.pow(2)

        if diag:
            if x1.size(-2) != x2.size(-2):
                raise ValueError(
                    'diag=True needs the same number of rows in x1 and x2, '
                    f'got {x1.size(-2)} and {x2.size(-2)}.'
                )
            diff = x1 - x2
            d2 = (diff * diff).sum(-1)  # (..., n)
            rr = r1 * r2  # (..., n, M)
            base = torch.exp(-d2.unsqueeze(-1) * inv_two_ell2)  # (..., n, M)
            return rr.sum(-1) * (rr * base).sum(-1)

        d2 = _squared_distances(x1, x2)  # (..., n, m)
        weighted = None
        for k in range(self.num_components):
            term = (
                r1[..., :, k].unsqueeze(-1)
                * r2[..., :, k].unsqueeze(-2)
                * torch.exp(-d2 * inv_two_ell2[k])
            )
            weighted = term if weighted is None else weighted + term
        membership = r1 @ r2.transpose(-1, -2)
        return membership * weighted
