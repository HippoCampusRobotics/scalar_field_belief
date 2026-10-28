# Visibility-graph GP (visGP) after Gilbert and Datta (2024), Biometrics:
# https://academic.oup.com/biometrics/article/doi/10.1093/biomtc/ujae089/7753413
# Authors' R implementation: https://github.com/bjg345/visGP

"""Configuration of the visGP belief.

All values are fixed for the lifetime of a belief; nothing is fitted.
Positions are in meters, values in the measurement unit of the scalar field
("units"). `signal_variance` and `measurement_noise_variance` are variances
(units^2), not standard deviations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class VisGPConfig:
    """Fixed parameters of the visGP belief.

    Parameters
    ----------
    signal_variance
        Variance of the isotropic Matern-3/2 parent covariance [units^2].
    lengthscale
        Lengthscale `ell` of the parent covariance [m]:
        `C(d) = signal_variance * (1 + r) * exp(-r)`, `r = sqrt(3) d / ell`.
    measurement_noise_variance
        Variance of the independent Gaussian sensor noise added to every
        observation [units^2]. Must be positive.
    reference_spacing
        Target spacing of the generated cell-centred reference grid [m]. Not
        used when explicit reference locations are given.
    prior_mean
        Constant prior mean [units].
    max_reference_nodes
        Dense size limit. A configured grid (before on-wall nodes are
        dropped) or an explicit set with more nodes is rejected; the
        resolution is never changed silently.
    max_neighbors
        Neighbour cap `k` of the nearest-clique rule for positions outside the
        reference set.
    covsel_tolerance
        Relative tolerance on the reproduced entries: covariance selection
        requires `max |L - K|` over the diagonal and the visible pairs to be
        at most `covsel_tolerance * signal_variance`.
    covsel_precision_tolerance
        Tolerance on the zero pattern: covariance selection also requires
        the normalised precision `|P_ij| / sqrt(P_ii P_jj)`, `P = inv(L)`, of
        every non-visible pair to be at most this value. It stops when both
        conditions hold.
    covsel_max_iterations
        Maximum number of sweeps over all cliques. Not converging within it is
        an error.
    max_cliques
        Maximum number of maximal cliques of the visibility graph. More cliques
        are an error (the dense solver is meant for small graphs).
    max_joint_query_size
        Largest batch accepted by a full-covariance (joint) query.
    """

    signal_variance: float
    lengthscale: float
    measurement_noise_variance: float
    reference_spacing: float
    prior_mean: float = 0.0
    max_reference_nodes: int = 200
    max_neighbors: int = 8
    covsel_tolerance: float = 1e-10
    covsel_precision_tolerance: float = 1e-9
    covsel_max_iterations: int = 500
    max_cliques: int = 2000
    max_joint_query_size: int = 2000

    def validate(self) -> None:
        """Raise ValueError for invalid values, TypeError for wrong types."""
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type == 'int':
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f'{f.name} must be an int, got {value!r}.')
                if value < 1:
                    raise ValueError(f'{f.name} must be >= 1, got {value}.')
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f'{f.name} must be a number, got {value!r}.')
            if not math.isfinite(value):
                raise ValueError(f'{f.name} must be finite, got {value}.')
            if f.name != 'prior_mean' and value <= 0.0:
                raise ValueError(f'{f.name} must be positive, got {value}.')
