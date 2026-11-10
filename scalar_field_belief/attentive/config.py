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

"""Configuration of the Attentive Kernel (AK) GP belief.

Units: positions in meters, values in the measurement unit of the scalar
field ("units"). Amplitude and noise settings are VARIANCES (units^2), not
standard deviations. The primitive lengthscales are in normalized input
units: the belief maps positions with a fixed square normalizer of side
`L = max(x extent, y extent)` [m], so a normalized lengthscale `ell`
corresponds to `ell * L` meters on both axes.

All values are fixed for the lifetime of a belief. The network weights,
the amplitude, the sensor-noise variance and the constant mean are learned
from the measurements (unless `optimize_parameters` is False); the
transforms and the primitive lengthscales are never fitted.

The initial values are given in physical units, so the prior and all
frozen-mode predictions do not depend on `target_scale`. `target_offset` is
the constant prior mean (and stays the mean in frozen mode). Trained results
depend on both: `target_offset` and `target_scale` (together with the
learning rates) set the optimization dynamics and the starting constant
mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

_INT_FIELDS = (
    'num_components',
    'hidden_width',
    'seed',
    'min_distinct_locations',
    'initial_training_steps',
    'optimize_every',
    'training_steps_per_event',
    'max_joint_query_size',
    'torch_threads',
)
_NONNEGATIVE_INT_FIELDS = ('seed', 'torch_threads')
_BOOL_FIELDS = ('optimize_parameters',)
_SIGNED_FLOAT_FIELDS = ('target_offset',)
_NONNEGATIVE_FLOAT_FIELDS = ('distinct_location_separation',)
_MAX_SEED = 2**63 - 1


def _check_field(name: str, value) -> None:
    """Type and sign of one field (ranges between fields are separate)."""
    if name in _BOOL_FIELDS:
        if not isinstance(value, bool):
            raise TypeError(f'{name} must be a bool, got {value!r}.')
        return
    if name in _INT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f'{name} must be an int, got {value!r}.')
        lower = 0 if name in _NONNEGATIVE_INT_FIELDS else 1
        if value < lower:
            raise ValueError(f'{name} must be >= {lower}, got {value}.')
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be a number, got {value!r}.')
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite, got {value}.')
    if name in _NONNEGATIVE_FLOAT_FIELDS and value < 0.0:
        raise ValueError(f'{name} must be >= 0, got {value}.')
    if name not in _SIGNED_FLOAT_FIELDS + _NONNEGATIVE_FLOAT_FIELDS and (
        value <= 0.0
    ):
        raise ValueError(f'{name} must be positive, got {value}.')


@dataclass(frozen=True, slots=True)
class AttentiveGPConfig:
    """Fixed settings of the AK GP belief.

    Parameters
    ----------
    num_components
        Number `M` of primitive RBF components and of network outputs.
    hidden_width
        Width `H` of both tanh hidden layers of the attention network.
    min_lengthscale, max_lengthscale
        Range of the fixed primitive lengthscales
        `linspace(min_lengthscale, max_lengthscale, M)` [normalized input
        units; multiply by `L` for meters]. They must be equal for `M = 1`.
    seed
        Seed of the network initialization (a local generator state; the
        global torch RNG is not changed).
    target_offset
        Offset of the fixed affine target transform [units]:
        `y_t = (y - target_offset) / target_scale`.
    target_scale
        Scale of the fixed affine target transform [units], > 0. It does not
        change the prior or frozen-mode predictions, but it changes training
        (see the module docstring); the reported loss is per observation in
        transformed units, and the physical negative log likelihood per
        observation is `loss + log(target_scale)`.
    init_amplitude
        Initial kernel amplitude `alpha` [units^2] (prior variance
        `k(u, u)`).
    init_noise_variance
        Initial sensor-noise variance [units^2]; must exceed
        `min_noise_variance`.
    min_noise_variance
        Lower bound of the learned sensor-noise variance [units^2], > 0.
    optimize_parameters
        False: frozen mode. Parameters stay at their initial values; every
        measurement is still conditioned on.
    min_distinct_locations
        Warm-up: optimization starts once this many distinct locations have
        been measured. Repeated observations at one location are data but
        not new coverage.
    distinct_location_separation
        Distance [m] for the warm-up rule: a position within this distance
        of an already counted location is a repeat, otherwise it is counted
        as a new location (greedy, in observation order). 0 counts only
        exactly equal positions as repeats.
    initial_training_steps
        Adam steps of the first optimization event.
    optimize_every
        Later events run after every `optimize_every` accepted
        measurements.
    training_steps_per_event
        Adam steps of each later event.
    learning_rate_network
        Adam learning rate of the attention-network weights.
    learning_rate_hyperparameters
        Adam learning rate of the amplitude, the noise variance and the
        constant mean (raw parameters in transformed units).
    max_joint_query_size
        Largest batch accepted by `query_with_covariance`.
    torch_threads
        torch intra-op threads for all torch work of the belief (adds,
        training, factorization, queries); the previous count is restored
        afterwards, also on errors. 0 leaves torch's setting unchanged. The
        matrices are small, so multi-threading overhead dominates and one
        thread is typically several times faster; with one thread the
        trained parameters do not depend on the core count (summation
        order). torch's thread count is process-global, so use the belief
        from one thread at a time.
    """

    num_components: int = 10
    hidden_width: int = 10
    min_lengthscale: float = 0.02
    max_lengthscale: float = 0.5
    seed: int = 0
    target_offset: float = 0.0
    target_scale: float = 0.01
    init_amplitude: float = 1.0e-4
    init_noise_variance: float = 1.0e-6
    min_noise_variance: float = 1.0e-8
    optimize_parameters: bool = True
    min_distinct_locations: int = 10
    distinct_location_separation: float = 0.05
    initial_training_steps: int = 100
    optimize_every: int = 1
    training_steps_per_event: int = 5
    learning_rate_network: float = 3.0e-3
    learning_rate_hyperparameters: float = 3.0e-2
    max_joint_query_size: int = 2000
    torch_threads: int = 1

    def validate(self) -> None:
        """Raise ValueError for invalid values, TypeError for wrong types."""
        for f in fields(self):
            _check_field(f.name, getattr(self, f.name))
        if self.seed > _MAX_SEED:
            raise ValueError(f'seed must be <= {_MAX_SEED}, got {self.seed}.')
        if self.min_lengthscale > self.max_lengthscale:
            raise ValueError(
                'min_lengthscale must not exceed max_lengthscale, got '
                f'{self.min_lengthscale} > {self.max_lengthscale}.'
            )
        if self.num_components == 1 and (
            self.min_lengthscale != self.max_lengthscale
        ):
            raise ValueError(
                'num_components = 1 needs min_lengthscale == max_lengthscale.'
            )
        if self.init_noise_variance <= self.min_noise_variance:
            raise ValueError(
                'init_noise_variance must exceed min_noise_variance, got '
                f'{self.init_noise_variance} <= {self.min_noise_variance}.'
            )

    def primitive_lengthscales(self) -> list[float]:
        """Primitive lengthscales [normalized input units], ascending."""
        m = self.num_components
        if m == 1:
            return [float(self.min_lengthscale)]
        step = (self.max_lengthscale - self.min_lengthscale) / (m - 1)
        values = [self.min_lengthscale + k * step for k in range(m)]
        values[-1] = float(self.max_lengthscale)
        return [float(v) for v in values]
