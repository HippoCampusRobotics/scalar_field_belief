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

"""Exact GP belief with the Attentive Kernel (AK).

Model, in transformed units (float64):

    u = normalized position (fixed square normalizer, side L [m])
    y_t = (y - target_offset) / target_scale      (fixed, never refitted)
    f_t ~ GP(c_t, alpha_t * k_AK(u, u')),   y_t = f_t(u) + e,
    e ~ N(0, sigma2_t) independent.

The attention network, `alpha_t`, `sigma2_t` and `c_t` are learned by
full-batch exact marginal likelihood (empirical Bayes); the primitive
lengthscales and both transforms are fixed. Physical values:
mean = target_offset + target_scale * mean_t, (co)variances and noise
variance = target_scale^2 * (.)_t.

Every accepted measurement is conditioned on immediately. Optimization
events start after a warm-up of `min_distinct_locations` distinct locations
(positions farther than `distinct_location_separation` from every location
counted before), then follow a fixed cadence with a bounded number of Adam
steps;
network, parameters and optimizer state are kept across events. After each
change the noisy training covariance is factorized once (exact Cholesky, no
jitter) and queries reuse that factorization. `add_measurement` is atomic:
on a failure the previous state is restored and the error is raised.

The GPyTorch/linear_operator settings
used during training and torch's intra-op thread count (`torch_threads`,
set for all torch work of the belief, then restored) are process-global, so
one belief should only be used from one thread at a time. The belief may be
called inside `torch.no_grad()` or `torch.inference_mode()`: model
construction, reset and every add run with gradients enabled and outside
inference mode; queries never record gradients.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import math
import time
from dataclasses import dataclass

import gpytorch
import linear_operator
import numpy as np
import torch

from scalar_field_belief.attentive.config import AttentiveGPConfig
from scalar_field_belief.attentive.kernel import AttentiveKernel
from scalar_field_belief.models import ExactFieldGP
from scalar_field_belief.transforms import InputNormalizer2d

DTYPE = torch.float64

# Latent variances in [-VARIANCE_ROUNDOFF_TOLERANCE * alpha_t, 0) are
# roundoff and returned as 0 (counted); more negative values are an error.
VARIANCE_ROUNDOFF_TOLERANCE = 1e-9

_QUERY_CHUNK = 1024  # query rows per block of a marginal query


class AttentiveGPError(RuntimeError):
    """Numerical failure: factorization, optimization or variance."""


@dataclass(frozen=True)
class MarginalPrediction:
    """Result of `AttentiveGPBelief.predict_marginal`.

    `mean` [units] and `variance` [units^2] are latent (no sensor noise).
    `variance_clamps` counts roundoff-negative variances returned as 0.
    """

    mean: np.ndarray
    variance: np.ndarray
    variance_clamps: int


@dataclass(frozen=True)
class _Posterior:
    """Factorization of the noisy training covariance for one state."""

    version: int
    chol: torch.Tensor | None  # lower Cholesky of K_XX + sigma2_t I
    weights: torch.Tensor | None  # (K_XX + sigma2_t I)^-1 (y_t - c_t)
    mean_t: float
    amplitude_t: float
    noise_t: float


@contextlib.contextmanager
def exact_gpytorch_settings():
    """Dense Cholesky in every GPyTorch/linear_operator solve, no jitter.

    No CG/Lanczos (and therefore no random probe vectors from the global
    torch RNG). A failed Cholesky raises `NotPSDError`: the single retry
    adds a jitter of exactly 0 (`cholesky_max_tries(0)` is not usable with
    linear_operator 0.6.1).
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(gpytorch.settings.max_cholesky_size(10**9))
        stack.enter_context(
            gpytorch.settings.fast_computations(
                covar_root_decomposition=False, log_prob=False, solves=False
            )
        )
        stack.enter_context(gpytorch.settings.fast_pred_var(False))
        stack.enter_context(
            linear_operator.settings.cholesky_jitter(
                float_value=0.0, double_value=0.0
            )
        )
        stack.enter_context(linear_operator.settings.cholesky_max_tries(1))
        yield


@contextlib.contextmanager
def torch_threads(num_threads: int):
    """Run with `num_threads` torch intra-op threads (0: unchanged); the
    previous count is restored, also on exceptions."""
    previous = torch.get_num_threads()
    if num_threads <= 0 or num_threads == previous:
        yield
        return
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@contextlib.contextmanager
def _trainable():
    """Gradients enabled and inference mode off (model tensors created or
    trained here must stay usable for autograd)."""
    with torch.inference_mode(False), torch.enable_grad():
        yield


def _as_positions(xy) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f'Expected xy with shape (N, 2), got {xy.shape}.')
    if not np.all(np.isfinite(xy)):
        raise ValueError('xy must contain only finite values.')
    return xy


def _check_range(name: str, values) -> tuple[float, float]:
    lo, hi = (float(v) for v in values)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f'{name} must be finite with max > min, got {values}.')
    return lo, hi


class AttentiveGPBelief:
    """Exact GP belief over a 2D domain with the Attentive Kernel.

    Parameters
    ----------
    config
        Model, transform and training settings (validated here).
    x_range, y_range
        Domain bounds `(min, max)` [m]. They define the fixed input
        normalizer: a square of side `L = max(x extent, y extent)` anchored
        at `(x_min, y_min)`, so `u` lies in `[0, x extent / L] x
        [0, y extent / L]` and a normalized lengthscale `ell` is `ell * L`
        meters on both axes. Positions outside the bounds are accepted.

    Notes
    -----
    The belief receives positions and values only (no geometry, walls or
    source information). The latent prior of the seeded initial model is
    available immediately. It may be used inside `torch.no_grad()` or
    `torch.inference_mode()` contexts.
    """

    def __init__(self, config: AttentiveGPConfig, x_range, y_range):
        if not isinstance(config, AttentiveGPConfig):
            raise TypeError('config must be an AttentiveGPConfig.')
        config.validate()
        self.config = config
        self.x_range = _check_range('x_range', x_range)
        self.y_range = _check_range('y_range', y_range)
        x_min, y_min = self.x_range[0], self.y_range[0]
        self.input_scale = max(
            self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0]
        )
        self.normalizer = InputNormalizer2d(
            x_min=x_min,
            x_max=x_min + self.input_scale,
            y_min=y_min,
            y_max=y_min + self.input_scale,
        )
        self._lengthscales = config.primitive_lengthscales()
        self._scale2 = float(config.target_scale) ** 2
        self._state_version = 0
        self.reset()

    # ------------------------------------------------------------- model
    def _build_model(self) -> None:
        """Seeded initial model, likelihood, MLL and a fresh optimizer."""
        cfg = self.config
        kernel = AttentiveKernel(
            self._lengthscales,
            hidden_width=cfg.hidden_width,
            input_dim=2,
            seed=cfg.seed,
            dtype=DTYPE,
        )
        covar_module = gpytorch.kernels.ScaleKernel(kernel)
        min_noise_t = cfg.min_noise_variance / self._scale2
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(min_noise_t)
        )
        model = ExactFieldGP(None, None, likelihood, covar_module).double()
        with torch.no_grad():
            # GPyTorch stores constraint bounds in the default dtype; set
            # the configured floor exactly in float64.
            likelihood.noise_covar.raw_noise_constraint.lower_bound.fill_(
                min_noise_t
            )
        covar_module.outputscale = torch.tensor(
            cfg.init_amplitude / self._scale2, dtype=DTYPE
        )
        likelihood.noise = torch.tensor(
            cfg.init_noise_variance / self._scale2, dtype=DTYPE
        )
        named = [*model.named_parameters(), *model.named_buffers()]
        wrong = [name for name, t in named if t.dtype != DTYPE]
        if wrong:
            raise RuntimeError(f'expected float64 model tensors: {wrong}')

        network = list(kernel.network.parameters())
        network_ids = {id(p) for p in network}
        others = [
            (name, p)
            for name, p in model.named_parameters()
            if id(p) not in network_ids
        ]
        if len(network) + len(others) != len(list(model.parameters())) or (
            len(others) != 3
        ):
            raise RuntimeError(
                'unexpected parameter layout: '
                f'{[name for name, _ in model.named_parameters()]}'
            )
        self.optimizer = torch.optim.Adam(
            [
                {'params': network, 'lr': cfg.learning_rate_network},
                {
                    'params': [p for _, p in others],
                    'lr': cfg.learning_rate_hyperparameters,
                },
            ]
        )
        model.eval()
        self.model = model
        self.likelihood = likelihood
        self.kernel = kernel
        self._mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self._initial_network = [p.detach().clone() for p in network]

    def reset(self) -> None:
        """Remove all measurements and restore the seeded initial model,
        a fresh optimizer and zero counters (global RNG unchanged)."""
        with _trainable():
            self._reset()

    def _reset(self) -> None:
        self._build_model()
        self._xy = np.empty((0, 2))
        self._values = np.empty(0)
        self._train_x = torch.empty((0, 2), dtype=DTYPE)
        self._train_y = torch.empty(0, dtype=DTYPE)
        self._locations = np.empty((0, 2))  # counted distinct locations
        self._num_events = 0
        self._num_steps = 0
        self._since_event = 0
        self._events: tuple = ()
        self._last_add: dict | None = None
        self._last_failure: str | None = None
        self._total_conditioning_ms = 0.0
        self._total_optimization_ms = 0.0
        self._set_model_data()
        self._commit_posterior()

    def _set_model_data(self) -> None:
        if len(self._train_y) == 0:
            self.model.train_inputs = None
            self.model.train_targets = None
            self.model.prediction_strategy = None
        else:
            self.model.set_train_data(
                inputs=self._train_x, targets=self._train_y, strict=False
            )

    # ------------------------------------------------------------- state
    def has_model(self) -> bool:
        """Always True: the initial model's prior is available at once."""
        return True

    @property
    def num_measurements(self) -> int:
        return int(len(self._values))

    @property
    def xy_train_phys(self) -> np.ndarray:
        """Stored measurement positions `(N, 2)` [m] (a copy)."""
        return self._xy.copy()

    @property
    def y_train(self) -> np.ndarray:
        """Stored measurement values `(N,)` [units] (a copy)."""
        return self._values.copy()

    @property
    def num_distinct_locations(self) -> int:
        return len(self._locations)

    @property
    def phase(self) -> str:
        """'frozen' (optimization off), 'warm-up' (no event yet) or
        'training'."""
        if not self.config.optimize_parameters:
            return 'frozen'
        return 'warm-up' if self._num_events == 0 else 'training'

    @property
    def state_version(self) -> int:
        """Changes whenever data or parameters change (cache key)."""
        return self._state_version

    @property
    def events(self) -> tuple:
        """Record of every optimization event since the last reset."""
        return copy.deepcopy(self._events)

    def primitive_lengthscales(self) -> np.ndarray:
        """Primitive lengthscales [normalized input units]."""
        return np.array(self._lengthscales)

    def primitive_lengthscales_m(self) -> np.ndarray:
        """Primitive lengthscales [m] (the same on both axes)."""
        return np.array(self._lengthscales) * self.input_scale

    def measurement_noise_variance(self) -> float:
        """Learned sensor-noise variance [units^2]."""
        return self._scale2 * self._posterior.noise_t

    def model_description(self) -> dict:
        """Settings derived from the configuration and bounds, and the
        digest of the seeded initial network (it depends on the torch
        initialization)."""
        cfg = self.config
        h = cfg.hidden_width
        lengthscales_m = self.primitive_lengthscales_m().tolist()
        return {
            'kernel': 'k = alpha <r(u), r(v)> sum_m r_m(u) r_m(v) '
            'exp(-|u - v|^2 / (2 ell_m^2)), r = softmax / L2 norm',
            'network': f'Linear(2, {h})-tanh-Linear({h}, {h})-tanh-'
            f'Linear({h}, {cfg.num_components})-softmax, shared w = z',
            'dtype': 'float64',
            'input_normalizer': {
                'x_min': self.normalizer.x_min,
                'x_max': self.normalizer.x_max,
                'y_min': self.normalizer.y_min,
                'y_max': self.normalizer.y_max,
                'side_m': self.input_scale,
                'u_range': [
                    (self.x_range[1] - self.x_range[0]) / self.input_scale,
                    (self.y_range[1] - self.y_range[0]) / self.input_scale,
                ],
            },
            'primitive_lengthscales': list(self._lengthscales),
            'primitive_lengthscales_m': lengthscales_m,
            'target_transform': 'y_t = (y - target_offset) / target_scale '
            '(fixed)',
            'initial_transformed': {
                'amplitude': cfg.init_amplitude / self._scale2,
                'noise_variance': cfg.init_noise_variance / self._scale2,
                'noise_variance_floor': cfg.min_noise_variance / self._scale2,
                'constant_mean': 0.0,
            },
            'optimizer': 'Adam, one instance kept across events; groups: '
            f'network weights lr {cfg.learning_rate_network}, amplitude, '
            'noise variance and constant mean (raw, transformed units) lr '
            f'{cfg.learning_rate_hyperparameters}',
            'loss': 'negative exact marginal log likelihood per observation '
            '(transformed targets)',
            'warm_up': f'first event after {cfg.min_distinct_locations} '
            'distinct locations (greedy, in observation order: farther than '
            f'{cfg.distinct_location_separation} m from every counted one)',
            'schedule': f'first event {cfg.initial_training_steps} steps, '
            f'then {cfg.training_steps_per_event} steps after every '
            f'{cfg.optimize_every} accepted measurements',
            'torch_threads': f'{cfg.torch_threads} for adds, training, '
            'factorization and queries (0: torch setting unchanged)',
            'prediction': 'exact Cholesky of K_XX + noise I per state, no '
            f'jitter; variances below -{VARIANCE_ROUNDOFF_TOLERANCE:g} * '
            'alpha_t are an error, roundoff-negative ones are set to 0',
            'initial_network_sha256': self.initial_state_digest(),
        }

    def initial_state_digest(self) -> str:
        """sha256 over the seeded initial network parameters (float64
        bytes, in parameter order)."""
        digest = hashlib.sha256()
        for p in self._initial_network:
            digest.update(p.numpy().tobytes())
        return digest.hexdigest()

    def status(self) -> dict:
        """Counts, phase, losses and timings (JSON-friendly)."""
        cfg = self.config
        if self.phase == 'training':
            due_in = max(cfg.optimize_every - self._since_event, 0)
        else:
            due_in = None
        return {
            'ready': True,
            'phase': self.phase,
            'num_measurements': self.num_measurements,
            'num_distinct_locations': self.num_distinct_locations,
            'min_distinct_locations': cfg.min_distinct_locations,
            'optimization_events': self._num_events,
            'optimization_steps': self._num_steps,
            'measurements_since_last_event': self._since_event,
            'measurements_until_next_event': due_in,
            'last_event': copy.deepcopy(self._events[-1])
            if self._events
            else None,
            'last_add': copy.deepcopy(self._last_add),
            'last_rejected': self._last_failure,
            'total_conditioning_ms': self._total_conditioning_ms,
            'total_optimization_ms': self._total_optimization_ms,
            'state_version': self._state_version,
        }

    def fitted_summary(self) -> dict:
        """Current amplitude, noise and mean in physical units, and the
        change of the network weights from their seeded initial values."""
        post = self._posterior
        offset, scale = self.config.target_offset, self.config.target_scale
        with torch.no_grad():
            current = torch.cat(
                [
                    p.detach().reshape(-1)
                    for p in self.kernel.network.parameters()
                ]
            )
            initial = torch.cat([p.reshape(-1) for p in self._initial_network])
            change = float(torch.linalg.vector_norm(current - initial))
            norm0 = float(torch.linalg.vector_norm(initial))
        amplitude = self._scale2 * post.amplitude_t
        noise = self._scale2 * post.noise_t
        return {
            'amplitude': amplitude,
            'amplitude_std': math.sqrt(amplitude),
            'noise_variance': noise,
            'noise_std': math.sqrt(noise),
            'constant_mean': offset + scale * post.mean_t,
            'transformed': {
                'amplitude': post.amplitude_t,
                'noise_variance': post.noise_t,
                'constant_mean': post.mean_t,
            },
            'network_parameters': int(current.numel()),
            'network_change_l2': change,
            'network_initial_l2': norm0,
            'network_change_relative': change / norm0 if norm0 > 0 else None,
        }

    # -------------------------------------------------------- measurements
    def add_measurement(self, x: float, y: float, value: float) -> None:
        """Store one measurement, optimize if an event is due, condition.

        Atomic: on any failure the previous data, parameters, optimizer
        state, counters and factorization are restored and the error is
        raised (ValueError for non-finite input, AttentiveGPError for a
        numerical failure).
        """
        x, y, value = float(x), float(y), float(value)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError('Measurement x/y must be finite.')
        if not math.isfinite(value):
            raise ValueError('Measurement value must be finite.')
        with _trainable(), torch_threads(self.config.torch_threads):
            snapshot = self._capture()
            try:
                self._add(x, y, value)
            except BaseException as exc:
                self._restore(snapshot)
                self._last_failure = (
                    f'measurement #{self.num_measurements + 1} at ({x}, {y}): '
                    f'{type(exc).__name__}: {exc}'
                )
                raise

    def _add(self, x: float, y: float, value: float) -> None:
        t0 = time.perf_counter()
        self._append(x, y, value)
        optimization_ms = None
        if self._event_due():
            t1 = time.perf_counter()
            self._run_event()
            optimization_ms = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()
        self._commit_posterior()
        t3 = time.perf_counter()
        total_ms = (t3 - t0) * 1e3
        conditioning_ms = total_ms - (optimization_ms or 0.0)
        self._last_add = {
            'measurement': self.num_measurements,
            'conditioning_ms': conditioning_ms,
            'factorization_ms': (t3 - t2) * 1e3,
            'optimization_ms': optimization_ms,
        }
        self._total_conditioning_ms += conditioning_ms
        self._total_optimization_ms += optimization_ms or 0.0
        self._last_failure = None

    def _append(self, x: float, y: float, value: float) -> None:
        cfg = self.config
        u = self.normalizer.transform(np.array([[x, y]]))
        self._xy = np.vstack([self._xy, [[x, y]]])
        self._values = np.append(self._values, value)
        self._train_x = torch.cat(
            [self._train_x, torch.as_tensor(u, dtype=DTYPE)]
        )
        y_t = (value - cfg.target_offset) / cfg.target_scale
        self._train_y = torch.cat(
            [self._train_y, torch.tensor([y_t], dtype=DTYPE)]
        )
        counted = self._locations
        distance = np.hypot(counted[:, 0] - x, counted[:, 1] - y)
        if not np.any(distance <= cfg.distinct_location_separation):
            self._locations = np.vstack([counted, [[x, y]]])
        self._since_event += 1
        self._set_model_data()

    def _event_due(self) -> bool:
        cfg = self.config
        if not cfg.optimize_parameters:
            return False
        if self._num_events == 0:
            return len(self._locations) >= cfg.min_distinct_locations
        return self._since_event >= cfg.optimize_every

    def _training_loss(self) -> torch.Tensor:
        """Negative exact marginal log likelihood per observation
        (transformed targets): the only loss evaluation of an event.
        Zero gradients are valid (e.g. the network for M = 1); only
        non-finite losses or gradients fail an event."""
        output = self.model(self._train_x)
        return -self._mll(output, self._train_y)

    def _run_event(self) -> None:
        """Bounded full-batch Adam on the exact marginal likelihood."""
        cfg = self.config
        first = self._num_events == 0
        steps = (
            cfg.initial_training_steps
            if first
            else cfg.training_steps_per_event
        )
        named = list(self.model.named_parameters())
        t0 = time.perf_counter()
        step = 0
        loss_before = None
        self.model.train()
        try:
            with exact_gpytorch_settings():
                for step in range(steps):
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = self._training_loss()
                    loss_value = float(loss.detach())
                    if not math.isfinite(loss_value):
                        raise AttentiveGPError(f'non-finite loss {loss_value}')
                    if step == 0:
                        loss_before = loss_value
                    loss.backward()
                    bad = [
                        name
                        for name, p in named
                        if p.grad is not None
                        and not bool(torch.isfinite(p.grad).all())
                    ]
                    if bad:
                        raise AttentiveGPError(f'non-finite gradient in {bad}')
                    if step == 0:
                        no_gradient = [n for n, p in named if p.grad is None]
                    self.optimizer.step()
                with torch.no_grad():
                    loss_after = float(self._training_loss())
                if not math.isfinite(loss_after):
                    raise AttentiveGPError(f'non-finite loss {loss_after}')
        except RuntimeError as exc:  # incl. NotPSDError and LinAlgError
            raise AttentiveGPError(
                f'optimization event {self._num_events + 1} failed at step '
                f'{step + 1}/{steps} with {self.num_measurements} '
                f'measurements: {type(exc).__name__}: {exc}'
            ) from exc
        finally:
            self.model.eval()
        self._num_events += 1
        self._num_steps += steps
        self._since_event = 0
        event = {
            'event': self._num_events,
            'measurement': self.num_measurements,
            'initial': first,
            'steps': steps,
            'loss_before': loss_before,
            'loss_after': loss_after,
            'optimization_ms': (time.perf_counter() - t0) * 1e3,
            'parameters_without_gradient': no_gradient,
        }
        self._events = (*self._events, event)

    def _capture(self) -> dict:
        return {
            'model': {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()
            },
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            'xy': self._xy,
            'values': self._values,
            'train_x': self._train_x,
            'train_y': self._train_y,
            'locations': self._locations,
            'num_events': self._num_events,
            'num_steps': self._num_steps,
            'since_event': self._since_event,
            'events': self._events,
            'posterior': self._posterior,
            'state_version': self._state_version,
        }

    def _restore(self, snapshot: dict) -> None:
        self.model.load_state_dict(snapshot['model'])
        self.optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
        self.model.eval()
        self._xy = snapshot['xy']
        self._values = snapshot['values']
        self._train_x = snapshot['train_x']
        self._train_y = snapshot['train_y']
        self._locations = snapshot['locations']
        self._num_events = snapshot['num_events']
        self._num_steps = snapshot['num_steps']
        self._since_event = snapshot['since_event']
        self._events = snapshot['events']
        self._set_model_data()
        self.model.prediction_strategy = None
        self._posterior = snapshot['posterior']
        self._state_version = snapshot['state_version']

    # ------------------------------------------------------- conditioning
    def _commit_posterior(self) -> None:
        """Factorize the current state; replaces the cache on success."""
        posterior = self._factorize(self._state_version + 1)
        self.model.prediction_strategy = None
        self._posterior = posterior
        self._state_version = posterior.version

    def _factorize(self, version: int) -> _Posterior:
        try:
            with torch.no_grad():
                return self._factorize_state(version)
        except AttentiveGPError:
            raise
        except RuntimeError as exc:
            raise AttentiveGPError(
                f'factorization with {len(self._train_y)} measurements '
                f'failed: {type(exc).__name__}: {exc}'
            ) from exc

    def _factorize_state(self, version: int) -> _Posterior:
        amplitude_t = float(self.model.covar_module.outputscale)
        noise_t = float(self.likelihood.noise)
        mean_t = float(self.model.mean_module.constant)
        for name, v in (
            ('amplitude', amplitude_t),
            ('noise variance', noise_t),
            ('constant mean', mean_t),
        ):
            if not math.isfinite(v):
                raise AttentiveGPError(f'non-finite {name}: {v}')
        n = len(self._train_y)
        if n == 0:
            return _Posterior(version, None, None, mean_t, amplitude_t, noise_t)
        cov = self.model.covar_module.forward(self._train_x, self._train_x)
        if not bool(torch.isfinite(cov).all()):
            raise AttentiveGPError('non-finite training covariance.')
        cov = cov + noise_t * torch.eye(n, dtype=DTYPE)
        chol, info = torch.linalg.cholesky_ex(cov)
        if int(info) != 0:
            raise AttentiveGPError(
                'Cholesky factorization of the noisy training covariance '
                f'failed: leading minor {int(info)} of {n} is not '
                'positive definite (no jitter is added).'
            )
        residual = (self._train_y - mean_t).unsqueeze(-1)
        weights = torch.cholesky_solve(residual, chol).squeeze(-1)
        if not bool(torch.isfinite(weights).all()):
            raise AttentiveGPError('non-finite posterior weights.')
        return _Posterior(version, chol, weights, mean_t, amplitude_t, noise_t)

    # ------------------------------------------------------------- queries
    def _inputs(self, xy) -> torch.Tensor:
        u = self.normalizer.transform(_as_positions(xy))
        return torch.as_tensor(u, dtype=DTYPE)

    def _check_variance(self, variance: np.ndarray, what: str) -> np.ndarray:
        tolerance = VARIANCE_ROUNDOFF_TOLERANCE * self._posterior.amplitude_t
        if not np.all(np.isfinite(variance)):
            raise AttentiveGPError(f'non-finite {what}.')
        if variance.size and float(variance.min()) < -tolerance:
            raise AttentiveGPError(
                f'negative {what} {float(variance.min()):.3e} (transformed '
                f'units) beyond roundoff tolerance {tolerance:.3e}.'
            )
        return variance < 0.0

    def predict_marginal(self, xy) -> MarginalPrediction:
        """Latent posterior mean [units] and variance [units^2] in blocks
        (no `(N, N)` matrix). Does not change the belief."""
        u = self._inputs(xy)
        post = self._posterior
        covar = self.model.covar_module
        n = len(u)
        mean_t = np.empty(n)
        var_t = np.empty(n)
        threads = torch_threads(self.config.torch_threads)
        with torch.no_grad(), threads:
            for start in range(0, n, _QUERY_CHUNK):
                block = slice(start, start + _QUERY_CHUNK)
                q = u[block]
                prior = covar.forward(q, q, diag=True)
                if post.chol is None:
                    mean_t[block] = post.mean_t
                    var_t[block] = prior.numpy()
                    continue
                k_qx = covar.forward(q, self._train_x)
                mean_t[block] = (post.mean_t + k_qx @ post.weights).numpy()
                v = torch.linalg.solve_triangular(
                    post.chol, k_qx.transpose(-1, -2), upper=False
                )
                var_t[block] = (prior - (v * v).sum(0)).numpy()
        if not np.all(np.isfinite(mean_t)):
            raise AttentiveGPError('non-finite posterior mean.')
        negative = self._check_variance(var_t, 'posterior variance')
        var_t[negative] = 0.0
        offset, scale = self.config.target_offset, self.config.target_scale
        return MarginalPrediction(
            mean=offset + scale * mean_t,
            variance=self._scale2 * var_t,
            variance_clamps=int(negative.sum()),
        )

    def query(self, xy) -> tuple[np.ndarray, np.ndarray]:
        """Latent posterior mean `(N,)` [units] and variance `(N,)`
        [units^2]. Raises ValueError for a wrong shape or non-finite
        input, AttentiveGPError for a variance below the roundoff
        tolerance."""
        prediction = self.predict_marginal(xy)
        return prediction.mean, prediction.variance

    def query_observation(self, xy) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean `(N,)` [units] and variance `(N,)` [units^2] of a
        new noisy observation: latent variance plus the learned sensor-noise
        variance."""
        prediction = self.predict_marginal(xy)
        return (
            prediction.mean,
            prediction.variance + self.measurement_noise_variance(),
        )

    def query_with_covariance(self, xy) -> tuple[np.ndarray, np.ndarray]:
        """Latent posterior mean `(N,)` [units] and covariance `(N, N)`
        [units^2], symmetrized, not clipped. Raises ValueError for more than
        `max_joint_query_size` positions, AttentiveGPError for a diagonal
        entry below the roundoff tolerance."""
        xy = _as_positions(xy)
        n = len(xy)
        if n > self.config.max_joint_query_size:
            raise ValueError(
                f'joint query of {n} positions exceeds max_joint_query_size '
                f'= {self.config.max_joint_query_size}; use query() for '
                'marginal variances.'
            )
        u = self._inputs(xy)
        post = self._posterior
        covar = self.model.covar_module
        threads = torch_threads(self.config.torch_threads)
        with torch.no_grad(), threads:
            cov = covar.forward(u, u)
            mean = torch.full((n,), post.mean_t, dtype=DTYPE)
            if post.chol is not None:
                k_qx = covar.forward(u, self._train_x)
                mean = mean + k_qx @ post.weights
                v = torch.linalg.solve_triangular(
                    post.chol, k_qx.transpose(-1, -2), upper=False
                )
                cov = cov - v.transpose(-1, -2) @ v
            cov = 0.5 * (cov + cov.transpose(-1, -2))
        mean_t, cov_t = mean.numpy(), cov.numpy()
        if not (np.all(np.isfinite(mean_t)) and np.all(np.isfinite(cov_t))):
            raise AttentiveGPError('non-finite joint posterior.')
        self._check_variance(np.diag(cov_t), 'posterior variance')
        offset, scale = self.config.target_offset, self.config.target_scale
        return offset + scale * mean_t, self._scale2 * cov_t

    def attention_representations(self, xy) -> np.ndarray:
        """L2-normalized attention representations `(N, M)` at physical
        positions (diagnostics; component indices carry no physical
        meaning). Does not change the belief."""
        u = self._inputs(xy)
        threads = torch_threads(self.config.torch_threads)
        with torch.no_grad(), threads:
            return self.kernel.representations(u).numpy().copy()
