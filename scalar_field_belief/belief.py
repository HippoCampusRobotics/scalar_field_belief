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

"""Exact-GP belief model for a 2D scalar field.

This module contains the core, ROS-independent belief implementation. The class
stores scalar measurements in physical coordinates, refits an exact Gaussian
process according to a configurable policy, and answers posterior queries at
physical `(x, y)` positions.

In wall-aware mode, the GP inputs are augmented with smooth features of a
static wall geometry. Walls, measurements, and queries must all use the same
physical frame and units (meters).

The ROS node is responsible for translating messages and services to this plain
Python interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gpytorch
import numpy as np
import torch

from scalar_field_belief.config import BeliefConfig
from scalar_field_belief.geometry import WallGeometry
from scalar_field_belief.models import (
    ExactFieldGP,
    build_covar_module,
    get_spatial_kernel,
    get_wall_kernel,
    initialize_model_hyperparameters,
)
from scalar_field_belief.transforms import (
    InputNormalizer2d,
    TargetStandardizer,
    build_gp_inputs,
)
from scalar_field_belief.wall_features import WallFeatureMap

# Tolerance for comparing configuration bounds with wall geometry bounds.
_BOUNDS_REL_TOL = 1e-9
_BOUNDS_ABS_TOL = 1e-9


@dataclass
class FitResult:
    did_refit: bool
    num_measurements: int


class ScalarFieldBelief:
    """Simple exact-GP belief over a 2D scalar field.

    The belief stores all measurements in physical coordinates, normalizes
    inputs before passing them to the GP, and standardizes targets
    during model fitting.

    Parameters
    ----------
    config
        Configuration for domain bounds, GP hyperparameters, refit behavior, and
        visualization settings.
    wall_geometry
        Static wall geometry in the same physical frame as measurements and
        queries. Required with at least one wall when
        `config.model_variant == 'wall_aware'`; its bounds must agree with the
        configuration bounds. Ignored in baseline mode.

    Raises
    ------
    ValueError
        If the configuration is invalid, or in wall-aware mode if the
        geometry is missing, has no walls, or its bounds differ from the
        configuration bounds.
    TypeError
        In wall-aware mode, if `wall_geometry` is not a `WallGeometry`.

    Notes
    -----
    Inputs are always represented in physical coordinates `(x, y)` at the public
    API boundary of this class.

    Internally, the GP operates on normalized inputs and standardized target
    values. In wall-aware mode, one dimensionless wall feature per wall,
    computed from the physical coordinates, is appended to the normalized
    inputs. Query results are transformed back to original measurement units.

    The query method returns the latent GP posterior mean and variance.
    """

    def __init__(
        self,
        config: BeliefConfig,
        *,
        wall_geometry: WallGeometry | None = None,
    ):
        config.validate()
        self.config = config
        self.normalizer = InputNormalizer2d(
            x_min=config.x_min,
            x_max=config.x_max,
            y_min=config.y_min,
            y_max=config.y_max,
        )

        # The wall feature map is fixed for the lifetime of the belief. It is
        # never constructed or evaluated in baseline mode.
        self.wall_geometry: WallGeometry | None = None
        self.wall_feature_map: WallFeatureMap | None = None
        if config.model_variant == 'wall_aware':
            self._check_wall_geometry(wall_geometry)
            self.wall_geometry = wall_geometry
            self.wall_feature_map = WallFeatureMap(
                geometry=wall_geometry,
                beta=config.wall_beta,
                rho=config.wall_rho,
            )

        self.reset()

    @property
    def wall_feature_dim(self) -> int:
        """Number of wall features `M` (zero in baseline mode)."""
        if self.wall_feature_map is None:
            return 0
        return self.wall_feature_map.num_walls

    def _check_wall_geometry(self, geometry: WallGeometry | None) -> None:
        """Validate the wall geometry required by wall-aware mode."""
        if geometry is None:
            raise ValueError(
                "model_variant 'wall_aware' requires a wall_geometry."
            )
        if not isinstance(geometry, WallGeometry):
            raise TypeError('wall_geometry must be a WallGeometry.')
        if len(geometry.walls) == 0:
            raise ValueError(
                f"Wall geometry '{geometry.name}' has no walls; "
                "model_variant 'wall_aware' requires at least one wall."
            )

        config_bounds = (*self.config.x_range, *self.config.y_range)
        geometry_bounds = (*geometry.x_range, *geometry.y_range)
        if not all(
            math.isclose(a, b, rel_tol=_BOUNDS_REL_TOL, abs_tol=_BOUNDS_ABS_TOL)
            for a, b in zip(config_bounds, geometry_bounds, strict=True)
        ):
            raise ValueError(
                f"Wall geometry '{geometry.name}' bounds "
                f'x={geometry.x_range}, y={geometry.y_range} do not match '
                f'the configured bounds x={self.config.x_range}, '
                f'y={self.config.y_range}.'
            )

    def reset(self) -> None:
        """Clear all stored measurements and fitted model state.

        After reset, the belief has no fitted model. A query will fail until
        enough measurements have been added to trigger the first refit. The
        configured wall geometry and wall feature map are kept.
        """
        self.xy_train_phys = np.empty((0, 2), dtype=float)
        self.y_train = np.empty((0,), dtype=float)
        self.new_since_last_fit = 0
        self.model: ExactFieldGP | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.standardizer: TargetStandardizer | None = None

    def add_measurement(self, x: float, y: float, value: float) -> FitResult:
        """Add one scalar measurement and refit if required/according to
        configured refit policy.

        Parameters
        ----------
        x
            Physical x coordinate of the measurement.
        y
            Physical y coordinate of the measurement.
        value
            Scalar measurement value.

        Returns
        -------
        FitResult
            Information about whether the model was refitted and how many
            measurements are stored.

        Raises
        ------
        ValueError
            If `x`, `y`, or `value` is not finite.
        """
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError('Measurement x/y must be finite.')
        if not np.isfinite(value):
            raise ValueError('Measurement value must be finite.')

        xy = np.array([[x, y]], dtype=float)
        val = np.array([value], dtype=float)
        self.xy_train_phys = np.vstack([self.xy_train_phys, xy])
        self.y_train = np.concatenate([self.y_train, val])
        self.new_since_last_fit += 1

        did_refit = self.maybe_refit()
        if not did_refit and self.has_model():
            self._condition_on_data()

        return FitResult(
            did_refit=did_refit, num_measurements=len(self.y_train)
        )

    def maybe_refit(self) -> bool:
        """Refit the GP if the configured refit policy says so."""
        if len(self.y_train) == 0:
            return False

        should_refit = False
        if self.config.refit_policy == 'every_measurement':
            should_refit = True
        elif self.config.refit_policy == 'every_k_measurements':
            should_refit = self.new_since_last_fit >= self.config.refit_every_k
        else:
            raise ValueError(
                f'Unknown refit_policy: {self.config.refit_policy}'
            )

        if not should_refit:
            return False

        self._fit_model()
        self.new_since_last_fit = 0
        return True

    def has_model(self) -> bool:
        """Return whether a fitted model and its associated transforms exist."""
        return (
            self.model is not None
            and self.likelihood is not None
            and self.standardizer is not None
        )

    def query(self, xy_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Query latent posterior mean and variance at physical positions.

        Parameters
        ----------
        xy_phys
            Physical query positions with shape `(N, 2)`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Posterior latent mean and variance at the query positions. Both are
            returned in original measurement units. The variance is in
            measurement units squared.

        Raises
        ------
        RuntimeError
            If no fitted model is available yet.
        ValueError
            If `xy_phys` does not have shape `(N, 2)` or contains non-finite
            values.

        Notes
        -----
        This method does not refit the model and does not update the target
        standardizer. It only evaluates the currently fitted belief state.
        """
        if not self.has_model():
            raise RuntimeError('Belief has no fitted model yet.')

        xy_phys = np.asarray(xy_phys, dtype=float)
        if xy_phys.ndim != 2 or xy_phys.shape[1] != 2:
            raise ValueError(
                f'xy_phys must have shape (N, 2), got {xy_phys.shape}'
            )
        if not np.all(np.isfinite(xy_phys)):
            raise ValueError('xy_phys must contain only finite values.')

        query_x = self._build_query_tensor(xy_phys)

        assert self.model is not None
        assert self.likelihood is not None
        assert self.standardizer is not None

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(query_x)
            mean_norm = posterior.mean
            var_norm = posterior.variance

        mean, var = self.standardizer.inverse_transform_mean_var(
            mean_norm, var_norm
        )
        return mean.detach().cpu().numpy(), var.detach().cpu().numpy()

    def query_with_covariance(
        self, xy_phys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query posterior mean and full covariance at physical positions.

        Unlike `query()`, this returns the full (N, N) covariance matrix, not
        just the diagonal.

        Parameters
        ----------
        xy_phys
            Physical query positions with shape `(N, 2)`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Posterior mean `(N,)` and posterior covariance `(N, N)`, in
            original measurement units.

        Raises
        ------
        RuntimeError
            If no fitted model is available yet.
        ValueError
            If `xy_phys` does not have shape `(N, 2)` or contains non-finite
            values.
        """
        if not self.has_model():
            raise RuntimeError('Belief has no fitted model yet.')

        xy_phys = np.asarray(xy_phys, dtype=float)
        if xy_phys.ndim != 2 or xy_phys.shape[1] != 2:
            raise ValueError(
                f'xy_phys must have shape (N, 2), got {xy_phys.shape}'
            )
        if not np.all(np.isfinite(xy_phys)):
            raise ValueError('xy_phys must contain only finite values.')

        query_x = self._build_query_tensor(xy_phys)

        assert self.model is not None
        assert self.likelihood is not None
        assert self.standardizer is not None

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(query_x)
            mean_norm = posterior.mean
            covar_norm = posterior.covariance_matrix

        mean = mean_norm * self.standardizer.std + self.standardizer.mean
        covar = self.standardizer.inverse_transform_covar(covar_norm)
        return mean.detach().cpu().numpy(), covar.detach().cpu().numpy()

    def measurement_noise_variance(self) -> float:
        """Return the GP likelihood noise variance (sigma_n^2) in physical
        measurement units.

        Raises
        ------
        RuntimeError
            If no fitted model is available yet.
        """
        if not self.has_model():
            raise RuntimeError('Belief has no fitted model yet.')

        assert self.likelihood is not None
        assert self.standardizer is not None

        noise_norm = self.likelihood.noise.detach()
        return float((noise_norm * self.standardizer.std**2).item())

    def fitted_hyperparameters_physical(self) -> dict[str, float]:
        """Return the currently fitted GP kernel hyperparameters in physical
        units: lengthscale_x_m/lengthscale_y_m (meters, via the domain
        extent), outputscale (measurement units squared, via
        standardizer.std**2 - same convention as measurement_noise_variance()),
        and noise_variance (measurement units squared).

        In wall-aware mode, the dictionary additionally contains
        wall_feature_lengthscale_0, wall_feature_lengthscale_1, ... in wall
        order. These are dimensionless (the wall features are dimensionless)
        and are not scaled by the domain extent.

        Raises
        ------
        RuntimeError
            If no fitted model is available yet.
        """
        if not self.has_model():
            raise RuntimeError('Belief has no fitted model yet.')
        assert self.model is not None
        assert self.standardizer is not None

        spatial_kernel = get_spatial_kernel(self.model.covar_module)
        lengthscale_norm = spatial_kernel.lengthscale.detach()
        x_extent = self.normalizer.x_max - self.normalizer.x_min
        y_extent = self.normalizer.y_max - self.normalizer.y_min
        std2 = float(self.standardizer.std**2)
        hyperparameters = {
            'lengthscale_x_m': float(lengthscale_norm[0, 0]) * x_extent,
            'lengthscale_y_m': float(lengthscale_norm[0, 1]) * y_extent,
            'outputscale': float(self.model.covar_module.outputscale.detach())
            * std2,
            'noise_variance': self.measurement_noise_variance(),
        }

        wall_kernel = get_wall_kernel(self.model.covar_module)
        if wall_kernel is not None:
            wall_lengthscale = wall_kernel.lengthscale.detach().reshape(-1)
            for index, value in enumerate(wall_lengthscale.tolist()):
                hyperparameters[f'wall_feature_lengthscale_{index}'] = value
        return hyperparameters

    def _condition_on_data(self) -> None:
        """Update the current model's conditioning set from all stored measurements.

        Unlike `_fit_model`, this does not rebuild the model and does not
        touch kernel or likelihood hyperparameters. It does recompute the
        target standardizer from all current measurements (a cheap mean/std
        recompute, not a gradient-based operation): freezing the
        standardizer between refits previously caused a severe scale
        mismatch once the model's fixed `init_outputscale`/`init_noise`
        (defined in standardized target units) were interpreted against a
        stale standardizer fit from very few points. Recomputing it here on
        every measurement, while leaving kernel hyperparameters untouched,
        keeps posterior queries reflecting every measurement between
        refits, while `refit_policy`/`refit_every_k` continue to control
        only the (expensive) hyperparameter re-optimization cadence.
        """
        assert self.model is not None

        train_x, train_y = self._build_train_tensors()
        self.model.set_train_data(inputs=train_x, targets=train_y, strict=False)

    def _fit_model(self) -> None:
        """Rebuild the exact GP against all currently stored measurements.

        The exact GP is always rebuilt from scratch whenever a refit is
        triggered. This is simple and robust for small domains, but it is not
        intended as a scalable online GP update method. -> Future work.

        If `config.optimize_hyperparameters` is `False`, the kernel
        hyperparameters stay fixed at their `init_*` values (no Adam
        training loop) and only the training data changes, so the posterior
        is conditioned on the new data without ever re-optimizing the prior.

        Between refits, `_condition_on_data` keeps the existing model
        conditioned on newly added measurements without rebuilding the model
        or touching its kernel/likelihood hyperparameters; this method is
        only reached when the refit policy actually fires, and additionally
        (re-)optimizes those hyperparameters when `optimize_hyperparameters`
        is `True`.
        """
        train_x, train_y = self._build_train_tensors()

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            self.config.device
        )
        covar_module = build_covar_module(
            self.config.kernel_type,
            model_variant=self.config.model_variant,
            wall_feature_dim=self.wall_feature_dim,
        ).to(self.config.device)
        model = ExactFieldGP(train_x, train_y, likelihood, covar_module).to(
            self.config.device
        )
        initialize_model_hyperparameters(
            model=model,
            likelihood=likelihood,
            train_x=train_x,
            init_lengthscale=(
                self.config.init_lengthscale_x,
                self.config.init_lengthscale_y,
            ),
            init_outputscale=self.config.init_outputscale,
            init_noise=self.config.init_noise,
            init_wall_lengthscale=self.config.init_wall_lengthscale,
        )

        if self.config.optimize_hyperparameters:
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.config.learning_rate
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for _ in range(self.config.training_iter):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

        self.model = model
        self.likelihood = likelihood

    def _build_train_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build GP training inputs and standardized training targets.

        This method is used by both `_fit_model` and `_condition_on_data`. It
        also updates `self.standardizer`, because target standardization is
        recomputed from all current measurements every time this is called.
        """
        if len(self.y_train) == 0:
            raise RuntimeError('Cannot build tensors without training data.')

        train_x = self._build_input_tensor(self.xy_train_phys)
        y_train_tensor = torch.as_tensor(
            self.y_train,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        self.standardizer = TargetStandardizer.fit(y_train_tensor)
        train_y = self.standardizer.transform(y_train_tensor)

        return train_x, train_y

    def _build_query_tensor(self, query_xy_phys: np.ndarray) -> torch.Tensor:
        """Build GP query inputs for posterior evaluation.

        This method is intentionally read-only with respect to the belief state.
        In particular, it does not rebuild training tensors and does not refit
        the target standardizer.
        """
        query_xy_phys = np.asarray(query_xy_phys, dtype=float)
        if query_xy_phys.ndim != 2 or query_xy_phys.shape[1] != 2:
            raise ValueError(
                f'query_xy_phys must have shape (N, 2), got {query_xy_phys.shape}'
            )
        if not np.all(np.isfinite(query_xy_phys)):
            raise ValueError('query_xy_phys must contain only finite values.')

        return self._build_input_tensor(query_xy_phys)

    def _build_input_tensor(self, xy_phys: np.ndarray) -> torch.Tensor:
        """Convert physical `(N, 2)` positions to the GP model input tensor.

        This is the single input preparation path for training and query
        inputs. The result has shape `(N, 2)` in baseline mode and
        `(N, 2 + M)` in wall-aware mode, with the configured dtype and device.
        """
        gp_inputs = build_gp_inputs(
            xy_phys=xy_phys,
            input_normalizer=self.normalizer,
            wall_feature_map=self.wall_feature_map,
        )
        return torch.as_tensor(
            gp_inputs.to_model_array(self.config.model_variant),
            dtype=self.config.dtype,
            device=self.config.device,
        )
