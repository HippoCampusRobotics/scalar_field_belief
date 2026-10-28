# Visibility-graph GP (visGP) after Gilbert and Datta (2024), Biometrics:
# https://academic.oup.com/biometrics/article/doi/10.1093/biomtc/ujae089/7753413
# Authors' R implementation: https://github.com/bjg345/visGP

"""visGP belief: latent visibility-graph GP with independent sensor noise.

Model (all values in physical units, no target standardisation):

    f ~ GP(prior_mean, C_vis),   y_i = f(x_i) + e_i,
    e_i ~ N(0, measurement_noise_variance) independent,

with `C_vis` from covariance selection on a fixed reference set (see
`covariance` and `extension`). Kernel parameters, the reference set, the
visibility graph and `L` are fixed at construction; measurements never change
them. Each `add_measurement` re-conditions on all stored measurements (batch
Cholesky; not an incremental update). Queries return the latent mean and the
latent variance/covariance (without sensor noise) and do not change state.

Positions: non-finite input raises `ValueError`. Positions outside the
closed domain rectangle or within the contact tolerance of a wall are
invalid: `query` returns NaN mean and variance there, `add_measurement` and
`query_with_covariance` raise `ValueError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from scalar_field_belief.geometry import WallGeometry
from scalar_field_belief.visgp.config import VisGPConfig
from scalar_field_belief.visgp.covariance import (
    covariance_selection,
    matern32,
)
from scalar_field_belief.visgp.extension import (
    ProcessExtension,
    location_key,
)
from scalar_field_belief.visgp.visibility import (
    ReferenceGrid,
    as_positions,
    check_reference_locations,
    contact_tolerance,
    inside_bounds_mask,
    on_wall_mask,
    reference_grid,
    visibility_graph,
    wall_array,
)

# Posterior variances in [-VARIANCE_CLAMP_TOLERANCE * signal_variance, 0) are
# roundoff and returned as 0 (counted); more negative values are an error.
VARIANCE_CLAMP_TOLERANCE = 1e-12

_QUERY_CHUNK = 512  # unique positions per block of a marginal query


@dataclass(frozen=True)
class MarginalPrediction:
    """Result of `VisGPBelief.predict_marginal`.

    `mean` [units] and `variance` [units^2] are latent and NaN where `valid`
    is False. `residual_clamps` / `variance_clamps` count unique positions
    whose F(x) / posterior variance was a roundoff-negative value set to 0.
    """

    mean: np.ndarray
    variance: np.ndarray
    valid: np.ndarray
    residual_clamps: int
    variance_clamps: int


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class VisGPBelief:
    """Dense latent visGP belief over a 2D domain with known straight walls.

    Parameters
    ----------
    config
        Fixed model and solver settings (validated here).
    geometry
        Domain bounds and walls in meters (same frame as all positions).
    reference_locations
        Optional explicit `(m, 2)` reference set V. Default: the cell-centred
        grid of `config.reference_spacing` (nodes on walls dropped). It
        depends only on the geometry and the configuration.

    Raises
    ------
    TypeError, ValueError
        For an invalid configuration, geometry or reference set (including
        a grid larger than `max_reference_nodes`).
    CovarianceSelectionError
        If covariance selection does not converge or fails.
    """

    def __init__(
        self,
        config: VisGPConfig,
        geometry: WallGeometry,
        reference_locations=None,
    ):
        if not isinstance(config, VisGPConfig):
            raise TypeError('config must be a VisGPConfig.')
        config.validate()
        if not isinstance(geometry, WallGeometry):
            raise TypeError('geometry must be a WallGeometry.')
        self.config = config
        self.geometry = geometry
        self.walls = _read_only(wall_array(geometry))
        self.contact_tolerance = contact_tolerance(geometry)

        self.reference_grid: ReferenceGrid | None = None
        if reference_locations is None:
            self.reference_grid = reference_grid(
                geometry,
                config.reference_spacing,
                config.max_reference_nodes,
                self.walls,
                self.contact_tolerance,
            )
            locations = self.reference_grid.locations
        else:
            locations = check_reference_locations(
                reference_locations,
                geometry,
                config.max_reference_nodes,
                self.walls,
                self.contact_tolerance,
            )
        self.reference_locations = _read_only(np.array(locations, dtype=float))
        self.adjacency = _read_only(
            visibility_graph(
                self.reference_locations, self.walls, self.contact_tolerance
            )
        )
        self.parent_covariance = _read_only(
            matern32(
                self.reference_locations,
                self.reference_locations,
                config.signal_variance,
                config.lengthscale,
            )
        )
        self.covariance_selection = covariance_selection(
            self.parent_covariance,
            self.adjacency,
            scale=config.signal_variance,
            tolerance=config.covsel_tolerance,
            precision_tolerance=config.covsel_precision_tolerance,
            max_iterations=config.covsel_max_iterations,
            max_cliques=config.max_cliques,
        )
        self.L = _read_only(self.covariance_selection.L)
        self.extension = ProcessExtension(
            self.reference_locations,
            self.adjacency,
            self.parent_covariance,
            self.walls,
            self.contact_tolerance,
            config.signal_variance,
            config.lengthscale,
            config.max_neighbors,
        )
        self.reset()

    # ------------------------------------------------------------- state
    def reset(self) -> None:
        """Remove all measurements and the posterior. Geometry, reference
        set, visibility graph and L are kept."""
        m = len(self.reference_locations)
        self.xy_train_phys = np.empty((0, 2))
        self.y_train = np.empty(0)
        self._location_ids: dict[tuple[float, float], int] = {}
        self._train_ids = np.empty(0, dtype=np.intp)
        self._train_weights = np.empty((0, m))
        self._train_residual = np.empty(0)
        self._train_clamped = np.empty(0, dtype=bool)
        self._chol = None
        self._alpha = np.empty(0)
        self._L_Bt = np.empty((m, 0))

    def has_model(self) -> bool:
        """Always True: the fixed prior is available before any measurement."""
        return True

    def measurement_noise_variance(self) -> float:
        """Sensor noise variance [units^2] (configured, not fitted)."""
        return float(self.config.measurement_noise_variance)

    @property
    def num_measurements(self) -> int:
        return len(self.y_train)

    def status(self) -> dict:
        """Readiness and size information (JSON-friendly)."""
        return {
            'ready': True,
            'num_measurements': self.num_measurements,
            'num_locations': len(self._location_ids),
            'num_reference_nodes': len(self.reference_locations),
            'training_residual_clamps': int(self._train_clamped.sum()),
            'covariance_selection': self.covariance_selection.summary(),
        }

    # ------------------------------------------------------------ validity
    def valid_mask(self, xy) -> np.ndarray:
        """True where a position is inside the domain and off every wall.
        Raises ValueError for non-finite input or a wrong shape."""
        xy = as_positions(xy)
        return inside_bounds_mask(xy, self.geometry) & ~on_wall_mask(
            xy, self.walls, self.contact_tolerance
        )

    # -------------------------------------------------------- measurements
    def add_measurement(self, x: float, y: float, value: float) -> None:
        """Store one measurement at its actual position and re-condition.

        Raises ValueError for non-finite input and for a position outside
        the domain or on a wall; the state is then unchanged.
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError('Measurement x/y must be finite.')
        if not math.isfinite(value):
            raise ValueError('Measurement value must be finite.')
        xy = np.array([[float(x), float(y)]]) + 0.0
        if not self.valid_mask(xy)[0]:
            raise ValueError(
                f'measurement position ({x}, {y}) is outside the domain or '
                f'within {self.contact_tolerance:.3g} m of a wall.'
            )

        key = location_key(x, y)
        weights = self._train_weights
        residual = self._train_residual
        clamped = self._train_clamped
        location_id = self._location_ids.get(key)
        if location_id is None:
            rows = self.extension.rows(xy)
            location_id = len(residual)
            weights = np.vstack([weights, rows.weights])
            residual = np.concatenate([residual, rows.residual])
            clamped = np.concatenate([clamped, rows.clamped])

        train_xy = np.vstack([self.xy_train_phys, xy])
        train_y = np.concatenate([self.y_train, [float(value)]])
        train_ids = np.append(self._train_ids, location_id)
        chol, alpha, L_Bt = self._condition(
            train_ids, weights, residual, train_y
        )

        self._location_ids[key] = location_id
        self._train_weights = weights
        self._train_residual = residual
        self._train_clamped = clamped
        self.xy_train_phys = train_xy
        self.y_train = train_y
        self._train_ids = train_ids
        self._chol, self._alpha, self._L_Bt = chol, alpha, L_Bt

    def _condition(self, ids, weights, residual, values):
        """Cholesky of the noisy training covariance and its solve."""
        B = weights[ids]
        same = ids[:, None] == ids[None, :]
        K = (B @ self.L) @ B.T + np.where(same, residual[ids][:, None], 0.0)
        K = 0.5 * (K + K.T)
        K[np.diag_indices_from(K)] += self.config.measurement_noise_variance
        chol = linalg.cho_factor(K, lower=True, check_finite=False)
        alpha = linalg.cho_solve(
            chol, values - self.config.prior_mean, check_finite=False
        )
        return chol, alpha, self.L @ B.T

    # ------------------------------------------------------------- queries
    def _unique_rows(self, xy: np.ndarray):
        """Rows for distinct valid positions; stored training rows are
        reused so that one location never gets two different rows."""
        n = len(xy)
        ids = np.array(
            [self._location_ids.get(location_key(x, y), -1) for x, y in xy],
            dtype=np.intp,
        )
        weights = np.empty((n, len(self.reference_locations)))
        residual = np.empty(n)
        clamped = np.empty(n, dtype=bool)
        known = ids >= 0
        weights[known] = self._train_weights[ids[known]]
        residual[known] = self._train_residual[ids[known]]
        clamped[known] = self._train_clamped[ids[known]]
        if not known.all():
            rows = self.extension.rows(xy[~known])
            weights[~known] = rows.weights
            residual[~known] = rows.residual
            clamped[~known] = rows.clamped
        return ids, weights, residual, clamped

    def _cross_covariance(self, ids, weights, residual) -> np.ndarray:
        """C_vis between query rows and all stored measurements."""
        same = ids[:, None] == self._train_ids[None, :]
        return weights @ self._L_Bt + np.where(same, residual[:, None], 0.0)

    def predict_marginal(self, xy) -> MarginalPrediction:
        """Latent posterior mean and marginal variance with diagnostics.

        Memory grows linearly with the number of positions (processed in
        blocks); no (N, N) matrix is formed.
        """
        xy = as_positions(xy) + 0.0
        n = len(xy)
        valid = self.valid_mask(xy)
        mean = np.full(n, np.nan)
        var = np.full(n, np.nan)
        residual_clamps = 0
        variance_clamps = 0
        if valid.any():
            unique, inverse = np.unique(xy[valid], axis=0, return_inverse=True)
            u_mean = np.empty(len(unique))
            u_var = np.empty(len(unique))
            s2 = self.config.signal_variance
            for start in range(0, len(unique), _QUERY_CHUNK):
                block = slice(start, start + _QUERY_CHUNK)
                ids, B, F, clamped = self._unique_rows(unique[block])
                residual_clamps += int(clamped.sum())
                prior = np.einsum('ij,ij->i', B @ self.L, B) + F
                if self.num_measurements == 0:
                    u_mean[block] = self.config.prior_mean
                    u_var[block] = prior
                    continue
                k_zx = self._cross_covariance(ids, B, F)
                u_mean[block] = self.config.prior_mean + k_zx @ self._alpha
                v = linalg.solve_triangular(
                    self._chol[0], k_zx.T, lower=True, check_finite=False
                )
                u_var[block] = prior - np.einsum('ij,ij->j', v, v)
            negative = u_var < 0.0
            if np.any(u_var < -VARIANCE_CLAMP_TOLERANCE * s2):
                raise RuntimeError(
                    'negative posterior variance '
                    f'{float(u_var.min()):.3e} beyond roundoff.'
                )
            variance_clamps = int(negative.sum())
            u_var[negative] = 0.0
            inverse = inverse.reshape(-1)
            mean[valid] = u_mean[inverse]
            var[valid] = u_var[inverse]
        return MarginalPrediction(
            mean=mean,
            variance=var,
            valid=valid,
            residual_clamps=residual_clamps,
            variance_clamps=variance_clamps,
        )

    def query(self, xy) -> tuple[np.ndarray, np.ndarray]:
        """Latent posterior mean `(N,)` [units] and variance `(N,)`
        [units^2]; NaN at invalid positions. Raises ValueError for
        non-finite input or a wrong shape."""
        prediction = self.predict_marginal(xy)
        return prediction.mean, prediction.variance

    def query_with_covariance(self, xy) -> tuple[np.ndarray, np.ndarray]:
        """Latent posterior mean `(N,)` and full covariance `(N, N)`.

        Repeated positions give identical rows. Raises ValueError for
        invalid positions or more than `max_joint_query_size` positions.
        """
        xy = as_positions(xy) + 0.0
        n = len(xy)
        if n > self.config.max_joint_query_size:
            raise ValueError(
                f'joint query of {n} positions exceeds max_joint_query_size '
                f'= {self.config.max_joint_query_size}; use query() for '
                'marginal variances.'
            )
        invalid = np.flatnonzero(~self.valid_mask(xy))
        if invalid.size:
            raise ValueError(
                f'query position {xy[invalid[0]].tolist()} (row '
                f'{invalid[0]}) is outside the domain or on a wall.'
            )
        unique, inverse = np.unique(xy, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        u_ids, u_B, u_F, _ = self._unique_rows(unique)
        ids, B, F = u_ids[inverse], u_B[inverse], u_F[inverse]
        same = inverse[:, None] == inverse[None, :]
        cov = (B @ self.L) @ B.T + np.where(same, F[:, None], 0.0)
        mean = np.full(n, float(self.config.prior_mean))
        if self.num_measurements > 0:
            k_zx = self._cross_covariance(ids, B, F)
            mean = mean + k_zx @ self._alpha
            v = linalg.solve_triangular(
                self._chol[0], k_zx.T, lower=True, check_finite=False
            )
            cov = cov - v.T @ v
        return mean, 0.5 * (cov + cov.T)
