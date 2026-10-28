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

"""Thin backend adapters with one small contract.

    reset()
    add_measurement(x, y, value)   # assimilation finished on return
    is_ready() -> bool             # query() allowed
    query(xy (N, 2)) -> (mean (N,), variance (N,))
    num_assimilated() -> int
    effective_config() -> dict     # snapshot for session metadata

Attributes: `name` (CLI key), `label` (status text) and `display_name`
(short panel title, e.g. "GP baseline"). Optional capability:
`model_grid() -> ModelGridOverlay | None`, the model's own node graph for the
estimate panel (GMRF: grid edges; visGP: visible reference lattice edges).

Mean is the latent posterior mean in scenario measurement units, variance the
latent marginal variance in units^2 (no unit conversion anywhere). Positions
the backend cannot represent (GMRF: hidden behind walls; visGP: on a wall)
are NaN.

Each adapter reads its own existing ROS parameter YAML (`/**: ros__parameters:`)
and keeps its own schema. The domain bounds are always replaced by the
scenario bounds and the known walls always come from the scenario; both
overrides are recorded in the snapshot.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from belief_playground.scenario import Domain

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GP_CONFIG = REPO_ROOT / 'config' / 'belief_params_default.yaml'
DEFAULT_GP_WALL_AWARE_CONFIG = (
    REPO_ROOT / 'config' / 'belief_params_wall_aware.yaml'
)
DEFAULT_GMRF_CONFIG = REPO_ROOT / 'gmrf' / 'ros' / 'belief_params_gmrf.yaml'
DEFAULT_VISGP_CONFIG = REPO_ROOT / 'config' / 'belief_params_visgp.yaml'
GMRF_MODULE_NAME = 'scalar_field_belief_gmrf_py'
BACKEND_NAMES = ('gp', 'gp_wall_aware', 'gmrf', 'visgp')


class NotReadyError(RuntimeError):
    """query() called before the backend can answer."""


@dataclasses.dataclass(frozen=True)
class ModelGridOverlay:
    """Display geometry of a backend's model grid [m].

    `segments` (E, 2, 2): drawable edges as ((x0, y0), (x1, y1)), i.e. model
    neighbour edges that are not cut by a wall and join two active nodes.
    `nodes` (M, 2): positions of the active nodes. `dx`, `dy`: node spacing.
    """

    segments: np.ndarray
    nodes: np.ndarray
    nx: int
    ny: int
    dx: float
    dy: float


def load_ros_params(path: str | Path) -> tuple[dict, dict]:
    """Read a ROS 2 parameter YAML with one node entry.

    Returns the `ros__parameters` dict and a file record (path, sha256,
    text) for the metadata.
    """
    path = Path(path).resolve()
    text = path.read_text()
    data = yaml.safe_load(text)
    if not (isinstance(data, dict) and len(data) == 1):
        raise ValueError(f'{path}: expected exactly one node entry.')
    ((_, body),) = data.items()
    if not (
        isinstance(body, dict) and isinstance(body.get('ros__parameters'), dict)
    ):
        raise ValueError(f'{path}: missing ros__parameters.')
    record = {
        'path': str(path),
        'sha256': hashlib.sha256(text.encode()).hexdigest(),
        'text': text,
    }
    return dict(body['ros__parameters']), record


def _bounds_override(params: dict, domain: Domain) -> dict:
    from_file = {k: params.get(k) for k in ('x_min', 'x_max', 'y_min', 'y_max')}
    used = {
        'x_min': domain.x_range[0],
        'x_max': domain.x_range[1],
        'y_min': domain.y_range[0],
        'y_max': domain.y_range[1],
    }
    changed = any(
        from_file[k] is None or float(from_file[k]) != used[k] for k in used
    )
    return {'from_file': from_file, 'used': used, 'changed': changed}


def _json_value(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------------
# GP
# --------------------------------------------------------------------------

# Parameters in the GP YAML that the playground does not use.
_GP_UNUSED_KEYS = {
    'belief_geometry': 'replaced by the scenario walls',
    'scenario_path': 'replaced by the scenario walls',
    'field_color_min': 'ROS visualization only',
    'field_color_max': 'ROS visualization only',
    'use_fixed_field_color_max': 'ROS visualization only',
    'z_offset': 'ROS visualization only',
}


def build_gp_config(params: dict, domain: Domain):
    """Map GP ROS parameters to a BeliefConfig (bounds from the scenario)."""
    from scalar_field_belief.config import BeliefConfig

    fields = {
        f.name: f for f in dataclasses.fields(BeliefConfig) if f.name != '_'
    }
    kwargs = {}
    missing = []
    for name, f in fields.items():
        if name in ('x_min', 'x_max', 'y_min', 'y_max'):
            continue
        if name == 'dtype':
            continue  # torch dtype keeps the BeliefConfig default (float64)
        if name in params:
            kwargs[name] = params[name]
        elif (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ):
            missing.append(name)
    if missing:
        raise ValueError(f'GP config is missing required parameters: {missing}')
    for name in ('training_iter', 'refit_every_k'):
        kwargs[name] = int(kwargs[name])
    bounds = _bounds_override(params, domain)
    kwargs.update(bounds['used'])
    cfg = BeliefConfig(**kwargs)
    cfg.validate()

    unknown = sorted(set(params) - set(fields) - set(_GP_UNUSED_KEYS))
    snapshot = {
        'effective': {name: _json_value(getattr(cfg, name)) for name in fields},
        'bounds_override': bounds,
        'unused_parameters': {
            k: reason for k, reason in _GP_UNUSED_KEYS.items() if k in params
        },
        'unknown_parameters_ignored': unknown,
    }
    return cfg, snapshot


class GPAdapter:
    """Adapter for scalar_field_belief.belief.ScalarFieldBelief."""

    name = 'gp'

    def __init__(
        self,
        domain: Domain,
        config_path: str | Path | None = None,
        name: str = 'gp',
    ):
        from scalar_field_belief.belief import ScalarFieldBelief
        from scalar_field_belief.geometry import WallGeometry, WallSegment

        params, file_record = load_ros_params(config_path or DEFAULT_GP_CONFIG)
        cfg, snapshot = build_gp_config(params, domain)
        if name == 'gp_wall_aware' and cfg.model_variant != 'wall_aware':
            raise ValueError(
                'backend gp_wall_aware needs model_variant: wall_aware, but '
                f'{file_record["path"]} has {cfg.model_variant!r}'
            )
        self.name = name

        geometry = None
        if cfg.model_variant == 'wall_aware':
            geometry = WallGeometry(
                name='scenario',
                x_range=domain.x_range,
                y_range=domain.y_range,
                walls=tuple(
                    WallSegment(start=a, end=b) for a, b in domain.walls
                ),
            )
        self.belief = ScalarFieldBelief(cfg, wall_geometry=geometry)
        self.label = f'GP ({cfg.model_variant}, {cfg.kernel_type})'
        self.display_name = 'GP ' + cfg.model_variant.replace('_', '-')
        self.unit = domain.unit
        self._snapshot = {
            'backend': name,
            'implementation': 'scalar_field_belief.belief.ScalarFieldBelief',
            'config_file': file_record,
            **snapshot,
            'walls_used': (
                [list(a) + list(b) for a, b in domain.walls]
                if geometry is not None
                else 'none (baseline variant does not use walls)'
            ),
            'readiness': 'no prediction until the first configured refit',
            'value_unit': domain.unit,
        }

    def reset(self) -> None:
        self.belief.reset()

    def add_measurement(self, x: float, y: float, value: float) -> None:
        self.belief.add_measurement(float(x), float(y), float(value))

    def is_ready(self) -> bool:
        return self.belief.has_model()

    def not_ready_reason(self) -> str:
        return 'waiting for fitted model'

    def num_assimilated(self) -> int:
        return int(len(self.belief.y_train))

    def query(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_ready():
            raise NotReadyError(self.not_ready_reason())
        xy = np.asarray(xy, dtype=float)
        mean, var = self.belief.query(xy)
        return (
            np.asarray(mean, dtype=float).reshape(len(xy)),
            np.asarray(var, dtype=float).reshape(len(xy)),
        )

    def effective_config(self) -> dict:
        return copy.deepcopy(self._snapshot)


# --------------------------------------------------------------------------
# GMRF
# --------------------------------------------------------------------------

# GmrfConfig fields that the GMRF ROS node reads (same names and units).
_GMRF_KEYS = (
    'nx',
    'ny',
    'prior_mean',
    'prior_sigma',
    'sigma_p_per_m',
    'sigma_s',
    'wall_endpoint_extension',
)
_GMRF_UNUSED_KEYS = {
    'frame_id': 'not used (single frame)',
    'walls': 'replaced by the scenario walls',
    'belief_geometry': 'replaced by the scenario walls',
    'scenario_path': 'replaced by the scenario walls',
    'publish_visualization': 'ROS visualization only',
    'visualization_grid_step': 'ROS visualization only',
    'z_offset': 'ROS visualization only',
    'field_color_min': 'ROS visualization only',
    'field_color_max': 'ROS visualization only',
    'use_fixed_field_color_max': 'ROS visualization only',
    'invalid_query_points': 'ROS node only',
}


def import_gmrf_module(module_dir: str | Path | None = None):
    """Import the pybind11 module, optionally from an explicit directory."""
    if module_dir is not None:
        module_dir = str(Path(module_dir).resolve())
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
    try:
        return importlib.import_module(GMRF_MODULE_NAME)
    except ImportError as exc:
        v = sys.version_info
        raise type(exc)(
            f'{exc} (this interpreter: {sys.executable}, Python {v.major}.{v.minor}; '
            f'{GMRF_MODULE_NAME} is built by the scalar_field_belief workspace '
            'build for the CMake Python, which must have the same major.minor '
            'version; rebuild the workspace and source its setup file)',
            name=exc.name,
        ) from exc


class GmrfAdapter:
    """Adapter for the C++ GmrfBelief through the pybind11 module."""

    name = 'gmrf'

    def __init__(
        self,
        domain: Domain,
        config_path: str | Path | None = None,
        module_dir: str | Path | None = None,
    ):
        gm = import_gmrf_module(module_dir)
        params, file_record = load_ros_params(
            config_path or DEFAULT_GMRF_CONFIG
        )
        missing = [k for k in _GMRF_KEYS if k not in params]
        if missing:
            raise ValueError(
                f'GMRF config is missing required parameters: {missing}'
            )

        cfg = gm.GmrfConfig()  # C++ defaults: stationary scalar model
        bounds = _bounds_override(params, domain)
        for k, v in bounds['used'].items():
            setattr(cfg, k, float(v))
        for k in _GMRF_KEYS:
            v = params[k]
            setattr(cfg, k, int(v) if k in ('nx', 'ny') else float(v))
        cfg.validate()

        seeds_flat = list(params.get('free_space_seeds') or [])
        if len(seeds_flat) % 2:
            raise ValueError(
                'free_space_seeds must be a flat [x, y, ...] list.'
            )
        seeds = [
            gm.Point2(float(seeds_flat[k]), float(seeds_flat[k + 1]))
            for k in range(0, len(seeds_flat), 2)
        ]
        walls = [
            gm.WallSegment(a[0], a[1], b[0], b[1]) for a, b in domain.walls
        ]
        self.belief = gm.GmrfBelief(cfg, walls, seeds)
        self.label = f'GMRF ({cfg.nx}x{cfg.ny} nodes)'
        self.display_name = 'GMRF'
        self.unit = domain.unit
        self._hidden_cells = 0
        self._model_grid = None  # built on first model_grid() call

        unknown = sorted(
            set(params)
            - set(_GMRF_KEYS)
            - set(_GMRF_UNUSED_KEYS)
            - {'x_min', 'x_max', 'y_min', 'y_max', 'free_space_seeds'}
        )
        self._snapshot = {
            'backend': 'gmrf',
            'implementation': f'{GMRF_MODULE_NAME} ({gm.__file__})',
            'config_file': file_record,
            'effective': {
                **{
                    k: getattr(cfg, k)
                    for k in ('x_min', 'x_max', 'y_min', 'y_max')
                },
                **{k: getattr(cfg, k) for k in _GMRF_KEYS},
                'other_fields': 'C++ GmrfConfig defaults (stationary scalar model)',
                'dx': self.belief.dx,
                'dy': self.belief.dy,
                'free_space_seeds': [[p.x, p.y] for p in seeds],
                'measurement_noise_variance': self.belief.measurement_noise_variance(),
            },
            'bounds_override': bounds,
            'grid_note': 'nx/ny kept from the config file; dx/dy follow the scenario bounds',
            'unused_parameters': {
                k: reason
                for k, reason in _GMRF_UNUSED_KEYS.items()
                if k in params
            },
            'unknown_parameters_ignored': unknown,
            'walls_used': [list(a) + list(b) for a, b in domain.walls],
            'readiness': 'prior posterior available immediately',
            'value_unit': domain.unit,
        }

    def reset(self) -> None:
        self.belief.reset()

    def add_measurement(self, x: float, y: float, value: float) -> None:
        self.belief.add_measurement(float(x), float(y), float(value))
        self.belief.update()

    def is_ready(self) -> bool:
        return True

    def not_ready_reason(self) -> str:
        return ''

    def num_assimilated(self) -> int:
        return int(self.belief.num_measurements())

    @property
    def hidden_cells(self) -> int:
        """Query points of the last query() the GMRF cannot represent."""
        return self._hidden_cells

    def query(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.ascontiguousarray(xy, dtype=float)
        mean = np.full(len(xy), np.nan)
        var = np.full(len(xy), np.nan)
        ok = np.asarray(self.belief.representable_mask(xy), dtype=bool)
        self._hidden_cells = int((~ok).sum())
        if ok.any():
            mean[ok], var[ok] = self.belief.query_mean_variance(xy[ok])
        return mean, var

    def effective_config(self) -> dict:
        return copy.deepcopy(self._snapshot)

    def model_grid(self) -> ModelGridOverlay | None:
        """Model nodes and uncut edges (fixed for this configuration; reset()
        keeps grid, walls and free-space mask). Built once, then cached."""
        if self._model_grid is None:
            if not hasattr(self.belief, 'model_grid'):
                return None  # older module without the bulk accessor
            g = self.belief.model_grid()
            xy = np.asarray(g['node_xy'], dtype=float)
            edges = np.asarray(g['edges'], dtype=np.int64).reshape(-1, 2)
            active = np.asarray(g['active'], dtype=bool)
            self._model_grid = ModelGridOverlay(
                segments=np.stack([xy[edges[:, 0]], xy[edges[:, 1]]], axis=1),
                nodes=xy[active],
                nx=int(g['nx']),
                ny=int(g['ny']),
                dx=float(self.belief.dx),
                dy=float(self.belief.dy),
            )
        return self._model_grid


# --------------------------------------------------------------------------
# visGP
# --------------------------------------------------------------------------

# Parameters in the visGP YAML that the playground does not use.
_VISGP_UNUSED_KEYS = {
    'frame_id': 'not used (single frame)',
    'walls': 'replaced by the scenario walls',
    'belief_geometry': 'replaced by the scenario walls',
    'scenario_path': 'replaced by the scenario walls',
}


def _visgp_number(name: str, value, integer: bool):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f'visGP parameter {name} must be a number: {value!r}')
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f'visGP parameter {name} must be a number: {value!r}'
        ) from exc
    if integer:
        if not number.is_integer():
            raise ValueError(
                f'visGP parameter {name} must be an integer: {value!r}'
            )
        return int(number)
    return number


def build_visgp_config(params: dict, domain: Domain):
    """Map visGP parameters to a VisGPConfig (bounds from the scenario)."""
    from scalar_field_belief.visgp import VisGPConfig

    fields = {f.name: f for f in dataclasses.fields(VisGPConfig)}
    kwargs = {}
    missing = []
    for name, f in fields.items():
        if name in params:
            kwargs[name] = _visgp_number(
                name, params[name], integer=f.type in ('int', int)
            )
        elif f.default is dataclasses.MISSING:
            missing.append(name)
    if missing:
        raise ValueError(
            f'visGP config is missing required parameters: {missing}'
        )
    cfg = VisGPConfig(**kwargs)
    cfg.validate()
    bounds = _bounds_override(params, domain)
    unknown = sorted(
        set(params)
        - set(fields)
        - set(_VISGP_UNUSED_KEYS)
        - {'x_min', 'x_max', 'y_min', 'y_max'}
    )
    snapshot = {
        'effective': {
            **bounds['used'],
            **{name: getattr(cfg, name) for name in fields},
        },
        'bounds_override': bounds,
        'unused_parameters': {
            k: reason for k, reason in _VISGP_UNUSED_KEYS.items() if k in params
        },
        'unknown_parameters_ignored': unknown,
    }
    return cfg, snapshot


class VisGPAdapter:
    """Adapter for scalar_field_belief.visgp.VisGPBelief (pure Python)."""

    name = 'visgp'

    def __init__(self, domain: Domain, config_path: str | Path | None = None):
        import scipy

        from scalar_field_belief.geometry import WallGeometry, WallSegment
        from scalar_field_belief.visgp import VisGPBelief

        params, file_record = load_ros_params(
            config_path or DEFAULT_VISGP_CONFIG
        )
        cfg, snapshot = build_visgp_config(params, domain)
        geometry = WallGeometry(
            name='scenario',
            x_range=domain.x_range,
            y_range=domain.y_range,
            walls=tuple(WallSegment(start=a, end=b) for a, b in domain.walls),
        )
        t0 = time.perf_counter()
        self.belief = VisGPBelief(cfg, geometry)
        construction_ms = (time.perf_counter() - t0) * 1e3
        belief = self.belief
        grid = belief.reference_grid
        num_nodes = len(belief.reference_locations)
        self.label = (
            f'visGP ({num_nodes} reference nodes, k={cfg.max_neighbors})'
        )
        self.display_name = 'visGP'
        self.unit = domain.unit
        self._last_query = self._query_counts(None)
        self._model_grid = None  # built on first model_grid() call
        reference_set = {
            'construction': 'cell-centred grid from the domain bounds and '
            'walls (nodes on walls dropped)',
            'nx': grid.nx,
            'ny': grid.ny,
            'dx': grid.dx,
            'dy': grid.dy,
            'num_nodes': num_nodes,
            'dropped_on_wall': grid.dropped_on_wall,
            'locations': belief.reference_locations.tolist(),
        }
        self._snapshot = {
            'backend': 'visgp',
            'implementation': 'scalar_field_belief.visgp.VisGPBelief',
            'config_file': file_record,
            'effective': {
                **snapshot['effective'],
                'parent_covariance': 'isotropic Matern-3/2',
                'neighbour_rule': 'nearest clique, scan stops at the first '
                'incompatible candidate',
                'contact_tolerance_m': belief.contact_tolerance,
                'reference_set': reference_set,
            },
            'bounds_override': snapshot['bounds_override'],
            'unused_parameters': snapshot['unused_parameters'],
            'unknown_parameters_ignored': snapshot[
                'unknown_parameters_ignored'
            ],
            'covariance_selection': {
                **belief.covariance_selection.summary(),
                'construction_ms': construction_ms,
                'scipy_version': scipy.__version__,
            },
            'walls_used': [list(a) + list(b) for a, b in domain.walls],
            'readiness': 'latent prior available immediately',
            'value_unit': domain.unit,
        }

    def reset(self) -> None:
        self.belief.reset()

    def add_measurement(self, x: float, y: float, value: float) -> None:
        self.belief.add_measurement(float(x), float(y), float(value))

    def is_ready(self) -> bool:
        return True

    def not_ready_reason(self) -> str:
        return ''

    def num_assimilated(self) -> int:
        return int(self.belief.num_measurements)

    @staticmethod
    def _query_counts(prediction) -> dict:
        if prediction is None:
            return {
                'cells': 0,
                'invalid_cells': 0,
                'residual_clamps': 0,
                'variance_clamps': 0,
            }
        return {
            'cells': int(prediction.valid.size),
            'invalid_cells': int((~prediction.valid).sum()),
            'residual_clamps': prediction.residual_clamps,
            'variance_clamps': prediction.variance_clamps,
        }

    @property
    def hidden_cells(self) -> int:
        """Query points of the last query() on a wall or outside (NaN)."""
        return self._last_query['invalid_cells']

    @property
    def last_query_counts(self) -> dict:
        """Cells, invalid cells and roundoff clamps (F, posterior variance)
        of the last query()."""
        return dict(self._last_query)

    def query(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.belief.predict_marginal(np.asarray(xy, dtype=float))
        self._last_query = self._query_counts(prediction)
        return prediction.mean, prediction.variance

    def effective_config(self) -> dict:
        snapshot = copy.deepcopy(self._snapshot)
        snapshot['last_query'] = self.last_query_counts
        return snapshot

    def model_grid(self) -> ModelGridOverlay | None:
        """Reference nodes and the visible edges between lattice neighbours
        (display only: a subset of the visibility graph, which also links
        distant visible nodes). Fixed for this configuration; cached."""
        grid = self.belief.reference_grid
        if self._model_grid is None and grid is not None:
            xy = self.belief.reference_locations
            index = {
                (i, j): k
                for k, (i, j) in enumerate(grid.lattice_index.tolist())
            }
            pairs = [
                (k, other)
                for (i, j), k in index.items()
                for other in (index.get((i + 1, j)), index.get((i, j + 1)))
                if other is not None and self.belief.adjacency[k, other]
            ]
            pairs = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
            self._model_grid = ModelGridOverlay(
                segments=np.stack([xy[pairs[:, 0]], xy[pairs[:, 1]]], axis=1),
                nodes=np.array(xy),
                nx=grid.nx,
                ny=grid.ny,
                dx=grid.dx,
                dy=grid.dy,
            )
        return self._model_grid


# --------------------------------------------------------------------------


class UnavailableBackend:
    """Placeholder for a backend that could not be constructed."""

    def __init__(self, name: str, error: str):
        self.name = name
        self.label = f'{name.upper()} (unavailable)'
        self.display_name = name.upper()
        self.unavailable_reason = error

    def reset(self) -> None:
        pass

    def add_measurement(self, x, y, value) -> None:
        raise RuntimeError(self.unavailable_reason)

    def is_ready(self) -> bool:
        return False

    def not_ready_reason(self) -> str:
        return self.unavailable_reason

    def num_assimilated(self) -> int:
        return 0

    def query(self, xy):
        raise NotReadyError(self.unavailable_reason)

    def effective_config(self) -> dict:
        return {'backend': self.name, 'unavailable': self.unavailable_reason}


def make_backends(
    names,
    domain: Domain,
    gp_config: str | Path | None = None,
    gmrf_config: str | Path | None = None,
    gmrf_module_dir: str | Path | None = None,
    gp_wall_aware_config: str | Path | None = None,
    visgp_config: str | Path | None = None,
) -> dict:
    """Construct the selected adapters; failures become UnavailableBackend."""
    backends = {}
    for name in names:
        if name not in BACKEND_NAMES:
            raise ValueError(
                f'unknown backend {name!r}; choose from {BACKEND_NAMES}'
            )
        try:
            if name == 'gp':
                backends[name] = GPAdapter(domain, gp_config)
            elif name == 'gp_wall_aware':
                backends[name] = GPAdapter(
                    domain,
                    gp_wall_aware_config or DEFAULT_GP_WALL_AWARE_CONFIG,
                    name=name,
                )
            elif name == 'visgp':
                backends[name] = VisGPAdapter(domain, visgp_config)
            else:
                backends[name] = GmrfAdapter(
                    domain, gmrf_config, gmrf_module_dir
                )
        except Exception as exc:  # noqa: BLE001 - report, keep the app usable
            backends[name] = UnavailableBackend(
                name, f'{type(exc).__name__}: {exc}'
            )
    return backends
