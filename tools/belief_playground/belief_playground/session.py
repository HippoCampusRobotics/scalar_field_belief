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

"""Experiment/session control (no GUI toolkit).

The `Session` owns the single ordered observation log. For every accepted
observation it determines the value once (truth sample + optional noise +
scenario clipping, or a manual value) and sends exactly that (x, y, value) to
every backend.

Backends are only touched by one worker thread, which processes an ordered
job queue. Every reset/undo/load starts a new *generation*: a "rebuild" job
resets all backends and replays the full retained log (stored values, no new
noise). Jobs and results of older generations are discarded, so stale results
can never overwrite a newer session, and no accepted observation is dropped.
The GUI (or a headless caller) only reads results via `poll()`.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import platform
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from belief_playground.scenario import DisplayGrid, Scenario

CSV_FIELDS = (
    'index',
    'x',
    'y',
    'value',
    'latent',
    'noise',
    'noise_std',
    'source',
    'clipped',
)
SESSION_FORMAT = 'belief_playground_session'
SESSION_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Observation:
    """One accepted observation. `value` is what every backend receives."""

    index: int  # 1-based position in the log
    x: float  # [m]
    y: float  # [m]
    value: float  # [unit]
    latent: float | None  # noiseless truth at (x, y); None for manual values
    noise: float  # noise added to `latent` before clipping (0 for manual)
    noise_std: float  # std used for this observation (0 for manual)
    source: str  # 'sample' | 'manual'
    clipped: bool  # scenario clip_range changed the sampled value


@dataclass
class BackendStatus:
    """Latest state of one backend (copied out by `Session.poll`)."""

    name: str
    generation: int = 0
    state: str = 'queued'  # queued | busy | ready | waiting | error
    n_assimilated: int = 0  # observations assimilated in this generation
    error: str | None = None  # assimilation failure (blocks further adds)
    failed_at: int | None = None  # observation index that failed
    query_error: str | None = None  # last query failure (transient)
    note: str = ''  # e.g. 'waiting for fitted model'
    # Wall-clock timings [ms] measured in the worker thread (see the GUI
    # status tooltip): update_ms covers the last assimilation job, which
    # assimilated `update_count` observations and, if `update_replay`, began
    # with backend.reset() (undo/reset/load: reset + replay of the log);
    # query_ms covers one backend.query() over `query_cells` display cells.
    update_ms: float | None = None
    update_count: int | None = None
    update_replay: bool | None = None
    query_ms: float | None = None
    query_cells: int | None = None
    mean: np.ndarray | None = None  # (N,) on the display grid
    var: np.ndarray | None = None  # (N,) on the display grid
    result_count: int | None = None  # observation count of mean/var
    result_id: int = 0  # increments whenever mean/var change


class Session:
    """Observation log, backend dispatch, worker thread, save/load."""

    def __init__(
        self,
        scenario: Scenario,
        backends: dict,
        grid: DisplayGrid,
        noise_std: float = 0.0,
        seed: int = 0,
    ):
        self.scenario = scenario
        self.backends = dict(backends)
        self.grid = grid
        self._check_noise_std(noise_std)
        self.noise_std = float(noise_std)
        # seed_in_use seeded the current generator; pending_seed (set_seed)
        # only takes effect on the next reset().
        self.seed_in_use = int(seed)
        self.pending_seed = int(seed)
        self._rng = np.random.default_rng(self.seed_in_use)
        self._noise_draws = 0  # draws from the current generator

        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._log: list[Observation] = []
        self._generation = 0
        self._pending = 0
        self._version = 0
        self._result_counter = 0
        self._status = {n: BackendStatus(name=n) for n in self.backends}
        self._jobs: queue.Queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name='belief-playground-worker', daemon=True
        )
        self._thread.start()
        with self._lock:
            self._start_generation_locked(())

    # ------------------------------------------------------------------ log
    @property
    def backend_names(self) -> list[str]:
        return list(self.backends)

    @property
    def observations(self) -> tuple[Observation, ...]:
        with self._lock:
            return tuple(self._log)

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def add_sample(self, x: float, y: float) -> Observation:
        """Sample the scenario at (x, y): latent + noise, then clip_range."""
        x, y = self._check_position(x, y)
        latent = float(self.scenario.truth(np.array([[x, y]]))[0])
        with self._lock:
            std = self.noise_std
            noise = 0.0
            if std > 0.0:
                noise = float(self._rng.normal(0.0, std))
                self._noise_draws += 1
            unclipped = latent + noise
            value = unclipped
            clip = self.scenario.clip_range
            if clip is not None:
                value = float(min(max(unclipped, clip[0]), clip[1]))
            obs = Observation(
                index=len(self._log) + 1,
                x=x,
                y=y,
                value=value,
                latent=latent,
                noise=noise,
                noise_std=std,
                source='sample',
                clipped=value != unclipped,
            )
            self._accept_locked(obs)
        return obs

    def add_manual(self, x: float, y: float, value: float) -> Observation:
        """Insert a user value as given (no noise, never clipped)."""
        x, y = self._check_position(x, y)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('manual value must be finite')
        with self._lock:
            obs = Observation(
                index=len(self._log) + 1,
                x=x,
                y=y,
                value=value,
                latent=None,
                noise=0.0,
                noise_std=0.0,
                source='manual',
                clipped=False,
            )
            self._accept_locked(obs)
        return obs

    def undo(self) -> Observation | None:
        """Drop the last observation; backends are reset and replayed."""
        with self._lock:
            if not self._log:
                return None
            removed = self._log.pop()
            self._start_generation_locked(tuple(self._log))
        return removed

    def reset(self) -> None:
        """Clear the log, re-seed the noise generator, reset all backends."""
        with self._lock:
            self._log.clear()
            self.seed_in_use = self.pending_seed
            self._rng = np.random.default_rng(self.seed_in_use)
            self._noise_draws = 0
            self._start_generation_locked(())

    def replace_log(self, observations) -> None:
        """Use a recorded log (e.g. loaded); values are replayed as stored."""
        obs = list(observations)
        for k, o in enumerate(obs, start=1):
            if o.index != k:
                raise ValueError(
                    f'observation indices must be 1..N, got {o.index} at {k}'
                )
            self._check_position(o.x, o.y)
            if not math.isfinite(o.value):
                raise ValueError(f'observation {k}: value must be finite')
        with self._lock:
            self._log = obs
            self._start_generation_locked(tuple(obs))

    def set_noise_std(self, std: float) -> None:
        """Noise std for subsequent samples (stored per observation)."""
        self._check_noise_std(std)
        with self._lock:
            self.noise_std = float(std)

    def set_seed(self, seed: int) -> None:
        """Seed used from the next reset() on (`pending_seed`)."""
        self.pending_seed = int(seed)

    @property
    def seed(self) -> int:
        """The seed the next reset() will use (= pending_seed)."""
        return self.pending_seed

    # --------------------------------------------------------------- worker
    def poll(self) -> tuple[int, dict[str, BackendStatus], int, int]:
        """(version, status copies, pending jobs, generation)."""
        with self._lock:
            status = {
                n: dataclasses.replace(s) for n, s in self._status.items()
            }
            return self._version, status, self._pending, self._generation

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until all queued jobs are processed (headless use/tests)."""
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
        return True

    def close(self, timeout: float = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._generation += 1  # make queued work stale
        self._jobs.put(None)
        self._thread.join(timeout)

    def _accept_locked(self, obs: Observation) -> None:
        self._log.append(obs)
        self._enqueue_locked(('add', self._generation, obs))

    def _enqueue_locked(self, job) -> None:
        if self._closed:
            raise RuntimeError('session is closed')
        self._pending += 1
        self._jobs.put(job)

    def _start_generation_locked(self, observations: tuple) -> None:
        self._generation += 1
        gen = self._generation
        self._status = {
            n: BackendStatus(name=n, generation=gen) for n in self.backends
        }
        self._version += 1
        self._enqueue_locked(('rebuild', gen, observations))

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            kind, gen, payload = job
            try:
                if gen == self.generation:
                    if kind == 'rebuild':
                        for name in self.backends:
                            self._assimilate(gen, name, payload, reset=True)
                    else:
                        for name in self.backends:
                            self._assimilate(gen, name, (payload,), reset=False)
                    if self._jobs.empty():
                        for name in self.backends:
                            self._query(gen, name)
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                for name in self.backends:
                    self._publish(
                        gen, name, error=f'internal: {exc!r}', state='error'
                    )
            finally:
                with self._idle:
                    self._pending -= 1
                    self._idle.notify_all()

    def _publish(self, gen: int, name: str, **changes) -> BackendStatus | None:
        """Apply changes only if `gen` is still current. Returns a copy."""
        with self._lock:
            if gen != self._generation:
                return None
            st = self._status[name]
            for k, v in changes.items():
                setattr(st, k, v)
            self._version += 1
            return dataclasses.replace(st)

    def _get_status(self, gen: int, name: str) -> BackendStatus | None:
        with self._lock:
            if gen != self._generation:
                return None
            return dataclasses.replace(self._status[name])

    def _assimilate(
        self, gen: int, name: str, observations, reset: bool
    ) -> None:
        backend = self.backends[name]
        unavailable = getattr(backend, 'unavailable_reason', None)
        if unavailable:
            self._publish(gen, name, state='error', error=unavailable)
            return
        st = self._get_status(gen, name)
        if st is None or (st.error is not None and not reset):
            return  # stale, or blocked at its last good count until rebuild
        self._publish(gen, name, state='busy')
        t0 = time.perf_counter()
        done = st.n_assimilated
        try:
            if reset:
                backend.reset()
                done = 0
            for obs in observations:
                if gen != self.generation:
                    return
                try:
                    backend.add_measurement(obs.x, obs.y, obs.value)
                except Exception as exc:  # noqa: BLE001
                    self._publish(
                        gen,
                        name,
                        state='error',
                        n_assimilated=done,
                        failed_at=obs.index,
                        error=f'observation #{obs.index} failed: '
                        f'{type(exc).__name__}: {exc}',
                    )
                    return
                done = obs.index
        except Exception as exc:  # noqa: BLE001 - reset failure
            self._publish(
                gen, name, state='error', error=f'reset failed: {exc}'
            )
            return
        ms = (time.perf_counter() - t0) * 1e3
        self._publish(
            gen,
            name,
            n_assimilated=done,
            update_ms=ms,
            update_count=len(observations),
            update_replay=bool(reset),
            state='queued',
        )

    def _query(self, gen: int, name: str) -> None:
        backend = self.backends[name]
        st = self._get_status(gen, name)
        if st is None or st.error is not None:
            return
        if not backend.is_ready():
            self._publish(
                gen,
                name,
                state='waiting',
                note=backend.not_ready_reason(),
                mean=None,
                var=None,
                result_count=None,
                query_ms=None,
                query_cells=None,
                result_id=self._next_result_id(),
            )
            return
        self._publish(gen, name, state='busy')
        t0 = time.perf_counter()
        try:
            mean, var = backend.query(self.grid.xy)
            n = len(self.grid.xy)
            mean = np.asarray(mean, dtype=float)
            var = np.asarray(var, dtype=float)
            if mean.shape != (n,) or var.shape != (n,):
                raise ValueError(
                    f'query returned shapes {mean.shape}, {var.shape}; expected ({n},)'
                )
        except Exception as exc:  # noqa: BLE001
            self._publish(
                gen,
                name,
                state='error',
                query_ms=None,
                query_cells=None,
                query_error=f'query failed: {type(exc).__name__}: {exc}',
            )
            return
        ms = (time.perf_counter() - t0) * 1e3
        self._publish(
            gen,
            name,
            state='ready',
            note='',
            mean=mean,
            var=var,
            result_count=st.n_assimilated,
            query_ms=ms,
            query_cells=n,
            query_error=None,
            result_id=self._next_result_id(),
        )

    def _next_result_id(self) -> int:
        with self._lock:
            self._result_counter += 1
            return self._result_counter

    # ---------------------------------------------------------- validation
    def _check_position(self, x, y) -> tuple[float, float]:
        x, y = float(x), float(y)
        if not self.scenario.domain.contains(x, y):
            raise ValueError(
                f'position ({x}, {y}) is outside the scenario domain'
            )
        return x, y

    @staticmethod
    def _check_noise_std(std) -> None:
        if not (math.isfinite(float(std)) and float(std) >= 0.0):
            raise ValueError(f'noise std must be finite and >= 0, got {std}')

    # ------------------------------------------------------------ save/load
    def metadata(self, extra: dict | None = None) -> dict:
        unit = self.scenario.domain.unit
        g = self.grid
        meta = {
            'format': SESSION_FORMAT,
            'format_version': SESSION_FORMAT_VERSION,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'num_observations': len(self.observations),
            'scenario': self.scenario.record(),
            'units': {'position': 'm', 'value': unit, 'variance': f'{unit}^2'},
            'noise': {
                'current_std': self.noise_std,
                'seed_in_use': self.seed_in_use,
                'pending_seed': self.pending_seed,
                'draws_from_current_generator': self._noise_draws,
                'generator': 'numpy.random.default_rng(seed_in_use), re-seeded '
                'with pending_seed on reset; undo/load do not '
                'touch it, loaded observations keep their stored noise',
                'applies_to': 'sampled observations only; manual values are '
                'used as typed (no noise, no clipping)',
                'clip_range': (
                    list(self.scenario.clip_range)
                    if self.scenario.clip_range is not None
                    else None
                ),
                'note': 'per-observation noise and noise_std are in the CSV',
            },
            'display_grid': {
                'layout': 'cell centres, row = y (np.meshgrid(xs, ys))',
                'nx': len(g.xs),
                'ny': len(g.ys),
                'x_first': float(g.xs[0]),
                'x_last': float(g.xs[-1]),
                'y_first': float(g.ys[0]),
                'y_last': float(g.ys[-1]),
                'extent': list(g.extent),
            },
            'backends': {
                n: b.effective_config() for n, b in self.backends.items()
            },
            'code_versions': code_versions(),
            'environment': environment_versions(),
        }
        if extra:
            meta.update(extra)
        return meta

    def save(
        self, csv_path: str | Path, extra: dict | None = None
    ) -> tuple[Path, Path]:
        """Write the log as CSV and the metadata as JSON next to it."""
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = csv_path.with_suffix('.json')
        obs = self.observations
        with csv_path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(CSV_FIELDS)
            for o in obs:
                w.writerow(
                    [
                        o.index,
                        repr(o.x),
                        repr(o.y),
                        repr(o.value),
                        '' if o.latent is None else repr(o.latent),
                        repr(o.noise),
                        repr(o.noise_std),
                        o.source,
                        int(o.clipped),
                    ]
                )
        meta = self.metadata(extra)
        meta['csv_file'] = csv_path.name
        meta['num_observations'] = len(obs)
        json_path.write_text(json.dumps(meta, indent=2))
        return csv_path, json_path

    def load(self, csv_path: str | Path) -> list[str]:
        """Replace the log with a saved one and replay it. Returns warnings."""
        csv_path = Path(csv_path)
        observations = read_observations_csv(csv_path)
        warnings = []
        json_path = csv_path.with_suffix('.json')
        if json_path.is_file():
            meta = json.loads(json_path.read_text())
            saved = meta.get('scenario', {}).get('content_sha256')
            if saved != self.scenario.content_hash():
                warnings.append(
                    'scenario content differs from the saved session '
                    f'({meta.get("scenario", {}).get("path")}); values replayed '
                    'unchanged into the current scenario'
                )
            for name, backend in self.backends.items():
                saved_cfg = meta.get('backends', {}).get(name)
                current = backend.effective_config()
                if saved_cfg is None:
                    warnings.append(f'{name}: not part of the saved session')
                elif saved_cfg.get('effective') != current.get('effective'):
                    warnings.append(
                        f'{name}: effective configuration differs from the saved session'
                    )
        else:
            warnings.append(
                f'no metadata file {json_path.name}; cannot check scenario/config'
            )
        self.replace_log(observations)
        return warnings


def read_observations_csv(path: str | Path) -> list[Observation]:
    path = Path(path)
    out = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        missing = set(CSV_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path}: missing CSV columns {sorted(missing)}')
        for row in reader:
            out.append(
                Observation(
                    index=int(row['index']),
                    x=float(row['x']),
                    y=float(row['y']),
                    value=float(row['value']),
                    latent=float(row['latent'])
                    if row['latent'] != ''
                    else None,
                    noise=float(row['noise']),
                    noise_std=float(row['noise_std']),
                    source=row['source'],
                    clipped=bool(int(row['clipped'])),
                )
            )
    return out


def rmse_on_common_grid(
    truth: np.ndarray, means: dict[str, np.ndarray | None]
) -> tuple[dict[str, float], int]:
    """RMSE of each available mean against truth on the cells where truth and
    *all* available means are finite. Returns (rmse per name, cells used)."""
    truth = np.asarray(truth, dtype=float)
    valid = np.isfinite(truth)
    avail = {
        n: np.asarray(m, dtype=float) for n, m in means.items() if m is not None
    }
    for m in avail.values():
        valid &= np.isfinite(m)
    n = int(valid.sum())
    if n == 0:
        return {name: float('nan') for name in avail}, 0
    return {
        name: float(np.sqrt(np.mean((m[valid] - truth[valid]) ** 2)))
        for name, m in avail.items()
    }, n


# ------------------------------------------------------------- provenance
def _git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(
        ['git', '--no-optional-locks', *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return out.stdout.strip()


def git_state(path: Path) -> dict:
    """HEAD, dirty flag and diff summary of the repository containing path.
    Read-only (`--no-optional-locks`, never writes the index)."""
    try:
        top = _git(['rev-parse', '--show-toplevel'], path)
        root = Path(top)
        porcelain = _git(['status', '--porcelain'], root)
        try:
            head = _git(['rev-parse', 'HEAD'], root)
        except subprocess.CalledProcessError:
            head = None
        stat = _git(['diff', '--shortstat', 'HEAD'], root) if head else ''
        lines = porcelain.splitlines()
        return {
            'toplevel': top,
            'head': head,
            'dirty': bool(lines),
            'modified_tracked': sum(
                1 for ln in lines if not ln.startswith('??')
            ),
            'untracked': sum(1 for ln in lines if ln.startswith('??')),
            'diff_shortstat': stat,
        }
    except Exception as exc:  # noqa: BLE001
        return {'path': str(path), 'error': f'{type(exc).__name__}: {exc}'}


def code_versions() -> dict:
    import scalar_field_sim

    import scalar_field_belief

    dirs = {
        'belief_playground': Path(__file__).resolve().parent,
        'scalar_field_belief': Path(
            next(iter(scalar_field_belief.__path__))
        ).resolve(),
        'scalar_field_sim': Path(
            next(iter(scalar_field_sim.__path__))
        ).resolve(),
    }
    return {
        name: {'module_dir': str(d), **git_state(d)} for name, d in dirs.items()
    }


def environment_versions() -> dict:
    versions = {'python': platform.python_version(), 'numpy': np.__version__}
    for mod in ('torch', 'gpytorch', 'matplotlib', 'yaml'):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = None
    # GUI toolkit (distribution metadata only; does not import Qt).
    from importlib import metadata

    for dist in ('PySide6', 'PySide6-Essentials', 'pyqtgraph'):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return versions
