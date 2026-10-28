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

"""Command line: `ros2 run scalar_field_belief belief_playground [options]`
(installed) or `python -m belief_playground [options]` (direct)."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from belief_playground.backends import BACKEND_NAMES, make_backends
from belief_playground.launcher import (
    DEFAULT_SCENARIO,
    ResourceError,
    available_scenarios,
    default_configs,
    default_visgp_config,
    resolve_scenario,
    scenario_roots,
)
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import Session


def parse_args(argv=None, prog: str | None = None):
    p = argparse.ArgumentParser(
        prog=prog or 'python -m belief_playground',
        description='Click a scenario map to add scalar observations and compare '
        'the baseline GP, wall-aware GP, GMRF and visGP beliefs (Qt window, no '
        'ROS nodes).',
    )
    p.add_argument(
        '--scenario',
        default=DEFAULT_SCENARIO,
        help='scenario TOML path, or a short name of an installed '
        'scalar_field_sim scenario (default: %(default)s)',
    )
    p.add_argument(
        '--list-scenarios',
        action='store_true',
        help='list the installed scenario names and exit',
    )
    p.add_argument(
        '--backends',
        default='gp,gp_wall_aware,gmrf',
        help='comma-separated, ordered subset of '
        f'{",".join(BACKEND_NAMES)} (default: %(default)s)',
    )
    p.add_argument(
        '--gp-config',
        type=Path,
        default=None,
        help='GP ROS parameter YAML (default: config/belief_params_default.yaml)',
    )
    p.add_argument(
        '--gp-wall-aware-config',
        type=Path,
        default=None,
        help='wall-aware GP ROS parameter YAML, model_variant: wall_aware '
        '(default: config/belief_params_wall_aware.yaml)',
    )
    p.add_argument(
        '--gmrf-config',
        type=Path,
        default=None,
        help='GMRF ROS parameter YAML (default: config/belief_params_gmrf.yaml)',
    )
    p.add_argument(
        '--visgp-config',
        type=Path,
        default=None,
        help='visGP parameter YAML (default: config/belief_params_visgp.yaml)',
    )
    p.add_argument(
        '--gmrf-module-dir', type=Path, default=None, help=argparse.SUPPRESS
    )  # development: extra module directory
    p.add_argument(
        '--grid-step',
        type=float,
        default=0.05,
        help='display/evaluation grid spacing [m] (default %(default)s)',
    )
    p.add_argument(
        '--noise-std',
        type=float,
        default=0.0,
        help='observation noise std for sampled values [scenario unit] '
        '(default 0 = off)',
    )
    p.add_argument(
        '--seed',
        type=int,
        default=None,
        help='noise seed (default: scenario seed, else 0)',
    )
    p.add_argument(
        '--load',
        type=Path,
        default=None,
        help='session CSV to load and replay at start',
    )
    p.add_argument(
        '--out-dir',
        type=Path,
        default=Path('belief_playground_sessions'),
        help='start directory of the Save session / Export image dialogs '
        '(default ./%(default)s)',
    )
    p.add_argument(
        '--hide-measurements',
        action='store_true',
        help='hide measurement markers and labels (display only)',
    )
    p.add_argument(
        '--mean-range',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=None,
        help='fixed colour range for truth and all estimates '
        '(default: shared automatic range)',
    )
    p.add_argument(
        '--std-max',
        type=float,
        default=None,
        metavar='MAX',
        help='fixed upper colour limit for all standard deviations '
        '(default: shared automatic range from 0)',
    )
    args = p.parse_args(argv)
    if args.mean_range is not None:
        lo, hi = args.mean_range
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            p.error('--mean-range needs finite MIN < MAX')
    if args.std_max is not None and not (
        math.isfinite(args.std_max) and args.std_max > 0
    ):
        p.error('--std-max must be finite and > 0')
    return args


def build_session(args, installed: bool = False) -> Session:
    names = [n.strip() for n in args.backends.split(',') if n.strip()]
    if not names or len(set(names)) != len(names):
        raise SystemExit(
            f'--backends must list distinct names from {BACKEND_NAMES}'
        )
    scenario = Scenario(resolve_scenario(args.scenario))
    grid = make_display_grid(
        scenario.domain.x_range, scenario.domain.y_range, args.grid_step
    )
    gp_config, gmrf_config = args.gp_config, args.gmrf_config
    gp_wall_aware_config = args.gp_wall_aware_config
    if (
        gp_config is None
        or gmrf_config is None
        or ('gp_wall_aware' in names and gp_wall_aware_config is None)
    ):
        default_gp, default_gp_wall_aware, default_gmrf = default_configs(
            installed
        )
        gp_config = gp_config or default_gp
        gp_wall_aware_config = gp_wall_aware_config or default_gp_wall_aware
        gmrf_config = gmrf_config or default_gmrf
    visgp_config = getattr(args, 'visgp_config', None)
    if 'visgp' in names and visgp_config is None:
        visgp_config = default_visgp_config(installed)
    backends = make_backends(
        names,
        scenario.domain,
        gp_config,
        gmrf_config,
        args.gmrf_module_dir,
        gp_wall_aware_config=gp_wall_aware_config,
        visgp_config=visgp_config,
    )
    for name, b in backends.items():
        reason = getattr(b, 'unavailable_reason', None)
        if reason:
            print(
                f'[belief_playground] backend {name} unavailable: {reason}',
                file=sys.stderr,
            )
    seed = args.seed
    if seed is None:
        seed = scenario.config.seed if scenario.config.seed is not None else 0
    if scenario.config.measurement_noise_std > 0 and args.noise_std == 0:
        print(
            '[belief_playground] note: scenario measurement_noise_std='
            f'{scenario.config.measurement_noise_std} is not used; pass '
            '--noise-std to add noise',
            file=sys.stderr,
        )
    print(f'[belief_playground] scenario {scenario.path}', file=sys.stderr)
    return Session(
        scenario, backends, grid, noise_std=args.noise_std, seed=seed
    )


def _print_scenarios() -> int:
    roots = scenario_roots()
    if not roots:
        print(
            'no scenario directory found (source the workspace that installs '
            'scalar_field_sim)',
            file=sys.stderr,
        )
        return 1
    print('scenario directories: ' + ', '.join(map(str, roots)))
    for rel, _path in available_scenarios(roots):
        print(f'  {rel}')
    return 0


def main(argv=None, installed: bool = False) -> int:
    prog = (
        'ros2 run scalar_field_belief belief_playground' if installed else None
    )
    args = parse_args(argv, prog=prog)
    if args.list_scenarios:
        return _print_scenarios()
    try:
        args.scenario = resolve_scenario(args.scenario)
    except ResourceError as exc:
        print(f'belief_playground: error: {exc}', file=sys.stderr)
        return 2
    try:
        from belief_playground.gui import run_app
    except ImportError as exc:
        print(
            f'belief_playground: cannot import the Qt frontend ({exc}). It needs '
            f'PySide6 and pyqtgraph for this interpreter ({sys.executable}); '
            'see tools/belief_playground/README.md.',
            file=sys.stderr,
        )
        return 1
    try:
        session = build_session(args, installed=installed)
    except ResourceError as exc:
        print(f'belief_playground: error: {exc}', file=sys.stderr)
        return 2
    try:
        return run_app(session, args)
    finally:
        session.close()


def _main_interruptible(argv=None, installed: bool = False) -> int:
    """main(); Ctrl+C before the window is up exits quietly with 130."""
    try:
        return main(argv, installed=installed)
    except KeyboardInterrupt:
        print('belief_playground: interrupted', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(_main_interruptible())
