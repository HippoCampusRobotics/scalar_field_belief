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

"""Command line of the live viewer: `ros2 run scalar_field_belief
belief_live_viewer [options] [--ros-args ...]` (installed) or
`python -m belief_playground.live_main [options] [--ros-args ...]` (direct).

Without --scenario/--geometry/--bounds the domain is discovered from the
running system: the field server's `scenario_path` (domain, walls, unit and
ground truth), else the belief node's bounds and walls (no truth).
"""

from __future__ import annotations

import argparse
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from scalar_field_sim.config import resolve_package_path

from belief_playground.launcher import ResourceError, resolve_scenario
from belief_playground.live_data import DEFAULT_TRAJECTORY_MIN_STEP
from belief_playground.scenario import (
    UNKNOWN_UNIT,
    Domain,
    Scenario,
    make_display_grid,
)
from scalar_field_belief.geometry import load_wall_geometry_from_toml

NAME = 'belief_live_viewer'
POSE_TYPE_NAMES = ('odometry', 'pose', 'pose_cov')
BOUND_PARAMS = ('x_min', 'x_max', 'y_min', 'y_max')
DEFAULT_FRAME = 'map'
DOMAIN_ERRORS = (ResourceError, OSError, ValueError, KeyError, TypeError)


def parse_args(argv=None, prog: str | None = None):
    p = argparse.ArgumentParser(
        prog=prog or 'python -m belief_playground.live_main',
        description='Show the posterior mean and standard deviation published '
        'by one running field-belief node, with the travelled trajectory, '
        'measurement positions and walls. ROS arguments go after --ros-args '
        '(e.g. --ros-args -r __ns:=/bluerov00). Without --scenario, '
        '--geometry or --bounds the domain is read from the parameters of '
        'the running field server, else of the belief node.',
    )
    dom = p.add_mutually_exclusive_group()
    dom.add_argument(
        '--scenario',
        help='scenario TOML path, or a short name of an installed '
        'scalar_field_sim scenario: domain, walls, unit and ground truth',
    )
    dom.add_argument(
        '--geometry',
        type=Path,
        help='geometry or scenario TOML: domain and walls only, no truth',
    )
    dom.add_argument(
        '--bounds',
        type=float,
        nargs=4,
        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
        help='domain bounds [m], no walls, no truth',
    )
    p.add_argument(
        '--no-truth',
        action='store_true',
        help='do not show the scenario ground truth',
    )
    p.add_argument(
        '--field-server-node',
        default='/field_server',
        help='node whose scenario_path parameter gives the domain and truth '
        'in automatic mode (default: %(default)s)',
    )
    p.add_argument(
        '--belief-node',
        default='scalar_field_belief',
        help='belief node whose bounds, walls and frame_id are used if the '
        'field server has no scenario, and whose frame_id is the default '
        'frame; relative names resolve in the viewer namespace (default: '
        '%(default)s; GMRF node: gmrf_belief)',
    )
    p.add_argument(
        '--param-timeout',
        type=float,
        default=15.0,
        metavar='S',
        help='automatic mode: wait at most S seconds for these nodes '
        '(default %(default)s)',
    )
    p.add_argument(
        '--mean-topic',
        default='belief/mean_cloud',
        help='posterior mean PointCloud2 (default: %(default)s)',
    )
    p.add_argument(
        '--variance-topic',
        default='belief/variance_cloud',
        help='posterior variance PointCloud2 (default: %(default)s)',
    )
    p.add_argument(
        '--measurement-topic',
        default='ir_measurement',
        help='ScalarMeasurement topic, empty to disable (default: %(default)s)',
    )
    p.add_argument(
        '--pose-topic',
        default='odometry',
        help='vehicle pose topic, empty to disable (default: %(default)s)',
    )
    p.add_argument(
        '--pose-type',
        choices=POSE_TYPE_NAMES,
        default='odometry',
        help='message type of --pose-topic: nav_msgs/Odometry, '
        'geometry_msgs/PoseStamped or PoseWithCovarianceStamped '
        '(default: %(default)s)',
    )
    p.add_argument(
        '--frame-id',
        default=None,
        help='expected frame of all inputs; other frames are ignored, '
        "empty frames are accepted (default: the belief node's frame_id "
        'in automatic mode if available, else map)',
    )
    p.add_argument(
        '--method-label',
        default=None,
        help='override the automatically detected method name (default: '
        'from the parameters of the node that publishes the mean cloud, '
        'e.g. GP baseline, GP wall-aware, GMRF)',
    )
    p.add_argument(
        '--grid-step',
        type=float,
        default=0.02,
        help='display grid spacing [m] (default %(default)s)',
    )
    p.add_argument(
        '--mean-range',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=None,
        help='fixed colour range for truth and estimate '
        '(default: shared automatic range)',
    )
    p.add_argument(
        '--std-max',
        type=float,
        default=None,
        metavar='MAX',
        help='fixed colour range [0, MAX] for the standard deviation '
        '(default: automatic range from 0)',
    )
    p.add_argument(
        '--trajectory-min-step',
        type=float,
        default=DEFAULT_TRAJECTORY_MIN_STEP,
        metavar='M',
        help='store a trajectory point only after moving at least M [m] '
        '(default %(default)s)',
    )
    p.add_argument(
        '--stale-after',
        type=float,
        default=10.0,
        metavar='S',
        help='mark the pose stream as stale after S seconds without pose '
        'messages (default %(default)s); the belief clouds and measurements '
        'only arrive when there is news and are never marked stale',
    )
    p.add_argument(
        '--hide-measurements',
        action='store_true',
        help='start with the measurement positions hidden',
    )
    p.add_argument(
        '--hide-trajectory',
        action='store_true',
        help='start with the trajectory hidden',
    )
    args = p.parse_args(argv)
    _check_args(p, args)
    return args


def _check_ranges(p: argparse.ArgumentParser, args) -> None:
    if args.bounds is not None:
        x0, x1, y0, y1 = args.bounds
        if not all(map(math.isfinite, args.bounds)) or x1 <= x0 or y1 <= y0:
            p.error('--bounds needs finite XMIN < XMAX and YMIN < YMAX')
    if args.mean_range is not None:
        lo, hi = args.mean_range
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            p.error('--mean-range needs finite MIN < MAX')
    if args.std_max is not None and not (
        math.isfinite(args.std_max) and args.std_max > 0
    ):
        p.error('--std-max must be finite and > 0')


def _check_args(p: argparse.ArgumentParser, args) -> None:
    _check_ranges(p, args)
    for name in ('grid_step', 'stale_after', 'param_timeout'):
        value = getattr(args, name)
        if not (math.isfinite(value) and value > 0):
            p.error(f'--{name.replace("_", "-")} must be finite and > 0')
    if not (
        math.isfinite(args.trajectory_min_step)
        and args.trajectory_min_step >= 0
    ):
        p.error('--trajectory-min-step must be finite and >= 0')
    for name in (
        'frame_id',
        'mean_topic',
        'variance_topic',
        'field_server_node',
        'belief_node',
    ):
        if getattr(args, name) == '':
            p.error(f'--{name.replace("_", "-")} must not be empty')


@dataclass(frozen=True)
class DomainChoice:
    """Where the viewer's domain and frame come from."""

    domain: Domain
    scenario: Scenario | None  # gives the ground truth
    name: str  # window title
    source: str  # human-readable origin of the domain
    frame_id: str
    frame_source: str
    warnings: tuple[str, ...] = ()
    # Parameters already read from a belief node: (full name, parameters).
    belief: tuple[str, dict] | None = None


class NoDomainError(LookupError):
    """Discovery found no usable domain source."""


def has_domain_option(args) -> bool:
    return (
        args.scenario is not None
        or args.geometry is not None
        or args.bounds is not None
    )


def _walls(geometry) -> tuple:
    return tuple(
        (tuple(map(float, w.start)), tuple(map(float, w.end)))
        for w in geometry.walls
    )


def load_domain(args) -> tuple[Domain, Scenario | None, str]:
    """(domain, scenario or None, display name) from the domain option."""
    if args.scenario is not None:
        scenario = Scenario(resolve_scenario(args.scenario))
        return scenario.domain, scenario, scenario.name
    if args.geometry is not None:
        geom = load_wall_geometry_from_toml(args.geometry.expanduser())
        domain = Domain(geom.x_range, geom.y_range, _walls(geom), UNKNOWN_UNIT)
        return domain, None, geom.name or args.geometry.stem
    x0, x1, y0, y1 = args.bounds
    return Domain((x0, x1), (y0, y1), (), UNKNOWN_UNIT), None, 'bounds'


def explicit_domain(args) -> DomainChoice:
    """DomainChoice of --scenario/--geometry/--bounds (no discovery)."""
    domain, scenario, name = load_domain(args)
    if args.scenario is not None:
        source = f'--scenario {scenario.path}'
    elif args.geometry is not None:
        source = f'--geometry {args.geometry}'
    else:
        source = '--bounds'
    frame = args.frame_id or DEFAULT_FRAME
    frame_source = '--frame-id' if args.frame_id else 'default'
    return DomainChoice(domain, scenario, name, source, frame, frame_source)


def read_wall_file(path) -> tuple:
    """Walls of a geometry or scenario TOML as the GMRF node reads them.

    `[objects.walls] wall_vertices = [[x0, y0, x1, y1], ...]`; `[bounds]`
    is optional. A top-level `geometry_file` (relative to the file's
    directory) is followed one level.
    """
    path = Path(path)
    with path.open('rb') as file:
        data = tomllib.load(file)
    if 'geometry_file' in data:
        ref = data['geometry_file']
        if not isinstance(ref, str) or not ref:
            raise ValueError(
                f"{path}: 'geometry_file' must be a non-empty string"
            )
        ref_path = Path(ref) if Path(ref).is_absolute() else path.parent / ref
        with ref_path.open('rb') as file:
            data = tomllib.load(file)
        if 'geometry_file' in data:
            raise ValueError(f'{ref_path}: nested geometry_file reference')
        path = ref_path
    elif 'field_file' in data:
        raise ValueError(f"{path}: 'field_file' without 'geometry_file'")
    rows = data.get('objects', {}).get('walls', {}).get('wall_vertices', [])
    walls = []
    for k, row in enumerate(rows):
        ok = isinstance(row, list) and len(row) == 4
        ok = ok and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in row
        )
        if not ok or not all(map(math.isfinite, row)):
            raise ValueError(
                f'{path}: wall_vertices[{k}] must be 4 finite numbers'
            )
        x0, y0, x1, y1 = map(float, row)
        if (x0, y0) == (x1, y1):
            raise ValueError(f'{path}: wall_vertices[{k}] has zero length')
        walls.append(((x0, y0), (x1, y1)))
    return tuple(walls)


def belief_walls(params: dict, share_dir=None) -> tuple[tuple, str]:
    """Walls of a belief node from its parameters, by the node's own rule,
    and a short note on where they come from.

    GP node (has `model_variant`): only in `wall_aware` mode, from
    `belief_geometry` ('scenario': the file in `scenario_path`; otherwise
    that file), read with the GP's loader. GMRF node: the flat `walls`
    list [x0, y0, x1, y1, ...] plus the `belief_geometry` file, if set,
    read like the GMRF does (`read_wall_file`). Relative paths resolve
    against the scalar_field_sim share directory (or `share_dir`). Errors
    name the parameter.
    """
    walls, notes = [], []
    flat = params.get('walls') or []
    if len(flat) % 4:
        raise ValueError('walls must have 4 values per wall')
    for k in range(0, len(flat), 4):
        x0, y0, x1, y1 = map(float, flat[k : k + 4])
        walls.append(((x0, y0), (x1, y1)))
    if walls:
        notes.append(f'{len(walls)} walls from walls')
    selection = str(params.get('belief_geometry', '')).strip()
    gp = 'model_variant' in params
    if selection and (not gp or params['model_variant'] == 'wall_aware'):
        from_scenario = selection == 'scenario'
        param = 'scenario_path' if from_scenario else 'belief_geometry'
        file = (
            str(params.get('scenario_path', '')).strip()
            if from_scenario
            else selection
        )
        if not file:
            raise ValueError(
                "belief_geometry 'scenario' needs a non-empty scenario_path"
            )
        path = resolve_package_path(file, share_dir)
        try:
            if gp:
                geom_walls = _walls(load_wall_geometry_from_toml(path))
            else:
                geom_walls = read_wall_file(path)
        except DOMAIN_ERRORS as exc:
            raise ValueError(f'{param} {file!r} ({path}): {exc}') from exc
        walls.extend(geom_walls)
        notes.append(f'{len(geom_walls)} walls from {path}')
    return tuple(walls), ', '.join(notes)


def _belief_bounds(params: dict | None) -> tuple | None:
    if not params or not all(k in params for k in BOUND_PARAMS):
        return None
    return tuple(float(params[k]) for k in BOUND_PARAMS)


def _load_scenario(path: str, share_dir):
    """(Scenario or None, problem text or None) for a non-empty path."""
    try:
        return Scenario(resolve_package_path(path, share_dir)), None
    except DOMAIN_ERRORS as exc:
        return None, f'scenario_path {path!r} cannot be loaded here: {exc}'


def _belief_domain(bp, belief_node, share_dir, warnings) -> tuple:
    """(Domain, wall note) from the belief node's parameters."""
    x0, x1, y0, y1 = _belief_bounds(bp)
    if not all(map(math.isfinite, (x0, x1, y0, y1))) or (x1 <= x0 or y1 <= y0):
        raise ValueError(
            f'{belief_node} has invalid x_min/x_max/y_min/y_max '
            f'[{x0}, {x1}] x [{y0}, {y1}]'
        )
    try:
        walls, note = belief_walls(bp, share_dir)
    except DOMAIN_ERRORS as exc:
        walls, note = (), ''
        warnings.append(
            f'{belief_node} walls not loaded ({exc}); showing the bounds '
            'without walls'
        )
    return Domain((x0, x1), (y0, y1), walls, UNKNOWN_UNIT), note


def _frame(frame_id, bp, scenario, field_server, belief_node):
    if frame_id:
        return frame_id, '--frame-id'
    if bp and str(bp.get('frame_id', '')).strip():
        return str(bp['frame_id']).strip(), f'{belief_node} frame_id'
    if scenario is not None:
        return scenario.config.frame_id, f'{field_server} scenario frame_id'
    return DEFAULT_FRAME, 'default'


def resolve_auto_domain(
    field_server_params: dict | None,
    belief_params: dict | None,
    *,
    field_server: str = '/field_server',
    belief_node: str = 'scalar_field_belief',
    frame_id: str | None = None,
    share_dir=None,
) -> DomainChoice:
    """Domain from discovered parameters (None: node not reached).

    1. The field server's non-empty `scenario_path` (resolved like the
       scalar_field_sim nodes do): domain, walls, unit and ground truth. If
       it cannot be loaded here, a warning is recorded and 2. is used.
    2. Else the belief node's x_min/x_max/y_min/y_max and walls
       (`belief_walls`; without walls and with a warning if they cannot be
       loaded), without truth.

    The frame is `frame_id` if given, else the belief node's `frame_id`,
    else the scenario's `frame_id`, else 'map' (with a warning if the
    belief node was not reached). A warning also notes belief bounds that
    differ from a field-server domain. Raises NoDomainError if neither
    source is usable, ValueError for invalid belief bounds.
    """
    warnings = []
    path = str((field_server_params or {}).get('scenario_path', '')).strip()
    scenario, fs_problem = None, None
    if path:
        scenario, fs_problem = _load_scenario(path, share_dir)
    bp = belief_params or {}
    bounds = _belief_bounds(belief_params)
    if scenario is not None:
        domain, name = scenario.domain, scenario.name
        source = f'{field_server} scenario_path: {scenario.path}'
        shown = (*domain.x_range, *domain.y_range)
        if bounds is not None and any(
            abs(a - b) > 1e-9 for a, b in zip(bounds, shown, strict=True)
        ):
            warnings.append(
                f'{belief_node} bounds [{bounds[0]:g}, {bounds[1]:g}] x '
                f'[{bounds[2]:g}, {bounds[3]:g}] differ from the shown domain '
                f'[{shown[0]:g}, {shown[1]:g}] x [{shown[2]:g}, {shown[3]:g}] '
                f'({field_server}); belief values outside it are not shown'
            )
    elif bounds is not None:
        if fs_problem:
            warnings.append(
                f'{field_server} {fs_problem}; using {belief_node} instead'
            )
        domain, note = _belief_domain(bp, belief_node, share_dir, warnings)
        name = belief_node
        source = f'{belief_node} x_min/x_max/y_min/y_max' + (
            f' ({note})' if note else ''
        )
    else:
        fs_state = (
            'not reachable'
            if field_server_params is None
            else fs_problem or 'scenario_path is empty'
        )
        bn_state = (
            'not reachable'
            if belief_params is None
            else 'no x_min/x_max/y_min/y_max parameters'
        )
        raise NoDomainError(
            f'no domain found ({field_server}: {fs_state}; {belief_node}: '
            f'{bn_state}); start the simulation or estimator first, or pass '
            '--scenario, --geometry or --bounds'
        )
    frame, frame_source = _frame(
        frame_id, belief_params, scenario, field_server, belief_node
    )
    if not frame_id and belief_params is None:
        warnings.append(
            f'{belief_node} was not reached; frame {frame!r} is a fallback '
            f'({frame_source})'
        )
    return DomainChoice(
        domain,
        scenario,
        name,
        source,
        frame,
        frame_source,
        tuple(warnings),
        None if belief_params is None else (belief_node, belief_params),
    )


def split_ros_args(argv: list[str]) -> list[str]:
    """The viewer's own arguments: `argv` without the `--ros-args ...`
    sections (which rclpy reads). Raises on malformed ROS arguments."""
    from rclpy.utilities import remove_ros_args

    return remove_ros_args([NAME] + list(argv))[1:]


def _err(text: str) -> None:
    print(f'{NAME}: {text}', file=sys.stderr)


def _info(text: str) -> None:
    print(f'[{NAME}] {text}', file=sys.stderr)


def _discover(args, context, live_ros) -> DomainChoice | int:
    """Automatic mode: domain from the running nodes, or an exit code."""
    try:
        found = live_ros.discover_parameters(
            context,
            args.field_server_node,
            args.belief_node,
            need_belief=args.frame_id is None,
            timeout=args.param_timeout,
            log=_info,
        )
    except ValueError as exc:  # invalid node name
        _err(f'error: {exc}')
        return 2
    try:
        choice = resolve_auto_domain(
            found.field_server_params,
            found.belief_params,
            field_server=found.field_server,
            belief_node=found.belief_node,
            frame_id=args.frame_id,
        )
    except NoDomainError as exc:
        _err(f'error: {exc}')
        return 2
    except DOMAIN_ERRORS as exc:
        _err(f'error: cannot load the discovered domain: {exc}')
        return 2
    for warning in choice.warnings:
        _info(f'warning: {warning}')
    return choice


def _import_frontends():
    """(run_live_app, live_ros module), or an exit code."""
    try:
        from belief_playground.live_gui import run_live_app
    except ImportError as exc:
        _err(
            f'cannot import the Qt frontend ({exc}). It needs PySide6 and '
            f'pyqtgraph for this interpreter ({sys.executable}); see '
            'tools/belief_playground/README.md.'
        )
        return 1
    try:
        from belief_playground import live_ros
    except ImportError as exc:
        _err(
            f'cannot import the ROS interfaces ({exc}); source the workspace '
            'that builds scalar_field_interfaces'
        )
        return 1
    return run_live_app, live_ros


def _start(args, argv, choice, live_ros):
    """(DomainChoice, started RosSpinner) after the domain is known, or an
    exit code. Owns the rclpy context until the spinner exists."""
    import rclpy

    from belief_playground.live_data import LiveState

    try:
        context = live_ros.init_context([NAME] + argv)
    except Exception as exc:  # noqa: BLE001 - e.g. invalid --ros-args
        _err(f'cannot initialise ROS: {exc}')
        return 2
    try:
        if choice is None:
            choice = _discover(args, context, live_ros)
            if isinstance(choice, int):
                rclpy.try_shutdown(context=context)
                return choice
        state = LiveState(
            frame_id=choice.frame_id,
            trajectory_min_step=args.trajectory_min_step,
        )
        node = live_ros.LiveViewerNode(
            state,
            mean_topic=args.mean_topic,
            variance_topic=args.variance_topic,
            measurement_topic=args.measurement_topic,
            pose_topic=args.pose_topic,
            pose_type=args.pose_type,
            context=context,
            method_lookup=args.method_label is None,
            belief_node=args.belief_node,
            known_params=dict([choice.belief]) if choice.belief else None,
        )
    except BaseException:
        rclpy.try_shutdown(context=context)
        raise
    return choice, live_ros.RosSpinner(node).start()


def _run(argv, installed: bool) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        import rclpy.utilities  # noqa: F401
    except ImportError as exc:
        _err(f'cannot import rclpy ({exc}); source the ROS workspace')
        return 1
    prog = f'ros2 run scalar_field_belief {NAME}' if installed else None
    try:
        cli_args = split_ros_args(argv)
    except Exception as exc:  # noqa: BLE001 - malformed --ros-args
        _err(f'cannot initialise ROS: {exc}')
        return 2
    args = parse_args(cli_args, prog=prog)
    choice = None
    if has_domain_option(args):
        try:
            choice = explicit_domain(args)
        except DOMAIN_ERRORS as exc:
            _err(f'error: {exc}')
            return 2
    frontends = _import_frontends()
    if isinstance(frontends, int):
        return frontends
    run_live_app, live_ros = frontends
    started = _start(args, argv, choice, live_ros)
    if isinstance(started, int):
        return started
    choice, spinner = started
    try:
        _info(f'domain from {choice.source}')
        _info(f'frame {choice.frame_id!r} ({choice.frame_source})')
        for stream, st in spinner.state.stream_status().items():
            _info(f'{stream}: {st.topic or "off"}')
        grid = make_display_grid(
            choice.domain.x_range, choice.domain.y_range, args.grid_step
        )
        truth = None
        if choice.scenario is not None and not args.no_truth:
            truth = choice.scenario.truth(grid.xy)
        return run_live_app(
            state=spinner.state,
            domain=choice.domain,
            grid=grid,
            truth=truth,
            title=choice.name,
            method_label=args.method_label,
            mean_range=args.mean_range,
            std_max=args.std_max,
            show_measurements=not args.hide_measurements,
            show_trajectory=not args.hide_trajectory,
            stale_after=args.stale_after,
            domain_source=choice.source,
            frame_source=choice.frame_source,
            domain_warnings=choice.warnings,
            shutdown_hook=spinner.stop,
        )
    finally:
        spinner.stop()


def main(argv=None, installed: bool = False) -> int:
    """Run the viewer; Ctrl+C before the window is up exits with 130."""
    try:
        return _run(argv, installed)
    except KeyboardInterrupt:
        print(f'{NAME}: interrupted', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
