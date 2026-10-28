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

"""Resource lookup for the command line (scenarios and default configs).

This is the only playground module that may use the ROS package index, and it
imports `ament_index_python` lazily, only when an installed resource has to be
found. scenario/backends/session stay ROS-free; nothing here initialises rclpy
or creates nodes.

* Scenario short names (`01_single_wall`, or a relative path such as
  `tests/fields/01_single_wall`) are looked up recursively as `<name>.toml`
  in `share/scalar_field_sim/scenarios` of the sourced workspace. Explicit
  file paths are used as given.
* Default backend configs: the installed launcher uses
  `share/scalar_field_belief/config/`; `python -m belief_playground` from a
  source checkout uses the checkout's files (falling back to the share). The
  visGP config is looked up only when the visgp backend is selected.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from belief_playground.backends import (
    DEFAULT_GMRF_CONFIG,
    DEFAULT_GP_CONFIG,
    DEFAULT_GP_WALL_AWARE_CONFIG,
    DEFAULT_VISGP_CONFIG,
)

DEFAULT_SCENARIO = '01_single_wall'
GP_CONFIG_NAME = 'belief_params_default.yaml'
GP_WALL_AWARE_CONFIG_NAME = 'belief_params_wall_aware.yaml'
GMRF_CONFIG_NAME = 'belief_params_gmrf.yaml'
VISGP_CONFIG_NAME = 'belief_params_visgp.yaml'


class ResourceError(RuntimeError):
    """A scenario or default configuration could not be resolved."""


def package_share(package: str) -> Path | None:
    """share/<package> from the ament index, or None if not available."""
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        return None
    try:
        return Path(get_package_share_directory(package))
    except (PackageNotFoundError, ValueError, LookupError, OSError):
        return None  # not installed, or no AMENT_PREFIX_PATH at all


def scenario_roots() -> list[Path]:
    """Directories searched for scenario short names."""
    roots = []
    share = package_share('scalar_field_sim')
    if share is not None and (share / 'scenarios').is_dir():
        roots.append(share / 'scenarios')
    if not roots:
        # Without a sourced workspace: the scenarios next to an importable
        # scalar_field_sim source checkout (its package dir's sibling).
        try:
            import scalar_field_sim

            for p in scalar_field_sim.__path__:
                candidate = Path(p).resolve().parent / 'scenarios'
                if candidate.is_dir() and candidate not in roots:
                    roots.append(candidate)
        except ImportError:
            pass
    return roots


def available_scenarios(roots=None) -> list[tuple[str, Path]]:
    """(relative name without .toml, path) of every scenario file, sorted."""
    roots = scenario_roots() if roots is None else roots
    out = {}
    for root in roots:
        for path in sorted(Path(root).rglob('*.toml')):
            rel = path.relative_to(root).with_suffix('').as_posix()
            if any(
                part.startswith('.') for part in path.relative_to(root).parts
            ):
                continue
            out.setdefault(rel, path)
    return sorted(out.items())


def resolve_scenario(spec: str | Path, roots=None) -> Path:
    """Scenario file for an explicit path or a short name."""
    spec_str = str(spec)
    path = Path(spec_str).expanduser()
    if path.is_file():
        return path.resolve()
    if path.suffix == '.toml' or path.is_absolute():
        raise ResourceError(f'scenario file not found: {spec_str}')
    roots = scenario_roots() if roots is None else roots
    if not roots:
        raise ResourceError(
            f'cannot resolve scenario name {spec_str!r}: no scenario directory found '
            '(source the workspace that installs scalar_field_sim, or pass a TOML path)'
        )
    entries = available_scenarios(roots)
    name = spec_str.strip('/')
    exact = [p for rel, p in entries if rel == name]
    matches = exact or [p for rel, p in entries if rel.split('/')[-1] == name]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        rels = [rel for rel, p in entries if p in matches]
        raise ResourceError(
            f'scenario name {spec_str!r} is ambiguous; use one of: '
            + ', '.join(rels)
        )
    close = difflib.get_close_matches(
        name.split('/')[-1],
        sorted({rel.split('/')[-1] for rel, _ in entries}),
        n=5,
    )
    hint = f'; did you mean {", ".join(close)}?' if close else ''
    raise ResourceError(
        f'unknown scenario {spec_str!r} in {", ".join(map(str, roots))}{hint} '
        '(see --list-scenarios)'
    )


def default_configs(installed: bool) -> tuple[Path, Path, Path]:
    """(GP, wall-aware GP, GMRF config) used when no --gp-config/
    --gp-wall-aware-config/--gmrf-config is given."""
    if (
        not installed
        and DEFAULT_GP_CONFIG.is_file()
        and DEFAULT_GP_WALL_AWARE_CONFIG.is_file()
        and DEFAULT_GMRF_CONFIG.is_file()
    ):
        return (
            DEFAULT_GP_CONFIG,
            DEFAULT_GP_WALL_AWARE_CONFIG,
            DEFAULT_GMRF_CONFIG,
        )
    share = package_share('scalar_field_belief')
    if share is None:
        raise ResourceError(
            'cannot find share/scalar_field_belief (is the workspace sourced?); '
            'pass --gp-config, --gp-wall-aware-config and --gmrf-config '
            'explicitly'
        )
    return (
        share / 'config' / GP_CONFIG_NAME,
        share / 'config' / GP_WALL_AWARE_CONFIG_NAME,
        share / 'config' / GMRF_CONFIG_NAME,
    )


def default_visgp_config(installed: bool) -> Path:
    """visGP config used when no --visgp-config is given."""
    if not installed and DEFAULT_VISGP_CONFIG.is_file():
        return DEFAULT_VISGP_CONFIG
    share = package_share('scalar_field_belief')
    if share is None:
        raise ResourceError(
            'cannot find share/scalar_field_belief (is the workspace '
            'sourced?); pass --visgp-config explicitly'
        )
    return share / 'config' / VISGP_CONFIG_NAME


def main() -> int:
    """Entry point of the installed `belief_playground` executable."""
    from belief_playground.__main__ import _main_interruptible

    return _main_interruptible(installed=True)
