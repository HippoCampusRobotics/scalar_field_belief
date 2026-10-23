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

"""Scenario/config resolution of the launcher layer (no GUI, no ROS needed)."""

from __future__ import annotations

from pathlib import Path

import pytest
from belief_playground import launcher
from belief_playground.backends import (
    DEFAULT_GMRF_CONFIG,
    DEFAULT_GP_CONFIG,
    DEFAULT_GP_WALL_AWARE_CONFIG,
)


@pytest.fixture
def roots(tmp_path):
    a = tmp_path / 'share_a' / 'scenarios'
    for rel in (
        'tests/fields/01_single_wall.toml',
        'tests/fields/00_open.toml',
        'dup_one/field_x.toml',
        'dup_two/field_x.toml',
        'top.toml',
        '.hidden/secret.toml',
    ):
        p = a / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# scenario\n')
    return [a]


def test_short_and_relative_names(roots):
    root = roots[0]
    assert (
        launcher.resolve_scenario('01_single_wall', roots)
        == (root / 'tests/fields/01_single_wall.toml').resolve()
    )
    assert (
        launcher.resolve_scenario('tests/fields/00_open', roots)
        == (root / 'tests/fields/00_open.toml').resolve()
    )
    assert (
        launcher.resolve_scenario('top', roots) == (root / 'top.toml').resolve()
    )
    # A relative name disambiguates.
    assert (
        launcher.resolve_scenario('dup_two/field_x', roots)
        == (root / 'dup_two/field_x.toml').resolve()
    )


def test_ambiguous_and_unknown_names(roots):
    with pytest.raises(launcher.ResourceError) as e:
        launcher.resolve_scenario('field_x', roots)
    assert 'ambiguous' in str(e.value)
    assert 'dup_one/field_x' in str(e.value) and 'dup_two/field_x' in str(
        e.value
    )
    with pytest.raises(launcher.ResourceError) as e:
        launcher.resolve_scenario('01_singel_wall', roots)
    assert 'unknown scenario' in str(e.value) and '01_single_wall' in str(
        e.value
    )
    assert '--list-scenarios' in str(e.value)
    with pytest.raises(launcher.ResourceError):
        launcher.resolve_scenario('secret', roots)  # hidden directories skipped


def test_explicit_paths(roots, tmp_path, monkeypatch):
    f = tmp_path / 'mine.toml'
    f.write_text('# x\n')
    assert launcher.resolve_scenario(str(f), roots) == f.resolve()
    monkeypatch.chdir(tmp_path)
    assert launcher.resolve_scenario('mine.toml', roots) == f.resolve()
    with pytest.raises(launcher.ResourceError) as e:
        launcher.resolve_scenario(str(tmp_path / 'missing.toml'), roots)
    assert 'not found' in str(e.value)


def test_no_scenario_directory(monkeypatch):
    with pytest.raises(launcher.ResourceError) as e:
        launcher.resolve_scenario('01_single_wall', [])
    assert 'no scenario directory' in str(e.value)


def test_available_scenarios_lists_relative_names(roots):
    names = [rel for rel, _ in launcher.available_scenarios(roots)]
    assert names == sorted(names)
    assert 'tests/fields/01_single_wall' in names and not any(
        'secret' in n for n in names
    )


def test_installed_resources_come_from_the_ament_share(tmp_path, monkeypatch):
    shares = {
        'scalar_field_belief': tmp_path / 'share' / 'scalar_field_belief',
        'scalar_field_sim': tmp_path / 'share' / 'scalar_field_sim',
    }
    (shares['scalar_field_sim'] / 'scenarios' / 'tests').mkdir(parents=True)
    (
        shares['scalar_field_sim']
        / 'scenarios'
        / 'tests'
        / '01_single_wall.toml'
    ).write_text('')
    monkeypatch.setattr(launcher, 'package_share', lambda pkg: shares.get(pkg))
    gp, gp_wall_aware, gmrf = launcher.default_configs(installed=True)
    assert (
        gp
        == shares['scalar_field_belief']
        / 'config'
        / 'belief_params_default.yaml'
    )
    assert (
        gp_wall_aware
        == shares['scalar_field_belief']
        / 'config'
        / 'belief_params_wall_aware.yaml'
    )
    assert (
        gmrf
        == shares['scalar_field_belief'] / 'config' / 'belief_params_gmrf.yaml'
    )
    assert launcher.scenario_roots() == [
        shares['scalar_field_sim'] / 'scenarios'
    ]
    assert launcher.resolve_scenario('01_single_wall').parent.name == 'tests'
    # Direct use from the source checkout keeps the checkout's files.
    assert launcher.default_configs(installed=False) == (
        DEFAULT_GP_CONFIG,
        DEFAULT_GP_WALL_AWARE_CONFIG,
        DEFAULT_GMRF_CONFIG,
    )
    monkeypatch.setattr(launcher, 'package_share', lambda pkg: None)
    with pytest.raises(launcher.ResourceError):
        launcher.default_configs(installed=True)


def test_source_fallback_without_ament(monkeypatch):
    monkeypatch.setattr(launcher, 'package_share', lambda pkg: None)
    roots = launcher.scenario_roots()
    import scalar_field_sim

    src = (
        Path(next(iter(scalar_field_sim.__path__))).resolve().parent
        / 'scenarios'
    )
    assert roots == ([src] if src.is_dir() else [])


def test_package_share_without_a_sourced_workspace(monkeypatch):
    monkeypatch.delenv('AMENT_PREFIX_PATH', raising=False)
    assert launcher.package_share('scalar_field_sim') is None
