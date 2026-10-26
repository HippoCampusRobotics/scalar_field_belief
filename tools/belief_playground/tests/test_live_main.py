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

"""Live viewer command line (no window is opened)."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pytest
from belief_playground.live_main import (
    NoDomainError,
    belief_walls,
    explicit_domain,
    has_domain_option,
    load_domain,
    main,
    parse_args,
    read_wall_file,
    resolve_auto_domain,
)


@pytest.mark.parametrize(
    'argv',
    [
        ['--bounds', '0', '1', '0', '1', '--geometry', 'g.toml'],
        ['--scenario', 'a', '--bounds', '0', '1', '0', '1'],
        ['--bounds', '1', '0', '0', '1'],
        ['--bounds', '0', '1', '0', 'nan'],
        ['--bounds', '0', '1', '0'],
        ['--bounds', '0', '1', '0', '1', '--grid-step', '0'],
        ['--bounds', '0', '1', '0', '1', '--stale-after', '-1'],
        ['--bounds', '0', '1', '0', '1', '--trajectory-min-step', 'inf'],
        ['--bounds', '0', '1', '0', '1', '--mean-range', '1', '1'],
        ['--bounds', '0', '1', '0', '1', '--frame-id', ''],
        ['--bounds', '0', '1', '0', '1', '--pose-type', 'tf'],
        ['--param-timeout', '0'],
        ['--param-timeout', 'nan'],
        ['--belief-node', ''],
        ['--field-server-node', ''],
    ],
)
def test_invalid_arguments_are_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_bounds_domain_and_defaults():
    args = parse_args(['--bounds', '-1', '2', '0.5', '4'])
    domain, scenario, name = load_domain(args)
    assert (domain.x_range, domain.y_range, domain.walls) == (
        (-1.0, 2.0),
        (0.5, 4.0),
        (),
    )
    assert scenario is None and name == 'bounds'
    assert (args.mean_topic, args.variance_topic) == (
        'belief/mean_cloud',
        'belief/variance_cloud',
    )
    assert (args.measurement_topic, args.pose_topic, args.pose_type) == (
        'ir_measurement',
        'odometry',
        'odometry',
    )
    assert (args.frame_id, args.grid_step, args.method_label) == (
        None,
        0.02,
        None,
    )
    choice = explicit_domain(args)
    assert (choice.frame_id, choice.frame_source) == ('map', 'default')
    assert choice.source == '--bounds'


def test_without_domain_option_discovery_is_used():
    args = parse_args([])
    assert not has_domain_option(args)
    assert (args.field_server_node, args.belief_node, args.param_timeout) == (
        '/field_server',
        'scalar_field_belief',
        15.0,
    )
    assert has_domain_option(parse_args(['--bounds', '0', '1', '0', '1']))
    args = parse_args(['--bounds', '0', '1', '0', '1', '--frame-id', 'world'])
    assert explicit_domain(args).frame_id == 'world'


GEOMETRY = """
[general]
name = "two_walls"

[bounds]
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 4.0

[objects.walls]
wall_vertices = [[1.0, 0.5, 1.0, 3.5], [0.2, 2.0, 0.8, 2.0]]
"""

BELIEF = {
    'frame_id': 'world',
    'x_min': -1.0,
    'x_max': 2.0,
    'y_min': 0.5,
    'y_max': 4.0,
    'model_variant': 'baseline',
    'belief_geometry': 'scenario',
    'scenario_path': 'unused_in_baseline.toml',
}


def test_field_server_scenario_gives_domain_and_truth(offset_scenario_path):
    choice = resolve_auto_domain(
        {'scenario_path': str(offset_scenario_path)}, BELIEF
    )
    assert choice.scenario is not None
    assert choice.domain == choice.scenario.domain
    assert choice.domain.x_range == (-3.0, 1.0) and len(choice.domain.walls)
    assert choice.domain.unit == 'mV'
    assert (
        choice.source == f'/field_server scenario_path: {offset_scenario_path}'
    )
    assert (choice.frame_id, choice.frame_source) == (
        'world',
        'scalar_field_belief frame_id',
    )


def test_relative_scenario_path_uses_the_share_directory(
    offset_scenario_path,
):
    share = offset_scenario_path.parent
    choice = resolve_auto_domain(
        {'scenario_path': '  offset_rect.toml '},
        None,
        field_server='/sim/field_server',
        share_dir=share,
    )
    assert choice.scenario.path == offset_scenario_path.resolve()
    assert choice.source.startswith('/sim/field_server scenario_path: ')
    assert (choice.frame_id, choice.frame_source) == (
        'map',
        '/sim/field_server scenario frame_id',
    )
    assert any('was not reached' in w for w in choice.warnings)


@pytest.mark.parametrize('field_server', [None, {}, {'scenario_path': ' '}])
def test_belief_bounds_without_field_server_scenario(field_server):
    choice = resolve_auto_domain(
        field_server, BELIEF, belief_node='/uuv00/scalar_field_belief'
    )
    assert choice.scenario is None
    assert (choice.domain.x_range, choice.domain.y_range) == (
        (-1.0, 2.0),
        (0.5, 4.0),
    )
    assert choice.domain.walls == ()  # baseline: belief_geometry not used
    assert choice.source == '/uuv00/scalar_field_belief x_min/x_max/y_min/y_max'
    assert choice.frame_id == 'world'


def test_explicit_frame_wins(offset_scenario_path):
    choice = resolve_auto_domain(
        {'scenario_path': str(offset_scenario_path)}, BELIEF, frame_id='odom'
    )
    assert (choice.frame_id, choice.frame_source) == ('odom', '--frame-id')
    no_frame = dict(BELIEF, frame_id='')
    assert resolve_auto_domain(None, no_frame).frame_id == 'map'


def test_wall_aware_walls_from_scenario_or_file(tmp_path, offset_scenario_path):
    geometry = tmp_path / 'geom.toml'
    geometry.write_text(GEOMETRY)
    aware = dict(
        BELIEF,
        model_variant='wall_aware',
        scenario_path='offset_rect.toml',
    )
    walls, note = belief_walls(aware, share_dir=tmp_path)
    assert walls == (((0.0, 2.2), (0.0, 3.3)),)
    assert str(offset_scenario_path) in note
    walls, _ = belief_walls(
        dict(aware, belief_geometry=str(geometry)), share_dir=tmp_path
    )
    assert walls == (((1.0, 0.5), (1.0, 3.5)), ((0.2, 2.0), (0.8, 2.0)))
    choice = resolve_auto_domain(
        None, dict(aware, belief_geometry='geom.toml'), share_dir=tmp_path
    )
    assert len(choice.domain.walls) == 2 and 'geom.toml' in choice.source
    with pytest.raises(ValueError, match='scenario_path'):
        belief_walls(dict(aware, scenario_path=''))


def test_gmrf_flat_walls_and_geometry_file(tmp_path):
    geometry = tmp_path / 'geom.toml'
    geometry.write_text(GEOMETRY)
    gmrf = {k: BELIEF[k] for k in ('frame_id', 'x_min', 'x_max', 'y_min')}
    gmrf.update(y_max=4.0, walls=[0.0, 1.0, 1.0, 1.0], belief_geometry='')
    walls, note = belief_walls(gmrf)
    assert walls == (((0.0, 1.0), (1.0, 1.0)),) and note == '1 walls from walls'
    walls, _ = belief_walls(dict(gmrf, belief_geometry=str(geometry)))
    assert len(walls) == 3
    with pytest.raises(ValueError, match='4 values'):
        belief_walls(dict(gmrf, walls=[0.0, 1.0, 1.0]))


def test_nothing_usable_raises_with_what_was_tried():
    with pytest.raises(NoDomainError) as exc:
        resolve_auto_domain(None, None, belief_node='/a/scalar_field_belief')
    text = exc.value.args[0]
    assert '/field_server: not reachable' in text
    assert '/a/scalar_field_belief: not reachable' in text
    assert '--scenario, --geometry or --bounds' in text
    with pytest.raises(NoDomainError, match='scenario_path is empty'):
        resolve_auto_domain({'scenario_path': ''}, {'frame_id': 'map'})
    with pytest.raises(ValueError, match='invalid x_min/x_max/y_min/y_max'):
        resolve_auto_domain(None, dict(BELIEF, x_max=-5.0))


def test_ros_args_are_split_off():
    pytest.importorskip('rclpy')
    from belief_playground.live_main import split_ros_args

    argv = [
        '--bounds',
        '0',
        '1',
        '0',
        '1',
        '--ros-args',
        '-r',
        '__ns:=/uuv00',
        '-p',
        'use_sim_time:=true',
        '--',
        '--no-truth',
    ]
    assert split_ros_args(argv) == [
        '--bounds',
        '0',
        '1',
        '0',
        '1',
        '--no-truth',
    ]


@pytest.mark.parametrize('ros_args', [['-p', 'bad'], ['--bogus'], ['-r']])
def test_malformed_ros_args_exit_2_without_traceback(ros_args):
    pytest.importorskip('rclpy')
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ('DISPLAY', 'WAYLAND_DISPLAY')
    }
    out = subprocess.run(
        [
            sys.executable,
            '-m',
            'belief_playground.live_main',
            '--bounds',
            '0',
            '1',
            '0',
            '1',
            '--ros-args',
            *ros_args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 2, out.stdout + out.stderr
    assert 'cannot initialise ROS' in out.stderr
    assert 'Traceback' not in out.stderr


@pytest.mark.parametrize(
    'text',
    [
        '[bounds]\nx_min = 0.0\n',  # incomplete bounds
        '[bounds]\nx_min = "a"\nx_max = 1\ny_min = 0\ny_max = 1\n',
        'not toml [',
    ],
)
def test_malformed_geometry_exits_2_with_message(tmp_path, capsys, text):
    pytest.importorskip('rclpy')
    path = tmp_path / 'bad.toml'
    path.write_text(text)
    assert main(['--geometry', str(path)]) == 2
    err = capsys.readouterr().err
    assert 'belief_live_viewer: error:' in err


def test_auto_mode_times_out_with_exit_2(capsys, monkeypatch):
    pytest.importorskip('rclpy')
    pytest.importorskip('PySide6')
    if 'ROS_DOMAIN_ID' not in os.environ:
        monkeypatch.setenv('ROS_DOMAIN_ID', '97')
    tag = uuid.uuid4().hex[:10]
    code = main(
        [
            '--field-server-node',
            f'/no_field_server_{tag}',
            '--param-timeout',
            '0.5',
            '--ros-args',
            '-r',
            f'__ns:=/live_test_{tag}',
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert 'waiting up to 0.5 s' in err
    assert f'/no_field_server_{tag}: not reachable' in err
    assert f'/live_test_{tag}/scalar_field_belief: not reachable' in err


FRAMED_SCENARIO = """
[general]
name = "framed"

[frame]
frame_id = "world"

[bounds]
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 4.0

[objects.sources]
source_positions = [[1.0, 2.0]]

[simulation.source_defaults]
amplitude = 1.0
sigma_x = 0.3
sigma_y = 0.3
"""


def test_frame_from_scenario_when_belief_node_missing(tmp_path):
    path = tmp_path / 'framed.toml'
    path.write_text(FRAMED_SCENARIO)
    choice = resolve_auto_domain({'scenario_path': str(path)}, None)
    assert (choice.frame_id, choice.frame_source) == (
        'world',
        '/field_server scenario frame_id',
    )
    (warning,) = choice.warnings
    assert 'scalar_field_belief was not reached' in warning
    assert "frame 'world' is a fallback" in warning
    # Answered belief node: its frame, no warning.
    choice = resolve_auto_domain(
        {'scenario_path': str(path)}, dict(BELIEF, x_min=0.0, y_min=0.0)
    )
    assert choice.frame_id == 'world' and choice.warnings == ()


def test_belief_bounds_differing_from_the_scenario_are_reported(
    offset_scenario_path,
):
    choice = resolve_auto_domain(
        {'scenario_path': str(offset_scenario_path)}, BELIEF
    )
    (warning,) = choice.warnings
    assert 'scalar_field_belief bounds [-1, 2] x [0.5, 4]' in warning
    assert 'shown domain [-3, 1] x [2, 3.5]' in warning


@pytest.mark.parametrize(
    'path', ['/no/such/scenario.toml', 'relative_missing.toml']
)
def test_unloadable_field_server_scenario_falls_back(tmp_path, path):
    choice = resolve_auto_domain(
        {'scenario_path': path}, BELIEF, share_dir=tmp_path
    )
    assert choice.scenario is None and choice.domain.x_range == (-1.0, 2.0)
    (warning,) = choice.warnings
    assert warning.startswith(f"/field_server scenario_path '{path}'")
    assert warning.endswith('using scalar_field_belief instead')
    with pytest.raises(NoDomainError) as exc:
        resolve_auto_domain({'scenario_path': path}, None, share_dir=tmp_path)
    assert f"/field_server: scenario_path '{path}' cannot be loaded" in str(
        exc.value
    )


def test_malformed_field_server_scenario_falls_back(tmp_path):
    bad = tmp_path / 'bad.toml'
    bad.write_text('not toml [')
    choice = resolve_auto_domain({'scenario_path': str(bad)}, BELIEF)
    assert choice.scenario is None
    assert 'cannot be loaded here' in choice.warnings[0]


def test_unloadable_belief_walls_give_bounds_without_walls(tmp_path):
    aware = dict(
        BELIEF, model_variant='wall_aware', belief_geometry='missing.toml'
    )
    choice = resolve_auto_domain(None, aware, share_dir=tmp_path)
    assert choice.domain.walls == () and choice.domain.x_range == (-1.0, 2.0)
    walls_warning = [w for w in choice.warnings if 'walls not loaded' in w]
    assert (
        walls_warning and "belief_geometry 'missing.toml'" in walls_warning[0]
    )


WALLS_ONLY = """
[objects.walls]
wall_vertices = [[0.5, 0.0, 0.5, 1.0], [0, 2, 1, 2]]
"""


def test_gmrf_wall_files_need_no_bounds(tmp_path):
    walls_file = tmp_path / 'walls.toml'
    walls_file.write_text(WALLS_ONLY)
    expected = (((0.5, 0.0), (0.5, 1.0)), ((0.0, 2.0), (1.0, 2.0)))
    assert read_wall_file(walls_file) == expected
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'scenario.toml').write_text('geometry_file = "../walls.toml"\n')
    assert read_wall_file(sub / 'scenario.toml') == expected
    gmrf = {k: BELIEF[k] for k in ('frame_id', 'x_min', 'x_max', 'y_min')}
    gmrf.update(y_max=4.0, belief_geometry='walls.toml')
    choice = resolve_auto_domain(None, gmrf, share_dir=tmp_path)
    assert choice.domain.walls == expected and choice.warnings == ()
    # The GP's loader needs bounds, as the GP node does.
    gp = dict(BELIEF, model_variant='wall_aware', belief_geometry='walls.toml')
    with pytest.raises(ValueError, match="belief_geometry 'walls.toml'"):
        belief_walls(gp, share_dir=tmp_path)
    (tmp_path / 'nested.toml').write_text(
        'geometry_file = "sub/scenario.toml"\n'
    )
    with pytest.raises(ValueError, match='nested'):
        read_wall_file(tmp_path / 'nested.toml')
    (tmp_path / 'bad_row.toml').write_text(
        '[objects.walls]\nwall_vertices = [[0, 0, 0, 0]]\n'
    )
    with pytest.raises(ValueError, match='zero length'):
        read_wall_file(tmp_path / 'bad_row.toml')


@pytest.mark.parametrize('option', ['--belief-node', '--field-server-node'])
def test_invalid_node_names_exit_2(capsys, monkeypatch, option):
    pytest.importorskip('rclpy')
    pytest.importorskip('PySide6')
    if 'ROS_DOMAIN_ID' not in os.environ:
        monkeypatch.setenv('ROS_DOMAIN_ID', '97')
    code = main([option, 'a b', '--param-timeout', '0.5'])
    assert code == 2
    err = capsys.readouterr().err
    assert "invalid node name 'a b'" in err
    assert 'Traceback' not in err
