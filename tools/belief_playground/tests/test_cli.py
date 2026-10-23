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

"""Entry point tests (subprocesses and in-process, no ROS nodes)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _env(**extra):
    keep = (
        'HOME',
        'PATH',
        'PYTHONPATH',
        'DISPLAY',
        'XAUTHORITY',
        'QT_QPA_PLATFORM',
    )
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(extra)
    return env


def test_build_session_with_load_and_explicit_paths(single_wall_path, tmp_path):
    pytest.importorskip('scalar_field_belief_gmrf_py')
    from belief_playground.__main__ import build_session, parse_args

    csv = tmp_path / 's.csv'
    csv.write_text(
        'index,x,y,value,latent,noise,noise_std,source,clipped\n'
        '1,0.5,1.0,0.01,,0.0,0.0,manual,0\n'
    )
    args = parse_args(
        [
            '--scenario',
            str(single_wall_path),
            '--grid-step',
            '0.1',
            '--load',
            str(csv),
            '--hide-measurements',
            '--mean-range',
            '0',
            '0.05',
            '--std-max',
            '0.01',
        ]
    )
    assert (
        args.hide_measurements
        and args.mean_range == [0.0, 0.05]
        and args.std_max == 0.01
    )
    s = build_session(args)
    try:
        assert s.scenario.path == single_wall_path.resolve()
        assert not any(
            hasattr(b, 'unavailable_reason') for b in s.backends.values()
        )
        s.load(args.load)
        assert s.wait_idle(60)
        assert len(s.observations) == 1
    finally:
        s.close()


def test_default_backends_build_all_three_in_order(single_wall_path):
    pytest.importorskip('scalar_field_belief_gmrf_py')
    from belief_playground.__main__ import build_session, parse_args

    args = parse_args(
        ['--scenario', str(single_wall_path), '--grid-step', '0.5']
    )
    s = build_session(args)
    try:
        assert list(s.backends) == ['gp', 'gp_wall_aware', 'gmrf']
        assert not any(
            hasattr(b, 'unavailable_reason') for b in s.backends.values()
        )
        assert s.backends['gp'].belief.config.model_variant == 'baseline'
        wa = s.backends['gp_wall_aware'].belief.config
        assert wa.model_variant == 'wall_aware'
    finally:
        s.close()


def test_explicit_gp_and_gmrf_configs_need_no_default_lookup(
    single_wall_path, monkeypatch
):
    pytest.importorskip('scalar_field_belief_gmrf_py')
    import belief_playground.__main__ as cli
    from belief_playground.backends import (
        DEFAULT_GMRF_CONFIG,
        DEFAULT_GP_CONFIG,
    )
    from belief_playground.launcher import ResourceError

    def no_defaults(installed):
        raise ResourceError('default_configs must not be called')

    monkeypatch.setattr(cli, 'default_configs', no_defaults)
    args = cli.parse_args(
        [
            '--scenario',
            str(single_wall_path),
            '--grid-step',
            '0.5',
            '--backends',
            'gp,gmrf',
            '--gp-config',
            str(DEFAULT_GP_CONFIG),
            '--gmrf-config',
            str(DEFAULT_GMRF_CONFIG),
        ]
    )
    s = cli.build_session(args)
    try:
        assert list(s.backends) == ['gp', 'gmrf']
        assert not any(
            hasattr(b, 'unavailable_reason') for b in s.backends.values()
        )
    finally:
        s.close()


@pytest.mark.parametrize(
    'argv',
    [['--mean-range', '1', '0'], ['--std-max', '0'], ['--std-max', 'nan']],
)
def test_invalid_display_limits_are_rejected(argv):
    from belief_playground.__main__ import parse_args

    with pytest.raises(SystemExit):
        parse_args(argv)


def test_unknown_scenario_exits_with_message(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            '-m',
            'belief_playground',
            '--scenario',
            'no_such_scenario_xyz',
        ],
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert out.returncode != 0
    assert 'no_such_scenario_xyz' in out.stderr
    assert 'Traceback' not in out.stderr


def test_no_ros_modules_are_imported(single_wall_path):
    code = (
        'import sys, belief_playground.__main__ as m, belief_playground.launcher as l, '
        'belief_playground.backends as b, belief_playground.session, '
        'belief_playground.scenario;'
        f'l.resolve_scenario({str(single_wall_path)!r});'
        "bad=[k for k in sys.modules if k.split('.')[0] in "
        "('rclpy','rclcpp','ament_index_python','rosidl_runtime_py')];"
        'print(bad); sys.exit(1 if bad else 0)'
    )
    out = subprocess.run(
        [sys.executable, '-c', code],
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_gui_module_imports_no_ros():
    pytest.importorskip('PySide6')
    pytest.importorskip('pyqtgraph')
    code = (
        'import sys, belief_playground.gui;'
        "bad=[k for k in sys.modules if k.split('.')[0] in "
        "('rclpy','ament_index_python','matplotlib')];"
        'print(bad); sys.exit(1 if bad else 0)'
    )
    out = subprocess.run(
        [sys.executable, '-c', code],
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def _installed_launcher() -> Path | None:
    """lib/scalar_field_belief/belief_playground of the sourced workspace."""
    for prefix in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep):
        exe = Path(prefix) / 'lib' / 'scalar_field_belief' / 'belief_playground'
        if prefix and exe.is_file():
            return exe
    return None


@pytest.mark.skipif(
    shutil.which('ros2') is None or _installed_launcher() is None,
    reason='needs a sourced workspace with scalar_field_belief installed',
)
def test_installed_launcher_lists_scenarios_from_any_directory(tmp_path):
    out = subprocess.run(
        [
            'ros2',
            'run',
            'scalar_field_belief',
            'belief_playground',
            '--list-scenarios',
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr
    assert 'share/scalar_field_sim/scenarios' in out.stdout
    assert '01_single_wall' in out.stdout


@pytest.mark.skipif(
    os.environ.get('BELIEF_PLAYGROUND_GUI_SMOKE') != '1',
    reason='set BELIEF_PLAYGROUND_GUI_SMOKE=1 (on Xvfb) for the real-window smoke test',
)
def test_gui_smoke_on_display(single_wall_path, tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            str(HERE / 'gui_smoke.py'),
            '--scenario',
            str(single_wall_path),
            '--out-dir',
            str(tmp_path),
        ],
        env=_env(),
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert 'GUI smoke PASSED' in out.stdout


@pytest.mark.skipif(
    bool(
        not os.environ.get('DISPLAY')
        or os.environ.get('WAYLAND_DISPLAY')
        or os.environ.get('QT_QPA_PLATFORM') != 'xcb'
    ),
    reason='needs Xvfb (env -u WAYLAND_DISPLAY QT_QPA_PLATFORM=xcb xvfb-run)',
)
def test_ctrl_c_quits_the_window_cleanly(single_wall_path, tmp_path):
    pytest.importorskip('scalar_field_belief_gmrf_py')
    import signal
    import time

    proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'belief_playground',
            '--scenario',
            str(single_wall_path),
            '--backends',
            'gmrf',
            '--grid-step',
            '0.1',
            '--out-dir',
            str(tmp_path),
        ],
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
    )
    time.sleep(6)  # window up and event loop running
    assert proc.poll() is None, proc.communicate()
    proc.send_signal(signal.SIGINT)
    out, err = proc.communicate(timeout=30)
    assert proc.returncode in (0, 130), err
    assert 'Traceback' not in err and 'KeyboardInterrupt' not in err, err
