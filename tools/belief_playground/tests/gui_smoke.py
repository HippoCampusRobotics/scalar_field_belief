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

"""Short automated GUI smoke test of the real Qt window (baseline GP,
wall-aware GP and GMRF).

    env -u WAYLAND_DISPLAY QT_QPA_PLATFORM=xcb \\
      xvfb-run -a -s "-screen 0 1920x1080x24" \\
      python tests/gui_smoke.py --scenario 01_single_wall --out-dir DIR [--screenshots DIR]

Opens the real window, clicks a map with a real Qt mouse event, waits for
all backends, then undo, save, reset, load and close. Exits 0 on success,
1 on failure; a watchdog closes the window after --timeout seconds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from belief_playground.__main__ import build_session, parse_args
from belief_playground.gui import PlaygroundWindow, ensure_app
from PySide6 import QtCore
from PySide6.QtTest import QTest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='01_single_wall')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--backends', default='gp,gp_wall_aware,gmrf')
    ap.add_argument('--gmrf-module-dir', default=None)
    ap.add_argument('--screenshots', default=None, help='directory for PNGs')
    ap.add_argument('--timeout', type=float, default=120.0)
    a = ap.parse_args()
    app = ensure_app()
    if app.platformName() != 'xcb':
        print(
            f'GUI smoke FAILED: Qt platform {app.platformName()!r}, expected xcb '
            '(run under env -u WAYLAND_DISPLAY QT_QPA_PLATFORM=xcb xvfb-run)'
        )
        return 1
    argv = [
        '--scenario',
        a.scenario,
        '--backends',
        a.backends,
        '--grid-step',
        '0.05',
        '--out-dir',
        a.out_dir,
        '--noise-std',
        '0.0005',
        '--seed',
        '1',
    ]
    if a.gmrf_module_dir:
        argv += ['--gmrf-module-dir', a.gmrf_module_dir]
    session = build_session(parse_args(argv))
    win = PlaygroundWindow(session, out_dir=a.out_dir)
    win.show()
    t_start = time.monotonic()
    shots = Path(a.screenshots) if a.screenshots else None
    log = []

    def pump(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)

    def check(cond, msg):
        if not cond:
            raise AssertionError(msg)
        log.append(f'ok: {msg}')

    def wait_all(n):
        while True:
            if time.monotonic() - t_start > a.timeout:
                raise AssertionError(f'timeout waiting for n={n}')
            _, st, pending, _ = session.poll()
            if pending == 0 and all(
                s.n_assimilated == n and s.state in ('ready', 'waiting')
                for s in st.values()
            ):
                pump(0.3)
                return st
            pump(0.1)

    def shot(name):
        if shots is not None:
            shots.mkdir(parents=True, exist_ok=True)
            win.grab().save(str(shots / f'{name}.png'))

    ok = False
    try:
        wait_all(0)
        shot('smoke_0_open')
        names = session.backend_names
        panel = win.panels[(names[-1], 'mean')]
        for k, (x, y) in enumerate([(0.6, 2.2), (1.5, 3.0)], start=1):
            QTest.mouseClick(
                panel.view.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                panel.map_to_view(x, y),
            )
            pump(0.05)
            obs = session.observations
            check(len(obs) == k, f'click {k} added one observation')
            x0, x1 = panel.domain.x_range
            tol = 2 * (x1 - x0) / panel.view.width()  # 2 px
            check(
                abs(obs[-1].x - x) < tol and abs(obs[-1].y - y) < tol,
                f'click mapped to ({obs[-1].x:.4f}, {obs[-1].y:.4f}) ~ ({x}, {y})',
            )
        st = wait_all(2)
        check(
            all(
                s.result_count == 2 or s.state == 'waiting' for s in st.values()
            ),
            f'backends: { {n: (s.state, s.result_count) for n, s in st.items()} }',
        )
        shot('smoke_1_two_clicks')
        win.actions_by_name['Undo'].trigger()
        wait_all(1)
        check(len(session.observations) == 1, 'undo left one observation')
        path = win.save_session(
            Path(a.out_dir) / 'smoke_session.csv'
        )  # no dialog
        check(
            path is not None
            and path.is_file()
            and path.with_suffix('.json').is_file(),
            f'saved {path.name}',
        )
        saved = session.observations
        win.actions_by_name['Reset'].trigger()
        wait_all(0)
        check(session.observations == (), 'reset cleared the log')
        check(win.load_session(path), 'load accepted')
        wait_all(1)
        check(session.observations == saved, 'load restored the log')
        shot('smoke_2_loaded')
        ok = True
    except Exception as exc:  # noqa: BLE001
        log.append(f'FAIL: {exc!r}')
    finally:
        win.close()
        pump(0.1)
    log.append(f'worker stopped: {not session._thread.is_alive()}')
    ok = ok and not session._thread.is_alive()
    print('\n'.join(log))
    print(
        'GUI smoke',
        'PASSED' if ok else 'FAILED',
        f'in {time.monotonic() - t_start:.1f} s',
    )
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
