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

"""Shared fixtures. Run as documented in the README (GUI tests on Xvfb)."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

XVFB_HINT = (
    'run GUI tests on a virtual display: env -u WAYLAND_DISPLAY '
    'QT_QPA_PLATFORM=xcb xvfb-run -a -s "-screen 0 1920x1080x24" '
    'python -m pytest ...'
)


@pytest.fixture(scope='session')
def qapp():
    """The QApplication on Xvfb (xcb). Skips without a suitable display so
    that a plain test run never opens windows on a live desktop."""
    if (
        not os.environ.get('DISPLAY')
        or os.environ.get('WAYLAND_DISPLAY')
        or os.environ.get('QT_QPA_PLATFORM') != 'xcb'
    ):
        pytest.skip(XVFB_HINT)
    pytest.importorskip('PySide6')
    pytest.importorskip('pyqtgraph')
    from belief_playground.gui import ensure_app

    app = ensure_app()
    if app.platformName() != 'xcb':
        pytest.fail(
            f'Qt platform is {app.platformName()!r}, expected xcb; {XVFB_HINT}'
        )
    return app


def process_events(app, seconds: float = 0.05) -> None:
    deadline = time.monotonic() + seconds
    while True:
        app.processEvents()
        if time.monotonic() >= deadline:
            return
        time.sleep(0.005)


SIM_ROOT = Path(__file__).resolve().parents[4] / 'scalar_field_sim'
SIM_SCENARIOS = (
    Path(os.environ.get('BELIEF_PLAYGROUND_SIM_ROOT', SIM_ROOT))
    / 'scenarios'
    / 'tests'
    / 'fields'
)

# Rectangular, negative/nonzero origin, non-square cells; own unit; clipping.
OFFSET_SCENARIO = """
[general]
name = "offset_rect"

[units]
field_value = "mV"

[bounds]
x_min = -3.0
x_max = 1.0
y_min = 2.0
y_max = 3.5

[objects.sources]
source_positions = [[-2.0, 3.0]]

[simulation.source_defaults]
amplitude = 5.0
sigma_x = 0.8
sigma_y = 0.5

[simulation]
background_level = 0.5
clip_min = 0.0
clip_max = 3.0

[objects.walls]
wall_vertices = [[0.0, 2.2, 0.0, 3.3]]
"""


@pytest.fixture
def offset_scenario_path(tmp_path) -> Path:
    path = tmp_path / 'offset_rect.toml'
    path.write_text(OFFSET_SCENARIO)
    return path


@pytest.fixture
def single_wall_path() -> Path:
    path = SIM_SCENARIOS / '01_single_wall.toml'
    if not path.is_file():
        pytest.skip(f'scenario not found: {path}')
    return path


class FakeBackend:
    """Records every call; optional delay, readiness threshold and failure."""

    def __init__(
        self, name='fake', ready_after=0, delay=0.0, fail_value=None, unit='mV'
    ):
        self.name = name
        self.label = name.upper()
        self.unit = unit
        self.ready_after = ready_after
        self.delay = delay
        self.fail_value = fail_value
        self.state = []  # current belief content
        self.calls = []  # ('reset',) / ('add', x, y, v) in call order
        self.thread_ids = set()
        self.lock = threading.Lock()

    def reset(self):
        self.thread_ids.add(threading.get_ident())
        self.calls.append(('reset',))
        self.state = []

    def add_measurement(self, x, y, value):
        self.thread_ids.add(threading.get_ident())
        if self.delay:
            time.sleep(self.delay)
        if self.fail_value is not None and value == self.fail_value:
            raise ValueError('rejected by fake backend')
        self.calls.append(('add', x, y, value))
        self.state.append((x, y, value))

    def is_ready(self):
        return len(self.state) >= self.ready_after

    def not_ready_reason(self):
        return 'waiting for fitted model'

    def num_assimilated(self):
        return len(self.state)

    def query(self, xy):
        xy = np.asarray(xy, dtype=float)
        f = xy[:, 0] + 10.0 * xy[:, 1]  # orientation-revealing pattern
        return f + len(self.state), f**2

    def effective_config(self):
        return {
            'backend': self.name,
            'effective': {'ready_after': self.ready_after},
        }
