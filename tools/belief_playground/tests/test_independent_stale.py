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

"""Independent stale-result tests.

A deliberately slow, gated mock backend lets the test hold the worker
*inside* add_measurement() or query() of the old session, then reset/undo,
add a new observation, and release the gate. Results computed for the old
session must never become visible, and the final state must be exactly the
new session. The mock's mean encodes the sum of assimilated values, so a
result from the wrong log is detectable.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import Session

GATE_TIMEOUT = 10.0


class Gate:
    """Armed once: the next call blocks until released."""

    def __init__(self):
        self.armed = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()

    def pass_through(self):
        if self.armed.is_set():
            self.armed.clear()
            self.entered.set()
            assert self.release.wait(GATE_TIMEOUT), 'gate never released'


class GatedBackend:
    """Blocks in add_measurement/query/reset while the matching gate is armed.
    Mean = sum of assimilated values, variance = their count."""

    label = 'GATED'

    def __init__(self, name='gated'):
        self.name = name
        self.state = []
        self.add_gate, self.query_gate, self.reset_gate = Gate(), Gate(), Gate()

    def release_all(self):
        for g in (self.add_gate, self.query_gate, self.reset_gate):
            g.armed.clear()
            g.release.set()

    def reset(self):
        self.reset_gate.pass_through()
        self.state = []

    def add_measurement(self, x, y, value):
        self.add_gate.pass_through()
        self.state.append((x, y, value))

    def is_ready(self):
        return True

    def not_ready_reason(self):
        return ''

    def num_assimilated(self):
        return len(self.state)

    def query(self, xy):
        snapshot = list(self.state)
        self.query_gate.pass_through()
        n = len(np.asarray(xy))
        total = sum(v for *_, v in snapshot)
        return np.full(n, total), np.full(n, float(len(snapshot)))

    def effective_config(self):
        return {'backend': self.name, 'effective': {}}


def _hold_new_rebuild_and_check(s, b, old_gate, new_gen):
    """Release the old in-flight call, then hold the worker at the reset()
    of the new generation's rebuild: at that point the old call has fully
    returned (and published whatever it publishes), and the new session has
    not assimilated anything yet. Nothing from the old call may be visible."""
    b.reset_gate.armed.set()
    old_gate.release.set()
    assert b.reset_gate.entered.wait(GATE_TIMEOUT)
    _, status, _, gen = s.poll()
    st = status['g']
    assert gen == new_gen and st.generation == new_gen
    assert st.mean is None and st.var is None and st.result_count is None, st
    assert st.n_assimilated == 0, st
    assert st.update_ms is None and st.query_ms is None, st
    b.reset_gate.release.set()


class Watcher:
    """Polls the session from another thread and records every result."""

    def __init__(self, session):
        self.session = session
        self.seen = []  # (session generation, status generation, mean[0], count)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(5)

    def _run(self):
        while not self._stop.is_set():
            _, status, _, gen = self.session.poll()
            for st in status.values():
                if st.mean is not None:
                    self.seen.append(
                        (gen, st.generation, float(st.mean[0]), st.result_count)
                    )
            time.sleep(0.001)


@pytest.fixture
def gated_session(offset_scenario_path):
    sc = Scenario(offset_scenario_path)
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, 0.5)
    backend = GatedBackend()
    s = Session(sc, {'g': backend}, grid)
    assert s.wait_idle(10)
    yield s, backend
    backend.release_all()
    s.close()


def _final(s):
    assert s.wait_idle(20)
    _, status, pending, gen = s.poll()
    return status['g'], pending, gen


@pytest.mark.parametrize('where', ['add', 'query'])
def test_reset_during_slow_update_discards_old_results(gated_session, where):
    s, b = gated_session
    s.add_manual(0.5, 3.0, 1.0)  # old session: #1 = 1.0
    assert s.wait_idle(10)
    gate = b.add_gate if where == 'add' else b.query_gate
    gate.armed.set()
    s.add_manual(-1.0, 3.0, 10.0)  # old session: #2 = 10.0 -> worker blocks
    assert gate.entered.wait(GATE_TIMEOUT)
    old_gen = s.generation

    with Watcher(s) as w:
        s.reset()
        new = s.add_manual(-2.0, 2.5, 100.0)  # new session: only this one
        new_gen = s.generation
        assert new_gen > old_gen
        _hold_new_rebuild_and_check(s, b, gate, new_gen)
        st, pending, gen = _final(s)
        time.sleep(0.05)  # let the watcher see the final state

    assert pending == 0 and gen == new_gen
    assert b.state == [(new.x, new.y, 100.0)]
    assert s.observations == (new,)
    assert st.generation == new_gen and st.error is None
    assert st.n_assimilated == 1 and st.result_count == 1
    np.testing.assert_array_equal(st.mean, 100.0)  # only the new value
    np.testing.assert_array_equal(st.var, 1.0)

    # Nothing from the old session ever became visible after the reset:
    # every result shown for the new generation is computed from [100.0].
    after = [r for r in w.seen if r[0] >= new_gen]
    assert after, 'watcher saw no result for the new session'
    assert all(r[1] == new_gen for r in after)
    assert {(r[2], r[3]) for r in after} == {(100.0, 1)}


def test_undo_during_slow_update_replays_only_kept_values(gated_session):
    s, b = gated_session
    keep = [s.add_manual(0.5, 3.0, 1.0), s.add_manual(-0.5, 2.5, 2.0)]
    assert s.wait_idle(10)
    b.add_gate.armed.set()
    s.add_manual(-1.0, 3.0, 10.0)  # #3, blocks
    assert b.add_gate.entered.wait(GATE_TIMEOUT)
    with Watcher(s) as w:
        removed = s.undo()
        new_gen = s.generation
        _hold_new_rebuild_and_check(s, b, b.add_gate, new_gen)
        st, _, _ = _final(s)
        time.sleep(0.05)
    assert removed.value == 10.0
    assert s.observations == tuple(keep)
    assert b.state == [(o.x, o.y, o.value) for o in keep]
    assert st.n_assimilated == 2 and st.result_count == 2
    np.testing.assert_array_equal(st.mean, 3.0)
    after = [r for r in w.seen if r[0] >= new_gen]
    assert after and {(r[2], r[3]) for r in after} == {(3.0, 2)}


def test_reset_during_slow_replay_of_loaded_log(gated_session, tmp_path):
    s, b = gated_session
    obs = [
        s.add_manual(x, 3.0, v)
        for x, v in ((0.5, 1.0), (-0.5, 2.0), (-1.5, 4.0))
    ]
    assert s.wait_idle(10)
    csv_path, _ = s.save(tmp_path / 's.csv')
    s.reset()
    assert s.wait_idle(10)
    b.add_gate.armed.set()
    s.load(csv_path)  # replay blocks in its first add_measurement
    assert b.add_gate.entered.wait(GATE_TIMEOUT)
    s.reset()
    new = s.add_manual(0.0, 2.2, 7.0)
    _hold_new_rebuild_and_check(s, b, b.add_gate, s.generation)
    st, _, _ = _final(s)
    assert b.state == [(new.x, new.y, 7.0)]
    assert st.n_assimilated == 1 and st.result_count == 1
    np.testing.assert_array_equal(st.mean, 7.0)
    assert len(obs) == 3  # the loaded log is gone, not merged
    assert s.observations == (new,)
