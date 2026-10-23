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

"""Controller tests with fake backends (no estimators)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import (
    Session,
    read_observations_csv,
    rmse_on_common_grid,
)
from conftest import FakeBackend


def _session(path, backends, noise_std=0.0, seed=0, step=0.25):
    sc = Scenario(path)
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, step)
    return Session(sc, backends, grid, noise_std=noise_std, seed=seed)


@pytest.fixture
def two_fakes():
    return {'a': FakeBackend('a'), 'b': FakeBackend('b')}


def test_one_click_one_observation_same_noisy_value_everywhere(
    offset_scenario_path, two_fakes
):
    s = _session(offset_scenario_path, two_fakes, noise_std=0.2, seed=7)
    try:
        obs = s.add_sample(-1.0, 2.5)
        assert s.wait_idle(10)
        assert len(s.observations) == 1
        expected_latent = s.scenario.truth(np.array([[-1.0, 2.5]]))[0]
        noise = np.random.default_rng(7).normal(0.0, 0.2)  # drawn once
        assert obs.latent == pytest.approx(expected_latent)
        assert obs.noise == pytest.approx(noise)
        assert obs.value == pytest.approx(expected_latent + noise)
        for fake in two_fakes.values():
            assert fake.state == [(obs.x, obs.y, obs.value)]
            assert fake.calls.count(('add', obs.x, obs.y, obs.value)) == 1
        # Backends are only touched by the worker thread.
        assert len(two_fakes['a'].thread_ids | two_fakes['b'].thread_ids) == 1
        _, status, pending, _ = s.poll()
        assert pending == 0
        assert {n: st.n_assimilated for n, st in status.items()} == {
            'a': 1,
            'b': 1,
        }
        assert all(st.result_count == 1 for st in status.values())
    finally:
        s.close()


def test_clipping_only_for_samples_and_manual_values_unchanged(
    offset_scenario_path, two_fakes
):
    s = _session(offset_scenario_path, two_fakes)
    try:
        hot = s.add_sample(-2.0, 3.0)  # latent ~5.5 mV, clip_max 3.0
        assert hot.latent > 3.0 and hot.value == 3.0 and hot.clipped
        manual = s.add_manual(-2.0, 3.0, 10.0)
        assert manual.value == 10.0 and not manual.clipped
        assert manual.latent is None and manual.noise == 0.0
        neg = s.add_manual(0.5, 2.5, -4.0)
        assert neg.value == -4.0
        s.wait_idle(10)
        assert [v for *_, v in two_fakes['a'].state] == [3.0, 10.0, -4.0]
    finally:
        s.close()


def test_invalid_positions_and_values_are_not_accepted(
    offset_scenario_path, two_fakes
):
    s = _session(offset_scenario_path, two_fakes)
    try:
        for x, y in [(1.5, 2.5), (-1.0, 1.9), (float('nan'), 2.5)]:
            with pytest.raises(ValueError):
                s.add_sample(x, y)
        with pytest.raises(ValueError):
            s.add_manual(0.0, 2.5, float('inf'))
        s.wait_idle(10)
        assert s.observations == ()
        assert two_fakes['a'].state == []
    finally:
        s.close()


def test_undo_and_reset_replay_stored_values_in_order(
    offset_scenario_path, two_fakes
):
    s = _session(offset_scenario_path, two_fakes, noise_std=0.5, seed=1)
    try:
        obs = [s.add_sample(x, 2.5) for x in (-2.5, -1.5, -0.5, 0.5)]
        s.wait_idle(10)
        rng_state = s._rng.bit_generator.state
        removed = s.undo()
        s.wait_idle(10)
        assert removed == obs[-1]
        assert s.observations == tuple(obs[:3])
        # Replay uses stored values and does not draw noise.
        assert s._rng.bit_generator.state == rng_state
        for fake in two_fakes.values():
            assert fake.state == [(o.x, o.y, o.value) for o in obs[:3]]
            last_reset = max(
                i for i, c in enumerate(fake.calls) if c == ('reset',)
            )
            assert fake.calls[last_reset + 1 :] == [
                ('add', o.x, o.y, o.value) for o in obs[:3]
            ]

        s.reset()
        s.wait_idle(10)
        assert s.observations == ()
        assert all(f.state == [] for f in two_fakes.values())
        # Reset re-seeds: the same clicks reproduce the same noisy values.
        again = [s.add_sample(x, 2.5) for x in (-2.5, -1.5)]
        assert [o.value for o in again] == [o.value for o in obs[:2]]
        assert (
            s.undo() is not None and s.undo() is not None and s.undo() is None
        )
    finally:
        s.close()


def test_stale_results_never_overwrite_a_new_generation(offset_scenario_path):
    slow = FakeBackend('slow', delay=0.2)
    s = _session(offset_scenario_path, {'slow': slow})
    try:
        s.wait_idle(10)
        for x in (-2.0, -1.0, 0.0):
            s.add_sample(x, 2.5)
        s.reset()  # while the worker is still busy with the old generation
        s.add_manual(0.5, 3.0, 1.25)
        assert s.wait_idle(20)
        _, status, _, gen = s.poll()
        st = status['slow']
        assert st.generation == gen
        assert st.n_assimilated == 1 and st.result_count == 1
        assert slow.state == [(0.5, 3.0, 1.25)]
        # The result belongs to the new state (fake mean = pattern + count).
        f = s.grid.xy[:, 0] + 10.0 * s.grid.xy[:, 1]
        np.testing.assert_allclose(st.mean, f + 1)
    finally:
        s.close()


def test_no_accepted_observation_is_dropped_under_rapid_clicks(
    offset_scenario_path,
):
    slow = FakeBackend('slow', delay=0.01)
    fast = FakeBackend('fast')
    s = _session(offset_scenario_path, {'slow': slow, 'fast': fast})
    try:
        xs = np.linspace(-2.9, 0.9, 25)
        obs = [s.add_sample(x, 3.0) for x in xs]
        assert s.wait_idle(30)
        for fake in (slow, fast):
            assert fake.state == [(o.x, o.y, o.value) for o in obs]
        _, status, _, _ = s.poll()
        assert all(st.result_count == 25 for st in status.values())
    finally:
        s.close()


def test_readiness_and_backend_errors_are_reported(offset_scenario_path):
    lazy = FakeBackend('lazy', ready_after=2)
    picky = FakeBackend('picky', fail_value=2.0)
    s = _session(offset_scenario_path, {'lazy': lazy, 'picky': picky})
    try:
        s.wait_idle(10)
        _, st, _, _ = s.poll()
        assert st['lazy'].state == 'waiting'
        assert st['lazy'].note == 'waiting for fitted model'
        assert st['lazy'].mean is None
        assert st['picky'].state == 'ready' and st['picky'].mean is not None

        s.add_manual(0.0, 3.0, 1.0)
        s.add_manual(0.0, 3.0, 2.0)  # picky rejects this one
        s.add_manual(0.0, 3.0, 3.0)
        s.wait_idle(10)
        _, st, _, _ = s.poll()
        assert st['lazy'].state == 'ready' and st['lazy'].n_assimilated == 3
        assert st['picky'].state == 'error'
        assert st['picky'].failed_at == 2 and st['picky'].n_assimilated == 1
        assert 'observation #2' in st['picky'].error
        # Maps stay at the last queried good state (label shows its count).
        assert st['picky'].result_count <= st['picky'].n_assimilated
        assert picky.state == [(0.0, 3.0, 1.0)]  # blocked after the failure

        s.undo()  # drop #3: rebuild still fails at #2
        s.undo()  # drop #2: rebuild succeeds
        s.wait_idle(10)
        _, st, _, _ = s.poll()
        assert st['picky'].error is None and st['picky'].n_assimilated == 1
        assert st['lazy'].state == 'waiting'  # back below its threshold
    finally:
        s.close()


def test_save_load_roundtrip_is_exact(
    offset_scenario_path, two_fakes, tmp_path
):
    s = _session(offset_scenario_path, two_fakes, noise_std=0.3, seed=5)
    try:
        obs = [
            s.add_sample(-2.0, 3.0),
            s.add_sample(-0.7, 2.1),
            s.add_manual(0.9, 3.4, -1.0 / 3.0),
        ]
        s.wait_idle(10)
        csv_path, json_path = s.save(tmp_path / 'run' / 's.csv')
        meta = json.loads(json_path.read_text())
        assert meta['num_observations'] == 3
        assert meta['scenario']['content_sha256'] == s.scenario.content_hash()
        assert 'offset_rect' in meta['scenario']['file_text']
        assert meta['scenario']['walls'] == [[0.0, 2.2, 0.0, 3.3]]
        assert (
            meta['units']['value'] == 'mV'
            and meta['units']['variance'] == 'mV^2'
        )
        assert (
            meta['noise']['seed_in_use'] == 5
            and meta['noise']['pending_seed'] == 5
        )
        assert meta['noise']['draws_from_current_generator'] == 2
        assert (
            meta['display_grid']['nx'] == 16 and meta['display_grid']['ny'] == 6
        )
        assert set(meta['backends']) == {'a', 'b'}
        assert (
            'head' in meta['code_versions']['scalar_field_sim']
            or 'error' in meta['code_versions']['scalar_field_sim']
        )
        assert read_observations_csv(csv_path) == obs

        s.reset()
        s.wait_idle(10)
        warnings = s.load(csv_path)
        s.wait_idle(10)
        assert warnings == []
        assert s.observations == tuple(obs)
        for fake in two_fakes.values():
            assert fake.state == [(o.x, o.y, o.value) for o in obs]
    finally:
        s.close()


def test_load_warns_when_scenario_differs(
    offset_scenario_path, single_wall_path, two_fakes, tmp_path
):
    s = _session(offset_scenario_path, two_fakes)
    s.add_manual(0.5, 2.5, 1.0)
    s.wait_idle(10)
    csv_path, _ = s.save(tmp_path / 's.csv')
    s.close()

    other = _session(single_wall_path, {'a': FakeBackend('a')})
    try:
        warnings = other.load(csv_path)
        other.wait_idle(10)
        assert any('scenario content differs' in w for w in warnings)
        assert len(other.observations) == 1
    finally:
        other.close()


def test_rmse_uses_common_valid_cells():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    a = np.array([0.0, 1.0, 2.0, 5.0])
    b = np.array([np.nan, 2.0, 2.0, 3.0])
    rmse, n = rmse_on_common_grid(truth, {'a': a, 'b': b, 'c': None})
    assert n == 3 and set(rmse) == {'a', 'b'}
    assert rmse['a'] == pytest.approx(np.sqrt(4.0 / 3.0))
    assert rmse['b'] == pytest.approx(np.sqrt(1.0 / 3.0))


def test_seed_in_use_vs_pending_seed(offset_scenario_path, two_fakes, tmp_path):
    s = _session(offset_scenario_path, two_fakes, noise_std=0.3, seed=5)
    try:
        first = s.add_sample(-1.0, 2.5)
        s.set_seed(9)  # pending only: the current generator keeps seed 5
        assert (s.seed_in_use, s.pending_seed) == (5, 9)
        second = s.add_sample(-1.0, 2.5)
        rng = np.random.default_rng(5)
        assert [first.noise, second.noise] == pytest.approx(
            [rng.normal(0, 0.3), rng.normal(0, 0.3)]
        )
        _, json_path = s.save(tmp_path / 's.csv')
        noise_meta = json.loads(json_path.read_text())['noise']
        assert (
            noise_meta['seed_in_use'] == 5 and noise_meta['pending_seed'] == 9
        )
        s.reset()
        assert s.seed_in_use == 9
        third = s.add_sample(-1.0, 2.5)
        assert third.noise == pytest.approx(
            np.random.default_rng(9).normal(0, 0.3)
        )
    finally:
        s.close()


def test_scenario_content_is_captured_at_load(offset_scenario_path):
    sc = Scenario(offset_scenario_path)
    digest, text = sc.content_hash(), sc.record()['file_text']
    offset_scenario_path.write_text(text.replace('offset_rect', 'changed'))
    assert sc.content_hash() == digest
    assert sc.record()['file_text'] == text
    assert Scenario(offset_scenario_path).content_hash() != digest


class QueryFailsAfterOne(FakeBackend):
    """Query works with 0 observations, raises once data are present."""

    def query(self, xy):
        if self.state:
            raise RuntimeError('query exploded')
        return super().query(xy)


def test_timing_fields_and_failed_query_clears_query_time(offset_scenario_path):
    s = _session(offset_scenario_path, {'q': QueryFailsAfterOne('q')})
    try:
        assert s.wait_idle(10)
        _, st, _, _ = s.poll()
        q = st['q']
        assert (
            q.update_replay is True and q.update_count == 0
        )  # initial rebuild
        assert q.query_ms is not None and q.query_cells == len(s.grid.xy)
        s.add_manual(-1.0, 3.0, 1.0)
        assert s.wait_idle(10)
        _, st, _, _ = s.poll()
        q = st['q']
        assert (
            q.update_replay is False
            and q.update_count == 1
            and q.update_ms is not None
        )
        assert 'query exploded' in q.query_error
        assert (
            q.query_ms is None and q.query_cells is None
        )  # no stale query time
    finally:
        s.close()
