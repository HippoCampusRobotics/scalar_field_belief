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

"""Offline correctness tests for `target_transform_mode` ('standardize' /
'fixed') of the scalar-field belief core (ROS-independent).

Run from the package root (no pytest needed)::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python -m unittest discover -s test -v

Numerical notes used for the tolerances below
---------------------------------------------
* All data tensors are float64 (`BeliefConfig.dtype`). The GPyTorch
  hyperparameters, however, are created in torch's default dtype (float32)
  by the production code (unchanged from the original code), so every
  hyperparameter value set from a Python float is quantized to float32 in
  raw (softplus) space: relative error up to ~1e-7. Tests that compare a GP
  against configured values therefore use ~5e-7 relative tolerance; tests
  that read the *actually stored* hyperparameters are compared at ~1e-9.
* The independent NumPy oracle (`gp_oracle.py`) solves the same well
  conditioned system (noise >= ~5e-3 vs outputscale ~1, N <= 20; condition
  number <~ 1e4), so float64 agreement is ~1e-12 relative to the prior
  variance; we allow 1e-9 relative.
* The physical mean is `offset + scale * m_t`. Its absolute round-off is
  dominated by `eps * |offset|` (e.g. ~1e-15 for offset 5), so mean
  tolerances are `1e-9 * scale + 64 * eps * max|value|`.
* `query()`/`query_with_covariance()` use `gpytorch.settings.fast_pred_var`
  (LOVE). For N far below `max_cholesky_size` (800) the cache is built from
  an exact Cholesky factorization, so it agrees with exact prediction to
  float64 round-off; compared at the same 1e-9 relative tolerances.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
# The package under test is the working tree next to this directory, so an
# installed (colcon overlay) copy cannot shadow it. SCALAR_FIELD_BELIEF_SRC
# may point at another source root (used for mutation checks).
PKG_ROOT = Path(os.environ.get('SCALAR_FIELD_BELIEF_SRC', TEST_DIR.parent))
for _p in (str(PKG_ROOT), str(TEST_DIR)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import gpytorch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import belief_scenarios as bs  # noqa: E402
import gp_oracle as go  # noqa: E402
from scalar_field_belief.belief import ScalarFieldBelief  # noqa: E402
from scalar_field_belief.config import (  # noqa: E402
    TARGET_TRANSFORM_MODES,
    BeliefConfig,
)
from scalar_field_belief.models import (  # noqa: E402
    ExactFieldGP,
    build_covar_module,
)
from scalar_field_belief.transforms import TargetStandardizer  # noqa: E402

EPS = np.finfo(float).eps

# Strictly positive fixed transform with a large offset relative to the
# scale, so a missing offset or a scale/scale**2 mix-up is obvious.
FIX_OFFSET = 5.0
FIX_SCALE = 0.003


def make_config(**overrides) -> BeliefConfig:
    kw = dict(bs.BASE_CONFIG)
    kw.update(overrides)
    return BeliefConfig(**kw)


def fixed_kwargs(offset=FIX_OFFSET, scale=FIX_SCALE) -> dict:
    return dict(
        target_transform_mode='fixed', target_offset=offset, target_scale=scale
    )


def standardize_kwargs() -> dict:
    return {'target_transform_mode': 'standardize'}


def data(seed: int, n: int, shift: float = 0.0):
    xy, v = bs.make_points(seed, n)
    return xy, v + shift


def mean_atol(scale: float, values) -> float:
    return 1e-9 * scale + 64 * EPS * float(np.max(np.abs(values)))


def effective_transform(
    belief: ScalarFieldBelief, values
) -> tuple[float, float]:
    """Expected (offset, scale) per the contract, computed independently."""
    cfg = belief.config
    if cfg.target_transform_mode == 'fixed':
        return float(cfg.target_offset), float(cfg.target_scale)
    return go.empirical_standardizer(values)


class AssertMixin:
    def assert_mean_close(self, actual, expected, scale, values, msg=''):
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=mean_atol(scale, values), err_msg=msg
        )

    def assert_cov_close(self, actual, expected, prior_var, msg='', rel=1e-9):
        np.testing.assert_allclose(
            actual, expected, rtol=rel, atol=rel * prior_var, err_msg=msg
        )


# ---------------------------------------------------------------------------
# Configuration / backward compatibility
# ---------------------------------------------------------------------------
class TestConfigBackwardCompatibility(unittest.TestCase):
    def test_package_under_test_location(self):
        import scalar_field_belief.belief as mod

        self.assertTrue(
            Path(mod.__file__).resolve().is_relative_to(PKG_ROOT.resolve()),
            mod.__file__,
        )

    def test_defaults_without_target_kwargs(self):
        cfg = make_config()
        self.assertEqual(cfg.target_transform_mode, 'fixed')
        self.assertEqual(cfg.target_offset, 0.0)
        self.assertEqual(cfg.target_scale, 0.01)
        cfg.validate()
        b = ScalarFieldBelief(cfg)
        expected = {'mode': 'fixed', 'offset': 0.0, 'scale': 0.01}
        self.assertEqual(b.target_transform_info(), expected)
        b.add_measurement(0.3, 1.1, 0.0021)
        self.assertTrue(b.has_model())
        self.assertEqual(b.target_transform_info(), expected)

    def test_positional_construction_in_original_field_order(self):
        # Original (HEAD) positional order; target_* must be keyword-only and
        # therefore must not shift positional arguments.
        order = [
            'frame_id', 'x_min', 'x_max', 'y_min', 'y_max', 'kernel_type',
            'training_iter', 'learning_rate', 'init_lengthscale_x',
            'init_lengthscale_y', 'init_outputscale', 'init_noise',
            'refit_policy', 'refit_every_k', 'optimize_hyperparameters',
            'publish_visualization', 'visualization_grid_step',
            'visualization_z_mode', 'visualization_height_scale',
        ]  # fmt: skip
        args = [bs.BASE_CONFIG[k] for k in order]
        cfg = BeliefConfig(*args, 'cpu', torch.float64)
        self.assertEqual(cfg, make_config())
        with self.assertRaises(TypeError):
            BeliefConfig(*args, 'cpu', torch.float64, 'fixed')

    def test_modes_constant(self):
        self.assertEqual(TARGET_TRANSFORM_MODES, ('standardize', 'fixed'))

    def test_config_is_frozen_startup_only(self):
        cfg = make_config(**fixed_kwargs())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.target_transform_mode = 'standardize'  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.target_scale = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 1. Transform math and physical units
# ---------------------------------------------------------------------------
class TestTargetStandardizerMath(unittest.TestCase):
    def setUp(self):
        self.mean = torch.tensor(5.0, dtype=torch.float64)
        self.std = torch.tensor(0.003, dtype=torch.float64)
        self.ts = TargetStandardizer(mean=self.mean, std=self.std)

    def test_round_trip(self):
        y = torch.tensor([4.99, 5.0, 5.0123, -1.0, 7.5], dtype=torch.float64)
        y_t = self.ts.transform(y)
        np.testing.assert_allclose(
            y_t.numpy(), (y.numpy() - 5.0) / 0.003, rtol=1e-14
        )
        back = self.ts.inverse_transform_mean(y_t)
        np.testing.assert_allclose(
            back.numpy(), y.numpy(), rtol=0, atol=8e-15 * 7.5
        )

    def test_inverse_variance_and_covariance(self):
        cov_t = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
        expected = cov_t.numpy() * 0.003**2
        np.testing.assert_allclose(
            self.ts.inverse_transform_covar(cov_t).numpy(), expected, rtol=1e-15
        )
        np.testing.assert_allclose(
            self.ts.inverse_transform_variance(torch.diagonal(cov_t)).numpy(),
            np.diag(expected),
            rtol=1e-15,
        )
        m, v = self.ts.inverse_transform_mean_var(
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            torch.tensor([1.0, 4.0], dtype=torch.float64),
        )
        np.testing.assert_allclose(m.numpy(), [5.0, 5.003], rtol=1e-15)
        np.testing.assert_allclose(v.numpy(), [9e-6, 3.6e-5], rtol=1e-14)
        self.assertAlmostEqual(
            float(self.ts.variance_scale()), 9e-6, delta=1e-20
        )

    def test_fit_population_std_and_fallback(self):
        y = torch.tensor([1.0, 2.0, 4.0, 7.0], dtype=torch.float64)
        ts = TargetStandardizer.fit(y)
        self.assertAlmostEqual(float(ts.mean), 3.5, places=14)
        self.assertAlmostEqual(
            float(ts.std), float(np.std([1, 2, 4, 7])), places=14
        )
        const = TargetStandardizer.fit(
            torch.full((5,), 0.0025, dtype=torch.float64)
        )
        self.assertEqual(float(const.std), 1.0)
        self.assertEqual(float(const.mean), 0.0025)
        tiny = TargetStandardizer.fit(
            torch.tensor([0.0, 1e-13], dtype=torch.float64)
        )
        self.assertEqual(float(tiny.std), 1.0)  # 5e-14 < 1e-12
        with self.assertRaises(ValueError):
            TargetStandardizer.fit(torch.zeros(2, 2, dtype=torch.float64))
        with self.assertRaises(TypeError):
            TargetStandardizer.fit(torch.tensor([1, 2, 3]))


class TestPhysicalUnits(AssertMixin, unittest.TestCase):
    """Belief outputs vs an independent closed-form oracle, both modes,
    with a non-trivial physical offset (~5) and scale (~1e-4..3e-3)."""

    def _run(self, mode_kwargs, kernel, opt, k=3, n=16, seed=3):
        cfg = make_config(
            kernel_type=kernel,
            optimize_hyperparameters=opt,
            refit_policy='every_k_measurements',
            refit_every_k=k,
            **mode_kwargs,
        )
        torch.manual_seed(0)
        b = ScalarFieldBelief(cfg)
        xy, v = data(seed, n, shift=FIX_OFFSET)
        grid, cov_pts = bs.query_points()
        checked = 0
        for i in range(n):
            b.add_measurement(*xy[i], v[i])
            if not b.has_model():
                continue
            vals = v[: i + 1]
            off, sc = effective_transform(b, vals)
            info = b.target_transform_info()
            self.assertAlmostEqual(
                info['offset'], off, delta=64 * EPS * abs(off)
            )
            self.assertAlmostEqual(info['scale'], sc, delta=1e-12 * sc)
            hp = go.model_hyper(b)
            om, oc = go.physical_posterior(
                xy[: i + 1], vals, grid, bs.DOMAIN, offset=off, scale=sc, **hp
            )
            prior_var = hp['outputscale'] * sc**2
            m, var = b.query(grid)
            cm, cov = b.query_with_covariance(grid)
            tag = f'{mode_kwargs} {kernel} opt={opt} N={i + 1}'
            self.assert_mean_close(m, om, sc, vals, tag)
            self.assert_mean_close(cm, om, sc, vals, tag)
            self.assert_cov_close(var, np.diag(oc), prior_var, tag)
            self.assert_cov_close(cov, oc, prior_var, tag)
            # query() diagonal consistency with query_with_covariance()
            self.assert_cov_close(var, np.diag(cov), prior_var, tag)
            np.testing.assert_allclose(
                cov, cov.T, rtol=0, atol=1e-12 * prior_var
            )
            # noise / outputscale / lengthscale physical conversion
            noise_t = float(b.likelihood.noise.detach().reshape(-1)[0])
            os_t = float(b.model.covar_module.outputscale.detach())
            # measurement_noise_variance() multiplies the float32 noise
            # tensor by a 0-dim float64 scale**2; torch type promotion keeps
            # the product float32 (unchanged from the original code), so the
            # result carries float32 rounding: allow one float32 ulp
            # (2**-23 ~ 1.2e-7 relative).
            self.assertAlmostEqual(
                b.measurement_noise_variance() / (noise_t * sc**2),
                1.0,
                delta=2.0**-23,
            )
            php = b.fitted_hyperparameters_physical()
            self.assertAlmostEqual(
                php['outputscale'] / (os_t * sc**2), 1.0, delta=1e-12
            )
            self.assertEqual(
                php['noise_variance'], b.measurement_noise_variance()
            )
            ls = hp['lengthscale']
            self.assertAlmostEqual(
                php['lengthscale_x_m'], ls[0] * 3.0, delta=1e-12
            )
            self.assertAlmostEqual(
                php['lengthscale_y_m'], ls[1] * 1.5, delta=1e-12
            )
            if not opt:
                # Configured transformed-unit parameters -> physical units.
                # 5e-7: float32 storage of GPyTorch raw parameters.
                self.assertAlmostEqual(
                    b.measurement_noise_variance() / (cfg.init_noise * sc**2),
                    1.0,
                    delta=5e-7,
                )
                self.assertAlmostEqual(
                    php['outputscale'] / (cfg.init_outputscale * sc**2),
                    1.0,
                    delta=5e-7,
                )
                self.assertEqual(hp['constant'], 0.0)
            checked += 1
        self.assertGreater(checked, n // 2)

    def test_all_modes_kernels_optimisation(self):
        for mode_kwargs in (standardize_kwargs(), fixed_kwargs()):
            for kernel in ('rbf', 'matern32', 'matern52'):
                for opt in (False, True):
                    with self.subTest(mode=mode_kwargs, kernel=kernel, opt=opt):
                        self._run(mode_kwargs, kernel, opt)

    def test_fixed_mode_prior_is_offset_and_scaled_outputscale(self):
        # Far from all data the latent posterior equals the prior:
        # physical mean = offset + scale * c, var = outputscale * scale**2.
        far = np.array([[1e4, 1e4]])
        for mode_kwargs in (standardize_kwargs(), fixed_kwargs()):
            b = ScalarFieldBelief(make_config(refit_every_k=2, **mode_kwargs))
            xy, v = data(5, 12, shift=FIX_OFFSET)
            for i in range(12):
                b.add_measurement(*xy[i], v[i])
                if not b.has_model():
                    continue
                off, sc = effective_transform(b, v[: i + 1])
                m, var = b.query(far)
                self.assertAlmostEqual(float(m[0]), off, delta=mean_atol(sc, v))
                self.assertAlmostEqual(
                    float(var[0]) / (b.config.init_outputscale * sc**2),
                    1.0,
                    delta=5e-7,
                )


# ---------------------------------------------------------------------------
# 2. Mode equivalence with consistently mapped parameters
# ---------------------------------------------------------------------------
class TestModeEquivalence(AssertMixin, unittest.TestCase):
    """standardize vs fixed with offset = empirical mean at final N and
    scale = r * empirical std; init_outputscale and init_noise of the fixed
    run are multiplied by (s_emp / s_fixed)**2 = 1/r**2 so the physical
    prior (mean offset, signal var, noise var) is identical. Optimisation is
    off, so the ConstantMean stays 0 in both (physical prior mean = offset).

    Mapped noise: 0.04 / r**2 with r in {2, 0.37, 1} -> {0.01, 0.292, 0.04},
    always >= 100x the GaussianLikelihood lower bound 1e-4, so the noise
    constraint is not active and the mapping is exact up to float32
    parameter storage (relative ~1e-7). That quantization perturbs the
    posterior by at most ~1e-7 * cond(K)-ish; observed ~1e-8. We assert
    agreement at 2e-6 relative (prior-variance / scale normalised), and
    additionally check each mode against the exact oracle at 1e-9.
    """

    def _pair(self, ratio, policy, k, kernel='rbf', n=15, seed=21):
        xy, v = data(seed, n)
        m_emp, s_emp = go.empirical_standardizer(v)
        s_fix = ratio * s_emp
        base = dict(
            kernel_type=kernel,
            refit_policy=policy,
            refit_every_k=k,
            optimize_hyperparameters=False,
            init_outputscale=1.0,
            init_noise=0.04,
        )
        std_b = ScalarFieldBelief(make_config(**base, **standardize_kwargs()))
        fix_kw = dict(base)
        fix_kw.update(
            init_outputscale=1.0 / ratio**2,
            init_noise=0.04 / ratio**2,
            **fixed_kwargs(offset=m_emp, scale=s_fix),
        )
        self.assertGreaterEqual(fix_kw['init_noise'], 100 * 1e-4)
        fix_b = ScalarFieldBelief(make_config(**fix_kw))
        for i in range(n):
            std_b.add_measurement(*xy[i], v[i])
            fix_b.add_measurement(*xy[i], v[i])
        self.assertTrue(std_b.has_model() and fix_b.has_model())
        return std_b, fix_b, xy, v, m_emp, s_emp

    def test_equivalent_physical_predictions(self):
        grid, cov_pts = bs.query_points()
        q = np.vstack([grid, cov_pts])
        for ratio in (1.0, 2.0, 0.37):
            for policy, k in (
                ('every_measurement', 1),
                ('every_k_measurements', 4),
            ):
                for kernel in ('rbf', 'matern52'):
                    with self.subTest(
                        ratio=ratio, policy=policy, k=k, kernel=kernel
                    ):
                        sb, fb, xy, v, m_emp, s_emp = self._pair(
                            ratio, policy, k, kernel
                        )
                        info = sb.target_transform_info()
                        self.assertAlmostEqual(
                            info['offset'], m_emp, delta=64 * EPS * 0.01
                        )
                        self.assertAlmostEqual(
                            info['scale'], s_emp, delta=1e-12 * s_emp
                        )
                        prior_var = s_emp**2  # init_outputscale 1 in std units
                        sm, sv = sb.query(q)
                        fm, fv = fb.query(q)
                        scm, scov = sb.query_with_covariance(q)
                        fcm, fcov = fb.query_with_covariance(q)
                        tol_m = 2e-6 * s_emp
                        np.testing.assert_allclose(fm, sm, rtol=0, atol=tol_m)
                        np.testing.assert_allclose(fcm, scm, rtol=0, atol=tol_m)
                        np.testing.assert_allclose(
                            fv, sv, rtol=2e-6, atol=2e-6 * prior_var
                        )
                        np.testing.assert_allclose(
                            fcov, scov, rtol=2e-6, atol=2e-6 * prior_var
                        )
                        # Physical hyperparameters agree (float32 storage).
                        sh = sb.fitted_hyperparameters_physical()
                        fh = fb.fitted_hyperparameters_physical()
                        for key in sh:
                            self.assertAlmostEqual(
                                fh[key] / sh[key], 1.0, delta=5e-7, msg=key
                            )
                        # Each mode vs the exact oracle with its own stored
                        # hyperparameters (tight).
                        for b in (sb, fb):
                            off, sc = effective_transform(b, v)
                            hp = go.model_hyper(b)
                            om, oc = go.physical_posterior(
                                xy, v, q, bs.DOMAIN, offset=off, scale=sc, **hp
                            )
                            bm, bc = b.query_with_covariance(q)
                            self.assert_mean_close(bm, om, sc, v)
                            self.assert_cov_close(
                                bc, oc, hp['outputscale'] * sc**2
                            )

    def test_modes_differ_before_final_n(self):
        # Documented modelling difference (not a defect): with the offset
        # matched only at the final N, earlier standardize predictions use a
        # different prior mean/variance than the fixed mode.
        xy, v = data(21, 15)
        m_emp, s_emp = go.empirical_standardizer(v)
        sb = ScalarFieldBelief(
            make_config(
                refit_policy='every_measurement', **standardize_kwargs()
            )
        )
        fb = ScalarFieldBelief(
            make_config(
                refit_policy='every_measurement',
                **fixed_kwargs(offset=m_emp, scale=s_emp),
            )
        )
        for i in range(5):
            sb.add_measurement(*xy[i], v[i])
            fb.add_measurement(*xy[i], v[i])
        far = np.array([[1e4, 1e4]])
        self.assertNotAlmostEqual(
            float(sb.query(far)[0][0]),
            float(fb.query(far)[0][0]),
            delta=1e-3 * s_emp,
        )


# ---------------------------------------------------------------------------
# 3. Streaming conditioning vs fresh conditioning
# ---------------------------------------------------------------------------
class TestStreamingVsFresh(AssertMixin, unittest.TestCase):
    """At every step the streaming belief (conditioning between refits,
    rebuilding at refits) must equal a GP built from scratch on the same
    accumulated data with the streaming model's current hyperparameters.
    """

    def _fresh_exact(self, b, xy, vals, q):
        off, sc = effective_transform(b, vals)
        dt = b.config.dtype
        train_x = torch.as_tensor(go.normalize_inputs(xy, bs.DOMAIN), dtype=dt)
        train_y = torch.as_tensor((np.asarray(vals) - off) / sc, dtype=dt)
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactFieldGP(
            train_x, train_y, lik, build_covar_module(b.config.kernel_type)
        )
        model.load_state_dict(b.model.state_dict(), strict=True)
        model.eval()
        lik.eval()
        qx = torch.as_tensor(go.normalize_inputs(q, bs.DOMAIN), dtype=dt)
        with torch.no_grad():
            post = model(qx)
            mean_t = post.mean.numpy()
            cov_t = post.covariance_matrix.numpy()
        os_t = float(model.covar_module.outputscale.detach())
        # Streaming model really holds the independently transformed data.
        # Absolute round-off of (v - off) / sc is ~eps * max|v| / sc.
        np.testing.assert_allclose(
            b.model.train_targets.detach().numpy(),
            train_y.numpy(),
            rtol=0,
            atol=64 * EPS * float(np.max(np.abs(vals))) / sc,
        )
        np.testing.assert_allclose(
            b.model.train_inputs[0].detach().numpy(),
            train_x.numpy(),
            rtol=0,
            atol=1e-15,
        )
        return off + sc * mean_t, sc**2 * cov_t, sc, os_t * sc**2

    def _stream(self, mode_kwargs, opt, k, kernel, n=14, seed=31):
        cfg = make_config(
            kernel_type=kernel,
            optimize_hyperparameters=opt,
            refit_policy='every_k_measurements',
            refit_every_k=k,
            training_iter=15,
            **mode_kwargs,
        )
        torch.manual_seed(0)
        b = ScalarFieldBelief(cfg)
        xy, v = data(seed, n, shift=1.0)
        grid, cov_pts = bs.query_points()
        q = np.vstack([grid, cov_pts])
        last_hyper = None
        last_transform = None
        n_cond = n_refit = 0
        for i in range(n):
            res = b.add_measurement(*xy[i], v[i])
            if not b.has_model():
                continue
            vals = v[: i + 1]
            hyper = {
                kk: p.detach().clone() for kk, p in b.model.state_dict().items()
            }
            if res.did_refit:
                n_refit += 1
            else:
                n_cond += 1
                # Conditioning must not touch hyperparameters.
                for kk in hyper:
                    torch.testing.assert_close(
                        hyper[kk], last_hyper[kk], rtol=0, atol=0
                    )
                if b.config.target_transform_mode == 'fixed':
                    self.assertEqual(b.target_transform_info(), last_transform)
            last_hyper = hyper
            last_transform = b.target_transform_info()
            fm, fc, sc, pv = self._fresh_exact(b, xy[: i + 1], vals, q)
            m, var = b.query(q)
            cm, cov = b.query_with_covariance(q)
            tag = (
                f'{mode_kwargs} opt={opt} k={k} N={i + 1} refit={res.did_refit}'
            )
            self.assert_mean_close(m, fm, sc, vals, tag)
            self.assert_mean_close(cm, fm, sc, vals, tag)
            self.assert_cov_close(var, np.diag(fc), pv, tag)
            self.assert_cov_close(cov, fc, pv, tag)
        self.assertGreater(n_cond, 0)
        self.assertGreater(n_refit, 1)
        return b

    def test_fresh_exactgp_with_copied_hyperparameters(self):
        for mode_kwargs in (
            standardize_kwargs(),
            fixed_kwargs(offset=1.002, scale=0.0015),
        ):
            for opt in (False, True):
                for kernel in ('rbf', 'matern32'):
                    with self.subTest(mode=mode_kwargs, opt=opt, kernel=kernel):
                        self._stream(mode_kwargs, opt, k=3, kernel=kernel)

    def test_fresh_belief_equivalence_no_optimisation(self):
        # Hyperparameters are constant without optimisation, so a belief that
        # refits on every measurement is a fully independent fresh reference.
        q = np.vstack(bs.query_points())
        for mode_kwargs in (
            standardize_kwargs(),
            fixed_kwargs(offset=1.002, scale=0.0015),
        ):
            with self.subTest(mode=mode_kwargs):
                sb = ScalarFieldBelief(
                    make_config(refit_every_k=4, **mode_kwargs)
                )
                xy, v = data(32, 13, shift=1.0)
                for i in range(13):
                    sb.add_measurement(*xy[i], v[i])
                    if not sb.has_model():
                        continue
                    fb = ScalarFieldBelief(
                        make_config(
                            refit_policy='every_measurement', **mode_kwargs
                        )
                    )
                    for j in range(i + 1):
                        fb.add_measurement(*xy[j], v[j])
                    self.assertEqual(
                        sb.target_transform_info(), fb.target_transform_info()
                    )
                    for a, b_ in zip(sb.query(q), fb.query(q)):
                        np.testing.assert_allclose(a, b_, rtol=1e-12, atol=0)
                    for a, b_ in zip(
                        sb.query_with_covariance(q), fb.query_with_covariance(q)
                    ):
                        np.testing.assert_allclose(
                            a, b_, rtol=1e-12, atol=1e-30
                        )

    def test_fresh_belief_at_refit_points_with_optimisation(self):
        # At a refit the streaming belief rebuilds from scratch on all data,
        # so it must equal a fresh belief that performs exactly one fit on
        # the same N measurements (deterministic Adam, same init).
        q = np.vstack(bs.query_points())
        k = 3
        for mode_kwargs in (
            standardize_kwargs(),
            fixed_kwargs(offset=1.002, scale=0.0015),
        ):
            with self.subTest(mode=mode_kwargs):
                kw = dict(
                    optimize_hyperparameters=True,
                    training_iter=15,
                    kernel_type='matern52',
                    **mode_kwargs,
                )
                torch.manual_seed(0)
                sb = ScalarFieldBelief(make_config(refit_every_k=k, **kw))
                xy, v = data(33, 12, shift=1.0)
                for i in range(12):
                    res = sb.add_measurement(*xy[i], v[i])
                    if not res.did_refit:
                        continue
                    torch.manual_seed(0)
                    fb = ScalarFieldBelief(
                        make_config(refit_every_k=i + 1, **kw)
                    )
                    for j in range(i + 1):
                        fres = fb.add_measurement(*xy[j], v[j])
                    self.assertTrue(fres.did_refit)
                    for a, b_ in zip(
                        sb.query_with_covariance(q), fb.query_with_covariance(q)
                    ):
                        np.testing.assert_allclose(
                            a, b_, rtol=1e-10, atol=1e-30
                        )
                    sh = sb.fitted_hyperparameters_physical()
                    fh = fb.fitted_hyperparameters_physical()
                    for key in sh:
                        self.assertAlmostEqual(
                            sh[key] / fh[key], 1.0, delta=1e-10
                        )


# ---------------------------------------------------------------------------
# 4. Edge cases and validation
# ---------------------------------------------------------------------------
class TestEdgeCases(AssertMixin, unittest.TestCase):
    def test_empty_belief(self):
        for mode_kwargs in (standardize_kwargs(), fixed_kwargs()):
            b = ScalarFieldBelief(make_config(**mode_kwargs))
            self.assertFalse(b.has_model())
            for call in (
                lambda: b.query(np.zeros((1, 2))),
                lambda: b.query_with_covariance(np.zeros((1, 2))),
                b.measurement_noise_variance,
                b.fitted_hyperparameters_physical,
            ):
                with self.assertRaises(RuntimeError):
                    call()
            info = b.target_transform_info()
            if b.config.target_transform_mode == 'fixed':
                self.assertEqual(
                    info,
                    {'mode': 'fixed', 'offset': FIX_OFFSET, 'scale': FIX_SCALE},
                )
            else:
                self.assertEqual(
                    info, {'mode': 'standardize', 'offset': None, 'scale': None}
                )

    def test_single_measurement(self):
        x, y, val = 0.3, 1.1, 5.0021
        for mode_kwargs in (standardize_kwargs(), fixed_kwargs()):
            b = ScalarFieldBelief(make_config(**mode_kwargs))
            res = b.add_measurement(x, y, val)
            self.assertTrue(res.did_refit)
            self.assertEqual(res.num_measurements, 1)
            off, sc = effective_transform(b, [val])
            if b.config.target_transform_mode == 'standardize':
                self.assertEqual((off, sc), (val, 1.0))
            info = b.target_transform_info()
            self.assertEqual((info['offset'], info['scale']), (off, sc))
            q = np.array([[x, y], [1.5, 0.7]])
            hp = go.model_hyper(b)
            om, oc = go.physical_posterior(
                [[x, y]], [val], q, bs.DOMAIN, offset=off, scale=sc, **hp
            )
            m, var = b.query(q)
            self.assert_mean_close(m, om, sc, [val])
            self.assert_cov_close(var, np.diag(oc), hp['outputscale'] * sc**2)
            if b.config.target_transform_mode == 'standardize':
                self.assertAlmostEqual(float(m[0]), val, delta=1e-14)

    def test_constant_targets(self):
        q = np.vstack(bs.query_points())
        xy, _ = data(41, 6)
        vals = np.full(6, 0.0025)
        for mode_kwargs in (
            standardize_kwargs(),
            fixed_kwargs(offset=0.002, scale=0.001),
        ):
            with self.subTest(mode=mode_kwargs):
                b = ScalarFieldBelief(make_config(**mode_kwargs))
                for i in range(6):
                    b.add_measurement(*xy[i], vals[i])
                    info = b.target_transform_info()
                    if b.config.target_transform_mode == 'fixed':
                        self.assertEqual(
                            (info['offset'], info['scale']), (0.002, 0.001)
                        )
                    else:
                        self.assertEqual(
                            (info['offset'], info['scale']), (0.0025, 1.0)
                        )
                off, sc = effective_transform(b, vals)
                m, var = b.query(q)
                cm, cov = b.query_with_covariance(q)
                self.assertTrue(
                    np.all(np.isfinite(m)) and np.all(np.isfinite(cov))
                )
                hp = go.model_hyper(b)
                om, oc = go.physical_posterior(
                    xy, vals, q, bs.DOMAIN, offset=off, scale=sc, **hp
                )
                self.assert_mean_close(m, om, sc, vals)
                self.assert_cov_close(cov, oc, hp['outputscale'] * sc**2)
                self.assert_cov_close(
                    var, np.diag(cov), hp['outputscale'] * sc**2
                )
                if b.config.target_transform_mode == 'standardize':
                    np.testing.assert_allclose(m, 0.0025, rtol=0, atol=1e-15)

    def test_invalid_modes(self):
        for bad in (
            'Fixed',
            'STANDARDIZE',
            'bogus',
            '',
            ' fixed',
            None,
            1,
            0.0,
            True,
            ['fixed'],
        ):
            with self.subTest(mode=bad):
                cfg = make_config(
                    target_transform_mode=bad,
                    target_offset=0.0,
                    target_scale=1.0,
                )
                with self.assertRaises(ValueError):
                    cfg.validate()
                with self.assertRaises(ValueError):
                    ScalarFieldBelief(cfg)

    def test_invalid_fixed_scale_and_offset(self):
        bad_scales = (
            0.0,
            -0.0,
            -1e-3,
            float('nan'),
            float('inf'),
            float('-inf'),
            None,
        )
        for scale in bad_scales:
            with self.subTest(scale=scale):
                cfg = make_config(
                    target_transform_mode='fixed',
                    target_offset=0.0,
                    target_scale=scale,
                )
                with self.assertRaises(ValueError):
                    cfg.validate()
                with self.assertRaises(ValueError):
                    ScalarFieldBelief(cfg)
        for offset in (float('nan'), float('inf'), float('-inf'), None):
            with self.subTest(offset=offset):
                cfg = make_config(
                    target_transform_mode='fixed',
                    target_offset=offset,
                    target_scale=1.0,
                )
                with self.assertRaises(ValueError):
                    ScalarFieldBelief(cfg)
        with self.assertRaises(ValueError):  # both missing
            ScalarFieldBelief(
                make_config(
                    target_transform_mode='fixed',
                    target_offset=None,
                    target_scale=None,
                )
            )
        # Valid edge values.
        ScalarFieldBelief(make_config(**fixed_kwargs(offset=0.0, scale=1e-9)))
        ScalarFieldBelief(make_config(**fixed_kwargs(offset=-3.0, scale=1e6)))

    def test_standardize_ignores_offset_and_scale(self):
        q = np.vstack(bs.query_points())
        xy, v = data(42, 8)
        outs = []
        for extra in (
            {},
            dict(target_offset=float('nan'), target_scale=0.0),
            dict(target_offset=7.0, target_scale=-2.0),
            dict(target_offset=None, target_scale=float('inf')),
        ):
            # Every variant selects standardize; only offset/scale differ.
            extra = {**standardize_kwargs(), **extra}
            b = ScalarFieldBelief(make_config(refit_every_k=3, **extra))
            for i in range(8):
                b.add_measurement(*xy[i], v[i])
            self.assertEqual(b.target_transform_info()['mode'], 'standardize')
            outs.append(
                (
                    b.query(q),
                    b.query_with_covariance(q),
                    b.target_transform_info(),
                )
            )
        for o in outs[1:]:
            self.assertEqual(o[2], outs[0][2])
            for a, b_ in zip(o[0] + o[1], outs[0][0] + outs[0][1]):
                np.testing.assert_array_equal(a, b_)

    def test_non_finite_measurement_rejected(self):
        b = ScalarFieldBelief(make_config(**fixed_kwargs()))
        for args in (
            (float('nan'), 0.0, 1.0),
            (0.0, float('inf'), 1.0),
            (0.0, 0.0, float('nan')),
        ):
            with self.assertRaises(ValueError):
                b.add_measurement(*args)
        self.assertEqual(len(b.y_train), 0)


class TestTargetTransformInfo(unittest.TestCase):
    def test_info_lifecycle_standardize(self):
        b = ScalarFieldBelief(
            make_config(refit_every_k=3, **standardize_kwargs())
        )
        self.assertEqual(
            b.target_transform_info(),
            {'mode': 'standardize', 'offset': None, 'scale': None},
        )
        xy, v = data(51, 7)
        for i in range(7):
            b.add_measurement(*xy[i], v[i])
            info = b.target_transform_info()
            self.assertEqual(info['mode'], 'standardize')
            if i < 2:  # before first fit
                self.assertIsNone(info['offset'])
                self.assertIsNone(info['scale'])
            else:  # re-estimated on refits AND conditioning updates
                m, s = go.empirical_standardizer(v[: i + 1])
                self.assertAlmostEqual(info['offset'], m, delta=1e-17)
                self.assertAlmostEqual(info['scale'], s, delta=1e-12 * s)
                self.assertIsInstance(info['offset'], float)
                self.assertIsInstance(info['scale'], float)
        b.reset()
        self.assertFalse(b.has_model())
        self.assertEqual(
            b.target_transform_info(),
            {'mode': 'standardize', 'offset': None, 'scale': None},
        )

    def test_info_lifecycle_fixed_and_reset(self):
        q = np.vstack(bs.query_points())
        cfg = make_config(
            refit_every_k=3, **fixed_kwargs(offset=0.002, scale=0.001)
        )
        expected = {'mode': 'fixed', 'offset': 0.002, 'scale': 0.001}
        b = ScalarFieldBelief(cfg)
        self.assertEqual(b.target_transform_info(), expected)
        xy, v = data(52, 7)
        for i in range(7):
            b.add_measurement(*xy[i], v[i])
            self.assertEqual(b.target_transform_info(), expected)
        before = b.query_with_covariance(q)
        b.reset()
        self.assertFalse(b.has_model())
        self.assertEqual(b.target_transform_info(), expected)
        with self.assertRaises(RuntimeError):
            b.query(q)
        # After reset, replaying the same data reproduces the same belief and
        # matches a brand-new belief (transform constant across reset).
        fresh = ScalarFieldBelief(cfg)
        for i in range(7):
            b.add_measurement(*xy[i], v[i])
            fresh.add_measurement(*xy[i], v[i])
            self.assertEqual(b.target_transform_info(), expected)
        after = b.query_with_covariance(q)
        for a, c, f in zip(after, before, fresh.query_with_covariance(q)):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(a, f)


# ---------------------------------------------------------------------------
# 5. Default configuration: fixed transform, offset 0.0, scale 0.01
# ---------------------------------------------------------------------------
DEFAULT_OFFSET = 0.0
DEFAULT_SCALE = 0.01


class TestDefaultFixedTransform(AssertMixin, unittest.TestCase):
    """A config without target kwargs uses `y_t = (y - 0.0) / 0.01` for the
    whole lifetime of the belief.

    The data are sensor-like values of ~2e-3 (y_t ~ 0.2) without a shift.
    Their empirical std (~5e-4) differs from 0.01 by more than an order of
    magnitude, so an empirical (standardize) transform anywhere in the
    pipeline would change every compared quantity by far more than the
    tolerances. The expected offset and scale are the documented defaults,
    not values read back from the belief or its config.
    """

    EXPECTED_INFO = {
        'mode': 'fixed',
        'offset': DEFAULT_OFFSET,
        'scale': DEFAULT_SCALE,
    }

    def _check_against_oracle(self, b, xy, vals, q, opt, tag):
        vals = np.asarray(vals)
        self.assertEqual(b.target_transform_info(), self.EXPECTED_INFO, tag)
        # The GP is trained on (y - 0.0) / 0.01, i.e. targets of ~0.2.
        np.testing.assert_allclose(
            b.model.train_targets.detach().numpy(),
            (vals - DEFAULT_OFFSET) / DEFAULT_SCALE,
            rtol=1e-15,
            atol=0,
            err_msg=tag,
        )
        hp = go.model_hyper(b)
        om, oc = go.physical_posterior(
            xy,
            vals,
            q,
            bs.DOMAIN,
            offset=DEFAULT_OFFSET,
            scale=DEFAULT_SCALE,
            **hp,
        )
        prior_var = hp['outputscale'] * DEFAULT_SCALE**2
        m, var = b.query(q)
        cm, cov = b.query_with_covariance(q)
        self.assert_mean_close(m, om, DEFAULT_SCALE, vals, tag)
        self.assert_mean_close(cm, om, DEFAULT_SCALE, vals, tag)
        self.assert_cov_close(var, np.diag(oc), prior_var, tag)
        self.assert_cov_close(cov, oc, prior_var, tag)
        # Float32 product, see TestPhysicalUnits.
        self.assertAlmostEqual(
            b.measurement_noise_variance() / (hp['noise'] * DEFAULT_SCALE**2),
            1.0,
            delta=2.0**-23,
            msg=tag,
        )
        php = b.fitted_hyperparameters_physical()
        self.assertAlmostEqual(
            php['outputscale'] / prior_var, 1.0, delta=1e-12, msg=tag
        )
        if not opt:
            # Shipped values: physical prior std 0.01, noise std 1e-3.
            cfg = b.config
            self.assertAlmostEqual(
                b.measurement_noise_variance()
                / (cfg.init_noise * DEFAULT_SCALE**2),
                1.0,
                delta=5e-7,
                msg=tag,
            )
            # The last query point is far from all data: prior there.
            self.assertAlmostEqual(
                float(var[-1]) / (cfg.init_outputscale * DEFAULT_SCALE**2),
                1.0,
                delta=5e-7,
                msg=tag,
            )
            self.assertAlmostEqual(
                float(m[-1]), DEFAULT_OFFSET, delta=1e-12, msg=tag
            )

    def _run(self, opt):
        k = 3
        cfg = make_config(
            refit_policy='every_k_measurements',
            refit_every_k=k,
            optimize_hyperparameters=opt,
            training_iter=15,
        )
        torch.manual_seed(0)
        b = ScalarFieldBelief(cfg)
        self.assertEqual(b.target_transform_info(), self.EXPECTED_INFO)
        xy, v = bs.make_points(61, 16)
        self.assertTrue(np.all((v > 1e-3) & (v < 4e-3)))
        q = np.vstack([*bs.query_points(), [[1e4, 1e4]]])
        n_before_reset = 10
        n_cond = n_refit = 0
        for i in range(len(v)):
            if i == n_before_reset:
                b.reset()
                self.assertFalse(b.has_model())
                self.assertEqual(b.target_transform_info(), self.EXPECTED_INFO)
            start = 0 if i < n_before_reset else n_before_reset
            res = b.add_measurement(*xy[i], v[i])
            self.assertEqual(res.did_refit, (i - start + 1) % k == 0)
            self.assertEqual(b.target_transform_info(), self.EXPECTED_INFO)
            if not b.has_model():
                continue
            if res.did_refit:
                n_refit += 1
            else:
                n_cond += 1
            tag = f'opt={opt} N={i + 1 - start} after_reset={start > 0}'
            self._check_against_oracle(
                b, xy[start : i + 1], v[start : i + 1], q, opt, tag
            )
        # Refits before and after the reset, conditioning in between.
        self.assertEqual((n_refit, n_cond), (5, 7))

    def test_default_transform_through_refits_conditioning_and_reset(self):
        for opt in (False, True):
            with self.subTest(opt=opt):
                self._run(opt)


if __name__ == '__main__':
    unittest.main()
