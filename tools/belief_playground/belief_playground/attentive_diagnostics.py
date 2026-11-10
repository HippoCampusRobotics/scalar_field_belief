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

"""Headless numerical diagnostic of the Attentive Kernel (AK) GP belief.

    python -m belief_playground.attentive_diagnostics [--out-dir DIR]
        [--gp-attentive-config YAML] [--author-demo PATH]
        [--scenario 01_single_wall]

Kernel checks (K1-K6) use fixed synthetic inputs and an independent NumPy
kernel. --author-demo adds parity with the author's `demo/main.py`: its
sha256 is checked first, then its kernel definitions are extracted with
`ast` and only those definitions are executed. Belief checks (B1-B7) use
synthetic data in a 2 m x 4 m box and dense NumPy Gaussian oracles. S1 runs
the ordinary playground Session with gp, gp_attentive and visgp; its
reconstruction quality is reported, not judged (the only use of scenario
truth).

Every check ends as passed, failed or not run. Exit code 1 if a check
failed, 3 if none failed but some did not run, else 0; 2 is a usage error.
Output (default `./belief_playground_sessions/attentive_diagnostics/<UTC>/`):
report.json, report.md, arrays.npz and the S1 session (CSV + JSON). Qt is
never imported.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import gpytorch
import linear_operator
import numpy as np
import torch

import belief_playground
import scalar_field_belief
from belief_playground import launcher
from belief_playground.backends import make_backends
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import Session, rmse_on_common_grid
from scalar_field_belief import attentive
from scalar_field_belief.attentive import AttentiveGPBelief, AttentiveGPConfig

AUTHOR_SHA256 = (
    '9411046f00d3119dd1bee2c275d2567646dcfe718eb2cfc75c36684314ecd645'
)
AUTHOR_REVISION = '1ccb7255e1a09d215af39da3bb1efe6305cf4dea'
AUTHOR_NAMES = {'softplus', 'inv_softplus', 'constraint', 'unconstraint'}
AUTHOR_NAMES |= {'rbf', 'TwoHiddenLayerTanhNN', 'AK'}
PASSED, FAILED, NOT_RUN = 'passed', 'failed', 'not run'
EPS, DT = float(np.finfo(float).eps), torch.float64
BOX, SIDE = ((0.0, 2.0), (0.0, 4.0)), 4.0  # square normalizer side [m]
FAST = {'initial_training_steps': 20, 'training_steps_per_event': 5}
CADENCE = {'initial_training_steps': 3, 'training_steps_per_event': 2}
CADENCE.update(min_distinct_locations=4, optimize_every=2)
SCHEDULE = ('num_distinct_locations', 'optimization_events')
SCHEDULE += ('optimization_steps', 'phase')
COUNTERS = SCHEDULE + ('num_measurements', 'measurements_since_last_event')
D10 = ((0.3, 0.4), (0.7, 1.1), (0.4, 1.9), (0.8, 2.7), (0.3, 3.5))
D10 += ((1.3, 0.5), (1.7, 1.3), (1.2, 2.2), (1.6, 3.0), (1.4, 3.7))
T14 = [(*p, 0.0) for p in D10[:9]] + [(*D10[0], 2e-4), (*D10[3], -1e-4)]
T14 += [(*D10[9], 0.0), (*D10[5], 1e-4), (1.0, 3.9, 0.0)]
F8 = [(0.3, 0.4, 0), (0.7, 1.1, 0), (0.4, 1.9, 0), (1.3, 0.5, 0)]
F8 += [(1.6, 3.0, 0), (0.7, 1.1, 3e-4), (1.2, 2.2, 0), (0.4, 1.9, -2e-4)]
Q10 = np.array([(0.7, 1.1), (0.7, 1.1), (0.5, 1.5), (1.0, 2.0), (1.05, 3.65)])
Q10 = np.vstack([Q10, [(1.9, 3.9), (0, 0), (2, 4), (1.5, 1.0), (0.2, 3.0)]])
S1_POS = [
    (x, y if k % 2 == 0 else 4.0 - y)
    for k, x in enumerate((0.25, 0.75, 1.25, 1.75))
    for y in (0.5, 1.5, 2.5, 3.5)
] + [(0.25, 0.5)]
S1_STD_POINTS = np.array([(1.0, 3.65), (1.0, 0.35), (1.3, 2.0), (0.7, 2.0)])
FORBIDDEN = (
    'rclpy rclcpp ament_index_python rosidl_runtime_py PySide6 pyqtgraph '
    'matplotlib belief_playground scalar_field_sim scalar_field_belief.visgp '
    'scalar_field_belief_gmrf_py scalar_field_belief.wall_features '
    'scalar_field_belief.geometry'
).split()


@dataclass
class Check:
    name: str
    status: str = PASSED
    reason: str = ''
    details: dict = field(default_factory=dict)

    def expect(self, label: str, ok, value=None) -> None:
        self.details[label] = {'ok': bool(ok)}
        if value is not None:
            self.details[label]['value'] = value
        if not ok:
            self.status = FAILED


def run_check(fn, *args, status=PASSED, reason='') -> Check:
    """`fn(check, *args)` unless `status` says it cannot run; exceptions
    become a failed check. The title is `fn`'s docstring."""
    c = Check(fn.__doc__.strip().rstrip('.'), status, reason)
    if status == PASSED:
        print(f'[attentive_diagnostics] {c.name}', file=sys.stderr)
        try:
            fn(c, *args)
        except Exception as exc:  # noqa: BLE001 - report, run the others
            c.status, c.reason = FAILED, f'{type(exc).__name__}: {exc}'
    return c


# ------------------------------------------------------------- helpers
def f_syn(x, y):
    return float(0.008 + 0.004 * np.sin(1.3 * x + 0.7 * y))


def _data(rows) -> list:
    return [(x, y, f_syn(x, y) + d) for x, y, d in rows]


def _belief(**kw) -> AttentiveGPBelief:
    return AttentiveGPBelief(AttentiveGPConfig(**kw), *BOX)


def _feed(b, data) -> None:
    for x, y, v in data:
        b.add_measurement(x, y, v)


def _raises(fn, *args) -> str:
    """Name of the exception `fn(*args)` raises, else 'nothing'."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            fn(*args)
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__
    return 'nothing'


def _val(t) -> float:
    return float(t.detach())


def _rel(a, b, scale=1.0) -> float:
    diff = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return float(np.max(diff)) / scale


def _same(a, b) -> bool:
    """Bitwise equality of nested tensors/arrays/containers."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype:
            return False
        return a.shape == b.shape and bool(torch.equal(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(map(_same, a, b))
    return a == b


def _globals() -> dict:
    """Settings, torch threads, grad mode and global RNG states."""
    fc, lo = gpytorch.settings.fast_computations, linear_operator.settings
    flags = (fc.covar_root_decomposition, fc.log_prob, fc.solves)
    return {
        'max_cholesky_size': gpytorch.settings.max_cholesky_size.value(),
        'fast_pred_var': gpytorch.settings.fast_pred_var.on(),
        'fast_computations': [f.on() for f in flags],
        'jitter': [lo.cholesky_jitter.value(t) for t in (torch.float32, DT)],
        'cholesky_max_tries': lo.cholesky_max_tries.value(),
        'torch_threads': torch.get_num_threads(),
        'grad_enabled': torch.is_grad_enabled(),
        'rng': [
            torch.get_rng_state(),
            np.random.get_state(),
            random.getstate(),
        ],
    }


def _params(b) -> dict:
    return {k: v.detach().clone() for k, v in b.model.state_dict().items()}


def _state(b) -> dict:
    """Everything an add/query must leave unchanged (bitwise)."""
    st = b.status()
    return {
        'model': _params(b),
        'optimizer': copy.deepcopy(b.optimizer.state_dict()),
        'train': [b._train_x.clone(), b._train_y.clone()],
        'counters': [st[k] for k in (*COUNTERS, 'state_version')],
        'events': b.events,
        'prediction': list(b.query(Q10)),
        'globals': _globals(),
    }


def expected_schedule(points, cfg) -> list:
    """(distinct, events, steps, phase) after each add, from the documented
    rule (greedy distinct locations, warm-up, cadence)."""
    counted, events, steps, since, out = [], 0, 0, 0, []
    sep = cfg.distinct_location_separation
    for x, y in points:
        if all(math.hypot(x - a, y - b) > sep for a, b in counted):
            counted.append((x, y))
        since += 1
        first = events == 0 and len(counted) >= cfg.min_distinct_locations
        later = events > 0 and since >= cfg.optimize_every
        if cfg.optimize_parameters and (first or later):
            steps += (
                cfg.initial_training_steps
                if first
                else cfg.training_steps_per_event
            )
            events, since = events + 1, 0
        phase = 'training' if events else 'warm-up'
        out.append((len(counted), events, steps, phase))
    return out


def _observed(b) -> tuple:
    st = b.status()
    return tuple(st[k] for k in SCHEDULE)


# ------------------------------------------------- independent references
def _weights(kernel) -> list:
    n = kernel.network
    layers = (n.linear1, n.linear2, n.linear3)
    return [t.detach().numpy().copy() for m in layers for t in m.parameters()]


def np_kernel(w, ell, alpha, a, b) -> np.ndarray:
    def reps(u):
        h = np.tanh(np.tanh(u @ w[0].T + w[1]) @ w[2].T + w[3]) @ w[4].T + w[5]
        s = np.exp(h - h.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        return s / np.linalg.norm(s, axis=-1, keepdims=True)

    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = reps(a), reps(b)
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    base = np.exp(-d2[..., None] / (2.0 * np.asarray(ell) ** 2))
    return alpha * (ra @ rb.T) * np.einsum('im,jm,ijm->ij', ra, rb, base)


def oracle(b, data, q) -> SimpleNamespace:
    """Dense Gaussian posterior in physical units from the belief's current
    parameters, with an independent normalizer and kernel."""
    cfg, k = b.config, b.model.covar_module
    s2, kern = cfg.target_scale**2, k.base_kernel
    o = SimpleNamespace(alpha=s2 * _val(k.outputscale), kappa=1.0)
    o.sigma2 = s2 * _val(b.likelihood.noise)
    c0 = cfg.target_offset
    c0 += cfg.target_scale * _val(b.model.mean_module.constant)
    w, ell = _weights(kern), kern.primitive_lengthscales.numpy()
    origin = [BOX[0][0], BOX[1][0]]
    Q = (np.asarray(q, float) - origin) / SIDE
    cov, o.mean = np_kernel(w, ell, o.alpha, Q, Q), np.full(len(Q), c0)
    o.s_y = cfg.target_scale
    if data:
        X = (np.array([d[:2] for d in data]) - origin) / SIDE
        y = np.array([d[2] for d in data])
        Ky = np_kernel(w, ell, o.alpha, X, X) + o.sigma2 * np.eye(len(y))
        Kq = np_kernel(w, ell, o.alpha, Q, X)
        sol = np.linalg.solve(Ky, np.column_stack([y - c0, Kq.T]))
        o.mean, cov = c0 + Kq @ sol[:, 0], cov - Kq @ sol[:, 1:]
        o.kappa = float(np.linalg.cond(Ky))
        o.s_y = max(float(np.abs(y - c0).max()), cfg.target_scale)
    o.cov = 0.5 * (cov + cov.T)
    return o


def compare_posterior(c, tag, b, data, q=Q10) -> SimpleNamespace:
    """query / query_with_covariance / query_observation vs the oracle."""
    o = oracle(b, data, q)
    tau = max(1e-9, 100.0 * o.kappa * EPS)
    (mean, var), (mj, cov) = b.query(q), b.query_with_covariance(q)
    vo = b.query_observation(q)[1]
    errs = {'mean': _rel(mean, o.mean, o.s_y), 'cov': _rel(cov, o.cov, o.alpha)}
    errs['var'] = _rel(var, np.diag(o.cov), o.alpha)
    info = {**errs, 'tau_post': tau, 'kappa': o.kappa}
    c.expect(f'{tag}: vs oracle', max(errs.values()) <= tau, info)
    lam = float(np.linalg.eigvalsh(cov)[0])
    tol = max(1e-12, 10 * o.kappa * EPS) * len(q) * o.alpha
    dup = [k for k in range(1, len(q)) if np.array_equal(q[k], q[0])]
    rows = [[mean[k], var[k], cov[k]] for k in [0, *dup]]
    c.expect(
        f'{tag}: joint diag/mean == query, symmetric, PSD, duplicate rows',
        _rel(np.diag(cov), var, o.alpha) <= 1e-12
        and _rel(mj, mean, o.s_y) <= 1e-12
        and _rel(cov, cov.T, o.alpha) <= 1e-12
        and lam >= -tol
        and all(_same(r, rows[0]) for r in rows),
        {'lambda_min': lam, 'tolerance': tol},
    )
    c.expect(
        f'{tag}: observation var - latent var == sigma2',
        _rel(vo - var, o.sigma2, o.sigma2) <= 1e-9,
    )
    return o


# --------------------------------------------------------------- kernel
def _scaled(b, alpha=2.5, sharpen=False):
    sk = copy.deepcopy(b.model.covar_module)
    sk.outputscale = torch.tensor(alpha, dtype=torch.float64)
    if sharpen:
        with torch.no_grad():
            sk.base_kernel.network.linear3.weight.mul_(8.0)
            sk.base_kernel.network.linear3.bias.mul_(8.0)
    return sk, _val(sk.outputscale)


def _K(kernel, a, b=None, diag=False) -> np.ndarray:
    with torch.no_grad():
        return linear_operator.to_dense(kernel(a, b, diag=diag)).numpy()


def _kernel_inputs():
    U = np.random.default_rng(11).uniform([0, 0], [0.5, 1], (40, 2))
    V = np.random.default_rng(12).uniform([0, 0], [0.5, 1], (30, 2))
    return torch.as_tensor(np.vstack([U, U[:5]])), torch.as_tensor(V)


def check_structure(c) -> None:
    """K1 structure, dtype and parameter registration."""
    cfg = AttentiveGPConfig()
    b = AttentiveGPBelief(cfg, (-1.0, 1.5), (2.0, 5.0))
    H, M = cfg.hidden_width, cfg.num_components
    net, model = b.model.covar_module.base_kernel.network, b.model
    kinds = [type(m).__name__ for m in net]
    shapes = [tuple(m.weight.shape) for m in net if hasattr(m, 'weight')]
    c.expect(
        'exactly Linear-Tanh-Linear-Tanh-Linear-Softmax(dim=-1)',
        kinds == ['Linear', 'Tanh', 'Linear', 'Tanh', 'Linear', 'Softmax']
        and shapes == [(H, 2), (H, H), (M, H)]
        and net[-1].dim == -1,
        {'layers': kinds, 'shapes': shapes},
    )
    tensors = [*model.named_parameters(), *model.named_buffers()]
    bad = [n for n, t in tensors if t.dtype != torch.float64 or t.is_cuda]
    names = [n for n, _ in model.named_parameters()]
    c.expect(
        'float64 CPU; one amplitude (raw_outputscale), no lengthscale param',
        not bad
        and [n for n in names if 'outputscale' in n]
        == ['covar_module.raw_outputscale']
        and not [n for n in names if 'lengthscale' in n],
        {'not_float64_cpu': bad, 'parameters': names[:3]},
    )
    buffers = dict(model.named_buffers())
    bank = buffers.get('covar_module.base_kernel.primitive_lengthscales')
    ref = np.linspace(cfg.min_lengthscale, cfg.max_lengthscale, M)
    c.expect(
        'bank buffer == linspace(min, max, M), no grad',
        bank is not None
        and np.array_equal(bank, ref)
        and not bank.requires_grad,
    )
    groups = b.optimizer.param_groups
    ids = [{id(p) for p in g['params']} for g in groups]
    lrs = [cfg.learning_rate_network, cfg.learning_rate_hyperparameters]
    c.expect(
        'optimizer groups partition parameters (network | rest), lrs',
        len(groups) == 2
        and ids[0] == {id(p) for p in net.parameters()}
        and not ids[0] & ids[1]
        and ids[0] | ids[1] == {id(p) for p in model.parameters()}
        and [g['lr'] for g in groups] == lrs,
    )
    side, n = max(1.5 - -1.0, 5.0 - 2.0), b.normalizer
    c.expect(
        'square normalizer (x_min, x_min + L, y_min, y_min + L)',
        (n.x_min, n.x_max, n.y_min, n.y_max) == (-1.0, -1.0 + side, 2.0, 5.0),
    )
    c.details['bank [m], 2 m x 4 m box (L = 4 m)'] = (ref * SIDE).tolist()


def check_gram(c, kb) -> None:
    """K2 Gram symmetry and PSD with duplicate inputs."""
    U, _ = _kernel_inputs()
    noise_t = _val(kb.likelihood.noise)
    for tag in ('default', 'sharpened'):
        sk, alpha = _scaled(kb, sharpen=tag == 'sharpened')
        K = _K(sk, U)
        tau_k, tau_psd = 1e-10 * alpha, 1e-12 * len(U) * alpha
        ev = np.linalg.eigvalsh(0.5 * (K + K.T))
        Kn = K + noise_t * np.eye(len(U))
        info = int(torch.linalg.cholesky_ex(torch.as_tensor(Kn)).info)
        lam = float(np.linalg.eigvalsh(Kn)[0])
        values = {'lambda_min': ev[0], 'near_zero': int((ev <= tau_psd).sum())}
        values['duplicates_bitwise'] = bool(np.array_equal(K[40:], K[:5]))
        values.update(lambda_min_noisy=lam, sigma2_t=noise_t)
        c.expect(
            f'{tag}: symmetric, PSD, >= 5 null eigenvalues, dups, +noise PD',
            _rel(K, K.T) <= tau_k
            and ev[0] >= -tau_psd
            and (ev <= tau_psd).sum() >= 5
            and _rel(K[40:], K[:5]) <= tau_k
            and lam >= noise_t - tau_psd
            and info == 0,
            values,
        )


def check_diag(c, kb) -> None:
    """K3 k(u, u) = alpha, diagonal semantics and rectangular blocks."""
    U, V = _kernel_inputs()
    sk, alpha = _scaled(kb)
    tau, A = 1e-10 * alpha, U[:30]
    w = _weights(sk.base_kernel)
    ell = sk.base_kernel.primitive_lengthscales.numpy()
    self_diags = [_K(sk, U, diag=True), np.diag(_K(sk, U))]
    self_diags.append(_K(sk, U, U.clone(), diag=True))
    c.expect(
        'self diag (diag=True, diag of K, U.clone()) == alpha',
        max(_rel(d, alpha) for d in self_diags) <= tau,
    )
    dg = _K(sk, A, V, diag=True)
    ref = np.diag(np_kernel(w, ell, alpha, A.numpy(), V.numpy()))
    gap = float(np.abs(dg - alpha).min())
    c.expect(
        'row-matched diag == diag(K(A, B)) == NumPy, != alpha',
        _rel(dg, np.diag(_K(sk, A, V))) <= tau
        and _rel(dg, ref) <= tau
        and gap > 1e-3 * alpha,
        {'min_abs_diff_from_alpha': gap},
    )
    KUV, KW = _K(sk, U, V), _K(sk, torch.cat([U, V]))
    ref = np_kernel(w, ell, alpha, U.numpy(), V.numpy())
    c.expect(
        'K(U, V) (45, 30) == block of K(U u V) == NumPy; K(V, U) == K^T',
        KUV.shape == (45, 30)
        and _rel(KUV, KW[:45, 45:]) <= tau
        and _rel(KUV, ref) <= tau
        and _rel(_K(sk, V, U), KUV.T) <= tau,
    )


def check_kernel_batching(c, kb) -> None:
    """K4 batched inputs, permutation and last_dim_is_batch."""
    U, _ = _kernel_inputs()
    sk, alpha = _scaled(kb)
    Ub, P = U.reshape(3, 15, 2), np.random.default_rng(13).permutation(45)
    Kb, Db = _K(sk, Ub), _K(sk, Ub, diag=True)
    c.expect(
        'batch (3, 15, 15) == per batch, diag == alpha; K(PU) == P K P^T',
        Kb.shape == (3, 15, 15)
        and max(_rel(Kb[i], _K(sk, Ub[i])) for i in range(3)) <= 1e-10 * alpha
        and Db.shape == (3, 15)
        and _rel(Db, alpha) <= 1e-10 * alpha
        and _rel(_K(sk, U[P]), _K(sk, U)[np.ix_(P, P)]) <= 1e-10 * alpha,
    )
    raised = _raises(lambda: sk(U, last_dim_is_batch=True).to_dense())
    c.expect('last_dim_is_batch=True raises', raised != 'nothing', raised)


def load_author(data: bytes, path) -> dict:
    """Execute only the author's kernel definitions, extracted with ast,
    in a namespace with np, torch, F and Parameter."""
    nodes = [
        n
        for n in ast.parse(data, filename=str(path)).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and n.name in AUTHOR_NAMES
    ]
    missing = sorted(AUTHOR_NAMES - {n.name for n in nodes})
    if missing:
        raise ValueError(f'definitions not found in {path}: {missing}')
    ns = {'np': np, 'torch': torch, 'F': torch.nn.functional}
    ns.update(Parameter=torch.nn.Parameter, __name__='author_demo')
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec')
    exec(code, ns)  # noqa: S102
    return ns


def _author_kernel(ns, sk):
    base, options = sk.base_kernel, np.get_printoptions()
    try:
        with torch.random.fork_rng(devices=[]):
            with contextlib.redirect_stdout(io.StringIO()):
                ak = ns['AK'](
                    _val(sk.outputscale),
                    base.primitive_lengthscales.numpy().copy(),
                    2,
                    base.network.linear1.out_features,
                    base.num_components,
                )
    finally:
        np.set_printoptions(**options)
    src = dict(base.network.named_parameters())
    dst = dict(ak.nn.named_parameters())
    names = [f'linear{i}.{t}' for i in (1, 2, 3) for t in ('weight', 'bias')]
    if any(src[n].shape != dst[n].shape for n in names):
        raise ValueError('author network shapes differ')
    with torch.no_grad():
        for n in names:
            dst[n].copy_(src[n])
    return ak


def check_author(c, kb, author) -> None:
    """K5 parity with the author's AK for identical weights."""
    c.details['author'] = {k: v for k, v in author.items() if k != 'ns'}
    U, V = _kernel_inputs()
    A = U[:30]
    for tag in ('default', 'sharpened'):
        sk, alpha = _scaled(kb, sharpen=tag == 'sharpened')
        tau = 1e-10 * alpha
        ak = _author_kernel(author['ns'], sk)
        with torch.no_grad():
            if tag == 'default':
                mine = kb.attention_representations(SIDE * U.numpy())
            else:
                mine = sk.base_kernel.representations(U).numpy()
            KUU, KUV = ak(U, U).numpy(), ak(U, V).numpy()
            cross = np.diag(ak(A, V).numpy())
            reps = ak.get_representations(U).numpy()
            diag_U, diag_A = ak.diag(U).numpy()[:, 0], ak.diag(A).numpy()[:, 0]
        err = max(_rel(KUU, _K(sk, U)), _rel(KUV, _K(sk, U, V)))
        c.expect(
            f'{tag}: forward, representations, self and cross diagonals',
            err <= tau
            and _rel(reps, mine) <= 1e-12
            and _rel(_K(sk, U, diag=True), diag_U) <= tau
            and _rel(_K(sk, A, V, diag=True), cross) <= tau,
            {'max_abs_diff': err},
        )
        gap = float(np.abs(diag_A - cross).max())
        c.details[f'{tag}: author.diag(A) vs diag(K(A, B)) (reported)'] = {
            'max_abs_diff': gap,
            'author_diag_is_self_only': gap > 1e-3 * alpha,
        }


def check_reductions(c, kb) -> None:
    """K6 constant memberships and M = 1 reduce to stationary kernels."""
    U, V = _kernel_inputs()
    sk, alpha = _scaled(kb)
    M, f64 = sk.base_kernel.num_components, torch.float64
    last = sk.base_kernel.network.linear3
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.linspace(-1, 1, M, dtype=f64))
    s = np.exp(np.linspace(-1, 1, M))
    s /= s.sum()
    ell = sk.base_kernel.primitive_lengthscales.numpy()
    d2 = ((U.numpy()[:, None] - V.numpy()[None]) ** 2).sum(-1)
    rbfs = np.exp(-d2[..., None] / (2 * ell**2))
    ref = alpha * (rbfs * s**2 / (s**2).sum()).sum(-1)
    shift = torch.tensor([0.1, -0.05], dtype=f64)
    c.expect(
        'constant memberships: alpha * sum_m w_m RBF_m, shift invariant',
        _rel(_K(sk, U, V), ref) <= 1e-10 * alpha
        and _rel(_K(sk, U + shift, V + shift), _K(sk, U, V)) <= 1e-10 * alpha,
    )
    b1 = _belief(num_components=1, min_lengthscale=0.1, max_lengthscale=0.1)
    sk1, a1 = _scaled(b1)
    c.expect(
        'M = 1: alpha * RBF (NumPy)',
        _rel(_K(sk1, U, V), a1 * np.exp(-d2 / 0.02)) <= 1e-10 * a1,
    )
    sk1(U, V).to_dense().sum().backward()
    grads = [p.grad for p in sk1.base_kernel.network.parameters()]
    c.details['M = 1: network gradients exactly zero (reported)'] = all(
        g is None or float(g.abs().max()) == 0.0 for g in grads
    )


# --------------------------------------------------------------- belief
def check_units(c) -> None:
    """B1 units and fixed-state conditioning vs a dense Gaussian."""
    data, preds = _data(F8), {}
    transforms = {'T_a': (0.0, 0.01), 'T_b': (0.0, 0.003)}
    transforms['T_c'] = (0.004, 0.003)
    for tag, (offset, scale) in transforms.items():
        b = _belief(
            optimize_parameters=False, target_offset=offset, target_scale=scale
        )
        a0, n0 = b.config.init_amplitude, b.config.init_noise_variance
        (m, v), vo = b.query(Q10), b.query_observation(Q10)[1]
        fs = b.fitted_summary()
        c.expect(
            f'{tag} prior: mean, var, obs var, noise, fitted summary (1e-12)',
            b.has_model()
            and _rel(m, offset, scale) <= 1e-12
            and _rel(v, a0, a0) <= 1e-12
            and _rel(vo, a0 + n0, a0) <= 1e-12
            and _rel(b.measurement_noise_variance(), n0, n0) <= 1e-12
            and _rel(fs['noise_variance'], n0, n0) <= 1e-12
            and _rel(fs['amplitude'], a0, a0) <= 1e-12
            and _rel(fs['constant_mean'], offset, scale) <= 1e-12,
        )
        _feed(b, data)
        o = compare_posterior(c, f'{tag} after F8', b, data)
        preds[tag] = (*b.query(Q10), o.s_y, o.alpha)
    (ma, va, s_y, alpha), (mb, vb, _, _) = preds['T_a'], preds['T_b']
    err = {'mean': _rel(ma, mb, s_y), 'var': _rel(va, vb, alpha)}
    c.expect(
        'T_a == T_b physical predictions (1e-9)', max(err.values()) <= 1e-9, err
    )
    xy = np.zeros((b.config.max_joint_query_size + 1, 2))
    raised = _raises(b.query_with_covariance, xy)
    c.expect('oversized joint query raises', raised == 'ValueError', raised)


def check_repeats(c) -> None:
    """B2 repeats, one point, near duplicates and constant targets."""
    vals, p = (0.010, 0.012, 0.011), np.array([[0.6, 1.1]])
    for mode, count in (('frozen', 3), ('warm-up', 1)):
        b = _belief(optimize_parameters=mode == 'warm-up')
        a, s2 = b.config.init_amplitude, b.config.init_noise_variance
        for n in range(1, count + 1):
            b.add_measurement(0.6, 1.1, vals[n - 1])
            (m, v), vo = b.query(p), b.query_observation(p)[1]
            err = {'mean': _rel(m, a * sum(vals[:n]) / (s2 + n * a), 0.012)}
            err['var'] = _rel(v, a * s2 / (s2 + n * a), a)
            c.expect(
                f'{mode} n={n}: closed form, var < sigma2 < obs var, phase',
                max(err.values()) <= 1e-10
                and v[0] < s2 < vo[0]
                and b.phase == mode
                and b.status()['optimization_events'] == 0,
                err,
            )
    cases = {
        '10 repeats': ([(0.6, 1.1)] * 10, 1),
        '10 within 1e-7 m': ([(0.6 + 1e-8 * k, 1.1) for k in range(10)], 1),
        '3 spaced 0.06 m': ([(0.6 + 0.06 * k, 1.1) for k in range(3)], 3),
    }
    for label, (points, distinct) in cases.items():
        b = _belief()
        _feed(b, [(x, y, 0.01) for x, y in points])
        expected = (distinct, 0, 0, 'warm-up')
        c.expect(
            f'{label}: {distinct} distinct, 0 events', _observed(b) == expected
        )
    points, grid = [*D10, (1.0, 3.9), (0.6, 1.1)], make_display_grid(*BOX, 0.1)
    for value in (0.0, 0.01):
        b = _belief(**FAST)
        _feed(b, [(x, y, value) for x, y in points])
        pred, fs = b.predict_marginal(grid.xy), b.fitted_summary()
        losses = [e[k] for e in b.events for k in ('loss_before', 'loss_after')]
        values = {'clamps': pred.variance_clamps, 'amplitude': fs['amplitude']}
        values.update(noise=fs['noise_variance'], mean=fs['constant_mean'])
        c.expect(
            f'constant {value}: schedule, finite, 0 <= var <= alpha_fit',
            _observed(b) == expected_schedule(points, b.config)[-1]
            and np.all(np.isfinite(losses))
            and np.all(np.isfinite(pred.mean))
            and 0 <= pred.variance.min()
            and pred.variance.max() <= fs['amplitude'] * (1 + 1e-9),
            values,
        )


def _np_nlml(state, b, data) -> float:
    """NLML (transformed units) of `data` for a model state_dict."""
    m, cfg = copy.deepcopy(b.model), b.config
    m.load_state_dict(state)
    kern = m.covar_module.base_kernel
    X = (np.array([d[:2] for d in data]) - [BOX[0][0], BOX[1][0]]) / SIDE
    y = (np.array([d[2] for d in data]) - cfg.target_offset) / cfg.target_scale
    r = y - _val(m.mean_module.constant)
    w, ell = _weights(kern), kern.primitive_lengthscales.numpy()
    K = np_kernel(w, ell, _val(m.covar_module.outputscale), X, X)
    L = np.linalg.cholesky(K + _val(m.likelihood.noise) * np.eye(len(r)))
    z = np.linalg.solve(L, r)
    logdet = np.log(np.diag(L)).sum()
    return float(0.5 * z @ z + logdet + 0.5 * len(r) * math.log(2 * math.pi))


def _adam_ok(b, step) -> bool:
    states = [b.optimizer.state[p] for p in b.model.parameters()]
    return all(
        float(s['step']) == step
        and bool(torch.isfinite(s['exp_avg']).all())
        and bool((s['exp_avg'] != 0).any())
        for s in states
    ) and all(bool(torch.isfinite(p).all()) for p in b.model.parameters())


def check_learning(c, shared) -> None:
    """B3 warm-up, cadence, optimizer state and learning on T14."""
    g0, data, b = _globals(), _data(T14), _belief()
    init, seen, got, drift = _params(b), {}, [], []
    for k, (x, y, v) in enumerate(data, 1):
        b.add_measurement(x, y, v)
        got.append(_observed(b))
        seen[k] = _params(b)
        if not _same(g0, _globals()):
            drift.append(k)
        if k == 12:
            c.expect(
                'Adam after obs 12: finite, nonzero, step 100', _adam_ok(b, 100)
            )
            m = copy.deepcopy(b.model).train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(m.likelihood, m)
            with attentive.exact_gpytorch_settings():
                (-mll(m(m.train_inputs[0]), m.train_targets)).backward()
            grads = [p.grad for p in m.parameters()]
            c.expect(
                'independent MLL gradients (deepcopy) finite and nonzero',
                all(
                    g is not None
                    and bool(torch.isfinite(g).all())
                    and bool((g != 0).any())
                    for g in grads
                ),
            )
    c.expect('globals unchanged after each add', not drift, drift)
    expected = expected_schedule([d[:2] for d in data], b.config)
    c.expect('schedule (defaults, T14) == rule', got == expected, got)
    c.expect('Adam after obs 14: step 110 (kept)', _adam_ok(b, 110))
    trained = [n for n, _ in b.model.named_parameters()]
    c.expect(
        'every parameter changed at obs 12 and between obs 12 and 13',
        all(not torch.equal(init[n], seen[12][n]) for n in trained)
        and all(not torch.equal(seen[12][n], seen[13][n]) for n in trained),
    )
    change = b.fitted_summary()['network_change_relative']
    c.details['network ||dtheta|| / ||theta0|| (reported)'] = change
    nlml = [_np_nlml(s, b, data[:12]) for s in (init, seen[12])]
    c.expect('NumPy NLML on 12 obs lower after obs 12', nlml[1] < nlml[0], nlml)
    o = compare_posterior(c, 'after obs 14', b, data)
    var = b.query(Q10)[1]
    c.expect(
        '0 <= var <= alpha_phys',
        0 <= var.min() <= var.max() <= o.alpha * (1 + 1e-9),
    )
    b2 = _belief(**CADENCE)
    pts, got = [D10[i] for i in (0, 1, 2, 0, 3, 4, 1, 5, 6)], []
    for x, y in pts:
        b2.add_measurement(x, y, f_syn(x, y))
        got.append(_observed(b2))
    expected = expected_schedule(pts, b2.config)
    c.expect('cadence variant == rule', got == expected, got)
    shared['b3'] = (b, data)


def check_atomicity(c) -> None:
    """B4 failure atomicity (invalid input, NaN loss during an event)."""
    data, b = _data(T14), _belief(**FAST)
    _feed(b, data[:12])
    before = _state(b)
    for bad in ((0.5, 1, math.nan), (0.5, 1, math.inf), (math.nan, 1, 0.01)):
        raised = _raises(b.add_measurement, *bad)
        ok = raised == 'ValueError' and _same(before, _state(b))
        c.expect(f'{bad}: ValueError, state unchanged', ok, raised)
    calls, original = [0], b._training_loss

    def failing():
        calls[0] += 1
        return original() * (math.nan if calls[0] == 3 else 1.0)

    b._training_loss = failing
    try:
        b.add_measurement(*data[12])
        message = 'no exception'
    except attentive.AttentiveGPError as exc:
        message = str(exc)
    finally:
        del b._training_loss
    after = _state(b)
    differs = [k for k in before if not _same(before[k], after[k])]
    c.expect(
        'NaN on 3rd loss call: AttentiveGPError, full state bitwise equal',
        calls[0] == 3 and 'step 3' in message and not differs,
        {'message': message, 'differs': differs},
    )
    b.add_measurement(*data[12])
    ref = _belief(**FAST)
    _feed(ref, data[:13])
    s_b, s_ref = _state(b), _state(ref)
    keys = ('model', 'optimizer', 'train', 'prediction')
    c.expect(
        're-added state == belief that never failed (bitwise)',
        all(_same(s_b[k], s_ref[k]) for k in keys),
    )


def check_queries(c, shared) -> None:
    """B5 queries have no side effects; batching."""
    (b, data), grid = shared['b3'], make_display_grid(*BOX, 0.05).xy
    before = _state(b)
    out = [*b.query(grid), *b.query(grid), b.query_with_covariance(Q10)[1]]
    out += [*b.query_observation(grid), b.attention_representations(grid)]
    b.measurement_noise_variance(), b.status(), b.fitted_summary()
    reps, (m1, v1) = out[-1], out[:2]
    c.expect(
        'no state/global change, repeat bitwise, float64 NumPy outputs',
        _same(before, _state(b))
        and _same(out[:2], out[2:4])
        and all(
            isinstance(a, np.ndarray) and a.dtype == np.float64 for a in out
        ),
    )
    c.expect(
        'representations (N, M) >= 0 with unit row norms (1e-12)',
        reps.shape == (len(grid), b.config.num_components)
        and reps.min() >= 0
        and _rel(np.linalg.norm(reps, axis=1), 1.0) <= 1e-12,
    )
    o = oracle(b, data, Q10)
    P = np.random.default_rng(14).permutation(len(grid))
    block = np.arange(1000, 2100)
    for label, idx in (('permuted', P), ('rows 1000..2099', block)):
        m, v = b.query(grid[idx])
        c.expect(
            f'{label} == full query (1e-12)',
            _rel(m, m1[idx], o.s_y) <= 1e-12
            and _rel(v, v1[idx], o.alpha) <= 1e-12,
            {'bitwise': _same([m, v], [m1[idx], v1[idx]])},
        )
    st0, p = b.status(), np.array([[1.0, 2.0]])
    var0 = b.query(p)[1]
    b.add_measurement(1.0, 2.0, f_syn(1.0, 2.0))
    var1, st1 = b.query(p)[1], b.status()
    keys = (*COUNTERS[:2], 'num_measurements')
    c.expect(
        'one add lowers the variance at its location; counters advance',
        var1[0] < var0[0] and all(st1[k] == st0[k] + 1 for k in keys),
        [var0[0], var1[0]],
    )


def check_reset_replay(c) -> None:
    """B6 reset, replay and grad modes reproduce the seeded model."""
    g0, grid, data = _globals(), make_display_grid(*BOX, 0.1).xy, _data(T14)
    A, B = _belief(**FAST), _belief(**FAST)
    pA, mapsA = _params(A), list(A.query(grid))
    c.expect(
        'same seed: initial params bitwise equal; seed + 1 differs',
        _same(pA, _params(B))
        and not _same(pA, _params(_belief(seed=1, **FAST))),
    )
    _feed(B, data)
    pB, mapsB = _params(B), list(B.query(grid))
    B.reset()
    st, cfg = B.status(), B.config
    lrs = [cfg.learning_rate_network, cfg.learning_rate_hyperparameters]
    c.expect(
        'reset == fresh: params, optimizer, lrs, counters, prior maps',
        _same(_params(B), pA)
        and len(B.optimizer.state) == 0
        and [g['lr'] for g in B.optimizer.param_groups] == lrs
        and [st[k] for k in (*COUNTERS[1:3], 'num_measurements')] == [0, 0, 0]
        and _same(list(B.query(grid)), mapsA),
    )
    _feed(B, data)
    m, v = B.query(grid)
    c.expect(
        'replay == before reset (params bitwise, maps 1e-12)',
        _same(_params(B), pB)
        and _rel(m, mapsB[0], cfg.target_scale) <= 1e-12
        and _rel(v, mapsB[1], cfg.init_amplitude) <= 1e-12,
        {'maps_bitwise': _same([m, v], mapsB)},
    )
    modes = {
        'torch.no_grad()': torch.no_grad,
        'inference_mode': torch.inference_mode,
    }
    for label, mode in modes.items():
        with mode():
            other = _belief(**FAST)
            _feed(other, data)
        c.expect(
            f'inside {label} (incl. events) == outside, bitwise',
            _same(_params(other), pB)
            and other.status()['optimization_events'] == 3,
        )
    c.expect('RNG and settings unchanged', _same(g0, _globals()))


B7_CORE = """
import json, sys
import numpy as np
import scalar_field_belief.attentive as a
b = a.AttentiveGPBelief(a.AttentiveGPConfig(), (0.0, 2.0), (0.0, 4.0))
for x, y in ((0.3, 0.4), (0.7, 1.1), (1.3, 0.5)):
    b.add_measurement(x, y, 0.01)
m, v = b.query(np.array([[1.0, 2.0]]))
loaded = [n for n in sys.argv[1:] if n in sys.modules]
finite = bool(np.isfinite(m + v).all())
print(json.dumps({'loaded': loaded, 'file': a.__file__, 'finite': finite}))
"""
B7_PLAYGROUND = """
import json, sys
sys.modules['scalar_field_belief_gmrf_py'] = None
sys.modules['scalar_field_belief.visgp'] = None
from belief_playground.backends import make_backends
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.session import Session
sc = Scenario(sys.argv[1])
names = ['gp', 'gp_attentive']
b = make_backends(names, sc.domain, gp_attentive_config=sys.argv[2])
s = Session(sc, b, make_display_grid(sc.domain.x_range, sc.domain.y_range, 0.1))
s.add_sample(0.5, 1.0); s.add_sample(1.5, 3.0); idle = s.wait_idle(600)
_, st, _, _ = s.poll(); s.close()
print(json.dumps({'types': [type(x).__name__ for x in b.values()], 'idle': idle,
                  'errors': [v.error for v in st.values()],
                  'n': [v.n_assimilated for v in st.values()],
                  'walls': b['gp_attentive'].effective_config()['walls_used']}))
"""


def _subprocess(code, roots, args, cwd) -> dict:
    env = {k: v for k, v in os.environ.items() if k != 'AMENT_PREFIX_PATH'}
    env['PYTHONPATH'] = os.pathsep.join(map(str, roots))
    command = [sys.executable, '-W', 'ignore', '-c', code, *map(str, args)]
    out = subprocess.run(
        command, env=env, cwd=cwd, capture_output=True, text=True, timeout=900
    )
    if out.returncode:
        raise RuntimeError(out.stderr.strip().splitlines()[-1])
    return json.loads(out.stdout.strip().splitlines()[-1])


def check_imports(c, scenario_path, config_path, out) -> None:
    """B7 import hygiene and optional backends (subprocesses)."""
    import scalar_field_sim

    core = Path(next(iter(scalar_field_belief.__path__))).parent
    sim = Path(next(iter(scalar_field_sim.__path__))).parent
    play = Path(belief_playground.__file__).resolve().parents[1]
    r = _subprocess(B7_CORE, [core], FORBIDDEN, out)
    c.expect(
        'core: no forbidden modules, __file__ under the source path',
        not r['loaded'] and r['file'].startswith(str(core)) and r['finite'],
        r,
    )
    args = [scenario_path, config_path]
    r = _subprocess(B7_PLAYGROUND, [core, play, sim], args, out)
    c.expect(
        'gp + gp_attentive without GMRF and visGP; walls not used',
        r['types'] == ['GPAdapter', 'AttentiveGPAdapter']
        and r['idle']
        and r['errors'] == [None, None]
        and r['n'] == [2, 2]
        and str(r['walls']).startswith('none'),
        r,
    )


# -------------------------------------------------------------- session
def _wait(s) -> None:
    if not s.wait_idle(900):
        raise RuntimeError('session did not become idle')


def _maps(s, name) -> list:
    st = s.poll()[1][name]
    return [None if a is None else a.copy() for a in (st.mean, st.var)]


def check_session(c, sc, config_path, out, shared) -> None:
    """S1 ordinary playground Session with gp, gp_attentive and visgp."""
    names = ['gp', 'gp_attentive', 'visgp']
    backends = make_backends(names, sc.domain, gp_attentive_config=config_path)
    for n, b in backends.items():
        if getattr(b, 'unavailable_reason', None):
            raise RuntimeError(f'{n} unavailable: {b.unavailable_reason}')
    grid = make_display_grid(sc.domain.x_range, sc.domain.y_range, 0.05)
    s = Session(sc, backends, grid, noise_std=5e-4, seed=1)
    shared['sessions'].append(s)
    ak = backends['gp_attentive']
    belief, cfg = ak.belief, ak.belief.config
    _wait(s)
    st, prior = s.poll()[1], _maps(s, 'gp_attentive')
    c.expect(
        '(a) AK ready, prior mean == offset, var == init_amplitude; gp waiting',
        st['gp_attentive'].state == 'ready'
        and st['gp'].state == 'waiting'
        and _rel(prior[0], cfg.target_offset, cfg.target_scale) <= 1e-12
        and _rel(prior[1], cfg.init_amplitude, cfg.init_amplitude) <= 1e-12,
    )
    eff0, net = ak.effective_config()['effective'], belief.kernel.network
    net0 = [p.detach().clone() for p in net.parameters()]
    for x, y in S1_POS:
        s.add_sample(x, y)
    _wait(s)
    st, maps17 = s.poll()[1], {n: _maps(s, n) for n in names}
    values17, fs = [o.value for o in s.observations], belief.fitted_summary()
    (mean, var), alpha_fit = maps17['gp_attentive'], fs['amplitude']
    losses = [
        e[k] for e in belief.events for k in ('loss_before', 'loss_after')
    ]
    eff, expected = ak.effective_config(), expected_schedule(S1_POS, cfg)[-1]
    live = {'training_status', 'fitted_summary'}
    changed = [
        not torch.equal(a, p)
        for a, p in zip(net0, net.parameters(), strict=True)
    ]
    c.expect(
        '(b) 17 obs, maps, counters, network changed, effective unchanged',
        all(st[n].error is None and st[n].n_assimilated == 17 for n in names)
        and np.all(np.isfinite(mean))
        and 0 <= var.min() <= var.max() <= alpha_fit * (1 + 1e-9)
        and _observed(belief) == expected
        and all(changed)
        and np.all(np.isfinite(losses))
        and eff['effective'] == eff0
        and live <= set(eff)
        and not live & set(eff['effective']),
        {'observed': _observed(belief), 'expected': expected},
    )
    counters = [belief.status()[k] for k in COUNTERS]
    s_y = np.abs(np.array(values17) - fs['constant_mean']).max()
    s_y = max(float(s_y), cfg.target_scale)

    def same17():
        m, v = _maps(s, 'gp_attentive')
        ok = _rel(m, mean, s_y) <= 1e-12 and _rel(v, var, alpha_fit) <= 1e-12
        ok = ok and [belief.status()[k] for k in COUNTERS] == counters
        return ok, {'bitwise': _same([m, v], [mean, var])}

    s.add_sample(1.5, 2.0)
    _wait(s)
    s.undo()
    _wait(s)
    c.expect('(c) add (1.5, 2.0) + undo == 17-obs maps, counters', *same17())
    csv, js = s.save(out / 'session_s1.csv')
    s.reset()
    _wait(s)
    reset_ok = _same(_maps(s, 'gp_attentive'), prior)
    c.expect('(c) reset: maps bitwise == prior', reset_ok)
    load_warnings = s.load(csv)
    _wait(s)
    ok, info = same17()
    meta = json.loads(js.read_text())['backends']['gp_attentive']
    sha = hashlib.sha256(Path(meta['config_file']['path']).read_bytes())
    c.expect(
        '(c) load == 17-obs maps, no warning; JSON keys, config sha256',
        ok
        and not [w for w in load_warnings if 'gp_attentive' in w]
        and meta['config_file']['sha256'] == sha.hexdigest()
        and {'effective', *live} <= set(meta),
        {**info, 'warnings': load_warnings},
    )
    gp_only = make_backends(['gp'], sc.domain)
    s2 = Session(sc, gp_only, grid, noise_std=5e-4, seed=1)
    shared['sessions'].append(s2)
    for x, y in S1_POS:
        s2.add_sample(x, y)
    _wait(s2)
    c.expect(
        '(d) gp-only session: observation values and gp maps bitwise equal',
        [o.value for o in s2.observations] == values17
        and _same(_maps(s2, 'gp'), maps17['gp']),
    )
    truth, x = sc.truth(grid.xy), grid.xy[:, 0]
    means = {n: m[0] for n, m in maps17.items()}
    rmse, cells = rmse_on_common_grid(truth, means)
    bands = {}
    for n, (m, v) in maps17.items():
        for lo, hi in ((0.7, 1.0), (1.0, 1.3)):
            sel = (x > lo) & (x < hi)
            band = [np.nanmean(m[sel]), np.nanmean(np.sqrt(v[sel]))]
            bands[f'{n} x {lo}..{hi}'] = [float(b) for b in band]
    last_query = ak.effective_config()['last_query']  # the grid query
    stds = {n: np.sqrt(backends[n].query(S1_STD_POINTS)[1]) for n in names}
    reps, status = belief.attention_representations(grid.xy), belief.status()
    keys = ('optimization_events', 'optimization_steps', 'last_event')
    keys += ('last_add', 'total_conditioning_ms', 'total_optimization_ms')
    c.details['report (not judged)'] = {
        'rmse': rmse,
        'rmse_cells': cells,
        'band mean, mean std': bands,
        'std at': {'points': S1_STD_POINTS, **stds},
        'ak_status': {k: status[k] for k in keys},
        'ak_last_grid_query': last_query,
        'ak_fitted': belief.fitted_summary(),
        'mean_membership (no physical meaning)': reps.mean(0),
        'timings_ms': {n: [st[n].update_ms, st[n].query_ms] for n in names},
    }
    arrays = shared['arrays']
    arrays.update(s1_grid_xy=grid.xy, s1_truth=truth, s1_ak_membership=reps)
    for n, (m, v) in maps17.items():
        arrays[f's1_{n}_mean'], arrays[f's1_{n}_var'] = m, v


# --------------------------------------------------------------- report
def _json(value):
    if isinstance(value, (np.generic, np.ndarray, torch.Tensor)):
        return value.tolist()
    return str(value)


def _markdown(report: dict) -> str:
    s = report['summary']

    def dump(v):
        return json.dumps(v, default=_json)

    lines = ['# Attentive Kernel GP diagnostic', '', report['created_utc'], '']
    lines += [f'- passed ({len(s["passed"])}): ' + ', '.join(s['passed'])]
    for key in ('failed', 'not_run'):
        lines.append(f'- {key.replace("_", " ")} ({len(s[key])})')
        lines += [f'  - {e["name"]}: {e["reason"]}' for e in s[key]]
    lines += ['', '## Checks', '']
    for chk in report['checks']:
        reason = f' ({chk["reason"]})' if chk['reason'] else ''
        lines.append(f'- **{chk["status"].upper()}** {chk["name"]}{reason}')
        for label, d in chk['details'].items():
            if isinstance(d, dict) and 'ok' in d:
                v = f': {dump(d["value"])}' if 'value' in d else ''
                lines.append(f'  - {"ok" if d["ok"] else "FAILED"} {label}{v}')
            else:
                lines.append(f'  - {label}: {dump(d)}')
    env = json.dumps(report['environment'], indent=2)
    lines += ['', '## Environment', '', '```json', env, '```', '']
    return '\n'.join(lines)


def _author(path) -> tuple[dict, str, str]:
    """Author namespace for K5; the sha256 is checked before anything of
    the file is executed."""
    if path is None:
        return {}, NOT_RUN, '--author-demo not given'
    path = Path(path).expanduser().resolve()
    info = {'path': str(path), 'expected_sha256': AUTHOR_SHA256}
    info['revision'] = AUTHOR_REVISION
    try:
        data = path.read_bytes()
        info['sha256'] = hashlib.sha256(data).hexdigest()
        reason = f'sha256 of {path} differs from demo/main.py at revision '
        reason += f'{AUTHOR_REVISION}; parity is defined for that file'
        if info['sha256'] != AUTHOR_SHA256:
            return info, NOT_RUN, reason
        info['ns'] = load_author(data, path)
    except Exception as exc:  # noqa: BLE001
        return info, FAILED, f'{type(exc).__name__}: {exc}'
    return info, PASSED, ''


def _environment() -> dict:
    mods = (scalar_field_belief, attentive, belief_playground, torch, gpytorch)
    files = {m.__name__: getattr(m, '__file__', None) for m in mods}
    files['scalar_field_belief'] = next(iter(scalar_field_belief.__path__))
    versions = {m.__name__: m.__version__ for m in (np, torch, gpytorch)}
    versions['linear_operator'] = linear_operator.__version__
    versions['python'] = sys.version.split()[0]
    threads = torch.get_num_threads()
    return {'files': files, 'versions': versions, 'torch_threads': threads}


def _run_checks(args, out, shared) -> tuple[list, dict]:
    config, scenario, error = None, None, ''
    try:
        default = launcher.default_gp_attentive_config(False)
        config = Path(args.gp_attentive_config or default).resolve()
        scenario = Scenario(launcher.resolve_scenario(args.scenario))
    except (launcher.ResourceError, OSError, ValueError) as exc:
        error = f'{type(exc).__name__}: {exc}'
    author, a_status, a_reason = _author(args.author_demo)
    kb, s_status = _belief(), FAILED if error else PASSED
    checks = [run_check(check_structure)]
    checks += [run_check(f, kb) for f in (check_gram, check_diag)]
    checks.append(run_check(check_kernel_batching, kb))
    kw = {'status': a_status, 'reason': a_reason}
    checks.append(run_check(check_author, kb, author, **kw))
    checks.append(run_check(check_reductions, kb))
    checks += [run_check(f) for f in (check_units, check_repeats)]
    checks.append(run_check(check_learning, shared))
    checks.append(run_check(check_atomicity))
    b3 = (PASSED, '') if 'b3' in shared else (NOT_RUN, 'no B3 belief')
    checks.append(run_check(check_queries, shared, status=b3[0], reason=b3[1]))
    checks.append(run_check(check_reset_replay))
    path, kw = scenario.path if scenario else None, {'reason': error}
    checks.append(
        run_check(check_imports, path, config, out, **kw, status=s_status)
    )
    s1_args = (scenario, config, out, shared)
    checks.append(run_check(check_session, *s1_args, status=s_status, **kw))
    meta = {
        'author_demo': {k: v for k, v in author.items() if k != 'ns'} or None,
        'gp_attentive_config': str(config) if config else None,
        'scenario': str(scenario.path) if scenario else args.scenario,
    }
    return checks, meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog='python -m belief_playground.attentive_diagnostics',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--out-dir', type=Path, help='output directory')
    p.add_argument('--gp-attentive-config', type=Path, help='S1/B7 AK YAML')
    p.add_argument('--author-demo', type=Path, help="author's demo/main.py")
    p.add_argument('--scenario', default='01_single_wall', help='S1/B7')
    args = p.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    default_out = Path('belief_playground_sessions') / 'attentive_diagnostics'
    out = (args.out_dir or default_out / stamp).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shared = {'sessions': [], 'arrays': {}}
    try:
        checks, meta = _run_checks(args, out, shared)
    finally:
        for s in shared['sessions']:
            s.close()
    summary = {'passed': [c.name for c in checks if c.status == PASSED]}
    for key, status in (('failed', FAILED), ('not_run', NOT_RUN)):
        found = [c for c in checks if c.status == status]
        summary[key] = [{'name': c.name, 'reason': c.reason} for c in found]
    exit_code = 1 if summary['failed'] else 3 if summary['not_run'] else 0
    report = {'created_utc': stamp, 'environment': _environment(), **meta}
    report.update(exit_code=exit_code, summary=summary)
    report['checks'] = [vars(c) for c in checks]
    text = json.dumps(report, indent=2, default=_json)
    (out / 'report.json').write_text(text)
    (out / 'report.md').write_text(_markdown(report))
    np.savez_compressed(out / 'arrays.npz', **shared['arrays'])
    for chk in checks:
        reason = f': {chk.reason}' if chk.reason else ''
        print(f'{chk.status.upper():8s} {chk.name}{reason}')
    print(f'output: {out}')
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
