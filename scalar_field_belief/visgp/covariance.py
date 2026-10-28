# Visibility-graph GP (visGP) after Gilbert and Datta (2024), Biometrics:
# https://academic.oup.com/biometrics/article/doi/10.1093/biomtc/ujae089/7753413
# Authors' R implementation: https://github.com/bjg345/visGP

"""Parent covariance and covariance selection on the reference set.

`covariance_selection(K, G)` returns the unique symmetric positive definite
`L` with

    L_ii = K_ii,  L_ij = K_ij for edges (i, j) of G,
    inv(L)_ij = 0 for distinct non-adjacent i, j,

i.e. the maximum-determinant completion of K restricted to the diagonal and
the edges (Dempster 1972). It is computed per connected component of G:

* L is exactly zero between components;
* a complete component (in particular an isolated node) keeps L = K;
* any other component is solved by the iterative proportional scaling of
  Speed and Kiiveri (1986) over its maximal cliques, in covariance form,
  starting from `diag(K)`:

      W <- W + W[:, c] inv(W_cc) (K_cc - W_cc) inv(W_cc) W[c, :]

  Each update sets `W_cc = K_cc` and changes only the clique block of
  `inv(W)`, so `inv(W)` keeps exact zeros on non-edges in exact arithmetic.

No graph edges are added (no chordal completion), and nothing is masked,
clipped or jittered. Failure to converge within the iteration bound, too
many cliques, or a failed Cholesky factorisation raise
`CovarianceSelectionError`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from scalar_field_belief.visgp.visibility import connected_components

SQRT3 = math.sqrt(3.0)


def matern32(
    a: np.ndarray, b: np.ndarray, signal_variance: float, lengthscale: float
) -> np.ndarray:
    """Isotropic Matern-3/2 covariance `(len(a), len(b))` between positions
    [m]; distances are computed elementwise."""
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    r = SQRT3 * d / lengthscale
    return signal_variance * (1.0 + r) * np.exp(-r)


class CovarianceSelectionError(RuntimeError):
    """Covariance selection failed; `result` holds the last diagnostics (or
    None if it failed before iterating)."""

    def __init__(self, message: str, result=None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class CovarianceSelectionResult:
    """`L` and its diagnostics.

    Residuals: `diag_edge_residual` = max |L - K| over the diagonal and the
    edges divided by `scale` (signal variance); `nonedge_precision_residual`
    = max |P_ij| / sqrt(P_ii P_jj), P = inv(L), over distinct non-adjacent
    pairs within a component (across components P is exactly zero).
    `sweeps` is the largest number of IPS sweeps over all components (0 if
    every component is complete).
    """

    L: np.ndarray
    converged: bool
    sweeps: int
    diag_edge_residual: float
    nonedge_precision_residual: float
    min_eigenvalue: float
    condition_number: float
    num_nodes: int
    num_edges: int
    num_components: int
    num_isolated: int
    num_complete_components: int
    num_cliques: int
    max_clique_size: int
    elapsed_s: float

    def summary(self) -> dict:
        """JSON-friendly diagnostics without the matrix."""
        out = {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
            if k not in ('L',)
        }
        for k, v in out.items():
            if isinstance(v, (np.floating, np.integer, np.bool_)):
                out[k] = v.item()
        return out


def maximal_cliques(adjacency: np.ndarray, max_cliques: int) -> list:
    """Maximal cliques (sorted index arrays, in lexicographic order) by
    Bron-Kerbosch with pivoting; deterministic. Raises
    `CovarianceSelectionError` if there are more than `max_cliques`."""
    m = len(adjacency)
    neighbours = [
        frozenset(np.flatnonzero(adjacency[i]).tolist()) for i in range(m)
    ]
    cliques = []

    def expand(r: list, p: set, x: set) -> None:
        if not p and not x:
            cliques.append(sorted(r))
            if len(cliques) > max_cliques:
                raise CovarianceSelectionError(
                    f'visibility graph has more than max_cliques = '
                    f'{max_cliques} maximal cliques.'
                )
            return
        pivot = max(sorted(p | x), key=lambda u: len(p & neighbours[u]))
        for v in sorted(p - neighbours[pivot]):
            expand(r + [v], p & neighbours[v], x & neighbours[v])
            p = p - {v}
            x = x | {v}

    expand([], set(range(m)), set())
    cliques.sort()
    return [np.array(c, dtype=np.intp) for c in cliques]


def _cholesky(matrix: np.ndarray, what: str):
    try:
        return linalg.cho_factor(matrix, lower=True, check_finite=False)
    except linalg.LinAlgError as exc:
        raise CovarianceSelectionError(
            f'Cholesky factorisation of {what} failed: {exc}'
        ) from exc


def _nonedge_precision_residual(W: np.ndarray, adjacency: np.ndarray) -> float:
    nonedge = ~adjacency
    np.fill_diagonal(nonedge, False)
    if not nonedge.any():
        return 0.0
    factor = _cholesky(W, 'the completed covariance')
    P = linalg.cho_solve(factor, np.eye(len(W)), check_finite=False)
    P = 0.5 * (P + P.T)
    d = np.sqrt(np.diag(P))
    return float(np.max(np.abs(P[nonedge]) / np.outer(d, d)[nonedge]))


def _ips_component(K, adjacency, cliques, scale, tol, precision_tol, sweeps):
    """IPS on one component. Returns (W, converged, sweeps, residuals)."""
    W = np.diag(np.diag(K))
    edge = adjacency | np.eye(len(K), dtype=bool)
    edge_residual = math.inf
    precision_residual = math.inf
    for sweep in range(1, sweeps + 1):
        for c in cliques:
            ix = np.ix_(c, c)
            factor = _cholesky(W[ix], f'a clique block (sweep {sweep})')
            Y = linalg.cho_solve(factor, K[ix] - W[ix], check_finite=False)
            M = linalg.cho_solve(factor, Y.T, check_finite=False)
            W = W + W[:, c] @ M @ W[c, :]
            W = 0.5 * (W + W.T)
        edge_residual = float(np.max(np.abs(W - K)[edge])) / scale
        if edge_residual <= tol:
            precision_residual = _nonedge_precision_residual(W, adjacency)
            if precision_residual <= precision_tol:
                return W, True, sweep, edge_residual, precision_residual
    if math.isinf(precision_residual):  # not evaluated yet; report it
        try:
            precision_residual = _nonedge_precision_residual(W, adjacency)
        except CovarianceSelectionError:
            pass  # W not positive definite; stays inf
    return W, False, sweeps, edge_residual, precision_residual


def covariance_selection(
    K: np.ndarray,
    adjacency: np.ndarray,
    scale: float,
    tolerance: float,
    precision_tolerance: float,
    max_iterations: int,
    max_cliques: int,
) -> CovarianceSelectionResult:
    """`L = CovSel(K, G)` for the graph `adjacency` (see module docstring).

    `scale` normalises the diagonal/edge residual (the signal variance).
    Raises `CovarianceSelectionError` on non-convergence or failure.
    """
    t0 = time.perf_counter()
    K = np.asarray(K, dtype=float)
    adjacency = np.asarray(adjacency, dtype=bool)
    m = len(K)
    if K.shape != (m, m) or adjacency.shape != (m, m):
        raise ValueError('K and adjacency must be square of the same size.')
    if not np.array_equal(adjacency, adjacency.T) or adjacency.diagonal().any():
        raise ValueError('adjacency must be symmetric without self loops.')

    L = np.zeros_like(K)
    components = connected_components(adjacency)
    converged = True
    sweeps = 0
    edge_residual = 0.0
    precision_residual = 0.0
    n_cliques = 0
    max_clique = 1
    n_complete = 0
    for comp in components:
        ix = np.ix_(comp, comp)
        sub = adjacency[ix]
        size = len(comp)
        if sub.sum() == size * (size - 1):
            L[ix] = K[ix]
            n_complete += 1
            n_cliques += 1
            max_clique = max(max_clique, size)
            continue
        cliques = maximal_cliques(sub, max_cliques)
        n_cliques += len(cliques)
        if n_cliques > max_cliques:
            raise CovarianceSelectionError(
                f'visibility graph has more than max_cliques = {max_cliques} '
                'maximal cliques.'
            )
        max_clique = max(max_clique, max(len(c) for c in cliques))
        W, ok, used, r_edge, r_prec = _ips_component(
            K[ix],
            sub,
            cliques,
            scale,
            tolerance,
            precision_tolerance,
            max_iterations,
        )
        L[ix] = W
        converged &= ok
        sweeps = max(sweeps, used)
        edge_residual = max(edge_residual, r_edge)
        precision_residual = max(precision_residual, r_prec)

    eig = np.linalg.eigvalsh(L)
    result = CovarianceSelectionResult(
        L=L,
        converged=bool(converged),
        sweeps=int(sweeps),
        diag_edge_residual=float(edge_residual),
        nonedge_precision_residual=float(precision_residual),
        min_eigenvalue=float(eig[0]),
        condition_number=float(eig[-1] / eig[0]) if eig[0] > 0 else math.inf,
        num_nodes=int(m),
        num_edges=int(adjacency.sum() // 2),
        num_components=len(components),
        num_isolated=int(sum(len(c) == 1 for c in components)),
        num_complete_components=int(n_complete),
        num_cliques=int(n_cliques),
        max_clique_size=int(max_clique),
        elapsed_s=time.perf_counter() - t0,
    )
    if not converged:
        raise CovarianceSelectionError(
            f'covariance selection did not converge in {max_iterations} '
            f'sweeps: diagonal/edge residual {edge_residual:.3e} (tolerance '
            f'{tolerance:.1e}), non-edge precision residual '
            f'{precision_residual:.3e} (tolerance {precision_tolerance:.1e}).',
            result,
        )
    try:
        _cholesky(L, 'L')  # positive definiteness of the assembled matrix
    except CovarianceSelectionError as exc:
        raise CovarianceSelectionError(str(exc), result) from exc
    return result
