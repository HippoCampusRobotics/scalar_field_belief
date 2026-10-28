# Visibility-graph GP (visGP) after Gilbert and Datta (2024), Biometrics:
# https://academic.oup.com/biometrics/article/doi/10.1093/biomtc/ujae089/7753413
# Authors' R implementation: https://github.com/bjg345/visGP

"""Extension of the reference-set covariance to arbitrary valid positions.

For a position `x` with neighbour set `N(x)` (indices into the reference set
V), the row `b(x)` over V and the residual variance `F(x)` are

    b_N(x) = C(x, N) inv(C(N, N)),   b(x) = 0 outside N,
    F(x)   = C(x, x) - C(x, N) inv(C(N, N)) C(N, x),

with the parent covariance C. A position exactly in V has the unit row and
`F = 0`; a position without usable neighbours has `b = 0` and
`F = C(x, x)`. The process covariance is

    C_vis(x, z) = b(x) L b(z)^T + I[x = z] F(x),

where `I[x = z]` means the same spatial location (exact float64 equality of
both coordinates, see `location_key`), not the same observation.

Neighbour sets follow the nearest-clique rule (Algorithm 2 of the
supplement of Gilbert and Datta): the reference nodes visible from `x`
are sorted by (squared distance, reference index); among the first
`max_neighbors` of them, candidates are added in that order while the set
stays a clique of the visibility graph, and the scan stops at the first
candidate that is not adjacent to every node chosen so far.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from scalar_field_belief.visgp.covariance import matern32
from scalar_field_belief.visgp.visibility import visibility_matrix

# F(x) in [-RESIDUAL_CLAMP_TOLERANCE * signal_variance, 0) is roundoff and set
# to 0 (counted); a more negative value is an error.
RESIDUAL_CLAMP_TOLERANCE = 1e-12

_BLOCK = 512  # positions per vectorised neighbour search


def location_key(x: float, y: float) -> tuple[float, float]:
    """Identity of a spatial location: both float64 coordinates exactly
    (-0.0 and 0.0 are the same location)."""
    return (float(x) + 0.0, float(y) + 0.0)


@dataclass(frozen=True)
class ExtensionRows:
    """`weights` (n, m): rows b(x); `residual` (n,): F(x) [units^2];
    `clamped` (n,): F was a roundoff-negative value set to 0."""

    weights: np.ndarray
    residual: np.ndarray
    clamped: np.ndarray


class ProcessExtension:
    """Nearest-clique neighbour sets and rows b(x), F(x) for fixed V, G and
    parent covariance. Positions must be valid (inside the domain, off
    walls); the caller checks this."""

    def __init__(
        self,
        locations: np.ndarray,
        adjacency: np.ndarray,
        reference_covariance: np.ndarray,
        walls: np.ndarray,
        tolerance: float,
        signal_variance: float,
        lengthscale: float,
        max_neighbors: int,
    ):
        self.locations = locations
        self.adjacency = adjacency
        self.reference_covariance = reference_covariance
        self.walls = walls
        self.tolerance = float(tolerance)
        self.signal_variance = float(signal_variance)
        self.lengthscale = float(lengthscale)
        self.max_neighbors = int(max_neighbors)
        self._adjacent = adjacency.tolist()
        self._index = {
            location_key(x, y): i for i, (x, y) in enumerate(locations)
        }

    @property
    def num_reference(self) -> int:
        return len(self.locations)

    def reference_index(self, x: float, y: float) -> int | None:
        """Index of the reference node at exactly (x, y), else None."""
        return self._index.get(location_key(x, y))

    def neighbours(self, xy: np.ndarray) -> list[np.ndarray]:
        """N(x) for each position, nearest first ([i] for a reference node)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        out: list = [None] * len(xy)
        off = []
        for r, (x, y) in enumerate(xy):
            i = self.reference_index(x, y)
            if i is None:
                off.append(r)
            else:
                out[r] = np.array([i], dtype=np.intp)
        for start in range(0, len(off), _BLOCK):
            rows = off[start : start + _BLOCK]
            for r, nb in zip(
                rows, self._nearest_cliques(xy[rows]), strict=True
            ):
                out[r] = nb
        return out

    def _nearest_cliques(self, xy: np.ndarray) -> list[np.ndarray]:
        V = self.locations
        visible = visibility_matrix(xy, V, self.walls, self.tolerance)
        dx = xy[:, None, 0] - V[None, :, 0]
        dy = xy[:, None, 1] - V[None, :, 1]
        d2 = np.where(visible, dx * dx + dy * dy, np.inf)
        order = np.argsort(d2, axis=1, kind='stable')[:, : self.max_neighbors]
        result = []
        for r in range(len(xy)):
            chosen: list[int] = []
            for j in order[r].tolist():
                if not np.isfinite(d2[r, j]):
                    break  # no further visible candidate
                row = self._adjacent[j]
                if not all(row[c] for c in chosen):
                    break  # first incompatible candidate ends the scan
                chosen.append(j)
            result.append(np.array(chosen, dtype=np.intp))
        return result

    def rows(self, xy: np.ndarray) -> ExtensionRows:
        """b(x) and F(x) for each position; each row depends only on its own
        position. Raises RuntimeError if F is negative beyond roundoff."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        n, m = len(xy), self.num_reference
        weights = np.zeros((n, m))
        residual = np.empty(n)
        clamped = np.zeros(n, dtype=bool)
        s2 = self.signal_variance
        for r, nb in enumerate(self.neighbours(xy)):
            if nb.size == 0:
                residual[r] = s2
                continue
            if self.reference_index(*xy[r]) is not None:
                weights[r, nb[0]] = 1.0
                residual[r] = 0.0
                continue
            c_xn = matern32(xy[r], self.locations[nb], s2, self.lengthscale)[0]
            factor = linalg.cho_factor(
                self.reference_covariance[np.ix_(nb, nb)],
                lower=True,
                check_finite=False,
            )
            w = linalg.cho_solve(factor, c_xn, check_finite=False)
            f = s2 - float(c_xn @ w)
            if f < 0.0:
                if f < -RESIDUAL_CLAMP_TOLERANCE * s2:
                    raise RuntimeError(
                        f'negative residual variance F = {f:.3e} at '
                        f'{xy[r].tolist()} (neighbours {nb.tolist()}).'
                    )
                f = 0.0
                clamped[r] = True
            weights[r, nb] = w
            residual[r] = f
        return ExtensionRows(
            weights=weights, residual=residual, clamped=clamped
        )
