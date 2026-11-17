"""Spectral point descriptors built from Laplace-Beltrami eigenpairs.

All functions take an :class:`~meshharmonics.eigensolve.EigenResult` whose
vectors are mass-orthonormal (as returned by
:func:`~meshharmonics.eigensolve.laplacian_eigenpairs`).

* Heat kernel signature (Sun, Ovsjanikov & Guibas 2009).
* Wave kernel signature (Aubry, Schlickewei & Cremers 2011).
* Global point signature (Rustamov 2007).
* ShapeDNA (Reuter, Wolter & Peinecke 2006), heat trace, diffusion distance.

Descriptors are returned row-major: ``sig[v]`` is the feature vector of
vertex ``v``, ready to be fed into a learning pipeline.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .eigensolve import EigenResult

ZERO_TOL = 1e-8


def _nonzero_pairs(eig: EigenResult, tol: float = ZERO_TOL) -> List[Tuple[float, List[float]]]:
    scale = max((abs(v) for v in eig.values), default=1.0) or 1.0
    return [(lam, vec) for lam, vec in eig if lam > tol * scale]


def _first_nonzero(eig: EigenResult) -> float:
    pairs = _nonzero_pairs(eig)
    if not pairs:
        raise ValueError("eigen decomposition has no non-zero eigenvalues")
    return pairs[0][0]


def default_hks_times(eig: EigenResult, count: int = 16) -> List[float]:
    """Logarithmically spaced times ``[4 ln 10 / λ_max, 4 ln 10 / λ_1]``."""
    if count < 1:
        raise ValueError("count must be positive")
    lam_min = _first_nonzero(eig)
    lam_max = max(eig.values)
    t_min = 4.0 * math.log(10.0) / lam_max
    t_max = 4.0 * math.log(10.0) / lam_min
    if count == 1:
        return [math.sqrt(t_min * t_max)]
    ratio = (t_max / t_min) ** (1.0 / (count - 1))
    return [t_min * ratio ** i for i in range(count)]


def heat_trace(eig: EigenResult, t: float) -> float:
    """``Z(t) = sum_i exp(-λ_i t)`` (truncated to the available eigenpairs)."""
    return math.fsum(math.exp(-lam * t) for lam in eig.values)


def heat_kernel(eig: EigenResult, x: int, y: int, t: float) -> float:
    """``k_t(x, y) = sum_i exp(-λ_i t) φ_i(x) φ_i(y)``."""
    return math.fsum(math.exp(-lam * t) * vec[x] * vec[y] for lam, vec in eig)


def heat_kernel_signature(eig: EigenResult, times: Optional[Sequence[float]] = None,
                          normalize: str = "none") -> List[List[float]]:
    """``HKS(x, t) = k_t(x, x)`` for every vertex and time.

    ``normalize="trace"`` divides each time column by the heat trace, which
    removes the global scale dependence (scale-normalized HKS);
    ``"l2"`` scales each vertex row to unit Euclidean length.
    """
    if normalize not in ("none", "trace", "l2"):
        raise ValueError("normalize must be 'none', 'trace' or 'l2'")
    times = list(times) if times is not None else default_hks_times(eig)
    n = len(eig.vectors[0]) if eig.vectors else 0
    decays = [[math.exp(-lam * t) for lam in eig.values] for t in times]
    sig = [[0.0] * len(times) for _ in range(n)]
    for i, vec in enumerate(eig.vectors):
        for ti, row_decay in enumerate(decays):
            d = row_decay[i]
            if d == 0.0:
                continue
            for v in range(n):
                sig[v][ti] += d * vec[v] * vec[v]
    if normalize == "trace":
        traces = [sum(row) for row in decays]
        sig = [[s / z for s, z in zip(row, traces)] for row in sig]
    elif normalize == "l2":
        out = []
        for row in sig:
            length = math.sqrt(sum(s * s for s in row)) or 1.0
            out.append([s / length for s in row])
        sig = out
    return sig


def wave_kernel_signature(eig: EigenResult, n_energies: int = 16,
                          sigma_factor: float = 7.0) -> List[List[float]]:
    """Wave kernel signature over log-spaced energies.

    Energies ``e_k`` are evenly spaced in ``[log λ_1, log λ_max]`` (with a
    margin of ``2σ``), ``σ = sigma_factor * Δe``, and each column is
    normalized by ``sum_i exp(-(e - log λ_i)^2 / 2σ^2)``.
    """
    pairs = _nonzero_pairs(eig)
    if len(pairs) < 2:
        raise ValueError("WKS needs at least two non-zero eigenvalues")
    logs = [math.log(lam) for lam, _ in pairs]
    lo, hi = logs[0], logs[-1]
    delta = (hi - lo) / max(n_energies, 1)
    sigma = sigma_factor * delta if delta > 0 else 1.0
    start, stop = lo + 2.0 * sigma, hi - 2.0 * sigma
    if stop <= start:
        start, stop = lo, hi
    energies = ([0.5 * (start + stop)] if n_energies == 1 else
                [start + (stop - start) * k / (n_energies - 1) for k in range(n_energies)])
    n = len(pairs[0][1])
    sig = [[0.0] * n_energies for _ in range(n)]
    for ei, e in enumerate(energies):
        weights = [math.exp(-((e - lg) ** 2) / (2.0 * sigma * sigma)) for lg in logs]
        total = sum(weights) or 1.0
        for w, (_, vec) in zip(weights, pairs):
            if w < 1e-300:
                continue
            w /= total
            for v in range(n):
                sig[v][ei] += w * vec[v] * vec[v]
    return sig


def global_point_signature(eig: EigenResult, k: Optional[int] = None) -> List[List[float]]:
    """Rustamov's GPS embedding ``(φ_1/√λ_1, φ_2/√λ_2, ...)`` per vertex.

    Note that eigenvector signs (and bases of repeated eigenspaces) are not
    unique, so GPS coordinates are only defined up to those ambiguities.
    """
    pairs = _nonzero_pairs(eig)
    if k is not None:
        pairs = pairs[:k]
    if not pairs:
        raise ValueError("no non-zero eigenpairs available")
    n = len(pairs[0][1])
    scales = [1.0 / math.sqrt(lam) for lam, _ in pairs]
    return [[s * vec[v] for s, (_, vec) in zip(scales, pairs)] for v in range(n)]


def diffusion_distance(eig: EigenResult, x: int, y: int, t: float) -> float:
    """``D_t(x, y)^2 = sum_i exp(-2 λ_i t) (φ_i(x) - φ_i(y))^2``."""
    return math.sqrt(math.fsum(math.exp(-2.0 * lam * t) * (vec[x] - vec[y]) ** 2
                               for lam, vec in eig))


def shape_dna(eig: EigenResult, k: Optional[int] = None, area: Optional[float] = None,
              skip_zero: bool = True) -> List[float]:
    """ShapeDNA: the (optionally area-normalized) beginning of the spectrum.

    Multiplying by the surface ``area`` makes the descriptor scale invariant
    (eigenvalues scale as ``1/length^2``).
    """
    values = [lam for lam, _ in _nonzero_pairs(eig)] if skip_zero else list(eig.values)
    if k is not None:
        values = values[:k]
    if area is not None:
        values = [lam * area for lam in values]
    return values


def descriptor_distances(sig: Sequence[Sequence[float]], query: int,
                         metric: str = "l2") -> List[float]:
    """Distance from vertex ``query``'s descriptor to every vertex descriptor.

    ``metric="l2"`` is Euclidean; ``"log_l1"`` compares log-descriptors
    (robust for HKS, whose values span orders of magnitude).
    """
    ref = sig[query]
    out = []
    if metric == "l2":
        for row in sig:
            out.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(row, ref))))
    elif metric == "log_l1":
        lref = [math.log(max(a, 1e-300)) for a in ref]
        for row in sig:
            out.append(sum(abs(math.log(max(a, 1e-300)) - b) for a, b in zip(row, lref)))
    else:
        raise ValueError("metric must be 'l2' or 'log_l1'")
    return out


def most_similar_vertices(sig: Sequence[Sequence[float]], query: int, count: int = 5,
                          metric: str = "l2", exclude_self: bool = True) -> List[int]:
    dists = descriptor_distances(sig, query, metric)
    order = sorted(range(len(dists)), key=lambda v: (dists[v], v))
    if exclude_self:
        order = [v for v in order if v != query]
    return order[:count]
