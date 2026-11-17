"""Sparse symmetric eigensolvers in pure Python.

Two families are provided:

* :func:`lanczos` -- Lanczos iteration with full reorthogonalization on an
  arbitrary symmetric linear operator.  When an invariant subspace is hit
  the basis is continued with a fresh random vector, so repeated eigenvalues
  are found as well.  Running it to ``n`` steps yields the full spectrum.
* :func:`power_iteration` -- deflated (inverse) power iteration with
  Rayleigh-quotient convergence checks; slower but very simple.

:func:`eigsh` solves the generalized problem ``A x = lambda M x`` for a
diagonal ``M`` by symmetric reduction ``M^{-1/2} A M^{-1/2}`` and, for the
smallest eigenvalues, a shift-invert spectral transform backed by an
envelope Cholesky factorization.  :func:`laplacian_eigenpairs` is the
mesh-level convenience wrapper.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .linalg import (CSRMatrix, ConvergenceError, LinAlgError, SkylineCholesky, conjugate_gradient,
                     dot, norm, orthogonalize, tridiagonal_eigh)

Operator = Callable[[Sequence[float]], List[float]]


@dataclass
class EigenResult:
    """Eigenpairs sorted by ascending eigenvalue.

    ``vectors[i]`` belongs to ``values[i]``.  For generalized problems the
    vectors are ``M``-orthonormal.  ``residuals[i]`` is
    ``||A x - lambda M x|| / max(1, |lambda|)``.
    """

    values: List[float]
    vectors: List[List[float]]
    residuals: List[float] = field(default_factory=list)
    iterations: int = 0
    method: str = ""

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(zip(self.values, self.vectors))

    @property
    def max_residual(self) -> float:
        return max(self.residuals, default=0.0)


def _random_unit(n: int, rng: random.Random, basis: Sequence[Sequence[float]] = ()) -> List[float]:
    for _ in range(20):
        v = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        v = orthogonalize(v, basis)
        nv = norm(v)
        if nv > 1e-8:
            return [x / nv for x in v]
    raise LinAlgError("could not draw a vector outside the current basis")


def lanczos(
    op: Operator,
    n: int,
    k: int,
    which: str = "LA",
    tol: float = 1e-10,
    max_basis: Optional[int] = None,
    min_basis: Optional[int] = None,
    seed: int = 0,
    v0: Optional[Sequence[float]] = None,
) -> EigenResult:
    """Extreme eigenpairs of a symmetric operator by Lanczos iteration.

    Parameters
    ----------
    op:
        Callable computing ``A @ x`` for a symmetric ``A`` of size ``n``.
    k:
        Number of eigenpairs wanted.
    which:
        ``"LA"`` (largest algebraic) or ``"SA"`` (smallest algebraic).
    tol:
        Ritz pairs are accepted once ``|beta_m s_m| <= tol * max(1, |theta|)``.
    max_basis:
        Upper bound on the Krylov dimension (default ``n``).
    min_basis:
        Minimum basis size before convergence is tested; a generous minimum
        guards against missing members of nearly repeated eigenvalues.
    """
    if which not in ("LA", "SA"):
        raise ValueError("which must be 'LA' or 'SA'")
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n (k=%d, n=%d)" % (k, n))
    rng = random.Random(seed)
    max_basis = n if max_basis is None else min(max_basis, n)
    if min_basis is None:
        min_basis = min(n, max(2 * k + 10, 20))
    min_basis = min(max(min_basis, k), max_basis)
    if v0 is not None:
        nv = norm(v0)
        q = [x / nv for x in v0] if nv > 0 else _random_unit(n, rng)
    else:
        q = _random_unit(n, rng)
    basis: List[List[float]] = []
    alphas: List[float] = []
    betas: List[float] = []  # betas[j] couples basis[j] and basis[j+1]
    beta_last = 0.0
    ritz_vals: List[float] = []
    ritz_vecs: List[List[float]] = []
    converged = False
    check_every = max(1, k // 2)
    while len(basis) < max_basis:
        basis.append(q)
        w = op(q)
        alpha = dot(w, q)
        alphas.append(alpha)
        w = orthogonalize(w, basis)
        beta = norm(w)
        m = len(basis)
        restart = beta <= 1e-12 * max(1.0, abs(alpha))
        if m >= min_basis and (m % check_every == 0 or restart or m == max_basis):
            ritz_vals, ritz_vecs = tridiagonal_eigh(alphas, betas)
            idx = list(range(m)) if which == "SA" else list(range(m - 1, -1, -1))
            wanted = idx[:k]
            effective_beta = 0.0 if restart else beta
            ok = all(abs(effective_beta * ritz_vecs[i][m - 1]) <= tol * max(1.0, abs(ritz_vals[i]))
                     for i in wanted)
            if ok and len(wanted) == k:
                converged = True
                beta_last = beta
                break
        if m == max_basis:
            beta_last = beta
            break
        if restart:
            # invariant subspace found; continue in its orthogonal complement
            q = _random_unit(n, rng, basis)
            betas.append(0.0)
        else:
            q = [x / beta for x in w]
            betas.append(beta)
    m = len(basis)
    if not ritz_vecs or len(ritz_vals) != m:
        ritz_vals, ritz_vecs = tridiagonal_eigh(alphas, betas)
    order = list(range(m)) if which == "SA" else list(range(m - 1, -1, -1))
    chosen = order[:k]
    if len(chosen) < k:
        raise ConvergenceError("Krylov space too small for %d eigenpairs" % k)
    values, vectors, residuals = [], [], []
    for i in chosen:
        s = ritz_vecs[i]
        vec = [0.0] * n
        for coeff, b in zip(s, basis):
            if coeff != 0.0:
                vec = [v + coeff * bj for v, bj in zip(vec, b)]
        values.append(ritz_vals[i])
        vectors.append(vec)
        residuals.append(abs(beta_last * s[m - 1]) / max(1.0, abs(ritz_vals[i])))
    if not converged and m < n and max(residuals) > tol * 1e3:
        raise ConvergenceError("Lanczos did not converge within %d basis vectors "
                               "(max residual %.2e)" % (m, max(residuals)))
    pairs = sorted(zip(values, vectors, residuals), key=lambda t: t[0])
    return EigenResult([p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs],
                       iterations=m, method="lanczos")


def power_iteration(
    op: Operator,
    n: int,
    k: int = 1,
    tol: float = 1e-10,
    maxiter: int = 5000,
    seed: int = 0,
    guard: Optional[int] = None,
) -> EigenResult:
    """Dominant eigenpairs (largest ``|lambda|``) by block power (subspace) iteration.

    A block of ``k + guard`` vectors is multiplied by the operator,
    re-orthonormalized and rotated by a Rayleigh-Ritz step each sweep.  The
    guard vectors speed up convergence when ``lambda_k`` is close to
    ``lambda_{k+1}``.  Converged when every wanted residual
    ``||A x - theta x||`` is below ``tol * max(1, |theta|)``.
    """
    from .linalg import jacobi_eigh

    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    rng = random.Random(seed)
    p = min(n, k + (guard if guard is not None else max(2, k // 2)))
    block: List[List[float]] = []
    for _ in range(p):
        block.append(_random_unit(n, rng, block))
    worst = math.inf
    for it in range(1, maxiter + 1):
        images = [op(x) for x in block]
        h = [[0.5 * (dot(block[a], images[b]) + dot(block[b], images[a])) for b in range(p)]
             for a in range(p)]
        thetas, coeffs = jacobi_eigh(h)
        order = sorted(range(p), key=lambda i: -abs(thetas[i]))
        ritz, ritz_images, residuals = [], [], []
        for idx in order:
            c = coeffs[idx]
            x = [sum(c[a] * block[a][r] for a in range(p)) for r in range(n)]
            ax = [sum(c[a] * images[a][r] for a in range(p)) for r in range(n)]
            ritz.append(x)
            ritz_images.append(ax)
            res = norm([u - thetas[idx] * v for u, v in zip(ax, x)])
            residuals.append(res / max(1.0, abs(thetas[idx])))
        worst = max(residuals[:k])
        if worst <= tol:
            vals = [thetas[i] for i in order[:k]]
            pairs = sorted(zip(vals, ritz[:k], residuals[:k]), key=lambda t: t[0])
            return EigenResult([q[0] for q in pairs], [q[1] for q in pairs], [q[2] for q in pairs],
                               iterations=it, method="power")
        block = []
        for y in ritz_images:
            y = orthogonalize(y, block)
            ny = norm(y)
            block.append([v / ny for v in y] if ny > 1e-300 else _random_unit(n, rng, block))
    raise ConvergenceError("power iteration stalled (residual %.2e)" % worst)


def eigsh(
    a: CSRMatrix,
    k: int,
    mass: Optional[Sequence[float]] = None,
    which: str = "SM",
    sigma: Optional[float] = None,
    method: str = "lanczos",
    solver: str = "cholesky",
    tol: float = 1e-10,
    seed: int = 0,
    max_basis: Optional[int] = None,
) -> EigenResult:
    """Eigenpairs of the symmetric (generalized) problem ``A x = lambda M x``.

    Parameters
    ----------
    a:
        Symmetric sparse matrix.
    k:
        Number of eigenpairs.
    mass:
        Positive diagonal of ``M`` (identity when omitted).
    which:
        ``"SM"`` for the eigenvalues closest to ``sigma`` (shift-invert),
        ``"LM"``/``"LA"`` for the largest algebraic eigenvalues.
    sigma:
        Shift for ``"SM"``.  Defaults to a small negative value relative to
        the spectral scale, which keeps ``A - sigma M`` positive definite for
        PSD ``A``.
    method:
        ``"lanczos"`` or ``"power"`` (deflated (inverse) power iteration).
    solver:
        Inner solver for shift-invert: ``"cholesky"`` (envelope Cholesky,
        factor once) or ``"cg"`` (preconditioned conjugate gradients).
    """
    n = a.nrows
    if a.ncols != n:
        raise ValueError("matrix must be square")
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    if which == "LM":
        which = "LA"
    if which not in ("SM", "LA"):
        raise ValueError("which must be 'SM', 'LM' or 'LA'")
    if method not in ("lanczos", "power"):
        raise ValueError("method must be 'lanczos' or 'power'")
    m = [1.0] * n if mass is None else [float(x) for x in mass]
    if len(m) != n or any(x <= 0.0 for x in m):
        raise ValueError("mass must be a positive vector of length n")
    inv_sqrt = [1.0 / math.sqrt(x) for x in m]
    sqrt_m = [math.sqrt(x) for x in m]
    a_sym = a.symmetric_scaled(inv_sqrt)

    if which == "LA":
        if method == "lanczos":
            res = lanczos(a_sym.matvec, n, k, which="LA", tol=tol, seed=seed, max_basis=max_basis)
        else:
            # minus the largest absolute row sum bounds the spectrum from below
            shift = -max(sum(abs(v) for _, v in a_sym.row(i)) for i in range(n))
            # shift so the largest algebraic eigenvalues dominate in magnitude
            op = (lambda x: [y - shift * xi for y, xi in zip(a_sym.matvec(x), x)])
            res = power_iteration(op, n, k, tol=tol, seed=seed)
            res.values = [v + shift for v in res.values]
        values, sym_vectors = res.values, res.vectors
        order = sorted(range(len(values)), key=lambda i: values[i])
    else:
        if sigma is None:
            scale = sum(abs(x) for x in a_sym.diagonal()) / n or 1.0
            sigma = -1e-6 * scale
        shifted = a.add_diagonal([-sigma * x for x in m])
        if solver == "cholesky":
            solve = SkylineCholesky(shifted).solve
        elif solver == "cg":
            def solve(b, _A=shifted):
                return conjugate_gradient(_A, b, tol=min(tol, 1e-12), raise_on_failure=True).x
        else:
            raise ValueError("solver must be 'cholesky' or 'cg'")

        def op(x: Sequence[float]) -> List[float]:
            # symmetric form of (A - sigma M)^{-1} M
            y = solve([s * xi for s, xi in zip(sqrt_m, x)])
            return [s * yi for s, yi in zip(sqrt_m, y)]

        if method == "lanczos":
            res = lanczos(op, n, k, which="LA", tol=tol, seed=seed, max_basis=max_basis)
        else:
            res = power_iteration(op, n, k, tol=tol, seed=seed)
        values = [sigma + 1.0 / theta if theta != 0.0 else math.inf for theta in res.values]
        sym_vectors = res.vectors
        order = sorted(range(len(values)), key=lambda i: values[i])

    out_vals, out_vecs, out_res = [], [], []
    for i in order:
        lam = values[i]
        x = [c * v for c, v in zip(inv_sqrt, sym_vectors[i])]
        # M-normalize and fix the sign so the largest-magnitude entry is positive
        mnorm = math.sqrt(sum(mi * xi * xi for mi, xi in zip(m, x)))
        x = [xi / mnorm for xi in x]
        pivot = max(range(n), key=lambda j: abs(x[j]))
        if x[pivot] < 0:
            x = [-xi for xi in x]
        ax = a.matvec(x)
        r = norm([axi - lam * mi * xi for axi, mi, xi in zip(ax, m, x)])
        out_vals.append(lam)
        out_vecs.append(x)
        out_res.append(r / max(1.0, abs(lam)))
    return EigenResult(out_vals, out_vecs, out_res, iterations=res.iterations,
                       method="%s/%s" % (method, "shift-invert" if which == "SM" else "direct"))


def laplacian_eigenpairs(
    mesh,
    k: int = 10,
    mass: str = "voronoi",
    operator: str = "cotangent",
    method: str = "lanczos",
    tol: float = 1e-9,
    seed: int = 0,
) -> EigenResult:
    """The ``k`` smallest Laplace-Beltrami eigenpairs of a mesh.

    Solves ``L phi = lambda M phi`` with the cotangent stiffness and lumped
    mass; the first eigenvalue is ~0 with a constant eigenvector for every
    connected component.
    """
    from .laplacian import laplacian_pair  # local import avoids a cycle

    pair = laplacian_pair(mesh, mass=mass, operator=operator)
    k = min(k, mesh.n_vertices)
    return eigsh(pair.stiffness, k, mass=pair.mass, which="SM", method=method, tol=tol, seed=seed)

