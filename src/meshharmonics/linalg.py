"""Small dense and sparse linear algebra kernels written in pure Python.

Everything in this module operates on plain ``list`` objects so that the
library has no compiled dependencies.  The routines are tuned for the mesh
sizes that are practical in pure Python (a few thousand vertices), and they
favour numerical robustness over raw speed:

* :class:`CSRMatrix` -- compressed sparse row storage with duplicate summing.
* :func:`conjugate_gradient` -- Jacobi-preconditioned CG for SPD systems.
* :func:`reverse_cuthill_mckee` + :class:`SkylineCholesky` -- a bandwidth
  reducing ordering and an envelope Cholesky factorization, which together
  give fast repeated direct solves on mesh matrices.
* :func:`jacobi_eigh` and :func:`tridiagonal_eigh` -- dense symmetric
  eigensolvers used as building blocks and as test oracles.
"""

from __future__ import annotations

import math
import operator
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

Vector = List[float]
Matrix = List[List[float]]
LinearOperator = Callable[[Sequence[float]], Vector]

_mul = operator.mul


class LinAlgError(ArithmeticError):
    """Raised when a factorization or iterative method cannot proceed."""


class ConvergenceError(LinAlgError):
    """Raised when an iterative method exceeds its iteration budget."""


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def zeros(n: int) -> Vector:
    return [0.0] * n


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(map(_mul, a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def vsub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [ai - bi for ai, bi in zip(a, b)]


def hadamard(a: Sequence[float], b: Sequence[float]) -> Vector:
    return list(map(_mul, a, b))


def normalized(a: Sequence[float]) -> Vector:
    n = norm(a)
    if n == 0.0:
        raise LinAlgError("cannot normalize a zero vector")
    return [x / n for x in a]


def orthogonalize(v: Vector, basis: Iterable[Sequence[float]], passes: int = 2) -> Vector:
    """Remove the components of ``v`` along an orthonormal ``basis``.

    Two passes of classical Gram-Schmidt ("twice is enough") keep the result
    orthogonal to working precision.
    """
    basis = list(basis)
    for _ in range(passes):
        for q in basis:
            c = dot(v, q)
            if c != 0.0:
                v = [vi - c * qi for vi, qi in zip(v, q)]
    return v


# 3-vector helpers used heavily by the geometry code -------------------------


def sub3(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add3(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale3(s: float, a: Sequence[float]) -> Tuple[float, float, float]:
    return (s * a[0], s * a[1], s * a[2])


def dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm3(a: Sequence[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


# ---------------------------------------------------------------------------
# Dense matrices (lists of rows)
# ---------------------------------------------------------------------------


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def matvec(a: Matrix, x: Sequence[float]) -> Vector:
    return [dot(row, x) for row in a]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]


def cholesky(a: Matrix) -> Matrix:
    """Dense Cholesky factor ``L`` with ``a = L L^T`` (lower triangular)."""
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        li = low[i]
        for j in range(i + 1):
            lj = low[j]
            s = a[i][j] - sum(map(_mul, li[:j], lj[:j]))
            if i == j:
                if s <= 0.0:
                    raise LinAlgError("matrix is not positive definite")
                li[i] = math.sqrt(s)
            else:
                li[j] = s / lj[j]
    return low


def cholesky_solve(low: Matrix, b: Sequence[float]) -> Vector:
    n = len(low)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(low[i][k] * y[k] for k in range(i))) / low[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(low[k][i] * x[k] for k in range(i + 1, n))) / low[i][i]
    return x


def _sort_eigenpairs(values: Vector, vectors_by_col: Matrix) -> Tuple[Vector, List[Vector]]:
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    vals = [values[i] for i in order]
    vecs = [[vectors_by_col[r][i] for r in range(n)] for i in order]
    return vals, vecs


def jacobi_eigh(a: Matrix, tol: float = 1e-12, max_sweeps: int = 100) -> Tuple[Vector, List[Vector]]:
    """Eigen-decomposition of a small dense symmetric matrix (cyclic Jacobi).

    Returns ``(values, vectors)`` with values ascending and ``vectors[i]`` the
    unit eigenvector belonging to ``values[i]``.
    """
    n = len(a)
    m = [list(map(float, row)) for row in a]
    v = identity(n)
    total = sum(x * x for row in m for x in row) or 1.0
    for _ in range(max_sweeps):
        off = sum(m[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
        if off <= (tol * tol) * total:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = m[p][q]
                if abs(apq) < 1e-300:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * apq)
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for row in m:
                    akp, akq = row[p], row[q]
                    row[p] = c * akp - s * akq
                    row[q] = s * akp + c * akq
                rp, rq = m[p], m[q]
                for k in range(n):
                    apk, aqk = rp[k], rq[k]
                    rp[k] = c * apk - s * aqk
                    rq[k] = s * apk + c * aqk
                for row in v:
                    vkp, vkq = row[p], row[q]
                    row[p] = c * vkp - s * vkq
                    row[q] = s * vkp + c * vkq
    else:
        raise ConvergenceError("Jacobi eigenvalue iteration did not converge")
    return _sort_eigenpairs([m[i][i] for i in range(n)], v)


def tridiagonal_eigh(
    diag: Sequence[float], offdiag: Sequence[float], max_iter: int = 60
) -> Tuple[Vector, List[Vector]]:
    """Eigenpairs of a symmetric tridiagonal matrix via implicit QL shifts.

    ``diag`` has length ``n`` and ``offdiag`` length ``n - 1``.  Returns
    ascending eigenvalues and the matching unit eigenvectors.
    """
    n = len(diag)
    if len(offdiag) != max(n - 1, 0):
        raise ValueError("offdiag must have length len(diag) - 1")
    if n == 0:
        return [], []
    d = [float(x) for x in diag]
    e = [float(x) for x in offdiag] + [0.0]
    z = identity(n)
    for l in range(n):
        iterations = 0
        while True:
            m = l
            while m < n - 1:
                dd = abs(d[m]) + abs(d[m + 1])
                if abs(e[m]) <= 2.2e-16 * dd:
                    break
                m += 1
            if m == l:
                break
            iterations += 1
            if iterations > max_iter:
                raise ConvergenceError("tridiagonal QL iteration did not converge")
            g = (d[l + 1] - d[l]) / (2.0 * e[l])
            r = math.hypot(g, 1.0)
            g = d[m] - d[l] + e[l] / (g + math.copysign(r, g))
            s = c = 1.0
            p = 0.0
            underflow = False
            i = m - 1
            while i >= l:
                f = s * e[i]
                b = c * e[i]
                r = math.hypot(f, g)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] -= p
                    e[m] = 0.0
                    underflow = True
                    break
                s = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                for row in z:
                    f = row[i + 1]
                    row[i + 1] = s * row[i] + c * f
                    row[i] = c * row[i] - s * f
                i -= 1
            if underflow:
                continue
            d[l] -= p
            e[l] = g
            e[m] = 0.0
    return _sort_eigenpairs(d, z)


# ---------------------------------------------------------------------------
# Sparse matrices
# ---------------------------------------------------------------------------


class CSRMatrix:
    """Compressed sparse row matrix of floats.

    Instances are immutable by convention: arithmetic helpers return new
    matrices.  Column indices within each row are sorted and unique.
    """

    __slots__ = ("nrows", "ncols", "indptr", "indices", "data")

    def __init__(
        self,
        nrows: int,
        ncols: int,
        indptr: List[int],
        indices: List[int],
        data: List[float],
    ) -> None:
        if len(indptr) != nrows + 1:
            raise ValueError("indptr must have nrows + 1 entries")
        if len(indices) != len(data) or indptr[-1] != len(data):
            raise ValueError("indices/data length mismatch with indptr")
        self.nrows = nrows
        self.ncols = ncols
        self.indptr = indptr
        self.indices = indices
        self.data = data

    # construction -----------------------------------------------------
    @classmethod
    def from_triplets(
        cls,
        nrows: int,
        ncols: int,
        rows: Iterable[int],
        cols: Iterable[int],
        vals: Iterable[float],
        drop_zeros: bool = True,
    ) -> "CSRMatrix":
        """Build from COO triplets, summing duplicate entries."""
        acc: List[Dict[int, float]] = [dict() for _ in range(nrows)]
        for r, c, v in zip(rows, cols, vals):
            if not (0 <= r < nrows and 0 <= c < ncols):
                raise IndexError("triplet (%d, %d) outside %dx%d" % (r, c, nrows, ncols))
            row = acc[r]
            row[c] = row.get(c, 0.0) + float(v)
        indptr = [0]
        indices: List[int] = []
        data: List[float] = []
        for row in acc:
            for c in sorted(row):
                v = row[c]
                if drop_zeros and v == 0.0:
                    continue
                indices.append(c)
                data.append(v)
            indptr.append(len(data))
        return cls(nrows, ncols, indptr, indices, data)

    @classmethod
    def diagonal_matrix(cls, values: Sequence[float]) -> "CSRMatrix":
        n = len(values)
        return cls(n, n, list(range(n + 1)), list(range(n)), [float(v) for v in values])

    # properties ---------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nrows, self.ncols)

    @property
    def nnz(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return "CSRMatrix(shape=%r, nnz=%d)" % (self.shape, self.nnz)

    def row(self, i: int) -> Iterator[Tuple[int, float]]:
        start, end = self.indptr[i], self.indptr[i + 1]
        return zip(self.indices[start:end], self.data[start:end])

    def get(self, i: int, j: int) -> float:
        start, end = self.indptr[i], self.indptr[i + 1]
        lo, hi = start, end
        idx = self.indices
        while lo < hi:
            mid = (lo + hi) // 2
            if idx[mid] < j:
                lo = mid + 1
            else:
                hi = mid
        if lo < end and idx[lo] == j:
            return self.data[lo]
        return 0.0

    def triplets(self) -> Iterator[Tuple[int, int, float]]:
        for i in range(self.nrows):
            for j, v in self.row(i):
                yield i, j, v

    # products -----------------------------------------------------------
    def matvec(self, x: Sequence[float]) -> Vector:
        if len(x) != self.ncols:
            raise ValueError("dimension mismatch: %d vs %d" % (len(x), self.ncols))
        indptr, indices, data = self.indptr, self.indices, self.data
        out = [0.0] * self.nrows
        for i in range(self.nrows):
            s = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                s += data[k] * x[indices[k]]
            out[i] = s
        return out

    __matmul__ = matvec

    def quadratic_form(self, x: Sequence[float]) -> float:
        return dot(x, self.matvec(x))

    # transforms ---------------------------------------------------------
    def diagonal(self) -> Vector:
        return [self.get(i, i) for i in range(min(self.nrows, self.ncols))]

    def row_sums(self) -> Vector:
        return [sum(self.data[self.indptr[i] : self.indptr[i + 1]]) for i in range(self.nrows)]

    def scaled(self, alpha: float) -> "CSRMatrix":
        return CSRMatrix(self.nrows, self.ncols, list(self.indptr), list(self.indices),
                         [alpha * v for v in self.data])

    def symmetric_scaled(self, d: Sequence[float]) -> "CSRMatrix":
        """Return ``D A D`` for the diagonal matrix ``D = diag(d)``."""
        data = []
        for i in range(self.nrows):
            di = d[i]
            for k in range(self.indptr[i], self.indptr[i + 1]):
                data.append(di * self.data[k] * d[self.indices[k]])
        return CSRMatrix(self.nrows, self.ncols, list(self.indptr), list(self.indices), data)

    def add(self, other: "CSRMatrix", alpha: float = 1.0, beta: float = 1.0) -> "CSRMatrix":
        """Return ``alpha * self + beta * other``."""
        if self.shape != other.shape:
            raise ValueError("shape mismatch")
        rows, cols, vals = [], [], []
        for i, j, v in self.triplets():
            rows.append(i)
            cols.append(j)
            vals.append(alpha * v)
        for i, j, v in other.triplets():
            rows.append(i)
            cols.append(j)
            vals.append(beta * v)
        return CSRMatrix.from_triplets(self.nrows, self.ncols, rows, cols, vals, drop_zeros=False)

    def add_diagonal(self, d: Sequence[float]) -> "CSRMatrix":
        return self.add(CSRMatrix.diagonal_matrix(d))

    def to_dense(self) -> Matrix:
        out = [[0.0] * self.ncols for _ in range(self.nrows)]
        for i, j, v in self.triplets():
            out[i][j] = v
        return out

    def is_symmetric(self, tol: float = 1e-12) -> bool:
        if self.nrows != self.ncols:
            return False
        scale_ = max((abs(v) for v in self.data), default=0.0) or 1.0
        for i, j, v in self.triplets():
            if abs(v - self.get(j, i)) > tol * scale_:
                return False
        return True

    def permuted(self, perm: Sequence[int]) -> "CSRMatrix":
        """Symmetric permutation ``P A P^T`` where new index ``k`` is old ``perm[k]``."""
        inv = [0] * len(perm)
        for new, old in enumerate(perm):
            inv[old] = new
        rows, cols, vals = [], [], []
        for i, j, v in self.triplets():
            rows.append(inv[i])
            cols.append(inv[j])
            vals.append(v)
        return CSRMatrix.from_triplets(self.nrows, self.ncols, rows, cols, vals, drop_zeros=False)


# ---------------------------------------------------------------------------
# Iterative solver
# ---------------------------------------------------------------------------


@dataclass
class CGResult:
    x: Vector
    iterations: int
    residual_norm: float
    converged: bool


def conjugate_gradient(
    a: "CSRMatrix | LinearOperator",
    b: Sequence[float],
    x0: Optional[Sequence[float]] = None,
    tol: float = 1e-10,
    maxiter: Optional[int] = None,
    preconditioner: Optional[Sequence[float]] = None,
    raise_on_failure: bool = False,
) -> CGResult:
    """Preconditioned conjugate gradients for a symmetric positive (semi)definite system.

    ``a`` may be a :class:`CSRMatrix` or any callable computing ``A @ x``.
    ``preconditioner`` is an optional diagonal (Jacobi) preconditioner given
    as the *inverse* diagonal.  If ``a`` is a CSR matrix and no
    preconditioner is passed, Jacobi preconditioning is used automatically.
    Convergence is measured relative to ``||b||``.
    """
    n = len(b)
    if isinstance(a, CSRMatrix):
        op = a.matvec
        if preconditioner is None:
            diag = a.diagonal()
            preconditioner = [1.0 / d if d > 0.0 else 1.0 for d in diag]
    else:
        op = a
    if maxiter is None:
        maxiter = max(10 * n, 50)
    x = [0.0] * n if x0 is None else [float(v) for v in x0]
    r = vsub(b, op(x)) if x0 is not None else [float(v) for v in b]
    bnorm = norm(b)
    if bnorm == 0.0:
        return CGResult([0.0] * n, 0, 0.0, True)
    z = hadamard(preconditioner, r) if preconditioner is not None else list(r)
    p = list(z)
    rz = dot(r, z)
    rnorm = norm(r)
    it = 0
    while it < maxiter and rnorm > tol * bnorm:
        ap = op(p)
        pap = dot(p, ap)
        if pap <= 0.0:
            break
        alpha = rz / pap
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * api for ri, api in zip(r, ap)]
        rnorm = norm(r)
        it += 1
        if rnorm <= tol * bnorm:
            break
        z = hadamard(preconditioner, r) if preconditioner is not None else list(r)
        rz_new = dot(r, z)
        beta = rz_new / rz
        rz = rz_new
        p = [zi + beta * pi for zi, pi in zip(z, p)]
    converged = rnorm <= tol * bnorm
    if raise_on_failure and not converged:
        raise ConvergenceError("CG stalled at relative residual %.3e" % (rnorm / bnorm))
    return CGResult(x, it, rnorm, converged)


# ---------------------------------------------------------------------------
# Direct sparse solver: RCM ordering + envelope (skyline) Cholesky
# ---------------------------------------------------------------------------


def reverse_cuthill_mckee(a: CSRMatrix) -> List[int]:
    """Bandwidth-reducing permutation for a structurally symmetric matrix.

    Each connected component is started from a pseudo-peripheral vertex found
    by repeated breadth-first sweeps.
    """
    n = a.nrows
    adj = [[j for j, _ in a.row(i) if j != i] for i in range(n)]
    degree = [len(nbrs) for nbrs in adj]
    visited = [False] * n
    order: List[int] = []

    def bfs_levels(start: int) -> List[List[int]]:
        seen = {start}
        levels = [[start]]
        while True:
            nxt = []
            for u in levels[-1]:
                for w in adj[u]:
                    if w not in seen:
                        seen.add(w)
                        nxt.append(w)
            if not nxt:
                return levels
            levels.append(nxt)

    for seed in sorted(range(n), key=lambda i: degree[i]):
        if visited[seed]:
            continue
        start = seed
        depth = len(bfs_levels(start))
        for _ in range(8):
            last = bfs_levels(start)[-1]
            candidate = min(last, key=lambda i: degree[i])
            cdepth = len(bfs_levels(candidate))
            if cdepth <= depth:
                break
            start, depth = candidate, cdepth
        visited[start] = True
        queue = deque([start])
        while queue:
            u = queue.popleft()
            order.append(u)
            for w in sorted((w for w in adj[u] if not visited[w]), key=lambda i: degree[i]):
                visited[w] = True
                queue.append(w)
    order.reverse()
    return order


class SkylineCholesky:
    """Envelope Cholesky factorization ``P A P^T = L L^T`` of an SPD sparse matrix.

    Fill-in is confined to the row envelope, so the cost is governed by the
    profile of the reordered matrix.  With RCM ordering (the default) a
    triangle mesh with ``n`` vertices typically has an envelope of order
    ``n**1.5`` entries, which is very manageable for pure Python.
    """

    def __init__(self, a: CSRMatrix, ordering: Optional[Sequence[int]] = None) -> None:
        if a.nrows != a.ncols:
            raise ValueError("matrix must be square")
        n = a.nrows
        perm = list(ordering) if ordering is not None else reverse_cuthill_mckee(a)
        if sorted(perm) != list(range(n)):
            raise ValueError("ordering is not a permutation")
        self.n = n
        self.perm = perm
        pa = a.permuted(perm)
        first = [i for i in range(n)]
        for i in range(n):
            for j, _ in pa.row(i):
                if j < first[i]:
                    first[i] = j
        rows: List[Vector] = []
        for i in range(n):
            fi = first[i]
            dense_row = [0.0] * (i - fi + 1)
            for j, v in pa.row(i):
                if j <= i:
                    dense_row[j - fi] = v
            rows.append(dense_row)
        for i in range(n):
            fi = first[i]
            ri = rows[i]
            for j in range(fi, i):
                fj = first[j]
                rj = rows[j]
                k0 = fi if fi > fj else fj
                s = ri[j - fi]
                if k0 < j:
                    s -= sum(map(_mul, ri[k0 - fi : j - fi], rj[k0 - fj : j - fj]))
                ri[j - fi] = s / rj[j - fj]
            tail = ri[: i - fi]
            diag = ri[i - fi] - dot(tail, tail)
            if diag <= 0.0:
                raise LinAlgError("matrix is not positive definite (pivot %d = %.3e)" % (i, diag))
            ri[i - fi] = math.sqrt(diag)
        self.first = first
        self.rows = rows

    @property
    def envelope_size(self) -> int:
        return sum(len(r) for r in self.rows)

    def solve(self, b: Sequence[float]) -> Vector:
        n = self.n
        if len(b) != n:
            raise ValueError("dimension mismatch")
        first, rows, perm = self.first, self.rows, self.perm
        y = [float(b[perm[k]]) for k in range(n)]
        for i in range(n):
            fi = first[i]
            ri = rows[i]
            s = y[i]
            if fi < i:
                s -= sum(map(_mul, ri[: i - fi], y[fi:i]))
            y[i] = s / ri[i - fi]
        for i in range(n - 1, -1, -1):
            fi = first[i]
            ri = rows[i]
            xi = y[i] / ri[i - fi]
            y[i] = xi
            if xi != 0.0:
                for k in range(fi, i):
                    y[k] -= ri[k - fi] * xi
        x = [0.0] * n
        for k in range(n):
            x[perm[k]] = y[k]
        return x

    __call__ = solve

