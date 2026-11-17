import math
import random

import pytest

from meshharmonics import linalg as la


def random_spd(n, seed=0):
    rng = random.Random(seed)
    b = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    a = la.matmul(b, la.transpose(b))
    for i in range(n):
        a[i][i] += n
    return a


def test_csr_from_triplets_sums_duplicates_and_sorts():
    m = la.CSRMatrix.from_triplets(2, 3, [0, 0, 1, 0], [2, 0, 1, 2], [1.0, 2.0, 3.0, 4.0])
    assert m.indices == [0, 2, 1]
    assert m.get(0, 2) == 5.0 and m.get(1, 0) == 0.0
    assert m.matvec([1, 1, 1]) == [7.0, 3.0]
    assert m.to_dense() == [[2.0, 0.0, 5.0], [0.0, 3.0, 0.0]]


def test_csr_rejects_out_of_range_triplet():
    with pytest.raises(IndexError):
        la.CSRMatrix.from_triplets(2, 2, [2], [0], [1.0])


def test_csr_arithmetic_helpers():
    a = la.CSRMatrix.from_triplets(2, 2, [0, 0, 1], [0, 1, 1], [2.0, -1.0, 3.0])
    assert not a.is_symmetric()
    s = a.add(a, alpha=1.0, beta=-1.0)
    assert all(v == 0.0 for v in s.data)
    assert a.add_diagonal([1, 1]).diagonal() == [3.0, 4.0]
    assert a.symmetric_scaled([2.0, 0.5]).to_dense() == [[8.0, -1.0], [0.0, 0.75]]
    assert a.row_sums() == [1.0, 3.0]
    p = a.permuted([1, 0])
    assert p.to_dense() == [[3.0, 0.0], [-1.0, 2.0]]


def test_cholesky_dense_roundtrip():
    a = random_spd(6)
    low = la.cholesky(a)
    for row, ref in zip(la.matmul(low, la.transpose(low)), a):
        assert row == pytest.approx(ref)
    x = la.cholesky_solve(low, [1, 2, 3, 4, 5, 6])
    assert la.matvec(a, x) == pytest.approx([1, 2, 3, 4, 5, 6])
    with pytest.raises(la.LinAlgError):
        la.cholesky([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(la.LinAlgError):
        la.SkylineCholesky(la.CSRMatrix.from_triplets(2, 2, [0, 1, 0, 1], [0, 1, 1, 0], [1, 1, 2, 2]))


def test_jacobi_and_tridiagonal_agree():
    d, e = [2.0, -1.0, 3.0, 0.5, 4.0], [1.0, 0.3, -2.0, 0.7]
    dense = [[0.0] * 5 for _ in range(5)]
    for i in range(5):
        dense[i][i] = d[i]
    for i in range(4):
        dense[i][i + 1] = dense[i + 1][i] = e[i]
    v1, x1 = la.jacobi_eigh(dense)
    v2, x2 = la.tridiagonal_eigh(d, e)
    assert v1 == pytest.approx(v2, abs=1e-10)
    for lam, vec in zip(v2, x2):
        assert la.matvec(dense, vec) == pytest.approx([lam * x for x in vec], abs=1e-9)
        assert la.norm(vec) == pytest.approx(1.0)
    assert sum(v2) == pytest.approx(sum(d))


def test_tridiagonal_edge_cases():
    assert la.tridiagonal_eigh([], []) == ([], [])
    vals, vecs = la.tridiagonal_eigh([3.0], [])
    assert vals == [3.0] and vecs == [[1.0]]
    vals, _ = la.tridiagonal_eigh([1.0, 1.0, 1.0], [0.0, 0.0])
    assert vals == [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        la.tridiagonal_eigh([1.0, 2.0], [])


def test_conjugate_gradient_and_skyline_solve_match():
    n = 30
    rows, cols, vals = [], [], []
    for i in range(n):  # 1D Poisson + identity: SPD tridiagonal
        rows += [i]
        cols += [i]
        vals += [3.0]
        if i + 1 < n:
            rows += [i, i + 1]
            cols += [i + 1, i]
            vals += [-1.0, -1.0]
    a = la.CSRMatrix.from_triplets(n, n, rows, cols, vals)
    b = [math.sin(i) for i in range(n)]
    cg = la.conjugate_gradient(a, b, tol=1e-12)
    assert cg.converged and cg.iterations <= n
    chol = la.SkylineCholesky(a)
    x = chol.solve(b)
    assert x == pytest.approx(cg.x, abs=1e-9)
    assert a.matvec(x) == pytest.approx(b, abs=1e-10)
    assert la.conjugate_gradient(a, [0.0] * n).x == [0.0] * n


def test_rcm_reduces_envelope_on_scrambled_matrix():
    n = 40
    rng = random.Random(3)
    perm = list(range(n))
    rng.shuffle(perm)
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(perm[i]); cols.append(perm[i]); vals.append(4.0)
        if i + 1 < n:
            rows += [perm[i], perm[i + 1]]
            cols += [perm[i + 1], perm[i]]
            vals += [-1.0, -1.0]
    a = la.CSRMatrix.from_triplets(n, n, rows, cols, vals)
    order = la.reverse_cuthill_mckee(a)
    assert sorted(order) == list(range(n))
    natural = la.SkylineCholesky(a, ordering=list(range(n)))
    reordered = la.SkylineCholesky(a)
    assert reordered.envelope_size < natural.envelope_size
    assert reordered.envelope_size == 2 * n - 1
    b = [1.0] * n
    assert natural.solve(b) == pytest.approx(reordered.solve(b))


def test_orthogonalize_and_vector_helpers():
    q = [la.normalized([1.0, 1.0, 0.0])]
    assert la.dot(la.orthogonalize([1.0, 0.0, 0.0], q), q[0]) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(la.LinAlgError):
        la.normalized([0.0, 0.0])
