import math

import pytest

import meshharmonics as mh
from meshharmonics import linalg as la
from meshharmonics.eigensolve import eigsh, lanczos, power_iteration


def path_laplacian(n):
    rows, cols, vals = [], [], []
    for i in range(n):
        deg = (i > 0) + (i < n - 1)
        rows.append(i); cols.append(i); vals.append(float(deg))
        if i + 1 < n:
            rows += [i, i + 1]; cols += [i + 1, i]; vals += [-1.0, -1.0]
    return la.CSRMatrix.from_triplets(n, n, rows, cols, vals)


def path_eigs(n):
    return sorted(2 - 2 * math.cos(math.pi * k / n) for k in range(n))


def test_lanczos_full_spectrum_matches_analytic():
    n = 25
    A = path_laplacian(n)
    res = lanczos(A.matvec, n, n, which="SA")
    assert res.values == pytest.approx(path_eigs(n), abs=1e-9)


def test_lanczos_handles_exact_multiplicity():
    diag = [1.0, 1.0, 1.0, 2.0, 2.0, 5.0, 7.0, 7.0]
    A = la.CSRMatrix.diagonal_matrix(diag)
    res = lanczos(A.matvec, len(diag), 5, which="SA", min_basis=len(diag))
    assert res.values == pytest.approx([1, 1, 1, 2, 2])


@pytest.mark.parametrize("method", ["lanczos", "power"])
def test_eigsh_largest(method):
    n = 20
    res = eigsh(path_laplacian(n), 3, which="LM", method=method, tol=1e-7)
    assert res.values == pytest.approx(path_eigs(n)[-3:], rel=1e-6)


@pytest.mark.parametrize("method,solver", [("lanczos", "cholesky"), ("lanczos", "cg"),
                                           ("power", "cholesky")])
def test_eigsh_smallest_generalized(method, solver):
    n = 20
    A = path_laplacian(n)
    mass = [1.0 + 0.5 * (i % 3) for i in range(n)]
    res = eigsh(A, 4, mass=mass, method=method, solver=solver, tol=1e-10)
    dense = A.symmetric_scaled([1 / math.sqrt(m) for m in mass]).to_dense()
    ref, _ = la.jacobi_eigh(dense)
    assert res.values == pytest.approx(ref[:4], abs=1e-6)
    for i, vi in enumerate(res.vectors):
        for j, vj in enumerate(res.vectors):
            g = sum(a * m * b for a, m, b in zip(vi, mass, vj))
            assert g == pytest.approx(1.0 if i == j else 0.0, abs=1e-6)
    assert res.max_residual < 1e-5


def test_power_iteration_dominant():
    A = la.CSRMatrix.diagonal_matrix([1.0, -4.0, 3.0])
    res = power_iteration(A.matvec, 3, 2)
    assert sorted(abs(v) for v in res.values) == pytest.approx([3.0, 4.0])


def test_eigsh_argument_validation():
    A = path_laplacian(4)
    for kwargs in ({"which": "XX"}, {"method": "qr"}, {"mass": [1, 1, 1, 0]}, {"solver": "lu"}):
        with pytest.raises(ValueError):
            eigsh(A, 2, **kwargs)
    with pytest.raises(ValueError):
        eigsh(A, 9)


def test_sphere_spectrum_approaches_l_l_plus_1(sphere_eig):
    vals = sphere_eig.values
    assert vals[0] == pytest.approx(0.0, abs=1e-8)
    assert vals[1:4] == pytest.approx([2.0] * 3, rel=0.01)
    assert vals[4:9] == pytest.approx([6.0] * 5, rel=0.03)
    assert vals[9:16] == pytest.approx([12.0] * 7, rel=0.07)
    first = sphere_eig.vectors[0]
    assert max(first) - min(first) < 1e-6  # constant eigenvector


def test_disconnected_mesh_has_zero_per_component():
    a = mh.tetrahedron()
    verts = list(a.vertices) + [tuple(x + 3 for x in p) for p in a.vertices]
    faces = list(a.faces) + [tuple(i + 4 for i in f) for f in a.faces]
    res = mh.laplacian_eigenpairs(mh.TriMesh(verts, faces), k=3)
    assert res.values[:2] == pytest.approx([0.0, 0.0], abs=1e-8)
    assert res.values[2] > 0.1


def test_scaling_law(sphere):
    lam1 = mh.laplacian_eigenpairs(sphere, k=2).values[1]
    lam2 = mh.laplacian_eigenpairs(sphere.transformed(2.0), k=2).values[1]
    assert lam2 == pytest.approx(lam1 / 4.0, rel=1e-8)
