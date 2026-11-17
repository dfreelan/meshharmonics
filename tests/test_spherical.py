import math
import random

import pytest

import meshharmonics as mh
from meshharmonics import spherical as sh


def test_low_degree_closed_forms():
    rng = random.Random(1)
    c1 = math.sqrt(3 / (4 * math.pi))
    for _ in range(20):
        theta, phi = rng.uniform(0, math.pi), rng.uniform(-math.pi, math.pi)
        x, y, z = sh.spherical_to_cartesian(theta, phi)
        b = sh.sh_basis_angles(2, theta, phi)
        assert b[0] == pytest.approx(0.5 / math.sqrt(math.pi))
        assert b[1:4] == pytest.approx([c1 * y, c1 * z, c1 * x])
        assert b[6] == pytest.approx(math.sqrt(5 / (16 * math.pi)) * (3 * z * z - 1))
        assert b[4] == pytest.approx(0.5 * math.sqrt(15 / math.pi) * x * y)
        assert sh.sh_basis(2, (2 * x, 2 * y, 2 * z)) == pytest.approx(b)
    assert sh.real_sph_harm(1, 0, 0.0, 0.0) == pytest.approx(c1)


def test_addition_theorem_holds_for_high_degree():
    # sum_m Y_lm(d)^2 = (2l+1)/(4 pi) for every direction
    for l in (5, 20, 60):
        b = sh.sh_basis_angles(l, 0.7, 2.1)
        band = sum(v * v for v in b[l * l:(l + 1) ** 2])
        assert band == pytest.approx((2 * l + 1) / (4 * math.pi), rel=1e-10)


def test_quadrature_orthonormality():
    mesh = mh.icosphere(3)
    dirs, w = sh.sphere_quadrature(mesh)
    assert sum(w) == pytest.approx(4 * math.pi, rel=1e-9)
    lmax = 3
    basis = [sh.sh_basis(lmax, d) for d in dirs]
    nc = sh.n_coefficients(lmax)
    for i in range(nc):
        for j in range(nc):
            g = sum(wk * bk[i] * bk[j] for wk, bk in zip(w, basis))
            assert g == pytest.approx(1.0 if i == j else 0.0, abs=0.02)


def test_projection_and_fit_recover_coefficients():
    mesh = mh.icosphere(2)
    truth = [0.0] * sh.n_coefficients(3)
    truth[sh.sh_index(0, 0)] = 1.5
    truth[sh.sh_index(2, -1)] = -0.7
    truth[sh.sh_index(3, 2)] = 0.4
    exp = sh.SHExpansion(3, truth)
    dirs, w = sh.sphere_quadrature(mesh)
    values = [exp.evaluate(d) for d in dirs]
    fitted = sh.fit(dirs, values, 3)
    assert fitted.coefficients == pytest.approx(truth, abs=1e-9)
    projected = sh.project(dirs, values, w, 3)
    assert projected.coefficients == pytest.approx(truth, abs=0.03)
    assert exp.band_energies()[2] == pytest.approx(0.49)
    with pytest.raises(ValueError):
        sh.SHExpansion(2, [0.0] * 4)


def test_fit_needs_enough_samples():
    with pytest.raises(mh.linalg.LinAlgError):
        sh.fit([(0, 0, 1), (1, 0, 0)], [1.0, 2.0], 2)
    assert sh.fit([(0, 0, 1), (1, 0, 0)], [1.0, 2.0], 2, regularization=1e-3).lmax == 2


def test_shape_descriptor_rotation_invariant():
    mesh = mh.icosphere(3)
    verts = [(1.5 * x, y, 0.8 * z) for x, y, z in mesh.vertices]  # ellipsoid
    ell = mh.TriMesh(verts, mesh.faces)
    rot = sh.rotate_mesh(ell, sh.rotation_matrix((1, 2, 0.5), 0.9))
    d1 = sh.shape_descriptor(ell, lmax=4)
    d2 = sh.shape_descriptor(rot, lmax=4)
    assert d1 == pytest.approx(d2, abs=0.01)
    assert d1[2] > 0.05 and d1[1] < 0.01  # ellipsoids are even functions
    sphere_desc = sh.shape_descriptor(mh.icosphere(2), lmax=4)
    assert max(sphere_desc[1:]) < 1e-3
    fitted = sh.radial_expansion(ell, 4, method="fit")
    projected = sh.radial_expansion(ell, 4, method="project")
    dirs, _ = sh.sphere_quadrature(ell)
    radii = [math.dist(p, ell.centroid()) for p in ell.vertices]

    def max_err(e):
        return max(abs(e.evaluate(d) - r) for d, r in zip(dirs, radii))

    assert max_err(fitted) < max_err(projected) and max_err(fitted) < 0.05


def test_index_reference_and_angles():
    assert all(sh.sh_degree_order(sh.sh_index(l, m)) == (l, m) for l in range(6) for m in range(-l, l + 1))
    assert sh.sphere_laplacian_eigenvalues(2) == [0, 2, 2, 2, 6, 6, 6, 6, 6]
    r, theta, phi = sh.cartesian_to_spherical(0, 3, 0)
    assert (r, theta, phi) == pytest.approx((3, math.pi / 2, math.pi / 2))
    with pytest.raises(ValueError):
        sh.sh_basis(2, (0, 0, 0))
