import math

import pytest

import meshharmonics as mh
from meshharmonics.laplacian import cotangent_weights, laplacian_pair, mass_matrix


def test_cotangent_weights_on_square(square):
    w = cotangent_weights(square)
    # right isosceles triangles: diagonal weight 0, sides 1/2 (single cot(45)=1 halved)
    assert w[(0, 2)] == pytest.approx(0.0, abs=1e-12)
    assert w[(0, 1)] == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["icosphere", "torus", "grid"])
def test_stiffness_properties(name):
    mesh = getattr(mh, name)()
    L = mh.cotangent_laplacian(mesh)
    assert L.is_symmetric()
    assert max(abs(s) for s in L.row_sums()) < 1e-12
    assert L.matvec([1.0] * mesh.n_vertices) == pytest.approx([0.0] * mesh.n_vertices, abs=1e-12)
    assert all(d > 0 for d in L.diagonal())


def test_dirichlet_energy_of_linear_function_on_plane():
    g = mh.grid(5, 5, 2.0, 1.0)
    pair = laplacian_pair(g)
    u = [3.0 * x - y for x, y, _ in g.vertices]
    # |grad u|^2 = 10 over an area of 2  ->  integral = 20, energy = 10
    assert pair.dirichlet_energy(u) == pytest.approx(10.0)
    lap = pair.apply(u)
    interior = [v for v in range(g.n_vertices) if v not in set(g.boundary_vertices())]
    assert max(abs(lap[v]) for v in interior) < 1e-10


@pytest.mark.parametrize("kind", ["barycentric", "voronoi"])
def test_mass_sums_to_area(kind, sphere):
    m = mass_matrix(sphere, kind)
    assert sum(m) == pytest.approx(sphere.surface_area())
    assert all(x > 0 for x in m)
    obtuse = mh.TriMesh([(0, 0, 0), (4, 0, 0), (2, 0.5, 0)], [(0, 1, 2)])
    assert sum(mass_matrix(obtuse, kind)) == pytest.approx(obtuse.surface_area())


def test_uniform_laplacian_is_graph_laplacian(square):
    for bad in (lambda: mass_matrix(square, "consistent"), lambda: laplacian_pair(square, operator="x")):
        with pytest.raises(ValueError):
            bad()
    tet = mh.tetrahedron()
    L = mh.uniform_laplacian(tet).to_dense()
    assert L == [[3.0 if i == j else -1.0 for j in range(4)] for i in range(4)]


def test_mean_curvature_on_sphere():
    r = 2.0
    hn = mh.mean_curvature_normals(mh.icosphere(3, radius=r))
    mags = [math.sqrt(sum(c * c for c in v)) for v in hn]
    assert sum(mags) / len(mags) == pytest.approx(1.0 / r, rel=0.02)


def test_isolated_vertices_rejected():
    mesh = mh.TriMesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (5, 5, 5)], [(0, 1, 2)])
    with pytest.raises(ValueError):
        laplacian_pair(mesh)


def test_heat_operator_spd(sphere):
    pair = laplacian_pair(sphere)
    x = [math.sin(i) for i in range(sphere.n_vertices)]
    assert pair.heat_operator(0.01).quadratic_form(x) > 0 and pair.inner(x, x) > 0
