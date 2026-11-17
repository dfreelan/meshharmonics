import math

import pytest

import meshharmonics as mh
from meshharmonics.mesh import HalfEdgeMesh, MeshError, NonManifoldError

CLOSED = ["tetrahedron", "octahedron", "cube", "icosahedron", "icosphere", "torus"]


@pytest.mark.parametrize("name", CLOSED)
def test_closed_primitives_are_watertight_and_outward(name):
    mesh = getattr(mh, name)()
    report = mesh.analyze()
    assert report.is_watertight and report.n_boundary_edges == 0
    assert mesh.signed_volume() > 0
    assert report.genus == (1 if name == "torus" else 0)
    # Gauss-Bonnet: total angle defect = 2 pi chi
    assert sum(mesh.angle_defects()) == pytest.approx(2 * math.pi * report.euler_characteristic)


def test_validation_errors():
    with pytest.raises(MeshError):
        mh.TriMesh([(0, 0, 0)], [(0, 1, 2)])
    with pytest.raises(MeshError):
        mh.TriMesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 1)])
    with pytest.raises(MeshError):
        mh.TriMesh([(0, 0, float("nan"))], [])
    with pytest.raises(MeshError):
        mh.TriMesh([(0, 0)], [])


def test_square_geometry(square):
    assert square.surface_area() == pytest.approx(1.0)
    assert square.n_edges == 5
    assert square.boundary_edges() == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert square.boundary_loops() == [[0, 1, 2, 3]]
    assert square.centroid() == pytest.approx((0.5, 0.5, 0.0))
    for n in square.vertex_normals("angle"):
        assert n == pytest.approx((0, 0, 1))
    angles = square.corner_angles()[0]
    assert sum(angles) == pytest.approx(math.pi)
    assert square.bounding_box() == ((0, 0, 0), (1, 1, 0))
    with pytest.raises(ValueError):
        square.vertex_normals("bogus")


def test_grid_valence_and_boundary():
    g = mh.grid(4, 4)
    report = g.analyze()
    assert report.n_boundary_loops == 1 and report.n_boundary_edges == 16
    assert report.euler_characteristic == 1 and report.genus == 0
    v = report.valence
    assert v.minimum == 3 and v.maximum == 8
    assert sum(v.histogram.values()) == g.n_vertices
    assert v.mean == pytest.approx(2 * g.n_edges / g.n_vertices)


def test_icosphere_valence_stats(sphere):
    stats = sphere.valence_stats()
    assert stats.histogram == {5: 12, 6: sphere.n_vertices - 12}
    assert stats.irregular_interior == 12 and stats.irregular_boundary == 0


def test_cylinder_has_two_boundary_loops():
    report = mh.cylinder(8, 3).analyze()
    assert report.n_boundary_loops == 2 and report.euler_characteristic == 0
    assert report.genus == 0


def test_bowtie_vertex_detected():
    mesh = mh.TriMesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)],
                      [(0, 1, 2), (1, 3, 4)])
    report = mesh.analyze()
    assert report.non_manifold_vertices == [1]
    assert report.is_edge_manifold and not report.is_manifold
    with pytest.raises(NonManifoldError):
        mesh.halfedge()


def test_non_manifold_edge_detected():
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    mesh = mh.TriMesh(verts, [(0, 1, 2), (1, 0, 3), (0, 1, 4)])
    report = mesh.analyze()
    assert report.non_manifold_edges == [(0, 1)]
    assert not report.is_edge_manifold and report.genus is None
    with pytest.raises(NonManifoldError):
        mesh.oriented()


def test_orientation_repair():
    cube = mh.cube()
    faces = list(cube.faces)
    faces[3] = (faces[3][0], faces[3][2], faces[3][1])
    broken = mh.TriMesh(cube.vertices, faces)
    assert not broken.is_consistently_oriented() and broken.is_orientable()
    fixed = broken.oriented()
    assert fixed.is_consistently_oriented()
    assert abs(fixed.signed_volume()) == pytest.approx(1.0)
    assert mh.cube().flipped().signed_volume() == pytest.approx(-1.0)


def test_mobius_strip_is_not_orientable():
    n = 8
    verts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        for s in (-0.3, 0.3):
            verts.append(((1 + s * math.cos(t / 2)) * math.cos(t),
                          (1 + s * math.cos(t / 2)) * math.sin(t), s * math.sin(t / 2)))
    faces = []
    for i in range(n):
        a, b = 2 * i, 2 * i + 1
        if i + 1 < n:
            c, d = 2 * i + 2, 2 * i + 3
        else:
            c, d = 1, 0  # half twist
        faces += [(a, b, d), (a, d, c)]
    report = mh.TriMesh(verts, faces).analyze()
    assert not report.is_orientable and report.n_boundary_loops == 1
    assert report.euler_characteristic == 0


def test_components_isolated_and_cleaning():
    a = mh.tetrahedron()
    verts = list(a.vertices) + [tuple(x + 5 for x in p) for p in a.vertices] + [(9, 9, 9)]
    faces = list(a.faces) + [tuple(i + 4 for i in f) for f in a.faces] + [a.faces[0]]
    mesh = mh.TriMesh(verts, faces)
    report = mesh.analyze()
    assert report.n_components == 2 and report.isolated_vertices == [8]
    assert report.duplicate_faces == [8] and report.genus is None
    clean = mesh.cleaned()
    assert clean.n_vertices == 8 and clean.n_faces == 8
    flat = mh.TriMesh([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0)], [(0, 1, 2), (0, 1, 3)])
    assert flat.analyze().degenerate_faces == [0]


def test_halfedge_one_rings(sphere):
    he = sphere.halfedge()
    assert he.n_halfedges == 3 * sphere.n_faces
    adj = sphere.vertex_adjacency()
    for v in range(sphere.n_vertices):
        ring = he.one_ring(v)
        assert sorted(ring) == adj[v]
        assert len(he.vertex_faces(v)) == len(ring)
    grid = mh.grid(3, 3)
    ghe = grid.halfedge()
    corner = ghe.one_ring(0)
    assert ghe.is_boundary_vertex(0) and sorted(corner) == grid.vertex_adjacency()[0]
    with pytest.raises(NonManifoldError):
        HalfEdgeMesh(3, [(0, 1, 2), (0, 1, 2)])


def test_normalized_and_transforms(sphere):
    unit = sphere.transformed(3.0, (1, 2, 3)).normalized("area")
    assert unit.surface_area() == pytest.approx(1.0)
    assert unit.centroid() == pytest.approx((0, 0, 0), abs=1e-12)
    lo, hi = sphere.normalized("bbox").bounding_box()
    assert math.dist(lo, hi) == pytest.approx(1.0)
    report_lines = sphere.analyze().summary_lines()
    assert any("genus" in line for line in report_lines)
