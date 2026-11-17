import pytest

import meshharmonics as mh
from meshharmonics.io import MeshParseError


def test_obj_quads_with_attributes(data_dir):
    mesh = mh.load_mesh(data_dir / "quad_square.obj")
    assert mesh.n_vertices == 6 and mesh.n_faces == 4
    assert mesh.surface_area() == pytest.approx(2.0)
    assert mesh.analyze().n_boundary_loops == 1


def test_off_fixtures(data_dir):
    tet = mh.load_mesh(data_dir / "tetra.off")
    assert tet.analyze().is_watertight and tet.signed_volume() > 0
    fin = mh.load_mesh(data_dir / "fin.off")
    assert fin.analyze().non_manifold_edges == [(0, 1)]
    bow = mh.load_mesh(data_dir / "bowtie.obj")
    assert bow.analyze().non_manifold_vertices == [1]


def test_negative_indices_comments_and_continuations():
    text = "v 0 0 0 # origin\nv 1 0 0\nv 0 1 \\\n 0\nf -3 -2 -1\n# end\n"
    mesh = mh.loads(text, "obj")
    assert mesh.faces == [(0, 1, 2)] and mesh.vertices[2] == (0.0, 1.0, 0.0)


@pytest.mark.parametrize("fmt", ["obj", "off"])
def test_roundtrip(fmt, tmp_path):
    mesh = mh.torus(6, 4)
    path = tmp_path / ("t." + fmt)
    mh.save_mesh(mesh, path)
    back = mh.load_mesh(path)
    assert back.faces == mesh.faces
    for p, q in zip(back.vertices, mesh.vertices):
        assert p == pytest.approx(q, abs=1e-8)
    assert mh.loads(mh.dumps(mesh, fmt), fmt).n_faces == mesh.n_faces


def test_off_variants():
    coff = "COFF\n3 1 0\n0 0 0 255 0 0 255\n1 0 0 0 255 0 255\n0 1 0 0 0 255 255\n3 0 1 2 1 1 1\n"
    assert mh.loads(coff, "off").n_faces == 1
    inline = "OFF 4 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n"
    assert mh.loads(inline, "off").faces == [(0, 1, 2), (0, 2, 3)]


@pytest.mark.parametrize("text,fmt,line", [
    ("v 0 0\n", "obj", 1),
    ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n", "obj", 4),
    ("v 0 0 0\nf 0 1 2\n", "obj", 2),
    ("v a b c\n", "obj", 1),
    ("OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 5\n", "off", 6),
    ("OFF\n3 1 0\n0 0 0\n", "off", None),
    ("OFF\nx y\n", "off", 2),
    ("", "off", None),
])
def test_parse_errors_report_location(text, fmt, line):
    with pytest.raises(MeshParseError) as info:
        mh.loads(text, fmt)
    assert info.value.line == line


def test_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        mh.load_mesh(tmp_path / "mesh.ply")
