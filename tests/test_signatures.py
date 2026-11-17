import math

import pytest

import meshharmonics as mh
from meshharmonics import signatures as sig


def test_default_times_span(sphere_eig):
    times = sig.default_hks_times(sphere_eig, 5)
    assert len(times) == 5 and times == sorted(times)
    assert times[0] == pytest.approx(4 * math.log(10) / max(sphere_eig.values))
    assert times[-1] == pytest.approx(4 * math.log(10) / sphere_eig.values[1])


def test_hks_on_sphere_is_nearly_constant(sphere_eig):
    hks = sig.heat_kernel_signature(sphere_eig, [0.1, 0.5])
    for col in range(2):
        vals = [row[col] for row in hks]
        assert (max(vals) - min(vals)) / max(vals) < 0.05
    # integral of k_t(x, x) over the surface is the heat trace
    mass = mh.mass_matrix(mh.icosphere(2))
    total = sum(m * row[0] for m, row in zip(mass, hks))
    assert total == pytest.approx(sig.heat_trace(sphere_eig, 0.1), rel=1e-6)
    assert sig.heat_kernel(sphere_eig, 3, 3, 0.1) == pytest.approx(hks[3][0])


def test_hks_normalizations(sphere_eig):
    traced = sig.heat_kernel_signature(sphere_eig, [0.2], normalize="trace")
    raw = sig.heat_kernel_signature(sphere_eig, [0.2])
    assert traced[0][0] == pytest.approx(raw[0][0] / sig.heat_trace(sphere_eig, 0.2))
    unit = sig.heat_kernel_signature(sphere_eig, [0.2, 0.4], normalize="l2")
    assert math.hypot(*unit[5]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sig.heat_kernel_signature(sphere_eig, normalize="max")


def test_hks_distinguishes_ellipsoid_tips():
    base = mh.icosphere(2)
    ell = mh.TriMesh([(2.0 * x, y, z) for x, y, z in base.vertices], base.faces)
    eig = mh.laplacian_eigenpairs(ell, k=12)
    hks = sig.heat_kernel_signature(eig, [0.05])
    tip = max(range(ell.n_vertices), key=lambda v: ell.vertices[v][0])
    waist = min(range(ell.n_vertices), key=lambda v: abs(ell.vertices[v][0]))
    assert hks[tip][0] > hks[waist][0]  # heat lingers at high-curvature points
    other_tip = min(range(ell.n_vertices), key=lambda v: ell.vertices[v][0])
    assert other_tip in sig.most_similar_vertices(hks, tip, count=8)
    assert sig.descriptor_distances(hks, tip, metric="log_l1")[tip] == 0.0


def test_wks_gps_and_dna(sphere_eig, sphere):
    wks = sig.wave_kernel_signature(sphere_eig, 6)
    assert len(wks) == sphere.n_vertices and len(wks[0]) == 6
    gps = sig.global_point_signature(sphere_eig, k=3)
    # GPS coordinates of the l=1 band are linear functions: norms ~ constant
    norms = [math.sqrt(sum(c * c for c in row)) for row in gps]
    assert (max(norms) - min(norms)) / max(norms) < 0.05
    dna = sig.shape_dna(sphere_eig, k=3, area=sphere.surface_area())
    assert dna == pytest.approx([2 * 4 * math.pi] * 3, rel=0.03)
    assert sig.diffusion_distance(sphere_eig, 0, 0, 0.1) == 0.0
    assert sig.diffusion_distance(sphere_eig, 0, 50, 0.1) > 0.0


def test_signature_errors():
    trivial = mh.EigenResult([0.0], [[1.0, 1.0]])
    with pytest.raises(ValueError):
        sig.default_hks_times(trivial)
    with pytest.raises(ValueError):
        sig.wave_kernel_signature(trivial)
    with pytest.raises(ValueError):
        sig.descriptor_distances([[1.0]], 0, metric="cos")
