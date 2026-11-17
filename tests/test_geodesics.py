import math

import pytest

import meshharmonics as mh
from meshharmonics.geodesics import (compare_distances, face_gradients, vertex_divergence)
from meshharmonics.laplacian import cotangent_laplacian

from conftest import great_circle


def test_gradient_of_linear_function_is_exact():
    g = mh.grid(3, 3)
    u = [2 * x + 5 * y for x, y, _ in g.vertices]
    for grad in face_gradients(g, u):
        assert grad == pytest.approx((2, 5, 0))


def test_divergence_of_gradient_is_minus_laplacian(sphere):
    u = [x * y + z for x, y, z in sphere.vertices]
    div = vertex_divergence(sphere, face_gradients(sphere, u))
    lu = cotangent_laplacian(sphere).matvec(u)
    assert div == pytest.approx([-v for v in lu], abs=1e-10)


def test_dijkstra_and_path(square):
    dist, pred = mh.dijkstra_distances(square, 0)
    assert dist == pytest.approx([0, 1, math.sqrt(2), 1])
    assert mh.shortest_edge_path(square, 0, 2) == [0, 2]
    capped, _ = mh.dijkstra_distances(square, 0, max_distance=1.0)
    assert math.isinf(capped[2])
    with pytest.raises(IndexError):
        mh.dijkstra_distances(square, 9)


def test_heat_method_on_sphere(sphere):
    solver = mh.HeatGeodesicSolver(sphere)
    d = solver.distance(0)
    ref = [great_circle(sphere.vertices[0], p) for p in sphere.vertices]
    heat_err = compare_distances(d, ref, ignore=[0])
    graph_err = compare_distances(mh.dijkstra_distances(sphere, 0)[0], ref, ignore=[0])
    assert d[0] == 0.0
    assert heat_err.mean_relative < 0.06 and heat_err.max_absolute < 0.15
    assert heat_err.mean_relative < graph_err.mean_relative
    antipode = max(range(sphere.n_vertices), key=lambda v: ref[v])
    assert d[antipode] == pytest.approx(math.pi, rel=0.05)


def test_heat_method_on_plane_with_boundary():
    g = mh.grid(12, 12)
    center = 6 * 13 + 6
    cg = mh.HeatGeodesicSolver(g, solver="cg").distance([center])
    d = mh.heat_geodesic_distances(g, center)
    assert d == pytest.approx(cg, abs=1e-6)
    ref = [math.dist(p, g.vertices[center]) for p in g.vertices]
    assert compare_distances(d, ref, ignore=[center]).mean_relative < 0.08


def test_multiple_sources_and_dispatch(sphere):
    north = max(range(sphere.n_vertices), key=lambda v: sphere.vertices[v][2])
    south = min(range(sphere.n_vertices), key=lambda v: sphere.vertices[v][2])
    d = mh.geodesic_distances(sphere, [north, south])
    equator = min(range(sphere.n_vertices), key=lambda v: abs(sphere.vertices[v][2]))
    assert d[equator] == pytest.approx(math.pi / 2, rel=0.08)
    dj = mh.geodesic_distances(sphere, north, method="dijkstra")
    assert dj[north] == 0.0
    with pytest.raises(ValueError):
        mh.geodesic_distances(sphere, 0, method="exact")
    with pytest.raises(ValueError):
        mh.HeatGeodesicSolver(sphere, time_factor=0)


def test_farthest_point_sampling(sphere):
    samples = mh.farthest_point_sampling(sphere, 4)
    assert len(set(samples)) == 4 and samples[0] == 0
    a, b = sphere.vertices[samples[0]], sphere.vertices[samples[1]]
    assert great_circle(a, b) == pytest.approx(math.pi, abs=0.05)
    heat = mh.farthest_point_sampling(mh.grid(4, 4), 3, method="heat")
    assert len(heat) == 3
    with pytest.raises(ValueError):
        mh.farthest_point_sampling(sphere, 0)
