import math
import pathlib

import pytest

import meshharmonics as mh

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    return DATA


@pytest.fixture(scope="session")
def sphere():
    return mh.icosphere(2)


@pytest.fixture(scope="session")
def sphere_eig(sphere):
    return mh.laplacian_eigenpairs(sphere, k=16)


@pytest.fixture
def square():
    """Two triangles forming the unit square."""
    return mh.TriMesh([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [(0, 1, 2), (0, 2, 3)])


def great_circle(p, q):
    return math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(p, q)))))
