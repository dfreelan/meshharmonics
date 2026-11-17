"""Procedural meshes useful for testing and benchmarking spectral methods."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .linalg import norm3, scale3
from .mesh import Edge, TriMesh, edge_key


def tetrahedron() -> TriMesh:
    """Regular tetrahedron inscribed in the unit sphere, outward oriented."""
    s = 1.0 / math.sqrt(3.0)
    verts = [(s, s, s), (s, -s, -s), (-s, s, -s), (-s, -s, s)]
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    return TriMesh(verts, faces)


def octahedron() -> TriMesh:
    verts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    return TriMesh(verts, faces)


def cube(size: float = 1.0) -> TriMesh:
    """Axis-aligned cube centred at the origin, two triangles per side."""
    h = 0.5 * size
    verts = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
             (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    faces = []
    for a, b, c, d in quads:
        faces.append((a, b, c))
        faces.append((a, c, d))
    return TriMesh(verts, faces)


def icosahedron(radius: float = 1.0) -> TriMesh:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
           (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
           (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [scale3(radius / norm3(p), p) for p in raw]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    return TriMesh(verts, faces)


def subdivide_midpoint(mesh: TriMesh) -> TriMesh:
    """1-to-4 midpoint subdivision (each triangle split into four)."""
    verts = list(mesh.vertices)
    midpoint: Dict[Edge, int] = {}

    def mid(i: int, j: int) -> int:
        key = edge_key(i, j)
        if key not in midpoint:
            p, q = verts[i], verts[j]
            midpoint[key] = len(verts)
            verts.append(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2))
        return midpoint[key]

    faces: List[Tuple[int, int, int]] = []
    for a, b, c in mesh.faces:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
    return TriMesh(verts, faces, validate=False)


def icosphere(subdivisions: int = 2, radius: float = 1.0) -> TriMesh:
    """Geodesic sphere: icosahedron refined ``subdivisions`` times and projected.

    Vertex count is ``10 * 4**subdivisions + 2``.
    """
    if subdivisions < 0:
        raise ValueError("subdivisions must be non-negative")
    mesh = icosahedron()
    for _ in range(subdivisions):
        mesh = subdivide_midpoint(mesh)
        mesh = TriMesh([scale3(1.0 / norm3(p), p) for p in mesh.vertices], mesh.faces, validate=False)
    return TriMesh([scale3(radius, p) for p in mesh.vertices], mesh.faces)


def grid(nx: int = 8, ny: int = 8, width: float = 1.0, height: float = 1.0,
         alternate: bool = True) -> TriMesh:
    """Planar ``nx`` x ``ny`` cell grid in the z=0 plane with a single boundary loop.

    With ``alternate=True`` the diagonals alternate direction ("criss-cross"
    pattern), which avoids a directional bias in graph distances.
    """
    if nx < 1 or ny < 1:
        raise ValueError("grid needs at least one cell in each direction")
    verts = [(width * i / nx, height * j / ny, 0.0) for j in range(ny + 1) for i in range(nx + 1)]

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    faces = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            if alternate and (i + j) % 2:
                faces.extend([(a, b, d), (b, c, d)])
            else:
                faces.extend([(a, b, c), (a, c, d)])
    return TriMesh(verts, faces)


def torus(n_major: int = 16, n_minor: int = 8, major_radius: float = 1.0,
          minor_radius: float = 0.35) -> TriMesh:
    """Closed torus of genus one."""
    if n_major < 3 or n_minor < 3:
        raise ValueError("torus needs at least 3 segments in each direction")
    verts = []
    for i in range(n_major):
        u = 2.0 * math.pi * i / n_major
        for j in range(n_minor):
            v = 2.0 * math.pi * j / n_minor
            r = major_radius + minor_radius * math.cos(v)
            verts.append((r * math.cos(u), r * math.sin(u), minor_radius * math.sin(v)))

    def vid(i: int, j: int) -> int:
        return (i % n_major) * n_minor + (j % n_minor)

    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            a, b = vid(i, j), vid(i + 1, j)
            c, d = vid(i + 1, j + 1), vid(i, j + 1)
            faces.extend([(a, b, c), (a, c, d)])
    return TriMesh(verts, faces)


def cylinder(n_around: int = 16, n_height: int = 4, radius: float = 1.0,
             height: float = 2.0) -> TriMesh:
    """Open tube (two boundary loops), axis along z."""
    verts = []
    for j in range(n_height + 1):
        z = height * j / n_height - height / 2.0
        for i in range(n_around):
            a = 2.0 * math.pi * i / n_around
            verts.append((radius * math.cos(a), radius * math.sin(a), z))

    def vid(i: int, j: int) -> int:
        return j * n_around + (i % n_around)

    faces = []
    for j in range(n_height):
        for i in range(n_around):
            a, b = vid(i, j), vid(i + 1, j)
            c, d = vid(i + 1, j + 1), vid(i, j + 1)
            faces.extend([(a, b, c), (a, c, d)])
    return TriMesh(verts, faces)


PRIMITIVES = {
    "tetrahedron": tetrahedron,
    "octahedron": octahedron,
    "cube": cube,
    "icosahedron": icosahedron,
    "icosphere": icosphere,
    "grid": grid,
    "torus": torus,
    "cylinder": cylinder,
}
