"""Discrete Laplace-Beltrami operators on triangle meshes.

Conventions
-----------
The *stiffness* matrix ``L`` returned by :func:`cotangent_laplacian` is
symmetric positive semi-definite::

    L[i, j] = -(cot(alpha_ij) + cot(beta_ij)) / 2      (i != j, edge ij)
    L[i, i] = -sum_j L[i, j]

so that ``u^T L u`` approximates the Dirichlet energy ``∫ |∇u|^2``.  The
Laplace-Beltrami eigenproblem is the generalized problem
``L phi = lambda M phi`` with a (lumped, diagonal) mass matrix ``M``; on the
unit sphere the eigenvalues approach ``l (l + 1)``.  Boundaries receive
natural (Neumann) conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .linalg import CSRMatrix, cross3, dot3, norm3, sub3
from .mesh import Edge, TriMesh, edge_key

MASS_KINDS = ("barycentric", "voronoi")


def face_cotangents(mesh: TriMesh) -> List[Tuple[float, float, float]]:
    """Cotangent of the interior angle at each corner of each face.

    Degenerate (zero-area) corners get cotangent 0 so that they do not inject
    infinite weights; inspect :meth:`TriMesh.degenerate_faces` beforehand if
    that matters for your data.
    """
    p = mesh.vertices
    out = []
    for a, b, c in mesh.faces:
        cots = []
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            u, v = sub3(p[j], p[i]), sub3(p[k], p[i])
            s = norm3(cross3(u, v))
            cots.append(dot3(u, v) / s if s > 1e-300 else 0.0)
        out.append((cots[0], cots[1], cots[2]))
    return out


def cotangent_weights(mesh: TriMesh) -> Dict[Edge, float]:
    """Symmetric edge weights ``w_ij = (cot alpha + cot beta) / 2``.

    Boundary edges receive the single available cotangent halved.
    """
    weights: Dict[Edge, float] = {}
    for (a, b, c), (ca, cb, cc) in zip(mesh.faces, face_cotangents(mesh)):
        # the corner at a is opposite edge (b, c), and so on
        for (i, j), cot in (((b, c), ca), ((c, a), cb), ((a, b), cc)):
            key = edge_key(i, j)
            weights[key] = weights.get(key, 0.0) + 0.5 * cot
    return weights


def laplacian_from_weights(n: int, weights: Dict[Edge, float]) -> CSRMatrix:
    rows, cols, vals = [], [], []
    diag = [0.0] * n
    for (i, j), w in weights.items():
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((-w, -w))
        diag[i] += w
        diag[j] += w
    rows.extend(range(n))
    cols.extend(range(n))
    vals.extend(diag)
    return CSRMatrix.from_triplets(n, n, rows, cols, vals, drop_zeros=False)


def cotangent_laplacian(mesh: TriMesh, clamp_negative: bool = False) -> CSRMatrix:
    """Symmetric PSD cotangent stiffness matrix (see module docstring).

    ``clamp_negative=True`` zeroes negative edge weights (which occur on
    non-Delaunay edges) so that the operator satisfies a maximum principle;
    this changes the spectrum slightly but can help heat-diffusion stability.
    """
    weights = cotangent_weights(mesh)
    if clamp_negative:
        weights = {e: max(w, 0.0) for e, w in weights.items()}
    return laplacian_from_weights(mesh.n_vertices, weights)


def uniform_laplacian(mesh: TriMesh) -> CSRMatrix:
    """Combinatorial graph Laplacian ``D - A`` of the edge graph."""
    return laplacian_from_weights(mesh.n_vertices, {e: 1.0 for e in mesh.edge_faces()})


def mass_matrix(mesh: TriMesh, kind: str = "voronoi") -> List[float]:
    """Diagonal (lumped) mass matrix as a list of vertex areas.

    ``"barycentric"`` assigns one third of each face area to each corner.
    ``"voronoi"`` uses the mixed Voronoi areas of Meyer et al. (2003), which
    fall back to barycentric-like splits for obtuse triangles.  Both sum to
    the total surface area.
    """
    if kind not in MASS_KINDS:
        raise ValueError("kind must be one of %s" % (MASS_KINDS,))
    n = mesh.n_vertices
    out = [0.0] * n
    areas = mesh.face_areas()
    if kind == "barycentric":
        for (a, b, c), area in zip(mesh.faces, areas):
            third = area / 3.0
            out[a] += third
            out[b] += third
            out[c] += third
        return out
    p = mesh.vertices
    for (a, b, c), area, (ca, cb, cc) in zip(mesh.faces, areas, face_cotangents(mesh)):
        if area <= 0.0:
            continue
        corners = ((a, b, c, ca), (b, c, a, cb), (c, a, b, cc))
        obtuse = [cot < 0.0 for _, _, _, cot in corners]
        if any(obtuse):
            for (v, _, _, _), is_obtuse in zip(corners, obtuse):
                out[v] += area / 2.0 if is_obtuse else area / 4.0
            continue
        # Voronoi region: (|PR|^2 cot Q + |PQ|^2 cot R) / 8 for vertex P
        for v, q, r, _ in corners:
            cot_q = (ca, cb, cc)[(a, b, c).index(q)]
            cot_r = (ca, cb, cc)[(a, b, c).index(r)]
            pr = sub3(p[r], p[v])
            pq = sub3(p[q], p[v])
            out[v] += (dot3(pr, pr) * cot_q + dot3(pq, pq) * cot_r) / 8.0
    return out


@dataclass
class LaplacianPair:
    """Stiffness matrix ``L`` together with the lumped mass ``M`` (diagonal)."""

    stiffness: CSRMatrix
    mass: List[float]

    @property
    def n(self) -> int:
        return self.stiffness.nrows

    def apply(self, u: Sequence[float]) -> List[float]:
        """Pointwise Laplace-Beltrami ``-Δu ≈ M^{-1} L u`` (positive convention)."""
        lu = self.stiffness.matvec(u)
        return [x / m if m > 0 else 0.0 for x, m in zip(lu, self.mass)]

    def dirichlet_energy(self, u: Sequence[float]) -> float:
        return 0.5 * self.stiffness.quadratic_form(u)

    def inner(self, u: Sequence[float], v: Sequence[float]) -> float:
        """Mass-weighted ``L2`` inner product ``u^T M v``."""
        return math.fsum(a * m * b for a, m, b in zip(u, self.mass, v))

    def heat_operator(self, t: float) -> CSRMatrix:
        """Backward-Euler heat matrix ``M + t L`` (SPD for ``t > 0`` on a valid mesh)."""
        return self.stiffness.scaled(t).add_diagonal(self.mass)


def laplacian_pair(mesh: TriMesh, mass: str = "voronoi", operator: str = "cotangent") -> LaplacianPair:
    """Build the ``(L, M)`` pair for a mesh.

    ``operator`` is ``"cotangent"`` (geometric) or ``"uniform"`` (graph).
    """
    if operator == "cotangent":
        stiffness = cotangent_laplacian(mesh)
    elif operator == "uniform":
        stiffness = uniform_laplacian(mesh)
    else:
        raise ValueError("operator must be 'cotangent' or 'uniform'")
    masses = mass_matrix(mesh, mass) if operator == "cotangent" else [1.0] * mesh.n_vertices
    if operator == "cotangent":
        isolated = [i for i, f in enumerate(mesh.vertex_faces()) if not f]
        if isolated:
            raise ValueError("mesh has %d isolated vertices; remove them first "
                             "(TriMesh.without_unreferenced_vertices)" % len(isolated))
    return LaplacianPair(stiffness, masses)


def mean_curvature_normals(mesh: TriMesh) -> List[Tuple[float, float, float]]:
    """Discrete mean-curvature normal ``H n = (M^{-1} L x) / 2`` per vertex.

    The magnitude is the mean curvature ``H`` (1/r on a sphere of radius r);
    for outward-oriented closed meshes the vector points outward.
    """
    pair = laplacian_pair(mesh)
    coords = list(zip(*mesh.vertices)) if mesh.vertices else [(), (), ()]
    comps = [pair.apply(list(c)) for c in coords]
    return [(0.5 * x, 0.5 * y, 0.5 * z) for x, y, z in zip(*comps)]
