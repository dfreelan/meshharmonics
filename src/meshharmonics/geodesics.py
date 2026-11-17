"""Approximate geodesic distances on triangle meshes.

* :func:`dijkstra_distances` -- exact shortest paths in the edge graph.  An
  upper bound on the true geodesic distance; error depends on the mesh.
* :class:`HeatGeodesicSolver` -- the heat method of Crane, Weischedel &
  Wardetzky (2013): diffuse heat for a short time, normalize its gradient,
  and recover distance from a Poisson equation.  Both linear systems are
  factored once, so additional source sets are cheap.
* :func:`farthest_point_sampling` -- well-spread vertex subsets, a common
  preprocessing step for geometric learning.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .laplacian import face_cotangents, laplacian_pair
from .linalg import SkylineCholesky, conjugate_gradient, cross3, dot3, norm3, scale3, sub3
from .mesh import TriMesh

Sources = Union[int, Iterable[int]]
Vec3 = Tuple[float, float, float]


def _as_sources(sources: Sources, n: int) -> List[int]:
    out = [sources] if isinstance(sources, int) else list(sources)
    if not out:
        raise ValueError("at least one source vertex is required")
    for s in out:
        if not 0 <= s < n:
            raise IndexError("source vertex %d out of range" % s)
    return out


def dijkstra_distances(mesh: TriMesh, sources: Sources, max_distance: Optional[float] = None
                       ) -> Tuple[List[float], List[int]]:
    """Edge-graph shortest path lengths and predecessor array (-1 at sources/unreached)."""
    n = mesh.n_vertices
    srcs = _as_sources(sources, n)
    adj = mesh.vertex_adjacency()
    lengths = mesh.edge_lengths()
    dist = [math.inf] * n
    pred = [-1] * n
    heap: List[Tuple[float, int]] = []
    for s in srcs:
        dist[s] = 0.0
        heap.append((0.0, s))
    heapq.heapify(heap)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if max_distance is not None and d > max_distance:
            dist[u] = math.inf
            continue
        for w in adj[u]:
            nd = d + lengths[(u, w) if u < w else (w, u)]
            if nd < dist[w]:
                dist[w] = nd
                pred[w] = u
                heapq.heappush(heap, (nd, w))
    if max_distance is not None:
        for v in range(n):
            if dist[v] > max_distance:
                dist[v] = math.inf
                pred[v] = -1
    return dist, pred


def shortest_edge_path(mesh: TriMesh, source: int, target: int) -> List[int]:
    """Vertex sequence of an edge-graph shortest path (empty if unreachable)."""
    dist, pred = dijkstra_distances(mesh, source)
    if math.isinf(dist[target]):
        return []
    path = [target]
    while path[-1] != source:
        path.append(pred[path[-1]])
    path.reverse()
    return path


def face_gradients(mesh: TriMesh, u: Sequence[float]) -> List[Vec3]:
    """Gradient of the piecewise-linear interpolant of ``u`` in every face."""
    p = mesh.vertices
    out = []
    for a, b, c in mesh.faces:
        n = cross3(sub3(p[b], p[a]), sub3(p[c], p[a]))
        twice_area = norm3(n)
        if twice_area <= 0.0:
            out.append((0.0, 0.0, 0.0))
            continue
        nu = scale3(1.0 / twice_area, n)
        gx = gy = gz = 0.0
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            e = sub3(p[k], p[j])  # edge opposite corner i, counter-clockwise
            r = cross3(nu, e)
            gx += u[i] * r[0]
            gy += u[i] * r[1]
            gz += u[i] * r[2]
        out.append((gx / twice_area, gy / twice_area, gz / twice_area))
    return out


def vertex_divergence(mesh: TriMesh, field: Sequence[Vec3]) -> List[float]:
    """Integrated divergence of a per-face vector field at each vertex.

    Uses ``div_i = 1/2 sum_f cot θ_1 (e_1 · X_f) + cot θ_2 (e_2 · X_f)``, the
    discretization adjoint to :func:`face_gradients` under the cotangent
    Laplacian (``div(grad u) = -L u``).
    """
    p = mesh.vertices
    div = [0.0] * mesh.n_vertices
    for (a, b, c), cots, x in zip(mesh.faces, face_cotangents(mesh), field):
        cot = {a: cots[0], b: cots[1], c: cots[2]}
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            e1 = sub3(p[j], p[i])  # opposite corner k
            e2 = sub3(p[k], p[i])  # opposite corner j
            div[i] += 0.5 * (cot[k] * dot3(e1, x) + cot[j] * dot3(e2, x))
    return div


@dataclass
class HeatGeodesicSolver:
    """Prefactored heat-method geodesic distance solver.

    Parameters
    ----------
    mesh:
        A triangle mesh (boundaries allowed; Neumann conditions are used).
    time_factor:
        Diffusion time is ``time_factor * h^2`` with ``h`` the mean edge
        length; ``1.0`` is the value recommended by the original paper.
    mass:
        Lumped mass kind, ``"voronoi"`` or ``"barycentric"``.
    solver:
        ``"cholesky"`` (factor once, fast repeated queries) or ``"cg"``.
    """

    mesh: TriMesh
    time_factor: float = 1.0
    mass: str = "voronoi"
    solver: str = "cholesky"

    def __post_init__(self) -> None:
        if self.time_factor <= 0:
            raise ValueError("time_factor must be positive")
        if self.solver not in ("cholesky", "cg"):
            raise ValueError("solver must be 'cholesky' or 'cg'")
        pair = laplacian_pair(self.mesh, mass=self.mass)
        h = self.mesh.mean_edge_length()
        self.t = self.time_factor * h * h
        self._pair = pair
        self._heat = pair.heat_operator(self.t)
        # Tiny mass regularization makes the Neumann Poisson system definite.
        scale = sum(pair.stiffness.diagonal()) / max(sum(pair.mass), 1e-300)
        self._eps = 1e-10 * scale
        self._poisson = pair.stiffness.add_diagonal([self._eps * m for m in pair.mass])
        if self.solver == "cholesky":
            self._solve_heat = SkylineCholesky(self._heat).solve
            self._solve_poisson = SkylineCholesky(self._poisson).solve
        else:
            self._solve_heat = lambda b: conjugate_gradient(self._heat, b, tol=1e-12).x
            self._solve_poisson = lambda b: conjugate_gradient(self._poisson, b, tol=1e-12).x

    def distance(self, sources: Sources) -> List[float]:
        """Approximate geodesic distance from the given source vertex set."""
        n = self.mesh.n_vertices
        srcs = _as_sources(sources, n)
        delta = [0.0] * n
        for s in srcs:
            delta[s] = 1.0
        u = self._solve_heat(delta)
        grads = face_gradients(self.mesh, u)
        field = []
        for g in grads:
            length = norm3(g)
            field.append(scale3(-1.0 / length, g) if length > 0.0 else (0.0, 0.0, 0.0))
        div = vertex_divergence(self.mesh, field)
        # L phi = -div; project onto the range of L (orthogonal to constants)
        rhs = [-d for d in div]
        mean = sum(rhs) / n
        rhs = [r - mean for r in rhs]
        phi = self._solve_poisson(rhs)
        offset = sum(phi[s] for s in srcs) / len(srcs)
        return [max(0.0, x - offset) for x in phi]

    __call__ = distance


def heat_geodesic_distances(mesh: TriMesh, sources: Sources, time_factor: float = 1.0) -> List[float]:
    """One-shot convenience wrapper around :class:`HeatGeodesicSolver`."""
    return HeatGeodesicSolver(mesh, time_factor=time_factor).distance(sources)


def geodesic_distances(mesh: TriMesh, sources: Sources, method: str = "heat", **kwargs) -> List[float]:
    """Dispatch to ``"heat"`` or ``"dijkstra"``."""
    if method == "heat":
        return heat_geodesic_distances(mesh, sources, **kwargs)
    if method == "dijkstra":
        return dijkstra_distances(mesh, sources, **kwargs)[0]
    raise ValueError("method must be 'heat' or 'dijkstra'")


def farthest_point_sampling(mesh: TriMesh, count: int, seed_vertex: int = 0,
                            method: str = "dijkstra") -> List[int]:
    """Greedy farthest-point sampling under the chosen distance."""
    n = mesh.n_vertices
    if not 1 <= count <= n:
        raise ValueError("count must be between 1 and the number of vertices")
    solver = HeatGeodesicSolver(mesh) if method == "heat" else None

    def dist_from(v: int) -> List[float]:
        if solver is not None:
            return solver.distance(v)
        return dijkstra_distances(mesh, v)[0]

    samples = [seed_vertex]
    nearest = dist_from(seed_vertex)
    while len(samples) < count:
        nxt = max(range(n), key=lambda v: (nearest[v] if not math.isinf(nearest[v]) else 1e300, -v))
        if nearest[nxt] <= 0.0:
            break
        samples.append(nxt)
        d = dist_from(nxt)
        nearest = [min(a, b) for a, b in zip(nearest, d)]
    return samples


@dataclass
class DistanceError:
    mean_absolute: float
    max_absolute: float
    mean_relative: float


def compare_distances(approx: Sequence[float], reference: Sequence[float],
                      ignore: Iterable[int] = ()) -> DistanceError:
    """Error statistics of ``approx`` against ``reference`` (skipping ``ignore`` and zeros)."""
    skip = set(ignore)
    abs_err, rel_err = [], []
    for v, (a, r) in enumerate(zip(approx, reference)):
        if v in skip or math.isinf(r):
            continue
        e = abs(a - r)
        abs_err.append(e)
        if r > 0.0:
            rel_err.append(e / r)
    if not abs_err:
        raise ValueError("no comparable entries")
    return DistanceError(sum(abs_err) / len(abs_err), max(abs_err),
                         sum(rel_err) / len(rel_err) if rel_err else 0.0)
