"""Triangle meshes, half-edge connectivity and topology diagnostics.

:class:`TriMesh` is the central container: an indexed face set with cached
geometric quantities.  :meth:`TriMesh.analyze` produces a
:class:`TopologyReport` covering manifoldness, orientation, boundary loops,
connected components, Euler characteristic and valence statistics.
:class:`HalfEdgeMesh` provides ordered one-ring traversal for meshes that
are edge-manifold and consistently oriented.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .linalg import add3, cross3, dot3, norm3, scale3, sub3

Point = Tuple[float, float, float]
Face = Tuple[int, int, int]
Edge = Tuple[int, int]

_DEGENERATE_AREA = 1e-14


class MeshError(ValueError):
    """Raised for structurally invalid meshes (bad indices, repeated corners)."""


class NonManifoldError(MeshError):
    """Raised when an operation requires a manifold, consistently oriented mesh."""


def edge_key(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class ValenceStats:
    """Vertex valence (edge degree) statistics."""

    minimum: int
    maximum: int
    mean: float
    histogram: Dict[int, int]
    irregular_interior: int
    irregular_boundary: int

    def as_dict(self) -> dict:
        out = asdict(self)
        out["histogram"] = dict(sorted(self.histogram.items()))
        return out


@dataclass
class TopologyReport:
    """Summary of the combinatorial structure of a triangle mesh."""

    n_vertices: int
    n_faces: int
    n_edges: int
    n_boundary_edges: int
    n_boundary_loops: int
    n_components: int
    euler_characteristic: int
    genus: Optional[int]
    is_edge_manifold: bool
    is_vertex_manifold: bool
    is_consistently_oriented: bool
    is_orientable: bool
    is_closed: bool
    non_manifold_edges: List[Edge] = field(default_factory=list)
    non_manifold_vertices: List[int] = field(default_factory=list)
    isolated_vertices: List[int] = field(default_factory=list)
    degenerate_faces: List[int] = field(default_factory=list)
    duplicate_faces: List[int] = field(default_factory=list)
    valence: Optional[ValenceStats] = None

    @property
    def is_manifold(self) -> bool:
        return self.is_edge_manifold and self.is_vertex_manifold

    @property
    def is_watertight(self) -> bool:
        return self.is_closed and self.is_manifold and self.is_consistently_oriented

    def as_dict(self) -> dict:
        """JSON-friendly dictionary (dataclass fields plus derived flags)."""
        out = asdict(self)
        out["valence"] = self.valence.as_dict() if self.valence else None
        out.update(is_manifold=self.is_manifold, is_watertight=self.is_watertight)
        return out

    def summary_lines(self) -> List[str]:
        def yn(flag: bool) -> str:
            return "yes" if flag else "no"

        lines = [
            "vertices            %d" % self.n_vertices,
            "faces               %d" % self.n_faces,
            "edges               %d" % self.n_edges,
            "boundary edges      %d (%d loops)" % (self.n_boundary_edges, self.n_boundary_loops),
            "components          %d" % self.n_components,
            "euler char          %d" % self.euler_characteristic,
            "genus               %s" % ("n/a" if self.genus is None else self.genus),
            "manifold            %s (edges: %s, vertices: %s)"
            % (yn(self.is_manifold), yn(self.is_edge_manifold), yn(self.is_vertex_manifold)),
            "oriented            %s (orientable: %s)"
            % (yn(self.is_consistently_oriented), yn(self.is_orientable)),
            "closed              %s" % yn(self.is_closed),
        ]
        if self.valence is not None:
            v = self.valence
            lines.append("valence             min %d / mean %.3f / max %d" % (v.minimum, v.mean, v.maximum))
            lines.append("irregular vertices  %d interior, %d boundary"
                         % (v.irregular_interior, v.irregular_boundary))
        for label, items in (
            ("isolated vertices", self.isolated_vertices),
            ("degenerate faces", self.degenerate_faces),
            ("duplicate faces", self.duplicate_faces),
            ("non-manifold edges", self.non_manifold_edges),
            ("non-manifold verts", self.non_manifold_vertices),
        ):
            if items:
                lines.append("%-19s %d" % (label, len(items)))
        return lines


class TriMesh:
    """Indexed triangle mesh with cached geometry.

    Parameters
    ----------
    vertices:
        Sequence of 3D points.
    faces:
        Sequence of vertex index triples.  Orientation is taken as given;
        counter-clockwise faces seen from outside give outward normals.
    validate:
        Check index ranges and reject faces with repeated corners.
    """

    def __init__(self, vertices: Iterable[Sequence[float]], faces: Iterable[Sequence[int]],
                 validate: bool = True) -> None:
        self.vertices: List[Point] = []
        for k, v in enumerate(vertices):
            if len(v) != 3:
                raise MeshError("vertex %d does not have three coordinates" % k)
            p = (float(v[0]), float(v[1]), float(v[2]))
            if validate and not all(math.isfinite(c) for c in p):
                raise MeshError("vertex %d has non-finite coordinates" % k)
            self.vertices.append(p)
        self.faces: List[Face] = []
        n = len(self.vertices)
        for k, f in enumerate(faces):
            if len(f) != 3:
                raise MeshError("face %d is not a triangle" % k)
            tri = (int(f[0]), int(f[1]), int(f[2]))
            if validate:
                for idx in tri:
                    if not 0 <= idx < n:
                        raise MeshError("face %d references vertex %d (have %d)" % (k, idx, n))
                if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                    raise MeshError("face %d repeats a vertex: %r" % (k, tri))
            self.faces.append(tri)
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # basics
    # ------------------------------------------------------------------
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def __repr__(self) -> str:
        return "TriMesh(vertices=%d, faces=%d)" % (self.n_vertices, self.n_faces)

    def _cached(self, key: str, builder):
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    # ------------------------------------------------------------------
    # connectivity
    # ------------------------------------------------------------------
    def edge_faces(self) -> Dict[Edge, List[int]]:
        """Map each undirected edge to the faces containing it."""

        def build() -> Dict[Edge, List[int]]:
            out: Dict[Edge, List[int]] = {}
            for fi, (a, b, c) in enumerate(self.faces):
                for i, j in ((a, b), (b, c), (c, a)):
                    out.setdefault(edge_key(i, j), []).append(fi)
            return out

        return self._cached("edge_faces", build)

    def edges(self) -> List[Edge]:
        return self._cached("edges", lambda: sorted(self.edge_faces()))

    @property
    def n_edges(self) -> int:
        return len(self.edge_faces())

    def boundary_edges(self) -> List[Edge]:
        """Directed boundary edges, oriented as they appear in their face."""

        def build() -> List[Edge]:
            ef = self.edge_faces()
            out = []
            for fi, (a, b, c) in enumerate(self.faces):
                for i, j in ((a, b), (b, c), (c, a)):
                    if len(ef[edge_key(i, j)]) == 1:
                        out.append((i, j))
            return sorted(out)

        return self._cached("boundary_edges", build)

    def boundary_vertices(self) -> List[int]:
        return sorted({v for e in self.boundary_edges() for v in e})

    def boundary_loops(self) -> List[List[int]]:
        """Ordered vertex loops along the boundary.

        Loops follow the face orientation where it is consistent and fall
        back to undirected boundary edges otherwise (e.g. on a Moebius
        strip).  At non-manifold boundary vertices the smallest unused
        neighbour is taken, so every boundary edge is visited exactly once.
        """
        directed = self.boundary_edges()
        succ: Dict[int, List[int]] = {}
        adj: Dict[int, List[int]] = {}
        for i, j in directed:
            succ.setdefault(i, []).append(j)
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)
        unused = {edge_key(i, j) for i, j in directed}

        def step(cur: int) -> Optional[int]:
            cands = [w for w in succ.get(cur, ()) if edge_key(cur, w) in unused]
            if not cands:
                cands = [w for w in adj[cur] if edge_key(cur, w) in unused]
            return min(cands) if cands else None

        loops: List[List[int]] = []
        for start in sorted(adj):
            while True:
                nxt = step(start)
                if nxt is None:
                    break
                loop, cur = [start], start
                while nxt is not None:
                    unused.discard(edge_key(cur, nxt))
                    if nxt == start:
                        break
                    loop.append(nxt)
                    cur = nxt
                    nxt = step(cur)
                loops.append(loop)
        return loops

    def vertex_faces(self) -> List[List[int]]:
        def build() -> List[List[int]]:
            out: List[List[int]] = [[] for _ in range(self.n_vertices)]
            for fi, f in enumerate(self.faces):
                for v in f:
                    out[v].append(fi)
            return out

        return self._cached("vertex_faces", build)

    def vertex_adjacency(self) -> List[List[int]]:
        def build() -> List[List[int]]:
            nbrs: List[set] = [set() for _ in range(self.n_vertices)]
            for i, j in self.edge_faces():
                nbrs[i].add(j)
                nbrs[j].add(i)
            return [sorted(s) for s in nbrs]

        return self._cached("adjacency", build)

    def valences(self) -> List[int]:
        return [len(n) for n in self.vertex_adjacency()]

    def valence_stats(self) -> ValenceStats:
        vals = self.valences()
        used = [v for v, f in zip(vals, self.vertex_faces()) if f]
        boundary = set(self.boundary_vertices())
        irr_int = sum(1 for i, v in enumerate(vals)
                      if self.vertex_faces()[i] and i not in boundary and v != 6)
        irr_bnd = sum(1 for i in boundary if vals[i] != 4)
        if not used:
            return ValenceStats(0, 0, 0.0, {}, 0, 0)
        return ValenceStats(min(used), max(used), sum(used) / len(used), dict(Counter(used)),
                            irr_int, irr_bnd)

    def connected_components(self) -> List[List[int]]:
        """Vertex sets of the face-connected components (isolated vertices excluded)."""
        uf = _UnionFind(self.n_vertices)
        for a, b, c in self.faces:
            uf.union(a, b)
            uf.union(a, c)
        groups: Dict[int, List[int]] = {}
        vf = self.vertex_faces()
        for v in range(self.n_vertices):
            if vf[v]:
                groups.setdefault(uf.find(v), []).append(v)
        return sorted(groups.values(), key=lambda g: g[0])

    def euler_characteristic(self) -> int:
        used = sum(1 for f in self.vertex_faces() if f)
        return used - self.n_edges + self.n_faces

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def face_normal_unnormalized(self, fi: int) -> Point:
        a, b, c = self.faces[fi]
        p = self.vertices
        return cross3(sub3(p[b], p[a]), sub3(p[c], p[a]))

    def face_areas(self) -> List[float]:
        return self._cached("face_areas", lambda: [
            0.5 * norm3(self.face_normal_unnormalized(f)) for f in range(self.n_faces)])

    def face_normals(self) -> List[Point]:
        def build() -> List[Point]:
            out = []
            for f in range(self.n_faces):
                n = self.face_normal_unnormalized(f)
                length = norm3(n)
                out.append(scale3(1.0 / length, n) if length > 0.0 else (0.0, 0.0, 0.0))
            return out

        return self._cached("face_normals", build)

    def surface_area(self) -> float:
        return math.fsum(self.face_areas())

    def corner_angles(self) -> List[Tuple[float, float, float]]:
        """Interior angle at each corner of each face, in face vertex order."""

        def angle(p: Point, q: Point, r: Point) -> float:
            u, v = sub3(q, p), sub3(r, p)
            return math.atan2(norm3(cross3(u, v)), dot3(u, v))

        def build():
            p = self.vertices
            return [(angle(p[a], p[b], p[c]), angle(p[b], p[c], p[a]), angle(p[c], p[a], p[b]))
                    for a, b, c in self.faces]

        return self._cached("corner_angles", build)

    def angle_defects(self) -> List[float]:
        """Discrete Gaussian curvature (integrated): ``2*pi - sum(angles)``.

        Boundary vertices use ``pi`` instead of ``2*pi`` (geodesic curvature
        of the boundary).  By Gauss-Bonnet the sum equals ``2*pi*chi``.
        """
        totals = [0.0] * self.n_vertices
        for (a, b, c), (ta, tb, tc) in zip(self.faces, self.corner_angles()):
            totals[a] += ta
            totals[b] += tb
            totals[c] += tc
        boundary = set(self.boundary_vertices())
        vf = self.vertex_faces()
        out = []
        for v, t in enumerate(totals):
            if not vf[v]:
                out.append(0.0)
            else:
                out.append((math.pi if v in boundary else 2.0 * math.pi) - t)
        return out

    def vertex_normals(self, weighting: str = "area") -> List[Point]:
        """Per-vertex normals; ``weighting`` is ``"area"``, ``"angle"`` or ``"uniform"``."""
        if weighting not in ("area", "angle", "uniform"):
            raise ValueError("unknown weighting %r" % weighting)
        acc = [(0.0, 0.0, 0.0)] * self.n_vertices
        normals = self.face_normals()
        areas = self.face_areas()
        angles = self.corner_angles() if weighting == "angle" else None
        for fi, f in enumerate(self.faces):
            n = normals[fi]
            for corner, v in enumerate(f):
                if weighting == "area":
                    w = areas[fi]
                elif weighting == "angle":
                    w = angles[fi][corner]
                else:
                    w = 1.0
                acc[v] = add3(acc[v], scale3(w, n))
        out = []
        for n in acc:
            length = norm3(n)
            out.append(scale3(1.0 / length, n) if length > 0.0 else (0.0, 0.0, 0.0))
        return out

    def edge_lengths(self) -> Dict[Edge, float]:
        p = self.vertices
        return self._cached("edge_lengths", lambda: {
            e: norm3(sub3(p[e[0]], p[e[1]])) for e in self.edge_faces()})

    def mean_edge_length(self) -> float:
        lengths = self.edge_lengths()
        return math.fsum(lengths.values()) / len(lengths) if lengths else 0.0

    def bounding_box(self) -> Tuple[Point, Point]:
        if not self.vertices:
            raise MeshError("empty mesh has no bounding box")
        xs, ys, zs = zip(*self.vertices)
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def centroid(self) -> Point:
        """Area-weighted surface centroid (vertex mean if the mesh has no area)."""
        total = self.surface_area()
        if total <= 0.0:
            n = max(self.n_vertices, 1)
            xs, ys, zs = zip(*self.vertices) if self.vertices else ((0.0,), (0.0,), (0.0,))
            return (sum(xs) / n, sum(ys) / n, sum(zs) / n)
        cx = cy = cz = 0.0
        p = self.vertices
        for (a, b, c), area in zip(self.faces, self.face_areas()):
            cx += area * (p[a][0] + p[b][0] + p[c][0])
            cy += area * (p[a][1] + p[b][1] + p[c][1])
            cz += area * (p[a][2] + p[b][2] + p[c][2])
        s = 1.0 / (3.0 * total)
        return (cx * s, cy * s, cz * s)

    def signed_volume(self) -> float:
        """Signed enclosed volume (meaningful for closed, oriented meshes)."""
        p = self.vertices
        return math.fsum(dot3(p[a], cross3(p[b], p[c])) for a, b, c in self.faces) / 6.0

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def degenerate_faces(self, tol: float = _DEGENERATE_AREA) -> List[int]:
        scale = self.mean_edge_length() ** 2 or 1.0
        return [i for i, a in enumerate(self.face_areas()) if a <= tol * scale]

    def duplicate_faces(self) -> List[int]:
        seen: Dict[Tuple[int, int, int], int] = {}
        dups = []
        for fi, f in enumerate(self.faces):
            key = tuple(sorted(f))
            if key in seen:
                dups.append(fi)
            else:
                seen[key] = fi
        return dups

    def non_manifold_edges(self) -> List[Edge]:
        return sorted(e for e, fs in self.edge_faces().items() if len(fs) > 2)

    def non_manifold_vertices(self) -> List[int]:
        """Vertices whose link is not a single path or cycle ("bow-tie" vertices)."""
        links: List[List[Edge]] = [[] for _ in range(self.n_vertices)]
        for a, b, c in self.faces:
            links[a].append((b, c))
            links[b].append((c, a))
            links[c].append((a, b))
        bad = []
        for v, link in enumerate(links):
            if not link:
                continue
            deg: Counter = Counter()
            nodes = {}
            for i, j in link:
                deg[i] += 1
                deg[j] += 1
                nodes.setdefault(i, len(nodes))
                nodes.setdefault(j, len(nodes))
            uf = _UnionFind(len(nodes))
            for i, j in link:
                uf.union(nodes[i], nodes[j])
            components = len({uf.find(k) for k in range(len(nodes))})
            ends = sum(1 for d in deg.values() if d == 1)
            if components != 1 or any(d > 2 for d in deg.values()) or ends not in (0, 2):
                bad.append(v)
        return bad

    def is_consistently_oriented(self) -> bool:
        directed = Counter()
        for a, b, c in self.faces:
            directed[(a, b)] += 1
            directed[(b, c)] += 1
            directed[(c, a)] += 1
        return all(n == 1 for n in directed.values())

    def _orientation_flips(self) -> Optional[List[bool]]:
        """Per-face flip flags making orientation consistent, or ``None`` if impossible."""
        ef = self.edge_faces()
        flip: List[Optional[bool]] = [None] * self.n_faces

        def has_directed(fi: int, i: int, j: int, flipped: bool) -> bool:
            a, b, c = self.faces[fi]
            pairs = ((a, b), (b, c), (c, a))
            if flipped:
                pairs = tuple((y, x) for x, y in pairs)
            return (i, j) in pairs

        for seed in range(self.n_faces):
            if flip[seed] is not None:
                continue
            flip[seed] = False
            queue = deque([seed])
            while queue:
                fi = queue.popleft()
                a, b, c = self.faces[fi]
                for i, j in ((a, b), (b, c), (c, a)):
                    if flip[fi]:
                        i, j = j, i
                    fs = ef[edge_key(i, j)]
                    if len(fs) != 2:
                        continue
                    other = fs[0] if fs[1] == fi else fs[1]
                    # neighbour must traverse the shared edge as (j, i)
                    want = not has_directed(other, j, i, False)
                    if flip[other] is None:
                        flip[other] = want
                        queue.append(other)
                    elif flip[other] != want:
                        return None
        return [bool(f) for f in flip]

    def is_orientable(self) -> bool:
        return self._orientation_flips() is not None

    def analyze(self) -> TopologyReport:
        """Compute a full :class:`TopologyReport`."""
        nm_edges = self.non_manifold_edges()
        nm_verts = self.non_manifold_vertices()
        vf = self.vertex_faces()
        boundary = self.boundary_edges()
        components = self.connected_components()
        chi = self.euler_characteristic()
        loops = self.boundary_loops()
        is_closed = not boundary
        edge_manifold = not nm_edges
        orientable = edge_manifold and self.is_orientable()
        genus: Optional[int] = None
        if edge_manifold and not nm_verts and orientable and len(components) == 1:
            # chi = 2 - 2g - b for a connected orientable surface with b boundary loops
            g2 = 2 - chi - len(loops)
            genus = g2 // 2 if g2 >= 0 and g2 % 2 == 0 else None
        return TopologyReport(
            n_vertices=self.n_vertices,
            n_faces=self.n_faces,
            n_edges=self.n_edges,
            n_boundary_edges=len(boundary),
            n_boundary_loops=len(loops),
            n_components=len(components),
            euler_characteristic=chi,
            genus=genus,
            is_edge_manifold=edge_manifold,
            is_vertex_manifold=not nm_verts,
            is_consistently_oriented=self.is_consistently_oriented(),
            is_orientable=orientable,
            is_closed=is_closed,
            non_manifold_edges=nm_edges,
            non_manifold_vertices=nm_verts,
            isolated_vertices=[v for v in range(self.n_vertices) if not vf[v]],
            degenerate_faces=self.degenerate_faces(),
            duplicate_faces=self.duplicate_faces(),
            valence=self.valence_stats(),
        )

    # ------------------------------------------------------------------
    # derived meshes
    # ------------------------------------------------------------------
    def halfedge(self) -> "HalfEdgeMesh":
        return self._cached("halfedge", lambda: HalfEdgeMesh(self.n_vertices, self.faces))

    def flipped(self) -> "TriMesh":
        return TriMesh(self.vertices, [(a, c, b) for a, b, c in self.faces], validate=False)

    def oriented(self) -> "TriMesh":
        """Copy with faces flipped so that orientation is consistent per component."""
        if self.non_manifold_edges():
            raise NonManifoldError("cannot orient a mesh with non-manifold edges")
        flips = self._orientation_flips()
        if flips is None:
            raise NonManifoldError("mesh is not orientable")
        faces = [(a, c, b) if fl else (a, b, c) for (a, b, c), fl in zip(self.faces, flips)]
        return TriMesh(self.vertices, faces, validate=False)

    def transformed(self, scale: float = 1.0, translate: Sequence[float] = (0.0, 0.0, 0.0)) -> "TriMesh":
        tx, ty, tz = translate
        verts = [(scale * x + tx, scale * y + ty, scale * z + tz) for x, y, z in self.vertices]
        return TriMesh(verts, self.faces, validate=False)

    def normalized(self, mode: str = "area") -> "TriMesh":
        """Center at the centroid and rescale to unit surface area or unit bbox diagonal."""
        cx, cy, cz = self.centroid()
        centered = self.transformed(1.0, (-cx, -cy, -cz))
        if mode == "area":
            area = centered.surface_area()
            s = 1.0 / math.sqrt(area) if area > 0 else 1.0
        elif mode == "bbox":
            lo, hi = centered.bounding_box()
            diag = norm3(sub3(hi, lo))
            s = 1.0 / diag if diag > 0 else 1.0
        else:
            raise ValueError("mode must be 'area' or 'bbox'")
        return centered.transformed(s)

    def without_unreferenced_vertices(self) -> Tuple["TriMesh", List[int]]:
        """Drop vertices not used by any face; returns the mesh and old->new map (-1 = removed)."""
        used = sorted({v for f in self.faces for v in f})
        mapping = [-1] * self.n_vertices
        for new, old in enumerate(used):
            mapping[old] = new
        verts = [self.vertices[i] for i in used]
        faces = [(mapping[a], mapping[b], mapping[c]) for a, b, c in self.faces]
        return TriMesh(verts, faces, validate=False), mapping

    def cleaned(self) -> "TriMesh":
        """Remove degenerate and duplicate faces, then unreferenced vertices."""
        drop = set(self.degenerate_faces()) | set(self.duplicate_faces())
        faces = [f for i, f in enumerate(self.faces) if i not in drop]
        return TriMesh(self.vertices, faces, validate=False).without_unreferenced_vertices()[0]


class HalfEdgeMesh:
    """Half-edge connectivity for an edge-manifold, consistently oriented mesh.

    Half-edge ``3*f + k`` runs from corner ``k`` to corner ``k+1`` of face
    ``f``.  Boundary half-edges have ``twin == -1``.  For boundary vertices
    the stored outgoing half-edge is the boundary one, so a counter-clockwise
    sweep from it visits the whole fan.
    """

    def __init__(self, n_vertices: int, faces: Sequence[Face]) -> None:
        nh = 3 * len(faces)
        self.n_vertices = n_vertices
        self.n_faces = len(faces)
        self.origin = [0] * nh
        self.target = [0] * nh
        self.twin = [-1] * nh
        directed: Dict[Edge, int] = {}
        for fi, f in enumerate(faces):
            for k in range(3):
                h = 3 * fi + k
                i, j = f[k], f[(k + 1) % 3]
                self.origin[h] = i
                self.target[h] = j
                if (i, j) in directed:
                    raise NonManifoldError(
                        "directed edge (%d, %d) used twice: inconsistent orientation "
                        "or non-manifold edge" % (i, j))
                directed[(i, j)] = h
        for (i, j), h in directed.items():
            t = directed.get((j, i))
            if t is not None:
                self.twin[h] = t
        self.vertex_halfedge = [-1] * n_vertices
        for h in range(nh):
            v = self.origin[h]
            if self.vertex_halfedge[v] == -1 or self.twin[h] == -1:
                self.vertex_halfedge[v] = h
        for v in range(n_vertices):
            h0 = self.vertex_halfedge[v]
            if h0 == -1:
                continue
            count = sum(1 for _ in self.outgoing(v))
            total = sum(1 for h in range(nh) if self.origin[h] == v)
            if count != total:
                raise NonManifoldError("vertex %d is non-manifold (fan has %d of %d faces)"
                                       % (v, count, total))

    @property
    def n_halfedges(self) -> int:
        return len(self.origin)

    @staticmethod
    def next(h: int) -> int:
        return h - h % 3 + (h + 1) % 3

    @staticmethod
    def prev(h: int) -> int:
        return h - h % 3 + (h + 2) % 3

    @staticmethod
    def face(h: int) -> int:
        return h // 3

    def is_boundary_vertex(self, v: int) -> bool:
        h = self.vertex_halfedge[v]
        return h != -1 and self.twin[h] == -1

    def outgoing(self, v: int) -> Iterator[int]:
        """Outgoing half-edges of ``v`` in counter-clockwise order."""
        h0 = self.vertex_halfedge[v]
        if h0 == -1:
            return
        h = h0
        while True:
            yield h
            t = self.twin[self.prev(h)]
            if t == -1 or t == h0:
                return
            h = t

    def one_ring(self, v: int) -> List[int]:
        """Neighbouring vertices of ``v`` in counter-clockwise order."""
        hs = list(self.outgoing(v))
        ring = [self.target[h] for h in hs]
        if hs and self.is_boundary_vertex(v):
            ring.append(self.origin[self.prev(hs[-1])])
        return ring

    def vertex_faces(self, v: int) -> List[int]:
        return [self.face(h) for h in self.outgoing(v)]
