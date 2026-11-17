"""Readers and writers for Wavefront OBJ and OFF triangle meshes.

Polygonal faces are fan-triangulated on load.  Parsers report the file name
and line number of malformed input via :class:`MeshParseError`.
"""

from __future__ import annotations

import io as _io
import os
from typing import IO, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from .mesh import TriMesh

PathLike = Union[str, "os.PathLike[str]"]


class MeshParseError(ValueError):
    """Malformed mesh file."""

    def __init__(self, message: str, source: str = "<stream>", line: Optional[int] = None) -> None:
        self.source = source
        self.line = line
        where = source if line is None else "%s:%d" % (source, line)
        super().__init__("%s: %s" % (where, message))


def triangulate_polygon(indices: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Fan-triangulate a polygon given as vertex indices (n >= 3)."""
    if len(indices) < 3:
        raise ValueError("a polygon needs at least three vertices")
    first = indices[0]
    return [(first, indices[k], indices[k + 1]) for k in range(1, len(indices) - 1)]


def _logical_lines(stream: IO[str]) -> Iterator[Tuple[int, str]]:
    """Yield ``(line_number, text)`` joining backslash continuations."""
    buf = ""
    start = 0
    for lineno, raw in enumerate(stream, 1):
        line = raw.rstrip("\r\n")
        if not buf:
            start = lineno
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        yield start, buf + line
        buf = ""
    if buf:
        yield start, buf


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------


def _parse_obj_index(token: str, count: int, source: str, lineno: int) -> int:
    head = token.split("/", 1)[0]
    try:
        idx = int(head)
    except ValueError:
        raise MeshParseError("bad face index %r" % token, source, lineno) from None
    if idx > 0:
        resolved = idx - 1
    elif idx < 0:
        resolved = count + idx
    else:
        raise MeshParseError("OBJ indices are 1-based; got 0", source, lineno)
    if not 0 <= resolved < count:
        raise MeshParseError("face index %d out of range (%d vertices so far)" % (idx, count),
                             source, lineno)
    return resolved


def read_obj(stream: IO[str], source: str = "<stream>") -> TriMesh:
    """Parse an OBJ stream; only ``v`` and ``f`` records are interpreted.

    Texture/normal indices (``f 1/2/3``) are accepted and ignored, negative
    (relative) indices are supported, and lines, points (``p``) and other
    statements are skipped.
    """
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    for lineno, line in _logical_lines(stream):
        hash_pos = line.find("#")
        if hash_pos >= 0:
            line = line[:hash_pos]
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "v":
            if len(parts) < 4:
                raise MeshParseError("vertex needs three coordinates", source, lineno)
            try:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                raise MeshParseError("bad vertex coordinates", source, lineno) from None
        elif tag == "f":
            if len(parts) < 4:
                raise MeshParseError("face needs at least three vertices", source, lineno)
            idx = [_parse_obj_index(t, len(vertices), source, lineno) for t in parts[1:]]
            if len(set(idx)) != len(idx):
                raise MeshParseError("face repeats a vertex", source, lineno)
            faces.extend(triangulate_polygon(idx))
    return TriMesh(vertices, faces)


def write_obj(mesh: TriMesh, stream: IO[str], precision: int = 9,
              header: Optional[str] = "written by meshharmonics") -> None:
    fmt = "v %%.%dg %%.%dg %%.%dg\n" % (precision, precision, precision)
    if header:
        for line in header.splitlines():
            stream.write("# %s\n" % line)
    for x, y, z in mesh.vertices:
        stream.write(fmt % (x, y, z))
    for a, b, c in mesh.faces:
        stream.write("f %d %d %d\n" % (a + 1, b + 1, c + 1))


# ---------------------------------------------------------------------------
# OFF
# ---------------------------------------------------------------------------


def _off_tokens(stream: IO[str]) -> Iterator[Tuple[int, List[str]]]:
    for lineno, line in _logical_lines(stream):
        hash_pos = line.find("#")
        if hash_pos >= 0:
            line = line[:hash_pos]
        parts = line.split()
        if parts:
            yield lineno, parts


def read_off(stream: IO[str], source: str = "<stream>") -> TriMesh:
    """Parse an OFF stream (``OFF``, ``COFF``, ``NOFF``, ``CNOFF`` headers).

    Per-vertex colours/normals and per-face colours are ignored.  The counts
    may follow the keyword on the same line.  Polygons are fan-triangulated.
    """
    rows = _off_tokens(stream)
    try:
        lineno, parts = next(rows)
    except StopIteration:
        raise MeshParseError("empty file", source) from None
    keyword = parts[0].upper()
    extra_per_vertex = 0
    if keyword.endswith("OFF") and set(keyword[:-3]) <= set("STCN4"):
        if "4" in keyword or "ST" in keyword:
            raise MeshParseError("unsupported OFF variant %r" % parts[0], source, lineno)
        if "N" in keyword[:-3]:
            extra_per_vertex += 3
        if "C" in keyword[:-3]:
            extra_per_vertex += 4  # RGBA (3 also accepted below)
        parts = parts[1:]
        if not parts:
            try:
                lineno, parts = next(rows)
            except StopIteration:
                raise MeshParseError("missing element counts", source, lineno) from None
    try:
        nv, nf = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        raise MeshParseError("bad element counts %r" % " ".join(parts), source, lineno) from None
    if nv < 0 or nf < 0:
        raise MeshParseError("negative element counts", source, lineno)
    vertices: List[Tuple[float, float, float]] = []
    for _ in range(nv):
        try:
            lineno, parts = next(rows)
        except StopIteration:
            raise MeshParseError("expected %d vertices, found %d" % (nv, len(vertices)), source) from None
        if len(parts) < 3:
            raise MeshParseError("vertex needs three coordinates", source, lineno)
        if extra_per_vertex and len(parts) not in (3, 3 + extra_per_vertex, 2 + extra_per_vertex):
            raise MeshParseError("unexpected vertex record length %d" % len(parts), source, lineno)
        try:
            vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            raise MeshParseError("bad vertex coordinates", source, lineno) from None
    faces: List[Tuple[int, int, int]] = []
    for k in range(nf):
        try:
            lineno, parts = next(rows)
        except StopIteration:
            raise MeshParseError("expected %d faces, found %d" % (nf, k), source) from None
        try:
            count = int(parts[0])
            idx = [int(t) for t in parts[1 : 1 + count]]
        except ValueError:
            raise MeshParseError("bad face record", source, lineno) from None
        if count < 3 or len(idx) != count:
            raise MeshParseError("face record has wrong vertex count", source, lineno)
        for i in idx:
            if not 0 <= i < nv:
                raise MeshParseError("face index %d out of range" % i, source, lineno)
        if len(set(idx)) != len(idx):
            raise MeshParseError("face repeats a vertex", source, lineno)
        faces.extend(triangulate_polygon(idx))
    return TriMesh(vertices, faces)


def write_off(mesh: TriMesh, stream: IO[str], precision: int = 9) -> None:
    fmt = "%%.%dg %%.%dg %%.%dg\n" % (precision, precision, precision)
    stream.write("OFF\n%d %d %d\n" % (mesh.n_vertices, mesh.n_faces, mesh.n_edges))
    for x, y, z in mesh.vertices:
        stream.write(fmt % (x, y, z))
    for a, b, c in mesh.faces:
        stream.write("3 %d %d %d\n" % (a, b, c))


# ---------------------------------------------------------------------------
# dispatch helpers
# ---------------------------------------------------------------------------

_READERS = {".obj": read_obj, ".off": read_off}
_WRITERS = {".obj": write_obj, ".off": write_off}


def _format_for(path: PathLike, fmt: Optional[str], table: dict) -> str:
    ext = ("." + fmt.lower().lstrip(".")) if fmt else os.path.splitext(os.fspath(path))[1].lower()
    if ext not in table:
        raise ValueError("unsupported mesh format %r (supported: %s)"
                         % (ext or os.fspath(path), ", ".join(sorted(table))))
    return ext


def load_mesh(path: PathLike, fmt: Optional[str] = None) -> TriMesh:
    """Load an OBJ or OFF file, choosing the parser from the extension."""
    ext = _format_for(path, fmt, _READERS)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return _READERS[ext](fh, source=os.fspath(path))


def save_mesh(mesh: TriMesh, path: PathLike, fmt: Optional[str] = None) -> None:
    """Write an OBJ or OFF file, choosing the writer from the extension."""
    ext = _format_for(path, fmt, _WRITERS)
    with open(path, "w", encoding="utf-8") as fh:
        _WRITERS[ext](mesh, fh)


def loads(text: str, fmt: str) -> TriMesh:
    """Parse mesh data from a string (``fmt`` is ``"obj"`` or ``"off"``)."""
    ext = _format_for("", fmt, _READERS)
    return _READERS[ext](_io.StringIO(text), source="<string>")


def dumps(mesh: TriMesh, fmt: str) -> str:
    ext = _format_for("", fmt, _WRITERS)
    buf = _io.StringIO()
    _WRITERS[ext](mesh, buf)
    return buf.getvalue()


def write_scalar_field(path: PathLike, columns: Sequence[Sequence[float]],
                       names: Optional[Iterable[str]] = None) -> None:
    """Write per-vertex values as CSV; ``columns[k][v]`` is column ``k`` at vertex ``v``."""
    names = list(names) if names is not None else ["c%d" % k for k in range(len(columns))]
    if len(names) != len(columns):
        raise ValueError("names and columns differ in length")
    n = len(columns[0]) if columns else 0
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(["vertex"] + names) + "\n")
        for v in range(n):
            fh.write(",".join([str(v)] + ["%.10g" % col[v] for col in columns]) + "\n")
