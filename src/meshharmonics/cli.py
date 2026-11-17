"""Command-line interface: ``meshharmonics <command> [options]``.

Commands
--------
``info``       topology, manifoldness, boundary and valence report
``spectrum``   smallest Laplace-Beltrami eigenvalues (optionally ShapeDNA)
``hks``        heat kernel signature per vertex, written as CSV
``geodesic``   distances from source vertices (heat method or Dijkstra)
``sh``         rotation-invariant spherical-harmonic shape descriptor
``convert``    OBJ <-> OFF conversion (with optional cleaning)
``primitive``  write a procedural mesh (icosphere, torus, grid, ...)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from . import __version__
from .eigensolve import laplacian_eigenpairs
from .geodesics import HeatGeodesicSolver, dijkstra_distances
from .io import MeshParseError, load_mesh, save_mesh, write_scalar_field
from .mesh import MeshError
from .primitives import PRIMITIVES
from .signatures import default_hks_times, heat_kernel_signature, shape_dna
from .spherical import shape_descriptor


def _emit(args: argparse.Namespace, payload: dict, lines: Sequence[str]) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for line in lines:
            print(line)


def cmd_info(args: argparse.Namespace) -> int:
    mesh = load_mesh(args.mesh)
    report = mesh.analyze()
    payload = report.as_dict()
    lines = ["%s" % args.mesh] + ["  " + line for line in report.summary_lines()]
    if mesh.n_faces:
        lo, hi = mesh.bounding_box()
        payload.update(surface_area=mesh.surface_area(), mean_edge_length=mesh.mean_edge_length(),
                       bbox_min=list(lo), bbox_max=list(hi))
        lines.append("  surface area        %.6g" % mesh.surface_area())
        lines.append("  mean edge length    %.6g" % mesh.mean_edge_length())
        if report.is_watertight:
            payload["volume"] = mesh.signed_volume()
            lines.append("  enclosed volume     %.6g" % mesh.signed_volume())
    _emit(args, payload, lines)
    return 0


def cmd_spectrum(args: argparse.Namespace) -> int:
    mesh = load_mesh(args.mesh)
    eig = laplacian_eigenpairs(mesh, k=args.k, mass=args.mass, method=args.method)
    payload = {"eigenvalues": eig.values, "residuals": eig.residuals, "method": eig.method}
    lines = ["%3s  %16s  %10s" % ("i", "eigenvalue", "residual")]
    lines += ["%3d  %16.10g  %10.2e" % (i, lam, r)
              for i, (lam, r) in enumerate(zip(eig.values, eig.residuals))]
    if args.shape_dna:
        dna = shape_dna(eig, area=mesh.surface_area())
        payload["shape_dna"] = dna
        lines.append("shapeDNA (area-normalized): " + " ".join("%.6g" % x for x in dna))
    _emit(args, payload, lines)
    return 0


def cmd_hks(args: argparse.Namespace) -> int:
    mesh = load_mesh(args.mesh)
    eig = laplacian_eigenpairs(mesh, k=args.k, mass=args.mass)
    times = default_hks_times(eig, args.times)
    sig = heat_kernel_signature(eig, times, normalize=args.normalize)
    columns = [[row[t] for row in sig] for t in range(len(times))]
    names = ["t=%.6g" % t for t in times]
    if args.output:
        write_scalar_field(args.output, columns, names)
        print("wrote %d x %d HKS matrix to %s" % (len(sig), len(times), args.output))
    else:
        print(",".join(["vertex"] + names))
        for v, row in enumerate(sig):
            print(",".join([str(v)] + ["%.8g" % x for x in row]))
    return 0


def cmd_geodesic(args: argparse.Namespace) -> int:
    mesh = load_mesh(args.mesh)
    if args.method == "heat":
        dist = HeatGeodesicSolver(mesh, time_factor=args.time_factor).distance(args.source)
    else:
        dist = dijkstra_distances(mesh, args.source)[0]
    if args.output:
        write_scalar_field(args.output, [dist], ["distance"])
        print("wrote distances for %d vertices to %s" % (len(dist), args.output))
    else:
        finite = [d for d in dist if d != float("inf")]
        payload = {"sources": args.source, "method": args.method, "distances": dist}
        lines = ["vertex,distance"] + ["%d,%.8g" % (v, d) for v, d in enumerate(dist)]
        if finite:
            lines.append("# max %.6g, mean %.6g" % (max(finite), sum(finite) / len(finite)))
        _emit(args, payload, lines)
    return 0


def cmd_sh(args: argparse.Namespace) -> int:
    mesh = load_mesh(args.mesh)
    desc = shape_descriptor(mesh, lmax=args.lmax, normalize=not args.raw)
    payload = {"lmax": args.lmax, "band_norms": desc}
    lines = ["l=%d  %.8g" % (l, x) for l, x in enumerate(desc)]
    _emit(args, payload, lines)
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    mesh = load_mesh(args.input)
    if args.clean:
        mesh = mesh.cleaned()
    if args.orient:
        mesh = mesh.oriented()
    save_mesh(mesh, args.output)
    print("wrote %s (%d vertices, %d faces)" % (args.output, mesh.n_vertices, mesh.n_faces))
    return 0


def cmd_primitive(args: argparse.Namespace) -> int:
    factory = PRIMITIVES[args.kind]
    kwargs = {}
    if args.kind == "icosphere":
        kwargs["subdivisions"] = args.subdivisions
    elif args.kind == "grid":
        kwargs.update(nx=args.resolution, ny=args.resolution)
    elif args.kind == "torus":
        kwargs.update(n_major=2 * args.resolution, n_minor=args.resolution)
    elif args.kind == "cylinder":
        kwargs.update(n_around=2 * args.resolution, n_height=args.resolution)
    mesh = factory(**kwargs)
    save_mesh(mesh, args.output)
    print("wrote %s %s (%d vertices, %d faces)" % (args.kind, args.output, mesh.n_vertices, mesh.n_faces))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meshharmonics",
                                     description="Spectral geometry on triangle meshes.")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    p = sub.add_parser("info", help="topology and quality report")
    p.add_argument("mesh")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("spectrum", help="Laplace-Beltrami eigenvalues")
    p.add_argument("mesh")
    p.add_argument("-k", type=int, default=10, help="number of eigenpairs (default 10)")
    p.add_argument("--mass", choices=("voronoi", "barycentric"), default="voronoi")
    p.add_argument("--method", choices=("lanczos", "power"), default="lanczos")
    p.add_argument("--shape-dna", action="store_true", help="also print area-normalized ShapeDNA")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_spectrum)

    p = sub.add_parser("hks", help="heat kernel signature (CSV)")
    p.add_argument("mesh")
    p.add_argument("-k", type=int, default=30, help="eigenpairs used (default 30)")
    p.add_argument("--times", type=int, default=8, help="number of diffusion times")
    p.add_argument("--mass", choices=("voronoi", "barycentric"), default="voronoi")
    p.add_argument("--normalize", choices=("none", "trace", "l2"), default="none")
    p.add_argument("-o", "--output", help="CSV path (default: stdout)")
    p.set_defaults(func=cmd_hks)

    p = sub.add_parser("geodesic", help="geodesic distance from source vertices")
    p.add_argument("mesh")
    p.add_argument("-s", "--source", type=int, action="append", required=True,
                   help="source vertex (repeatable)")
    p.add_argument("--method", choices=("heat", "dijkstra"), default="heat")
    p.add_argument("--time-factor", type=float, default=1.0)
    p.add_argument("-o", "--output", help="CSV path (default: stdout)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_geodesic)

    p = sub.add_parser("sh", help="spherical-harmonic shape descriptor")
    p.add_argument("mesh")
    p.add_argument("--lmax", type=int, default=8)
    p.add_argument("--raw", action="store_true", help="do not normalize by the l=0 band")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_sh)

    p = sub.add_parser("convert", help="convert between OBJ and OFF")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--clean", action="store_true", help="drop degenerate/duplicate faces")
    p.add_argument("--orient", action="store_true", help="make face orientation consistent")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("primitive", help="write a procedural mesh")
    p.add_argument("kind", choices=sorted(PRIMITIVES))
    p.add_argument("output")
    p.add_argument("--subdivisions", type=int, default=2, help="icosphere refinement level")
    p.add_argument("--resolution", type=int, default=8, help="segments for grid/torus/etc.")
    p.set_defaults(func=cmd_primitive)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (MeshParseError, MeshError, ValueError, OSError) as exc:
        print("meshharmonics: error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
