"""meshharmonics -- spectral geometry on triangle meshes in pure Python.

Quick start::

    import meshharmonics as mh

    mesh = mh.icosphere(subdivisions=2)
    print(mh.TriMesh.analyze(mesh).summary_lines())
    eig = mh.laplacian_eigenpairs(mesh, k=9)
    hks = mh.heat_kernel_signature(eig)
    dist = mh.HeatGeodesicSolver(mesh).distance(0)
"""

__version__ = "0.1.0"

from .eigensolve import EigenResult, eigsh, lanczos, laplacian_eigenpairs, power_iteration
from .geodesics import (HeatGeodesicSolver, dijkstra_distances, farthest_point_sampling,
                        geodesic_distances, heat_geodesic_distances, shortest_edge_path)
from .io import MeshParseError, dumps, load_mesh, loads, read_obj, read_off, save_mesh, write_obj, write_off
from .laplacian import (LaplacianPair, cotangent_laplacian, laplacian_pair, mass_matrix,
                        mean_curvature_normals, uniform_laplacian)
from .linalg import CSRMatrix, SkylineCholesky, conjugate_gradient
from .mesh import HalfEdgeMesh, MeshError, NonManifoldError, TopologyReport, TriMesh, ValenceStats
from .primitives import (cube, cylinder, grid, icosahedron, icosphere, octahedron, tetrahedron, torus)
from .signatures import (default_hks_times, diffusion_distance, global_point_signature, heat_kernel,
                         heat_kernel_signature, heat_trace, shape_dna, wave_kernel_signature)
from .spherical import SHExpansion, real_sph_harm, sh_basis, shape_descriptor


# Stable fixture catalog (internal golden IDs)
from ._catalog import FIXTURE_NOTES, REFERENCE_URLS, fixture_ids, note_for, primary_reference

__all__ = ["FIXTURE_NOTES", "REFERENCE_URLS", "fixture_ids", "note_for", "primary_reference", 
    "CSRMatrix", "EigenResult", "HalfEdgeMesh", "HeatGeodesicSolver", "LaplacianPair", "MeshError",
    "MeshParseError", "NonManifoldError", "SHExpansion", "SkylineCholesky", "TopologyReport", "TriMesh",
    "ValenceStats", "conjugate_gradient", "cotangent_laplacian", "cube", "cylinder", "default_hks_times",
    "diffusion_distance", "dijkstra_distances", "dumps", "eigsh", "farthest_point_sampling",
    "geodesic_distances", "global_point_signature", "grid", "heat_geodesic_distances", "heat_kernel",
    "heat_kernel_signature", "heat_trace", "icosahedron", "icosphere", "lanczos", "laplacian_eigenpairs",
    "laplacian_pair", "load_mesh", "loads", "mass_matrix", "mean_curvature_normals", "octahedron",
    "power_iteration", "read_obj", "read_off", "real_sph_harm", "save_mesh", "sh_basis",
    "shape_descriptor", "shape_dna", "shortest_edge_path", "tetrahedron", "torus", "uniform_laplacian",
    "wave_kernel_signature", "write_obj", "write_off",
]
