# meshharmonics

**Spectral geometry on triangle meshes, in pure Python.**

`meshharmonics` gives you the core building blocks of spectral mesh processing:
robust OBJ/OFF loading, topology diagnostics, the cotangent Laplace-Beltrami
operator, sparse eigensolvers, real spherical harmonics, spectral point
descriptors (HKS / WKS / GPS / ShapeDNA) and geodesic distances. It has
**no required dependencies**. The sparse matrices, Cholesky factorization,
Lanczos iteration and tridiagonal QL solver are all included, so it installs
anywhere Python runs and is easy to read, debug and extend.

It suits prototyping geometric-learning features, teaching, test oracles for
GPU pipelines, and small to medium meshes (up to a few thousand vertices) where
a dependency-free tool is worth more than raw speed.

```
$ meshharmonics primitive icosphere sphere.obj --subdivisions 2
$ meshharmonics spectrum sphere.obj -k 9
  i        eigenvalue    residual
  0   1.172896686e-15    7.26e-15
  1       1.999990487    3.62e-11
  2       1.999990487    1.03e-12
  3       1.999990487    2.36e-12
  4       5.865268583    1.64e-11
  ...
```

## Installation

```bash
pip install -e .            # library + `meshharmonics` command
pip install -e '.[dev]'     # plus pytest
```

Requires Python 3.9+. No compiled extensions, no numpy.

## Quick tour

```python
import meshharmonics as mh

mesh = mh.load_mesh("bunny.obj")          # OBJ or OFF; polygons are fan-triangulated
report = mesh.analyze()
print("\n".join(report.summary_lines()))
# vertices / faces / edges, boundary loops, components, Euler characteristic,
# genus, edge- and vertex-manifoldness, orientation, valence histogram ...

# Laplace-Beltrami eigenpairs:  L phi = lambda M phi
eig = mh.laplacian_eigenpairs(mesh, k=30)
eig.values[:5], eig.max_residual

# Spectral descriptors (one row per vertex)
hks = mh.heat_kernel_signature(eig)                 # log-spaced default times
wks = mh.wave_kernel_signature(eig, n_energies=16)
gps = mh.global_point_signature(eig, k=10)
dna = mh.shape_dna(eig, k=10, area=mesh.surface_area())

# Geodesic distance (heat method, factor once and query many times)
solver = mh.HeatGeodesicSolver(mesh)
d0 = solver.distance(0)
d_multi = solver.distance([12, 40, 99])
samples = mh.farthest_point_sampling(mesh, 64)
```

## Modules

| module | contents |
| --- | --- |
| `meshharmonics.io` | `read_obj`, `read_off` (OFF/COFF/NOFF), writers, `load_mesh`/`save_mesh`, CSV export; errors carry file and line numbers |
| `meshharmonics.mesh` | `TriMesh` (areas, normals, angles, angle defects, volume, components), `TopologyReport`, `ValenceStats`, `HalfEdgeMesh` (ordered one-rings, boundary loops), orientation repair, cleaning |
| `meshharmonics.laplacian` | cotangent weights and stiffness matrix, uniform graph Laplacian, barycentric and mixed-Voronoi lumped mass, `LaplacianPair`, mean-curvature normals |
| `meshharmonics.eigensolve` | `lanczos` (full reorthogonalization, restarts on invariant subspaces), `power_iteration` (block subspace iteration with Rayleigh-Ritz), `eigsh` for generalized problems with shift-invert, `laplacian_eigenpairs` |
| `meshharmonics.spherical` | real orthonormal SH (stable recurrences), solid-angle sphere quadrature, projection, regularized least-squares fit, rotation-invariant shape descriptor |
| `meshharmonics.signatures` | HKS (with trace/L2 normalization), WKS, GPS, heat kernel and trace, diffusion distance, ShapeDNA, descriptor matching |
| `meshharmonics.geodesics` | Dijkstra distances and paths, heat-method geodesics, face gradient / vertex divergence operators, farthest point sampling, error statistics |
| `meshharmonics.linalg` | CSR matrices, preconditioned CG, reverse Cuthill-McKee ordering, envelope (skyline) Cholesky, Jacobi and tridiagonal QL eigensolvers |
| `meshharmonics.primitives` | tetrahedron, octahedron, cube, icosahedron, icosphere, grid, torus, cylinder, midpoint subdivision |
| `meshharmonics.cli` | the `meshharmonics` command |

## Conventions

* **Stiffness matrix.** `L` is symmetric positive semi-definite, with
  `L[i,j] = -(cot α + cot β)/2`, so `uᵀLu = ∫|∇u|²` for piecewise-linear `u`.
* **Eigenproblem.** `L φ = λ M φ` with a lumped diagonal `M`. Eigenvectors are
  `M`-orthonormal, and their signs are fixed so that the largest-magnitude entry
  is positive. On the unit sphere, `λ → l(l+1)` with multiplicity `2l+1`.
* **Boundaries** get natural (Neumann) conditions throughout.
* **Spherical harmonics** are real and orthonormal, without the Condon-Shortley
  phase. They are stored flat at index `l² + l + m`, so `Y₁₋₁, Y₁₀, Y₁₁ ∝ y, z, x`.
* **Descriptors** are returned row-major (`sig[vertex][feature]`), ready for
  feature matrices.

## How the eigensolver works

For the smallest eigenpairs, `eigsh` applies the shift-invert transform
`(L − σM)⁻¹M` in the symmetric form `M^{1/2}(L − σM)⁻¹M^{1/2}`. The shifted
matrix is factored once with an envelope Cholesky after reverse Cuthill-McKee
reordering. On a typical mesh the envelope grows roughly like `n^1.5`, so a
Lanczos sweep with full reorthogonalization converges in a few dozen steps.
Convergence is tested on the Ritz residuals. If Lanczos hits an invariant
subspace, it continues with a fresh random vector orthogonal to the basis, so
exactly repeated eigenvalues are recovered too. `solver="cg"` swaps the direct
solve for Jacobi-preconditioned conjugate gradients.

Approximate timings in CPython 3.12 on a laptop: 9 sphere eigenpairs on 162
vertices take about 0.2 s. The cost grows with mesh size; for meshes with
hundreds of thousands of vertices, use a compiled sparse eigensolver and keep
`meshharmonics` as the reference implementation.

## Mesh diagnostics

`TriMesh.analyze()` reports:

* edge-manifoldness (edges with more than two faces) and vertex-manifoldness
  (bow-tie vertices whose link is not a single path or cycle);
* consistent orientation versus orientability. `TriMesh.oriented()` repairs
  orientation per component and refuses non-orientable inputs such as a Möbius
  strip;
* boundary edges and ordered boundary loops;
* connected components, Euler characteristic and genus (for connected orientable
  manifolds, `χ = 2 − 2g − b`);
* isolated vertices, degenerate (zero-area) and duplicate faces;
* valence min/mean/max, histogram, and irregular interior (≠ 6) and boundary
  (≠ 4) vertices.

```
$ meshharmonics info tests/data/fin.off
tests/data/fin.off
  vertices            5
  faces               3
  edges               7
  boundary edges      6 (2 loops)
  components          1
  euler char          1
  genus               n/a
  manifold            no (edges: no, vertices: no)
  oriented            no (orientable: no)
  closed              no
  valence             min 2 / mean 2.800 / max 4
  irregular vertices  0 interior, 3 boundary
  non-manifold edges  1
  non-manifold verts  2
```

## Command line

```
meshharmonics info MESH [--json]
meshharmonics spectrum MESH [-k 10] [--mass voronoi|barycentric] [--method lanczos|power] [--shape-dna] [--json]
meshharmonics hks MESH [-k 30] [--times 8] [--normalize none|trace|l2] [-o out.csv]
meshharmonics geodesic MESH -s 0 [-s 17 ...] [--method heat|dijkstra] [--time-factor 1.0] [-o out.csv] [--json]
meshharmonics sh MESH [--lmax 8] [--raw] [--json]
meshharmonics convert IN OUT [--clean] [--orient]
meshharmonics primitive KIND OUT [--subdivisions 2] [--resolution 8]
```

`python -m meshharmonics` works as well. Errors exit with status 2 and print a
`file:line` location for parse failures.

## Accuracy notes

* Heat-method geodesics on a 162-vertex icosphere have a mean relative error of
  about 4% against exact great-circle distances. Edge-graph Dijkstra on the
  same mesh gives about 7.5%. Accuracy improves with refinement. Use a larger
  `time_factor` for smoother but more biased distances.
* `shape_descriptor` uses quadrature projection by default, which is fast and
  rotation-invariant up to sampling error. `radial_expansion(..., method="fit")`
  solves a regularized least-squares problem and reconstructs more accurately on
  irregular samplings.
* Eigenvectors inside a repeated eigenspace are not unique, and neither are GPS
  coordinates built from them. HKS and WKS are unaffected.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

The test suite uses tiny fixtures (procedural primitives plus a few
hand-written OBJ/OFF files in `tests/data`) and runs in about a second. It
checks against analytic references: the path-graph spectrum, sphere
eigenvalues, Gauss-Bonnet, SH addition theorem and quadrature
orthonormality, great-circle and Euclidean distances, and the
`div ∘ grad = −L` identity.

## References

* M. Meyer, M. Desbrun, P. Schröder, A. Barr. *Discrete Differential-Geometry
  Operators for Triangulated 2-Manifolds*, 2003.
* M. Reuter, F.-E. Wolter, N. Peinecke. *Laplace-Beltrami spectra as
  "Shape-DNA" of surfaces and solids*, 2006.
* R. Rustamov. *Laplace-Beltrami eigenfunctions for deformation invariant shape
  representation*, 2007.
* J. Sun, M. Ovsjanikov, L. Guibas. *A Concise and Provably Informative
  Multi-Scale Signature Based on Heat Diffusion*, 2009.
* M. Aubry, U. Schlickewei, D. Cremers. *The Wave Kernel Signature*, 2011.
* K. Crane, C. Weischedel, M. Wardetzky. *Geodesics in Heat*, 2013.

## License

MIT. See [LICENSE](LICENSE).
