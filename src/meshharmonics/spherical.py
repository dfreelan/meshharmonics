"""Real spherical harmonics: evaluation, projection, fitting and descriptors.

Real, orthonormal spherical harmonics without the Condon-Shortley phase
(the convention common in graphics)::

    Y_l^0     = K_l^0 P_l^0(cos θ)
    Y_l^m     = sqrt(2) K_l^m P_l^m(cos θ) cos(m φ)        m > 0
    Y_l^{-m}  = sqrt(2) K_l^m P_l^m(cos θ) sin(m φ)        m > 0

so that, e.g., ``Y_1^{-1}, Y_1^0, Y_1^1`` are proportional to ``y, z, x``.
Coefficients are stored flat at index ``l*l + l + m``.  Normalized
associated Legendre values are generated with stable three-term
recurrences, so moderate degrees (``l`` of a few hundred) are safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .linalg import LinAlgError, cholesky, cholesky_solve, norm3, sub3
from .mesh import TriMesh

FOUR_PI = 4.0 * math.pi


def sh_index(l: int, m: int) -> int:
    if l < 0 or abs(m) > l:
        raise ValueError("invalid (l, m) = (%d, %d)" % (l, m))
    return l * l + l + m


def sh_degree_order(index: int) -> Tuple[int, int]:
    """Inverse of :func:`sh_index`."""
    if index < 0:
        raise ValueError("index must be non-negative")
    l = int(math.isqrt(index)) if hasattr(math, "isqrt") else int(math.sqrt(index))
    return l, index - l * l - l


def n_coefficients(lmax: int) -> int:
    return (lmax + 1) ** 2


def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Return ``(r, theta, phi)``: radius, polar angle from +z, azimuth from +x."""
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0.0:
        return 0.0, 0.0, 0.0
    theta = math.acos(max(-1.0, min(1.0, z / r)))
    phi = math.atan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(theta: float, phi: float, r: float = 1.0) -> Tuple[float, float, float]:
    st = math.sin(theta)
    return (r * st * math.cos(phi), r * st * math.sin(phi), r * math.cos(theta))


def normalized_legendre(lmax: int, x: float) -> List[List[float]]:
    """Table ``Q[l][m] = K_l^m P_l^m(x)`` for ``0 <= m <= l <= lmax``.

    ``K_l^m = sqrt((2l+1)/(4π) (l-m)!/(l+m)!)``; no Condon-Shortley phase.
    """
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    if not -1.0 - 1e-12 <= x <= 1.0 + 1e-12:
        raise ValueError("x must lie in [-1, 1]")
    x = max(-1.0, min(1.0, x))
    s = math.sqrt(max(0.0, 1.0 - x * x))
    q = [[0.0] * (l + 1) for l in range(lmax + 1)]
    q[0][0] = math.sqrt(1.0 / FOUR_PI)
    for m in range(1, lmax + 1):
        q[m][m] = q[m - 1][m - 1] * s * math.sqrt((2.0 * m + 1.0) / (2.0 * m))
    for m in range(0, lmax):
        q[m + 1][m] = x * math.sqrt(2.0 * m + 3.0) * q[m][m]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            a = math.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
            b = math.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1.0) ** 2 - 1.0))
            q[l][m] = a * (x * q[l - 1][m] - b * q[l - 2][m])
    return q


def sh_basis_angles(lmax: int, theta: float, phi: float) -> List[float]:
    """All ``(lmax+1)^2`` real SH values at polar angle ``theta``, azimuth ``phi``."""
    q = normalized_legendre(lmax, math.cos(theta))
    out = [0.0] * n_coefficients(lmax)
    root2 = math.sqrt(2.0)
    cos_m = [math.cos(m * phi) for m in range(lmax + 1)]
    sin_m = [math.sin(m * phi) for m in range(lmax + 1)]
    for l in range(lmax + 1):
        base = l * l + l
        out[base] = q[l][0]
        for m in range(1, l + 1):
            out[base + m] = root2 * q[l][m] * cos_m[m]
            out[base - m] = root2 * q[l][m] * sin_m[m]
    return out


def sh_basis(lmax: int, direction: Sequence[float]) -> List[float]:
    """Real SH basis at a (not necessarily unit) direction vector."""
    r, theta, phi = cartesian_to_spherical(direction[0], direction[1], direction[2])
    if r == 0.0:
        raise ValueError("direction must be non-zero")
    return sh_basis_angles(lmax, theta, phi)


def real_sph_harm(l: int, m: int, theta: float, phi: float) -> float:
    """Single real spherical harmonic ``Y_l^m(theta, phi)``."""
    return sh_basis_angles(l, theta, phi)[sh_index(l, m)]


def sphere_laplacian_eigenvalues(lmax: int) -> List[float]:
    """Analytic unit-sphere spectrum ``l(l+1)`` with multiplicity ``2l+1``."""
    return [float(l * (l + 1)) for l in range(lmax + 1) for _ in range(2 * l + 1)]


@dataclass
class SHExpansion:
    """A band-limited function on the sphere given by real SH coefficients."""

    lmax: int
    coefficients: List[float]

    def __post_init__(self) -> None:
        if len(self.coefficients) != n_coefficients(self.lmax):
            raise ValueError("expected %d coefficients for lmax=%d"
                             % (n_coefficients(self.lmax), self.lmax))

    def coefficient(self, l: int, m: int) -> float:
        return self.coefficients[sh_index(l, m)]

    def evaluate(self, direction: Sequence[float]) -> float:
        basis = sh_basis(self.lmax, direction)
        return math.fsum(c * b for c, b in zip(self.coefficients, basis))

    def band_energies(self) -> List[float]:
        """``sum_m c_lm^2`` for each degree ``l`` (rotation invariant)."""
        return [math.fsum(self.coefficients[l * l + k] ** 2 for k in range(2 * l + 1))
                for l in range(self.lmax + 1)]

    def band_norms(self) -> List[float]:
        return [math.sqrt(e) for e in self.band_energies()]


def project(directions: Sequence[Sequence[float]], values: Sequence[float],
            weights: Sequence[float], lmax: int) -> SHExpansion:
    """Quadrature projection ``c_lm = sum_i w_i f_i Y_lm(d_i)``.

    ``weights`` should be solid-angle quadrature weights summing to ``4π``.
    """
    if not len(directions) == len(values) == len(weights):
        raise ValueError("directions, values and weights must have equal length")
    coeffs = [0.0] * n_coefficients(lmax)
    for d, f, w in zip(directions, values, weights):
        wf = w * f
        if wf == 0.0:
            continue
        for k, b in enumerate(sh_basis(lmax, d)):
            coeffs[k] += wf * b
    return SHExpansion(lmax, coeffs)


def fit(directions: Sequence[Sequence[float]], values: Sequence[float], lmax: int,
        regularization: float = 0.0, weights: Optional[Sequence[float]] = None) -> SHExpansion:
    """Weighted least-squares SH fit with optional Laplacian (smoothness) regularization.

    Minimizes ``sum_i w_i (f_i - sum c Y(d_i))^2 + reg * sum_lm (l(l+1))^2 c_lm^2``
    by solving the normal equations with a dense Cholesky factorization.
    """
    nc = n_coefficients(lmax)
    if len(directions) != len(values):
        raise ValueError("directions and values must have equal length")
    w = [1.0] * len(values) if weights is None else list(weights)
    gram = [[0.0] * nc for _ in range(nc)]
    rhs = [0.0] * nc
    for d, f, wi in zip(directions, values, w):
        b = sh_basis(lmax, d)
        for i in range(nc):
            wb = wi * b[i]
            if wb == 0.0:
                continue
            rhs[i] += wb * f
            row = gram[i]
            for j in range(i, nc):
                row[j] += wb * b[j]
    for i in range(nc):
        for j in range(i):
            gram[i][j] = gram[j][i]
        l, _ = sh_degree_order(i)
        gram[i][i] += regularization * float(l * (l + 1)) ** 2
    try:
        low = cholesky(gram)
    except LinAlgError as exc:
        raise LinAlgError("SH normal equations are singular; use more samples, a lower "
                          "lmax, or regularization > 0") from exc
    return SHExpansion(lmax, cholesky_solve(low, rhs))


def sphere_quadrature(mesh: TriMesh, center: Optional[Sequence[float]] = None
                      ) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """Unit directions and solid-angle weights for the vertices of a star-shaped mesh.

    Each triangle's solid angle (Van Oosterom-Strackee formula) is split
    equally among its corners, so the weights sum to ``4π`` for a closed
    mesh that is star-shaped with respect to ``center``.
    """
    c = tuple(center) if center is not None else mesh.centroid()
    rel = [sub3(p, c) for p in mesh.vertices]
    dirs = []
    for v in rel:
        r = norm3(v)
        if r == 0.0:
            raise ValueError("a vertex coincides with the projection center")
        dirs.append((v[0] / r, v[1] / r, v[2] / r))
    weights = [0.0] * mesh.n_vertices
    for a, b, cc in mesh.faces:
        pa, pb, pc = dirs[a], dirs[b], dirs[cc]
        triple = (pa[0] * (pb[1] * pc[2] - pb[2] * pc[1])
                  - pa[1] * (pb[0] * pc[2] - pb[2] * pc[0])
                  + pa[2] * (pb[0] * pc[1] - pb[1] * pc[0]))
        denom = 1.0 + (pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2]) \
            + (pb[0] * pc[0] + pb[1] * pc[1] + pb[2] * pc[2]) \
            + (pc[0] * pa[0] + pc[1] * pa[1] + pc[2] * pa[2])
        omega = abs(2.0 * math.atan2(triple, denom))
        for v in (a, b, cc):
            weights[v] += omega / 3.0
    return dirs, weights


def project_vertex_function(mesh: TriMesh, values: Sequence[float], lmax: int,
                            center: Optional[Sequence[float]] = None) -> SHExpansion:
    """Project a per-vertex function on a sphere-like mesh onto real SH."""
    dirs, weights = sphere_quadrature(mesh, center)
    return project(dirs, values, weights, lmax)


def radial_expansion(mesh: TriMesh, lmax: int, center: Optional[Sequence[float]] = None,
                     method: str = "fit", regularization: float = 1e-8) -> SHExpansion:
    """SH expansion of the radial function ``r(direction)`` of a star-shaped mesh."""
    c = tuple(center) if center is not None else mesh.centroid()
    radii = [norm3(sub3(p, c)) for p in mesh.vertices]
    if method == "project":
        return project_vertex_function(mesh, radii, lmax, c)
    if method != "fit":
        raise ValueError("method must be 'fit' or 'project'")
    dirs, weights = sphere_quadrature(mesh, c)
    return fit(dirs, radii, lmax, regularization=regularization, weights=weights)


def shape_descriptor(mesh: TriMesh, lmax: int = 8, normalize: bool = True,
                     method: str = "project") -> List[float]:
    """Rotation-invariant SH energy descriptor of a star-shaped mesh.

    The radial function is expanded in SH and the per-band norms are
    returned (optionally divided by the ``l = 0`` band so the descriptor is
    scale invariant).  Rotations mix coefficients only within a band, so the
    band norms are invariant up to sampling error.
    """
    norms = radial_expansion(mesh, lmax, method=method).band_norms()
    if normalize and norms[0] > 0.0:
        norms = [x / norms[0] for x in norms]
    return norms


def rotation_matrix(axis: Sequence[float], angle: float) -> List[List[float]]:
    """Rodrigues rotation matrix about ``axis`` by ``angle`` radians."""
    ax = norm3(axis)
    if ax == 0.0:
        raise ValueError("axis must be non-zero")
    x, y, z = (axis[0] / ax, axis[1] / ax, axis[2] / ax)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]


def rotate_mesh(mesh: TriMesh, matrix: Sequence[Sequence[float]]) -> TriMesh:
    verts = [tuple(sum(matrix[r][k] * p[k] for k in range(3)) for r in range(3)) for p in mesh.vertices]
    return TriMesh(verts, mesh.faces, validate=False)
