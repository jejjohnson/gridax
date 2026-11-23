"""The canonical vector calculus: divergence, curl, Laplacian and gradients.

On a C-grid the operands and the result sit on their staggered points, and a
3-D grid is a stack of levels: the operators are horizontal. On a collocated
grid everything is at T, with the stencils of ``gridax.analysis``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.analysis.differences import (
    adaptive_difference,
    difference as _collocated_difference,
)
from gridax._src.api import grid_dispatch
from gridax._src.fields.located import located
from gridax._src.grid import (
    CartesianAGrid1D,
    CartesianAGrid2D,
    CartesianAGrid3D,
    CartesianCGrid1D,
    CartesianCGrid2D,
    CartesianCGrid3D,
    SphericalAGrid2D,
    SphericalCGrid2D,
    SphericalCGrid3D,
)
from gridax._src.grid.points import Stagger
from gridax._src.operators._ghost import interior
from gridax._src.operators._utils import (
    _metric_like,
    _safe_div_cos,
    _spacing,
    _spherical_d2_lon,
    _spherical_laplacian_merid,
)
from gridax._src.operators.moves import _masked, difference
from gridax._src.operators.stencils import (
    _neighbours,
    diff_x_bwd_2d,
    diff_x_fwd_2d,
    diff_y_bwd_2d,
    diff_y_fwd_2d,
)
from gridax._src.operators.triples import point_name
from gridax._src.pytree import array_namespace

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


@grid_dispatch
@located("T", u="U", v="V")
def divergence(
    u: Float[Array, "..."], v: Float[Array, "..."], grid: Any, *, mask: Any = None
) -> Float[Array, "..."]:
    """The horizontal divergence of a vector, at T.

    ``div = δₓu + δᵧv``, with ``u`` at U and ``v`` at V; on a sphere
    ``1/(R cos φ) (∂λ u + ∂φ (v cos φ))``.

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch *grid"]
        The components, at U and V (at T on a collocated grid).
    grid : grid
        Two- or three-dimensional, Cartesian or spherical.
    mask : Mask2D or Mask3D, optional
        Zeroes the dry cells; on a collocated grid they are NaN, and a wet
        one next to them takes the widest stencil that fits.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At T, with a zero ghost ring on a C-grid.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 2.0, 2.0)
    >>> u = np.broadcast_to(grid.coords("x", at="U"), grid.shape)
    >>> gridax.divergence(u, 0.0 * u, grid)[2, 1:-1].tolist()
    [1.0, 1.0, 1.0, 1.0]
    """
    return divergence._impl(grid, u, v, mask=mask)


@grid_dispatch
@located("X", u="U", v="V")
def curl(
    u: Float[Array, "..."], v: Float[Array, "..."], grid: Any, *, mask: Any = None
) -> Float[Array, "..."]:
    """The vertical curl of a horizontal vector (the relative vorticity), at X.

    ``ζ = δₓv - δᵧu``; on a sphere ``1/(R cos φ) (∂λ v - ∂φ (u cos φ))``.

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch *grid"]
        The components, at U and V (at T on a collocated grid).
    grid : grid
        Two- or three-dimensional, Cartesian or spherical.
    mask : Mask2D or Mask3D, optional
        Zeroes the corners without four wet cells; on a collocated grid as in
        :func:`divergence`.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At X (T on a collocated grid), with a zero ghost ring on a C-grid.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 2.0, 2.0)
    >>> v = np.broadcast_to(grid.coords("x", at="V"), grid.shape)
    >>> gridax.curl(0.0 * v, v, grid)[2, 1:-1].tolist()
    [1.0, 1.0, 1.0, 1.0]
    """
    return curl._impl(grid, u, v, mask=mask)


@grid_dispatch
@located("at", f="at")
def laplacian(
    f: Float[Array, "..."],
    grid: Any,
    *,
    at: str | Stagger | None = None,
    mask: Any = None,
) -> Float[Array, "..."]:
    """The horizontal Laplacian of a scalar, at its own points.

    ``δₓδₓf + δᵧδᵧf``; on a sphere ``1/(R² cos² φ) ∂λλ f
    + 1/(R² cos φ) ∂φ (cos φ ∂φ f)``.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        The scalar, at ``at``.
    grid : grid
        Cartesian (1-D to 3-D) or spherical.
    at : str or Stagger, optional
        The point of a plain array, ``"T"`` by default. A `CField` has its
        own.
    mask : Mask1D, Mask2D or Mask3D, optional
        Zeroes the dry points of ``at``; on a collocated grid as in
        :func:`divergence`.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At ``at``, with a zero ghost ring on a C-grid.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid1D.from_interior(4, 4.0)
    >>> gridax.laplacian(grid.coords("x") ** 2, grid)[1:-1].tolist()
    [2.0, 2.0, 2.0, 2.0]
    """
    return laplacian._impl(grid, f, at=at, mask=mask)


@grid_dispatch
@located(("U", "V"), f="T")
def gradient(
    f: Float[Array, "..."], grid: Any, *, mask: Any = None
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """The horizontal gradient of a scalar: ``(∂ₓf, ∂ᵧf)``.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        The scalar, at T.
    grid : grid
        Two- or three-dimensional, Cartesian or spherical.
    mask : Mask2D or Mask3D, optional
        Zeroes the dry faces; on a collocated grid as in :func:`divergence`.

    Returns
    -------
    tuple of CField or Float[Array, "*batch *grid"]
        At U and V (both at T on a collocated grid).

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 2.0, 2.0)
    >>> h = np.broadcast_to(grid.coords("x"), grid.shape)
    >>> dx, dy = gridax.gradient(h, grid)
    >>> dx[2, 2].item(), dy[2, 2].item()
    (1.0, 0.0)
    """
    return gradient._impl(grid, f, mask=mask)


@grid_dispatch
@located(("U", "V"), psi="at")
def grad_perp(
    psi: Float[Array, "..."],
    grid: Any,
    *,
    at: str | Stagger | None = None,
    mask: Any = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """The rotated gradient of a streamfunction, ``(u, v) = (-∂ᵧψ, ∂ₓψ)``.

    On a C-grid without a mask the velocity is discretely non-divergent. From
    ψ at X it is the two-point differences, ``u = -δᵧψ`` and ``v = δₓψ``; from
    ψ at T the same on ψ averaged to the corners, which is a compact stencil
    over the six cells around a face.

    Parameters
    ----------
    psi : CField or Float[Array, "*batch *grid"]
        The streamfunction, at X or T.
    grid : grid
        Two- or three-dimensional, Cartesian or spherical.
    at : {"X", "T"}, optional
        The point of a plain array, ``"X"`` by default. A `CField` has its
        own.
    mask : Mask2D or Mask3D, optional
        Zeroes the dry faces; on a collocated grid as in :func:`divergence`.

    Returns
    -------
    tuple of CField or Float[Array, "*batch *grid"]
        At U and V (both at T on a collocated grid).

    Raises
    ------
    ValueError
        If ``psi`` is at another point.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 2.0, 2.0)
    >>> psi = np.broadcast_to(grid.coords("x", at="X"), grid.shape)
    >>> u, v = gridax.grad_perp(psi, grid)
    >>> abs(u[2, 2].item()), v[2, 2].item()
    (0.0, 1.0)
    """
    return grad_perp._impl(grid, psi, at=at, mask=mask)


# ----------------------------------------------------------------------
# C-grids
# ----------------------------------------------------------------------


def _levels(grid: Any, f: Any) -> Any:
    """The interior levels of a 3-D field; a 2-D one as it is.

    The horizontal stencils trim the last two axes; trimming z too lets
    `interior` pad a zero ghost slab.
    """
    return f[..., 1:-1, :, :] if len(grid.axes) == 3 else f


@divergence.register(CartesianCGrid2D)
@divergence.register(CartesianCGrid3D)
def _divergence_cartesian(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    xp = array_namespace(u, v)
    dx, dy = _spacing(grid, "x", u), _spacing(grid, "y", v)
    like = xp.zeros(u.shape, dtype=xp.result_type(u, v))
    u, v = _levels(grid, u), _levels(grid, v)
    # div[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx + (v[j+1/2, i] - v[j-1/2, i]) / dy
    out = interior(diff_x_bwd_2d(u) / dx + diff_y_bwd_2d(v) / dy, like)
    return _masked(out, mask, "T")


@curl.register(CartesianCGrid2D)
@curl.register(CartesianCGrid3D)
def _curl_cartesian(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    dx, dy = _spacing(grid, "x", v, at="X"), _spacing(grid, "y", u, at="X")
    like = u
    u, v = _levels(grid, u), _levels(grid, v)
    # ζ[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
    #                 - (u[j+1, i+1/2] - u[j, i+1/2]) / dy
    out = interior(diff_x_fwd_2d(v) / dx - diff_y_fwd_2d(u) / dy, like)
    return _masked(out, mask, "X")


@laplacian.register(CartesianCGrid1D)
def _laplacian_cartesian_1d(grid: Any, f: Any, *, at: Any, mask: Any) -> Any:
    at = "T" if at is None else point_name(at)
    # Each offset is sliced once (see `_Neighbours2D` for why)
    c, e, w = f[..., 1:-1], f[..., 2:], f[..., :-2]  # f[i], f[i+1], f[i-1]
    # ∇²f[i] = (f[i+1] - 2 f[i] + f[i-1]) / dx²
    out = interior(((e - c) - (c - w)) / _spacing(grid, "x", f, at=at) ** 2, f)
    return _masked(out, mask, at)


@laplacian.register(CartesianCGrid2D)
@laplacian.register(CartesianCGrid3D)
def _laplacian_cartesian(grid: Any, f: Any, *, at: Any, mask: Any) -> Any:
    at = "T" if at is None else point_name(at)
    dx, dy = _spacing(grid, "x", f, at=at), _spacing(grid, "y", f, at=at)
    # ∇²f[j, i] = (f[j, i+1] - 2 f[j, i] + f[j, i-1]) / dx²
    #           + (f[j+1, i] - 2 f[j, i] + f[j-1, i]) / dy²
    nb = _neighbours(_levels(grid, f))  # c, e, w, n, s views  [Ny-2, Nx-2]
    d2x = ((nb.e - nb.c) - (nb.c - nb.w)) / dx**2
    d2y = ((nb.n - nb.c) - (nb.c - nb.s)) / dy**2
    return _masked(interior(d2x + d2y, f), mask, at)


@gradient.register(CartesianCGrid2D)
@gradient.register(CartesianCGrid3D)
@gradient.register(SphericalCGrid2D)
@gradient.register(SphericalCGrid3D)
def _gradient_staggered(grid: Any, f: Any, *, mask: Any) -> Any:
    return (
        difference(f, grid, axis="x", src="T", dst="U", mask=mask),
        difference(f, grid, axis="y", src="T", dst="V", mask=mask),
    )


@grad_perp.register(CartesianCGrid2D)
@grad_perp.register(CartesianCGrid3D)
@grad_perp.register(SphericalCGrid2D)
@grad_perp.register(SphericalCGrid3D)
def _grad_perp_staggered(grid: Any, psi: Any, *, at: Any, mask: Any) -> Any:
    at = "X" if at is None else point_name(at)
    if at == "X":
        # u[j, i+1/2] = -(ψ[j+1/2, i+1/2] - ψ[j-1/2, i+1/2]) / dy
        u = -difference(psi, grid, axis="y", src="X", dst="U", mask=mask)
        # v[j+1/2, i] = (ψ[j+1/2, i+1/2] - ψ[j+1/2, i-1/2]) / dx
        return u, difference(psi, grid, axis="x", src="X", dst="V", mask=mask)
    if at != "T":
        raise ValueError(f"grad_perp takes a streamfunction at X or T, not {at}")
    p = _levels(grid, psi)
    # u[j, i+1/2] = -(ψ[j+1,i] + ψ[j+1,i+1] - ψ[j-1,i] - ψ[j-1,i+1]) / (4 dy)
    u = -(p[..., 2:, 1:-1] + p[..., 2:, 2:] - p[..., :-2, 1:-1] - p[..., :-2, 2:])
    u = u / (4.0 * _spacing(grid, "y", psi, at="U"))
    # v[j+1/2, i] = (ψ[j,i+1] + ψ[j+1,i+1] - ψ[j,i-1] - ψ[j+1,i-1]) / (4 dx)
    v = p[..., 1:-1, 2:] + p[..., 2:, 2:] - p[..., 1:-1, :-2] - p[..., 2:, :-2]
    if isinstance(grid, SphericalCGrid2D | SphericalCGrid3D):
        cos = _metric_like(grid._cos_lat("V"), psi)[1:-1, 1:-1]
        v = _safe_div_cos(v, cos, _metric_like(4.0 * grid.R * grid.dlon, psi))
    else:
        v = v / (4.0 * _spacing(grid, "x", psi, at="V"))
    return (
        _masked(interior(u, psi), mask, "U"),
        _masked(interior(v, psi), mask, "V"),
    )


@divergence.register(SphericalCGrid2D)
@divergence.register(SphericalCGrid3D)
def _divergence_spherical(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    like = u
    cos_V = _metric_like(grid._cos_lat("V"), v)
    cos_T = _metric_like(grid._cos_lat("T"), u)[1:-1, 1:-1]
    dlon, dlat = _metric_like(grid.dlon, u), _metric_like(grid.dlat, v)
    u, v = _levels(grid, u), _levels(grid, v)
    # div[j, i] = 1/(R cos φ_T) [du/dλ + d(v cos φ_V)/dφ]
    du_dlon = diff_x_bwd_2d(u) / dlon
    v_cos_N = v[..., 1:-1, 1:-1] * cos_V[1:-1, 1:-1]
    v_cos_S = v[..., :-2, 1:-1] * cos_V[:-2, 1:-1]
    dvc_dlat = (v_cos_N - v_cos_S) / dlat
    R = _metric_like(grid.R, u)
    out = interior(_safe_div_cos(du_dlon + dvc_dlat, cos_T, R), like)
    return _masked(out, mask, "T")


@curl.register(SphericalCGrid2D)
@curl.register(SphericalCGrid3D)
def _curl_spherical(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    like = u
    cos_U = _metric_like(grid._cos_lat("U"), u)
    cos_X = _metric_like(grid._cos_lat("X"), u)[1:-1, 1:-1]
    dlon, dlat = _metric_like(grid.dlon, v), _metric_like(grid.dlat, u)
    u, v = _levels(grid, u), _levels(grid, v)
    # ζ[j+1/2, i+1/2] = 1/(R cos φ_X) [dv/dλ - d(u cos φ_U)/dφ]
    dv_dlon = diff_x_fwd_2d(v) / dlon
    u_cos_N = u[..., 2:, 1:-1] * cos_U[2:, 1:-1]
    u_cos_S = u[..., 1:-1, 1:-1] * cos_U[1:-1, 1:-1]
    duc_dlat = (u_cos_N - u_cos_S) / dlat
    R = _metric_like(grid.R, u)
    out = interior(_safe_div_cos(dv_dlon - duc_dlat, cos_X, R), like)
    return _masked(out, mask, "X")


@laplacian.register(SphericalCGrid2D)
@laplacian.register(SphericalCGrid3D)
def _laplacian_spherical(grid: Any, f: Any, *, at: Any, mask: Any) -> Any:
    at = "T" if at is None else point_name(at)
    cos = _metric_like(grid._cos_lat(at), f)
    nb = _neighbours(_levels(grid, f))  # c, e, w, n, s views  [Ny-2, Nx-2]
    if at in ("T", "U", "W"):
        # the faces along y are the V rows: j (north) and j-1 (south)
        faces = _metric_like(grid._cos_lat("V"), f)
        north, south = faces[1:-1, 1:-1], faces[:-2, 1:-1]
    else:
        # between two V rows, the faces are the T rows: j+1 and j
        faces = _metric_like(grid._cos_lat("T"), f)
        north, south = faces[2:, 1:-1], faces[1:-1, 1:-1]
    cos = cos[1:-1, 1:-1]
    lon = _spherical_d2_lon(nb.c, nb.e, nb.w, cos, grid.dlon, grid.R)
    lat = _spherical_laplacian_merid(
        nb.c, nb.n, nb.s, north, south, cos, grid.dlat, grid.R
    )
    return _masked(interior(lon + lat, f), mask, at)


# ----------------------------------------------------------------------
# Collocated grids
# ----------------------------------------------------------------------


def _d(grid: Any, f: Any, axis: str, mask: Any, derivative: int = 1) -> Any:
    """``∂ⁿf`` along the coordinate of an axis: an angle, on a sphere."""
    kwargs: dict[str, Any] = {
        "axis": grid.axes.index(axis) - len(grid.axes),
        "derivative": derivative,
    }
    if grid._step(axis) is not None:
        kwargs["step"] = grid._step(axis)
        kwargs["periodic"] = axis in grid.periodic
    else:
        kwargs["coords"] = grid.coords(axis)
    kernel: Any = _collocated_difference
    if mask is not None:
        kernel = adaptive_difference
        kwargs["mask"] = ~mask.wet()
    return kernel(f, **kwargs)


def _only_t(name: str, at: Any) -> None:
    if at is not None and point_name(at) != "T":
        raise ValueError(f"{name}: a collocated grid has T points only, not {at}")


@divergence.register(CartesianAGrid2D)
@divergence.register(CartesianAGrid3D)
def _divergence_collocated(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    return _d(grid, u, "x", mask) + _d(grid, v, "y", mask)


@curl.register(CartesianAGrid2D)
@curl.register(CartesianAGrid3D)
def _curl_collocated(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    return _d(grid, v, "x", mask) - _d(grid, u, "y", mask)


@laplacian.register(CartesianAGrid1D)
def _laplacian_collocated_1d(grid: Any, f: Any, *, at: Any, mask: Any) -> Any:
    _only_t("laplacian", at)
    return _d(grid, f, "x", mask, 2)


@laplacian.register(CartesianAGrid2D)
@laplacian.register(CartesianAGrid3D)
def _laplacian_collocated(grid: Any, f: Any, *, at: Any, mask: Any) -> Any:
    _only_t("laplacian", at)
    return _d(grid, f, "y", mask, 2) + _d(grid, f, "x", mask, 2)


@gradient.register(CartesianAGrid2D)
@gradient.register(CartesianAGrid3D)
@gradient.register(SphericalAGrid2D)
def _gradient_collocated(grid: Any, f: Any, *, mask: Any) -> Any:
    return (
        difference(f, grid, axis="x", mask=mask),
        difference(f, grid, axis="y", mask=mask),
    )


@grad_perp.register(CartesianAGrid2D)
@grad_perp.register(CartesianAGrid3D)
@grad_perp.register(SphericalAGrid2D)
def _grad_perp_collocated(grid: Any, psi: Any, *, at: Any, mask: Any) -> Any:
    _only_t("grad_perp", at)
    dx, dy = gradient(psi, grid, mask=mask)
    return -dy, dx


def _sphere(grid: Any, like: Any) -> tuple[Any, Any]:
    """cos(lat), ``[Ny, 1]``, and the radius, in the dtype of a field."""
    return _metric_like(grid._cos_lat(), like), _metric_like(grid.R, like)


@divergence.register(SphericalAGrid2D)
def _divergence_collocated_sphere(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    cos, R = _sphere(grid, u)
    # 1/(R cos φ) [∂u/∂λ + ∂(v cos φ)/∂φ]
    total = _d(grid, u, "x", mask) + _d(grid, v * cos, "y", mask)
    return _safe_div_cos(total, cos, R)


@curl.register(SphericalAGrid2D)
def _curl_collocated_sphere(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    cos, R = _sphere(grid, u)
    # 1/(R cos φ) [∂v/∂λ - ∂(u cos φ)/∂φ]
    total = _d(grid, v, "x", mask) - _d(grid, u * cos, "y", mask)
    return _safe_div_cos(total, cos, R)


@laplacian.register(SphericalAGrid2D)
def _laplacian_collocated_sphere(grid: Any, f: Any, *, at: Any, mask: Any) -> Any:
    _only_t("laplacian", at)
    cos, R = _sphere(grid, f)
    xp = array_namespace(f)
    sin = _metric_like(xp.sin(xp.asarray(grid.coords("y")))[:, None], f)
    # 1/(R² cos² φ) ∂²f/∂λ² + 1/R² (∂²f/∂φ² - tan φ ∂f/∂φ)
    lon = _safe_div_cos(_d(grid, f, "x", mask, 2), cos, cos)
    lat = _d(grid, f, "y", mask, 2) - _safe_div_cos(
        sin * _d(grid, f, "y", mask), cos, xp.ones_like(cos)
    )
    return (lon + lat) / R**2
