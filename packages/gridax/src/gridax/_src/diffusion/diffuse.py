"""The canonical diffusive tendencies, `diffuse` and `biharmonic`.

Harmonic diffusion is in flux form: a face flux ``κ δh`` on the faces, zero on
dry and (with a closed wall) on edge faces, and its divergence. Each face flux
is used twice with opposite signs, so the tendency is conservative. Biharmonic
diffusion is two harmonic passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.advection.advect import Wall, check_wall
from gridax._src.api import grid_dispatch
from gridax._src.fields.located import located
from gridax._src.grid import (
    CartesianCGrid2D,
    CartesianCGrid3D,
    SphericalCGrid2D,
    SphericalCGrid3D,
)
from gridax._src.operators._ghost import interior
from gridax._src.operators._utils import _metric_like, _safe_div_cos, _spacing
from gridax._src.operators.calculus import _levels
from gridax._src.operators.stencils import _materialize, avg_x_fwd_2d, avg_y_fwd_2d
from gridax._src.pytree import array_namespace

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["biharmonic", "diffuse", "diffusive_fluxes"]


@grid_dispatch
@located("T", f="T", kappa="T")
def diffuse(
    f: Float[Array, "..."],
    grid: Any,
    *,
    kappa: float | Float[Array, "..."],
    mask: Any = None,
    wall: Wall = "closed",
) -> Float[Array, "..."]:
    """The harmonic diffusion of a tracer, ``∇·(κ∇f)``, at T.

    ``∂f/∂t = δₓ(κ δₓf) + δᵧ(κ δᵧf)``, with ``κ`` the mean of the two cells on
    each face; on a sphere with the metric of the fluxes and of their
    divergence. A 3-D grid is a stack of levels: the operator is horizontal.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        The tracer, at T.
    grid : grid
        A 2-D or 3-D C-grid.
    kappa : float, CField or Float[Array, "*batch *grid"]
        The diffusivity: a number, or a field at T.
    mask : Mask2D or Mask3D, optional
        The wet cells: no flux crosses a dry face, and the tendency is zero on
        dry cells. Dry values are read, and must be finite.
    wall : {"closed", "open"}, default "closed"
        ``"closed"``: no flux through the edge faces, and no ghost cell is
        read. ``"open"``: the edge faces take their flux from the ghost ring
        of ``f`` (and of a field ``kappa``), which the caller has filled; not
        with a mask, and Cartesian only.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At T, with a zero ghost ring.

    Raises
    ------
    NotImplementedError
        If the grid is not a 2-D or 3-D C-grid, or the wall is not available.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
    >>> x = np.broadcast_to(grid.coords("x"), grid.shape)
    >>> gridax.diffuse(x**2, grid, kappa=0.5)[2].tolist()
    [0.0, 1.5, 1.0, 1.0, -3.5, 0.0]
    """
    check_wall(wall, mask)
    return diffuse._impl(grid, f, kappa=kappa, mask=mask, wall=wall)


@grid_dispatch
@located("T", f="T", kappa4="T")
def biharmonic(
    f: Float[Array, "..."],
    grid: Any,
    *,
    kappa4: float | Float[Array, "..."],
    mask: Any = None,
) -> Float[Array, "..."]:
    """The biharmonic diffusion of a tracer, ``-κ₄ ∇²(∇²f)``, at T.

    Two passes of :func:`diffuse` with a unit diffusivity, so the walls and
    the coast are no-flux for ``f`` and for ``∇²f``, and no ghost cell is
    read. It damps a wave of wavenumber ``k`` at the rate ``κ₄ k⁴``.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        The tracer, at T.
    grid : grid
        A 2-D or 3-D C-grid.
    kappa4 : float, CField or Float[Array, "*batch *grid"]
        The biharmonic diffusivity, outside the divergence.
    mask : Mask2D or Mask3D, optional
        The wet cells. Dry values are not read.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At T, with a zero ghost ring.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
    >>> float(abs(gridax.biharmonic(np.ones(grid.shape), grid, kappa4=1.0)).max())
    0.0
    """
    return biharmonic._impl(grid, f, kappa4=kappa4, mask=mask)


@grid_dispatch
@located(("U", "V"), f="T", kappa="T")
def diffusive_fluxes(
    f: Float[Array, "..."],
    grid: Any,
    *,
    kappa: float | Float[Array, "..."],
    mask: Any = None,
    wall: Wall = "closed",
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """The fluxes whose divergence is :func:`diffuse`, ``κ δₓf`` and ``κ δᵧf``.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        The tracer, at T.
    grid : grid
        A 2-D or 3-D C-grid.
    kappa : float, CField or Float[Array, "*batch *grid"]
        The diffusivity: a number, or a field at T.
    mask : Mask2D or Mask3D, optional
        The wet cells: no flux crosses a dry face.
    wall : {"closed", "open"}, default "closed"
        As in :func:`diffuse`.

    Returns
    -------
    tuple
        The fluxes at U and V, with the sign of the gradient, zero on the
        faces :func:`diffuse` does not use.

    Raises
    ------
    NotImplementedError
        If the grid is not a 2-D or 3-D C-grid, or the wall is not available.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
    >>> x = np.broadcast_to(grid.coords("x"), grid.shape)
    >>> gridax.diffusive_fluxes(x, grid, kappa=0.5)[0][2].tolist()
    [0.0, 0.5, 0.5, 0.5, 0.0, 0.0]
    """
    check_wall(wall, mask)
    return diffusive_fluxes._impl(grid, f, kappa=kappa, mask=mask, wall=wall)


# ----------------------------------------------------------------------
# Kernels on [*batch, Ny, Nx]
# ----------------------------------------------------------------------


def _face_kappa(kappa: Any, wall: str = "closed") -> tuple[Any, Any]:
    """The diffusivity on the faces the kernels compute, the mean of two cells.

        κ[j, i+1/2] = (κ[j, i] + κ[j, i+1]) / 2
        κ[j+1/2, i] = (κ[j, i] + κ[j+1, i]) / 2

    ``wall="closed"`` gives the interior faces, ``[Ny-2, Nx-3]`` and
    ``[Ny-3, Nx-2]``, and reads no ghost cell; ``"open"`` gives every face,
    ``[Ny-2, Nx-1]`` and ``[Ny-1, Nx-2]``. A scalar is returned as it is.
    """
    xp = array_namespace(kappa)
    kappa = xp.asarray(kappa)
    if kappa.ndim < 2:
        return kappa, kappa
    if wall == "open":
        # κ[j, i+1/2], i = 0 .. Nx-2, and κ[j+1/2, i], j = 0 .. Ny-2
        kx = 0.5 * (kappa[..., 1:-1, :-1] + kappa[..., 1:-1, 1:])
        ky = 0.5 * (kappa[..., :-1, 1:-1] + kappa[..., 1:, 1:-1])
    else:
        # κ[j, i+1/2], i = 1 .. Nx-3, and κ[j+1/2, i], j = 1 .. Ny-3
        kx = avg_x_fwd_2d(kappa)[..., :, :-1]
        ky = avg_y_fwd_2d(kappa)[..., :-1, :]
    return kx, ky


def _fluxes(
    h: Any, kappa: Any, grid: Any, mu: Any, mv: Any, wall: str
) -> tuple[Any, Any]:
    """The face fluxes along x and y, on the faces of :func:`_face_kappa`.

    In the dtype of ``h``: a float64 ``kappa`` does not promote a float32 field.
    """
    kx, ky = _face_kappa(kappa, wall)
    # the cells after and before each face i+1/2, i = 0 .. Nx-2 (open) or
    # 1 .. Nx-3 (closed); one slice per offset, so the VJP has one pad each
    after, before = (
        (slice(1, None), slice(None, -1))
        if wall == "open"
        else (slice(2, -1), slice(1, -2))
    )
    c = slice(1, -1)
    # κ[j, i+1/2] (h[j, i+1] - h[j, i]), κ[j+1/2, i] (h[j+1, i] - h[j, i])
    fx = kx * (h[..., c, after] - h[..., c, before])
    fy = ky * (h[..., after, c] - h[..., before, c])
    if isinstance(grid, SphericalCGrid2D | SphericalCGrid3D):
        cos_u = _metric_like(grid._cos_lat("U"), h)[1:-1, 1:-2]
        fx = _safe_div_cos(fx, cos_u, _metric_like(grid.R * grid.dlon, h))
        fy = fy / _metric_like(grid.R * grid.dlat, h)
    else:
        fx, fy = fx / _spacing(grid, "x", h), fy / _spacing(grid, "y", h)
    fx, fy = fx.astype(h.dtype), fy.astype(h.dtype)
    if mu is not None:
        fx = fx * mu[..., c, before]
    if mv is not None:
        fy = fy * mv[..., before, c]
    return fx, fy


def _sides(fx: Any, fy: Any, wall: str) -> tuple[Any, Any, Any, Any]:
    """The flux on the east, west, north and south face of every interior cell."""
    if wall == "open":
        return fx[..., 1:], fx[..., :-1], fy[..., 1:, :], fy[..., :-1, :]
    xp = array_namespace(fx)
    lead = ((0, 0),) * (fx.ndim - 2)

    # the faces of a closed wall are zero, and enter as a pad, so that the
    # tendency and its VJP fuse into one pass with no flux buffer
    def pad(x: Any, y: tuple[int, int], along: tuple[int, int]) -> Any:
        return xp.pad(x, (*lead, y, along))

    none = (0, 0)
    return (
        pad(fx, none, (0, 1)),
        pad(fx, none, (1, 0)),
        pad(fy, (0, 1), none),
        pad(fy, (1, 0), none),
    )


def _wet(mask: Any, grid: Any) -> tuple[Any, Any, Any]:
    """The wet T, U and V points, on the interior levels."""
    if mask is None:
        return None, None, None
    return (
        mask.wet(),
        _levels(grid, mask.wet(at="U")),
        _levels(grid, mask.wet(at="V")),
    )


def _kappa_levels(grid: Any, kappa: Any) -> Any:
    """A 3-D diffusivity on the interior levels; any other as it is."""
    xp = array_namespace(kappa)
    kappa = xp.asarray(kappa)
    return _levels(grid, kappa) if kappa.ndim >= 3 else kappa


def _tendency(grid: Any, f: Any, kappa: Any, mask: Any, wall: str) -> Any:
    mh, mu, mv = _wet(mask, grid)
    h = _levels(grid, f)
    fx, fy = _fluxes(h, _kappa_levels(grid, kappa), grid, mu, mv, wall)
    east, west, north, south = _sides(fx, fy, wall)
    if isinstance(grid, SphericalCGrid2D | SphericalCGrid3D):
        cos_v = _metric_like(grid._cos_lat("V"), h)
        cos_t = _metric_like(grid._cos_lat("T"), h)[1:-1, 1:-1]
        dlon, dlat, R = (_metric_like(m, h) for m in (grid.dlon, grid.dlat, grid.R))
        # 1/(R cos φ_T) [ (Fλ_E - Fλ_W) / dλ + (cos φ_N Fφ_N - cos φ_S Fφ_S) / dφ ]
        du = (east - west) / dlon
        dv = (cos_v[1:-1, 1:-1] * north - cos_v[:-2, 1:-1] * south) / dlat
        out = _safe_div_cos(du + dv, cos_t, R)
    else:
        # (Fˣ_E - Fˣ_W) / dx + (Fʸ_N - Fʸ_S) / dy
        du = (east - west) / _spacing(grid, "x", h)
        dv = (north - south) / _spacing(grid, "y", h)
        out = du + dv
    out = interior(out, f)
    return out if mh is None else out * mh


@diffusive_fluxes.register(CartesianCGrid2D)
@diffusive_fluxes.register(CartesianCGrid3D)
@diffusive_fluxes.register(SphericalCGrid2D)
@diffusive_fluxes.register(SphericalCGrid3D)
def _diffusive_fluxes(
    grid: Any, f: Any, *, kappa: Any, mask: Any, wall: str
) -> tuple[Any, Any]:
    if isinstance(grid, SphericalCGrid2D | SphericalCGrid3D):
        _closed(wall)
    xp = array_namespace(f, kappa)
    _, mu, mv = _wet(mask, grid)
    h = _levels(grid, f)
    fx, fy = _fluxes(h, _kappa_levels(grid, kappa), grid, mu, mv, wall)
    # x faces on [1:-1, :-1] (open) or [1:-1, 1:-2] (closed); y likewise
    edge = (0, 1) if wall == "open" else (1, 2)
    levels = ((1, 1),) if len(grid.axes) == 3 else ()
    lead = (*((0, 0),) * (fx.ndim - 2 - len(levels)), *levels)
    return (
        xp.pad(fx, (*lead, (1, 1), edge)),
        xp.pad(fy, (*lead, edge, (1, 1))),
    )


@diffuse.register(CartesianCGrid2D)
@diffuse.register(CartesianCGrid3D)
def _diffuse_cartesian(grid: Any, f: Any, *, kappa: Any, mask: Any, wall: str) -> Any:
    return _tendency(grid, f, kappa, mask, wall)


@diffuse.register(SphericalCGrid2D)
@diffuse.register(SphericalCGrid3D)
def _diffuse_spherical(grid: Any, f: Any, *, kappa: Any, mask: Any, wall: str) -> Any:
    _closed(wall)
    return _tendency(grid, f, kappa, mask, wall)


def _closed(wall: str) -> None:
    if wall != "closed":
        raise NotImplementedError("diffuse: wall='open' is not defined on the sphere")


@biharmonic.register(CartesianCGrid2D)
@biharmonic.register(CartesianCGrid3D)
@biharmonic.register(SphericalCGrid2D)
@biharmonic.register(SphericalCGrid3D)
def _biharmonic(grid: Any, f: Any, *, kappa4: Any, mask: Any) -> Any:
    xp = array_namespace(f)
    if mask is not None:
        # land may hold NaN: zero it before any stencil reads it
        f = xp.where(mask.wet(), f, 0.0)
    # the second pass reads the first at four offsets: keep it as one buffer
    # rather than letting XLA recompute it per offset
    once = _materialize(_tendency(grid, f, 1.0, mask, "closed"))
    return -kappa4 * _tendency(grid, once, 1.0, mask, "closed")
