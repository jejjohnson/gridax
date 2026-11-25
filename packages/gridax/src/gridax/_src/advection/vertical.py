"""The vertical transport of a tracer: `advect_vertical` and `diffuse_vertical`.

`advect` and `diffuse` are horizontal, level by level. These are the two terms
along z that close a 3-D tracer equation, in flux form: a flux on the W faces,
used twice with opposite signs, so the sum over a column is what crosses its
bottom and its top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np
from pipekit_array.functions import namespace_of
from pipekit_array.hooks import set_at

from gridax._src.advection.advect import Wall, _open
from gridax._src.advection.reconstruct import axis_flux, check_scheme, extends
from gridax._src.api import grid_dispatch
from gridax._src.boundary.fill import BC
from gridax._src.fields.located import located
from gridax._src.grid import (
    CartesianCGrid3D,
    SphericalCGrid3D,
    StretchedCGrid3D,
)
from gridax._src.operators._utils import _spacing
from gridax._src.schemes.adaptive import policies
from gridax._src.schemes.base import Scheme

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["advect_vertical", "diffuse_vertical"]

#: Every interior level, and the level below it.
_HERE = (..., slice(1, -1), slice(1, -1), slice(1, -1))
_BELOW = (..., slice(0, -2), slice(1, -1), slice(1, -1))


@grid_dispatch
@located("T", q="T", w="W")
def advect_vertical(
    q: Float[Array, "..."],
    w: Float[Array, "..."],
    grid: Any,
    *,
    scheme: Scheme,
    mask: Any = None,
    wall: Wall = "closed",
    bc: BC | None = None,
) -> Float[Array, "..."]:
    """The vertical part of the tendency of a tracer, ``-∂(w q)/∂z``, at T.

    ``dq/dt[k] = -(w q̂[k+1/2] - w q̂[k-1/2]) / Δz``, with ``q̂`` the face values of
    ``q`` by ``scheme`` (:func:`reconstruct`). The horizontal part is
    :func:`advect`.

    Parameters
    ----------
    q : CField or Float[Array, "*batch Nz Ny Nx"]
        Cell averages at T, with one ghost level at the bottom and the top.
    w : CField or Float[Array, "*batch Nz Ny Nx"]
        The vertical velocity, at W: ``w[k]`` on the top face of ``T[k]``, so
        ``w[0]`` is on the floor and ``w[Nz-2]`` on the lid. A velocity at T is
        ``interpolate(w, grid, src="T", dst="W")`` inside, with ``w[0]`` for the
        caller to set.
    grid : CartesianCGrid3D or SphericalCGrid3D
        The grid.
    scheme : Scheme
        Any of `gridax.schemes`.
    mask : Mask3D, optional
        The wet cells. Dry values are not read, no flux crosses a dry face, and
        the tendency is zero on dry cells.
    wall : {"closed", "open"}, default "closed"
        ``"closed"``: no flux through the floor and the lid, whatever ``w``
        is there. ``"open"``: their flux is first-order upwind of the ghost
        levels, which the caller has filled.
    bc : BC, optional
        The condition of ``q`` at the bottom and the top, instead of ``wall``:
        as many ghost levels as the scheme reads are filled from it, and the
        flux through the floor and the lid is that of ``w`` there.

    Returns
    -------
    CField or Float[Array, "*batch Nz Ny Nx"]
        At T, on every interior cell, with a zero ghost ring.

    Raises
    ------
    TypeError
        If ``scheme`` is not a scheme.
    ValueError
        If ``wall`` is unknown, or ``bc`` comes with an open wall or lacks a face.
    NotImplementedError
        If the grid is not a 3-D C-grid with a uniform z.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> from gridax.schemes import Upwind
    >>> grid = gridax.CartesianCGrid3D.from_interior(2, 2, 4, 2.0, 2.0, 4.0)
    >>> q = np.broadcast_to(np.arange(6.0)[:, None, None], grid.shape)
    >>> w = np.ones(grid.shape)  # upwards
    >>> gridax.advect_vertical(q, w, grid, scheme=Upwind(1))[:, 1, 1].tolist()
    [0.0, -1.0, -1.0, -1.0, 3.0, 0.0]
    """
    check_scheme(scheme)
    _check_wall(wall)
    if extends(scheme, bc) and wall != "closed":
        raise ValueError("bc= (or periodic edges) says what the walls are: drop wall=")
    return advect_vertical._impl(grid, q, w, scheme=scheme, mask=mask, wall=wall, bc=bc)


@grid_dispatch
@located("T", f="T", kappa="T")
def diffuse_vertical(
    f: Float[Array, "..."],
    grid: Any,
    *,
    kappa: float | Float[Array, "..."],
    mask: Any = None,
    wall: Wall = "closed",
) -> Float[Array, "..."]:
    """The vertical diffusion of a tracer, ``∂(κ ∂f/∂z)/∂z``, at T.

    ``df/dt[k] = (κ ∂f/∂z[k+1/2] - κ ∂f/∂z[k-1/2]) / Δz[k]``, with ``κ`` the mean
    of the two cells on each face. The horizontal part is :func:`diffuse`.

    Parameters
    ----------
    f : CField or Float[Array, "*batch Nz Ny Nx"]
        The tracer, at T.
    grid : grid
        A 3-D C-grid; a stretched one has its own level thicknesses.
    kappa : float, CField or Float[Array, "*batch Nz Ny Nx"]
        The vertical diffusivity: a number, or a field at T.
    mask : Mask3D, optional
        The wet cells: no flux crosses a dry face, dry values are not read, and
        the tendency is zero on dry cells.
    wall : {"closed", "open"}, default "closed"
        ``"closed"``: no flux through the floor and the lid. ``"open"``: their
        flux is that of the ghost levels of ``f`` (and of a field ``kappa``),
        which the caller has filled, so a surface flux enters there.

    Returns
    -------
    CField or Float[Array, "*batch Nz Ny Nx"]
        At T, on every interior cell, with a zero ghost ring.

    Raises
    ------
    ValueError
        If ``wall`` is unknown.
    NotImplementedError
        If the grid is not a 3-D C-grid.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid3D.from_interior(2, 2, 4, 2.0, 2.0, 4.0)
    >>> z = np.broadcast_to(np.arange(6.0)[:, None, None], grid.shape)
    >>> gridax.diffuse_vertical(z**2, grid, kappa=0.5)[:, 1, 1].tolist()
    [0.0, 1.5, 1.0, 1.0, -3.5, 0.0]
    """
    _check_wall(wall)
    return diffuse_vertical._impl(grid, f, kappa=kappa, mask=mask, wall=wall)


def _check_wall(wall: str) -> None:
    if wall not in ("closed", "open"):
        raise ValueError(f"wall must be one of ['closed', 'open']; got {wall!r}")


def _column(flux: Any, f: Any, grid: Any, mask: Any) -> Any:
    """``(F[k+1/2] - F[k-1/2]) / Δz`` on the interior, zero elsewhere and on land."""
    xp = namespace_of(flux, f)
    step = _spacing(grid, "z", flux)
    if np.ndim(step):
        step = step[1:-1]
    values = (flux[_HERE] - flux[_BELOW]) / step
    out = set_at(xp.zeros(f.shape, dtype=xp.result_type(f, values)), _HERE, values)
    return out if mask is None else xp.where(mask.wet(), out, 0.0)


def _faces(flux: Any, wall: str) -> Any:
    """The flux, zero on the floor and the lid of a closed column."""
    if wall == "open":
        return flux
    for face in (0, flux.shape[-3] - 2):
        flux = set_at(flux, (..., face, slice(None), slice(None)), 0.0)
    return flux


@advect_vertical.register(CartesianCGrid3D)
@advect_vertical.register(SphericalCGrid3D)
def _advect_vertical(
    grid: Any, q: Any, w: Any, *, scheme: Scheme, mask: Any, wall: str, bc: BC | None
) -> Any:
    xp = namespace_of(q, w)
    if mask is not None:
        # the dry values enter neither the output nor its gradient
        if policies(scheme, bc, edges="narrow", dry="narrow")[2] != "none":
            q = xp.where(mask.wet(), q, 0.0)
        w = xp.where(mask.wet(at="W"), w, 0.0)
    flux = axis_flux(grid, q, w, axis="z", scheme=scheme, mask=mask, bc=bc)
    if mask is not None:
        flux = xp.where(mask.wet(at="W"), flux, 0.0)
    if not extends(scheme, bc):
        flux = _open(flux, w, q, -3) if wall == "open" else _faces(flux, wall)
    return -_column(flux, q, grid, mask)


@diffuse_vertical.register(CartesianCGrid3D)
@diffuse_vertical.register(SphericalCGrid3D)
@diffuse_vertical.register(StretchedCGrid3D)
def _diffuse_vertical(grid: Any, f: Any, *, kappa: Any, mask: Any, wall: str) -> Any:
    xp = namespace_of(f, kappa)
    step = f[..., 1:, :, :] - f[..., :-1, :, :]  # [Nz-1]: f[k+1] - f[k]
    kappa = xp.asarray(kappa)
    if kappa.ndim >= 3:
        # κ[k+1/2] = 1/2 (κ[k] + κ[k+1])
        kappa = 0.5 * (kappa[..., 1:, :, :] + kappa[..., :-1, :, :])
    if mask is not None:
        # a difference is linear: zero here keeps a NaN out of the gradient too
        wet = mask.wet(at="W")[..., :-1, :, :]
        step = xp.where(wet, step, 0.0)
        kappa = xp.where(wet, kappa, 0.0) if kappa.ndim >= 3 else kappa
    # F[k+1/2] = κ[k+1/2] (f[k+1] - f[k]) / Δz[k+1/2]
    between = _spacing(grid, "z", f, at="W")
    if np.ndim(between):
        between = between[:-1]
    inside = kappa * step / between
    flux = xp.concat([inside, xp.zeros_like(inside[..., :1, :, :])], axis=-3)
    flux = flux.astype(f.dtype) if hasattr(flux, "astype") else flux
    return _column(_faces(flux, wall), f, grid, mask)
