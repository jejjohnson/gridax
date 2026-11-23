"""The canonical advective tendency, `advect`.

The face fluxes come from the axis-generic `face_flux` with any scheme, and
each is used twice with opposite signs, so the tendency is conservative. With a
mask the dry values never enter: they are zeroed before any stencil reads them,
a tier that would read a dry cell is not used, and a dry face has no flux.
"""

from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING, Any, Literal

from jaxtyping import Float
import numpy as np

from gridax._src.advection.reconstruct import check_scheme, face_flux, hierarchy_of
from gridax._src.api import grid_dispatch
from gridax._src.backend import set_at
from gridax._src.fields.located import located
from gridax._src.grid import (
    CartesianCGrid1D,
    CartesianCGrid2D,
    CartesianCGrid3D,
    SphericalCGrid2D,
    SphericalCGrid3D,
)
from gridax._src.operators._utils import _metric_like, _safe_div_cos, _spacing
from gridax._src.pytree import array_namespace
from gridax._src.schemes.base import Scheme

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["advect"]

Wall = Literal["closed", "open"]


@grid_dispatch
@located("T", q="T", u="U", v="V")
def advect(
    q: Float[Array, "..."],
    u: Float[Array, "..."],
    v: Float[Array, "..."] | None,
    grid: Any,
    *,
    scheme: Scheme,
    mask: Any = None,
    wall: Wall = "closed",
) -> Float[Array, "..."]:
    """The finite-volume tendency of a tracer, ``-∇·(q u)``, at T.

    ``dq/dt = -(δₓ(u q̂) + δᵧ(v q̂))``, with ``q̂`` the face values of ``q`` by
    ``scheme`` (:func:`reconstruct`); on a sphere ``-1/(R cos φ) (∂λ(u q̂) +
    ∂φ(v q̂ cos φ))``. A 3-D grid is a stack of levels: the operator is
    horizontal.

    Parameters
    ----------
    q : CField or Float[Array, "*batch *grid"]
        Cell averages at T, with one ghost ring.
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity, at U and V; ``v`` is ``None`` on a 1-D grid.
    grid : grid
        A C-grid.
    scheme : Scheme
        Any of `gridax.schemes`.
    mask : Mask1D, Mask2D or Mask3D, optional
        The wet cells. Dry values are not read, no flux crosses a dry face,
        and the tendency is zero on dry cells.
    wall : {"closed", "open"}, default "closed"
        ``"closed"`` writes the cells that are two or more from the edge of
        the array, and takes no flux through the edge faces. ``"open"`` writes
        every interior cell, with the edge faces first-order upwind of the
        ghost ring, which the caller has filled (periodic, inflow, outflow);
        not with a mask.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At T, zero on the ghost ring.

    Raises
    ------
    TypeError
        If ``scheme`` is not a scheme, or ``v`` does not match the grid.
    NotImplementedError
        If the grid is not a C-grid, or for ``wall="open"`` with a mask.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> from gridax.schemes import Upwind
    >>> grid = gridax.CartesianCGrid1D.from_interior(6, 6.0)
    >>> q, u = np.arange(8.0), np.ones(8)
    >>> gridax.advect(q, u, None, grid, scheme=Upwind(1)).tolist()
    [0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0]
    """
    check_scheme(scheme)
    check_wall(wall, mask)
    return advect._impl(grid, q, u, v, scheme=scheme, mask=mask, wall=wall)


def check_wall(wall: str, mask: object) -> None:
    """Validate ``wall``, and reject an open wall with a mask."""
    if wall not in ("closed", "open"):
        raise ValueError(f"wall must be one of ['closed', 'open']; got {wall!r}")
    if wall == "open" and mask is not None:
        raise NotImplementedError(
            "wall='open' is not supported together with a mask; the open-wall "
            "ghost-flux path assumes a rectangular domain with no interior "
            "land. Use wall='closed' with the mask, or drop the mask."
        )


def _fluxes(
    grid: Any,
    q: Array,
    velocities: dict[str, Array],
    *,
    scheme: Scheme,
    mask: Any,
    wall: str,
) -> dict[str, Array]:
    """The flux through the faces along each horizontal axis, dry faces zeroed."""
    xp = array_namespace(q, *velocities.values())
    rank = len(grid.axes)
    points = {"x": "U", "y": "V"}
    wet: dict[str, Any] = {}
    if mask is not None:
        wet = {axis: mask.wet(at=points[axis]) for axis in velocities}
        # zero the dry inputs before any stencil reads them, so land values
        # (even NaN) reach neither the output nor its gradient
        q = xp.where(mask.wet(), q, 0.0)
        velocities = {a: xp.where(wet[a], c, 0.0) for a, c in velocities.items()}
    out = {}
    for axis, velocity in velocities.items():
        hierarchy = None
        if mask is not None:
            hierarchy = hierarchy_of(mask, axis, scheme)
        flux = face_flux(
            q,
            velocity,
            axis=grid.axes.index(axis) - rank,
            scheme=scheme,
            rank=rank,
            hierarchy=hierarchy,
            wet=None if mask is None else mask.wet(),
        )
        if mask is not None:
            # F[i+1/2] = 0 unless both cells of the face are wet
            flux = xp.where(wet[axis], flux, 0.0)
        if wall == "open":
            flux = _open(flux, velocity, q, grid.axes.index(axis) - rank)
        out[axis] = flux
    return out


def _open(flux: Array, velocity: Array, q: Array, position: int) -> Array:
    """``flux`` with its two edge faces replaced by first-order upwind ones.

    The ghost cell is read as it is, as the cell value outside the domain: the
    wrapped cell for a periodic axis, the boundary value for an inflow. So a
    Dirichlet inflow needs ``ghost = value``, and not the face reflection
    ``2 value - interior`` that `Dirichlet` writes for centred stencils.
    """
    xp = array_namespace(flux, velocity, q)

    def at(k: int) -> tuple[Any, ...]:
        return (..., k, *([slice(None)] * (-position - 1)))

    size = q.shape[position]
    for face in (0, size - 2):
        lo, hi, speed = q[at(face)], q[at(face + 1)], velocity[at(face)]
        # F[i+1/2] = u[i+1/2] (q[i] if u[i+1/2] >= 0 else q[i+1])
        flux = set_at(flux, at(face), xp.where(speed >= 0.0, lo, hi) * speed)
    return flux


def _written(grid: Any, wall: str) -> tuple[Any, ...]:
    """The cells the tendency is written on: every interior level, and their
    interior (an open wall) or the cells one further in."""
    g = 1 if wall == "open" else 2
    levels = (slice(1, -1),) if len(grid.axes) == 3 else ()
    return (..., *levels, *([slice(g, -g)] * (len(grid.axes) - len(levels))))


def _back(index: tuple[Any, ...], position: int) -> tuple[Any, ...]:
    """``index`` one cell back along an axis."""
    out = list(index)
    s = out[position]
    out[position] = slice(s.start - 1, s.stop - 1)
    return tuple(out)


def _components(grid: Any, u: Any, v: Any) -> dict[str, Any]:
    one = len(grid.axes) == 1
    if one != (v is None):
        what = "is None on a 1-D grid" if one else "is required on a 2-D or 3-D grid"
        raise TypeError(f"advect: v {what}")
    return {"x": u} if one else {"x": u, "y": v}


@advect.register(CartesianCGrid1D)
@advect.register(CartesianCGrid2D)
@advect.register(CartesianCGrid3D)
def _advect_cartesian(
    grid: Any,
    q: Any,
    u: Any,
    v: Any,
    *,
    scheme: Scheme,
    mask: Any,
    wall: str,
) -> Any:
    xp = array_namespace(q, u)
    fluxes = _fluxes(
        grid,
        q,
        _components(grid, u, v),
        scheme=scheme,
        mask=mask,
        wall=wall,
    )
    here = _written(grid, wall)
    terms = []
    for axis, flux in fluxes.items():
        position = grid.axes.index(axis) - len(grid.axes)
        # (F[i+1/2] - F[i-1/2]) / Δ
        step = _spacing(grid, axis, flux)
        terms.append((flux[here] - flux[_back(here, position)]) / step)
    total = -functools.reduce(operator.add, terms)
    out = set_at(xp.zeros(q.shape, dtype=xp.result_type(q, total)), here, total)
    return out if mask is None else xp.where(mask.wet(), out, 0.0)


@advect.register(SphericalCGrid2D)
@advect.register(SphericalCGrid3D)
def _advect_spherical(
    grid: Any,
    q: Any,
    u: Any,
    v: Any,
    *,
    scheme: Scheme,
    mask: Any,
    wall: str,
) -> Any:
    xp = array_namespace(q, u)
    fluxes = _fluxes(
        grid,
        q,
        _components(grid, u, v),
        scheme=scheme,
        mask=mask,
        wall=wall,
    )
    fe, fn = fluxes["x"], fluxes["y"]
    here = _written(grid, wall)
    plane = here[-2:]  # the cos(lat) arrays are 2-D
    cos_t = _metric_like(grid._cos_lat("T"), fe)[plane]
    cos_v = _metric_like(grid._cos_lat("V"), fn)
    dlon, dlat, R = (_metric_like(m, fe) for m in (grid.dlon, grid.dlat, grid.R))
    # -1/(R cos φ_T) [ (Fλ[i+1/2] - Fλ[i-1/2]) / dλ
    #                + (cos φ_N Fφ[j+1/2] - cos φ_S Fφ[j-1/2]) / dφ ]
    du = (fe[here] - fe[_back(here, -1)]) / dlon
    dv = (
        cos_v[plane] * fn[here] - cos_v[_back(plane, -2)] * fn[_back(here, -2)]
    ) / dlat
    total = -_safe_div_cos(du + dv, cos_t, R)
    out = set_at(xp.zeros(q.shape, dtype=xp.result_type(q, total)), here, total)
    return out if mask is None else xp.where(mask.wet(), out, 0.0)
