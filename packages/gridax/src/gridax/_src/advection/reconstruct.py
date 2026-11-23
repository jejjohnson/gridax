"""One axis-generic face flux (Layer 1) and the canonical ``reconstruct``.

``face_flux(q, u, axis=..., scheme=...)`` is the advective flux ``F[i+1/2] =
q[i+1/2] u[i+1/2]`` on the faces along ``axis`` of a C-grid array with one
ghost ring, on bare arrays. The face value comes from the widest tier of
``scheme`` that fits: inside the array, and with ``mask`` on wet cells too.
Upwind reconstructions narrow at a wall rather than shift, so they never read a
cell downwind of the face. Every ``Reconstruction1D/2D/3D`` method, and
:func:`reconstruct` and :func:`~gridax.advect`, are calls of this function.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.advection.flux import upwind_flux
from gridax._src.api import grid_dispatch
from gridax._src.backend import pad, set_at
from gridax._src.boundary.fill import (
    BC,
    extend,
    extend_wet,
    fill_coasts,
    wrapping,
)
from gridax._src.fields.located import located
from gridax._src.grid import (
    CartesianCGrid1D,
    CartesianCGrid2D,
    CartesianCGrid3D,
    SphericalCGrid2D,
    SphericalCGrid3D,
)
from gridax._src.grid.points import point_stagger
from gridax._src.mask.base import tiers_of
from gridax._src.operators._ghost import interior
from gridax._src.pytree import array_namespace
from gridax._src.schemes.adaptive import policies
from gridax._src.schemes.base import Ladder, Scheme, Tier
from gridax._src.stencils.staggered import along, along_wet, crop, wet_windows

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["face_flux", "reconstruct"]

_DIRECTIONS = {-1: "x", -2: "y", -3: "z"}

#: ``{size: mask}``, the output of ``mask.tiers``.
type Hierarchy = dict[int, Any]


@grid_dispatch
@located("axis", q="T", u="axis")
def reconstruct(
    q: Float[Array, "..."],
    u: Float[Array, "..."],
    grid: Any,
    *,
    axis: str,
    scheme: Scheme,
    mask: Any = None,
    bc: BC | None = None,
) -> Float[Array, "..."]:
    """The advective flux ``q̂ u`` on the faces along ``axis``.

    ``q̂`` is the face value of the cell averages ``q`` by ``scheme``. For flow
    towards ``+axis`` the face ``i + 1/2`` takes the cells ``i + lo … i + hi``
    of the widest tier of the scheme that fits, and for flow towards ``-axis``
    their mirror; a scheme that is not upwinded takes the same cells for
    either. A tier fits if its cells are inside the array, or ``bc`` says what
    is outside it, and, with ``mask``, wet; so no face reads a dry cell, and a
    face between two wet cells always has a flux.

    Parameters
    ----------
    q : CField or Float[Array, "*batch *grid"]
        Cell averages at T, with one ghost ring.
    u : CField or Float[Array, "*batch *grid"]
        The velocity normal to the faces: at U for x, V for y, W for z.
    grid : grid
        A C-grid. The metric does not enter.
    axis : str
        One of ``grid.axes``.
    scheme : Scheme
        Any of `gridax.schemes`.
    mask : Mask1D, Mask2D or Mask3D, optional
        The wet cells.
    bc : BC, optional
        The condition of ``q`` on the faces normal to ``axis``. The ghost ring
        of ``q`` is filled from it, and as many layers beyond as the widest
        tier reads, so every face takes that tier.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        The flux, at U, V or W, with a zero ghost ring.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> from gridax.schemes import Upwind
    >>> grid = gridax.CartesianCGrid1D.from_interior(4, 4.0)
    >>> q, u = np.arange(6.0), np.ones(6)
    >>> gridax.reconstruct(q, u, grid, axis="x", scheme=Upwind(1)).tolist()
    [0.0, 1.0, 2.0, 3.0, 4.0, 0.0]
    """
    check_scheme(scheme)
    return reconstruct._impl(grid, q, u, axis=axis, scheme=scheme, mask=mask, bc=bc)


@reconstruct.register(CartesianCGrid1D)
@reconstruct.register(CartesianCGrid2D)
@reconstruct.register(CartesianCGrid3D)
@reconstruct.register(SphericalCGrid2D)
@reconstruct.register(SphericalCGrid3D)
def _reconstruct_staggered(
    grid: Any,
    q: Any,
    u: Any,
    *,
    axis: str,
    scheme: Scheme,
    mask: Any,
    bc: BC | None,
) -> Any:
    if axis not in grid.axes:
        raise ValueError(f"axis must be one of {grid.axes}; got {axis!r}")
    flux = axis_flux(grid, q, u, axis=axis, scheme=scheme, mask=mask, bc=bc)
    if not extends(scheme, bc):
        return flux
    position = grid.axes.index(axis) - len(grid.axes)
    ring = (..., slice(1, -1), *(slice(None),) * (-position - 1))
    return interior(flux[ring], flux)


def extends(scheme: Scheme, bc: BC | None) -> bool:
    """Whether the fluxes read ghost layers that are filled for them."""
    _, edges, _ = policies(scheme, bc, edges="narrow", dry="narrow")
    return edges in ("extend", "periodic")


_FACES = {"x": "U", "y": "V", "z": "W"}


def axis_flux(
    grid: Any,
    q: Array,
    u: Array,
    *,
    axis: str,
    scheme: Scheme,
    mask: Any,
    bc: BC | None,
) -> Array:
    """The flux through the faces along ``axis``.

    With a zero ghost ring; where ghost layers are filled for it (`extends`)
    the first face (the wall on the low side) has its flux too.
    """
    rank = len(grid.axes)
    position = grid.axes.index(axis) - rank
    scheme, edges, dry = policies(scheme, bc, edges="narrow", dry="narrow")
    halo = scheme.halo(staggered=True)
    if mask is not None and dry == "bias":
        windows = wet_windows(scheme, "fwd", 0)
        halo = max(abs(k) for cells in windows for k in cells)
    if mask is not None and dry == "fill":
        assert bc is not None  # policies() asked for it
        stagger = point_stagger("T")
        q = fill_coasts(q, grid, bc, mask, stagger, width=grid.ghost, layers=halo)

    wet = None
    tiers: Hierarchy | None = None
    if mask is not None and dry in ("narrow", "bias"):
        wet = mask.wet()
    around: BC | None = None
    extra = 0
    size = q.shape[position]
    if edges in ("extend", "periodic"):
        around = wrapping([axis]) if edges == "periodic" else bc
        assert around is not None  # policies() asked for it
        # at least one layer, so the wall on the low side is an interior face
        extra = max(halo - grid.ghost, 1)
        q = extend(q, grid, around, point_stagger("T"), [axis], extra)
        widths = [(0, 0)] * u.ndim
        widths[position] = (extra, extra)
        u = pad(u, tuple(widths))
        wet = None if wet is None else extend_wet(wet, grid, around, [axis], extra)
    if wet is not None and scheme.upwinded:
        if around is None:
            tiers = hierarchy_of(mask, axis, scheme)
        else:
            tiers = tiers_of(wet, position, [tier.size for tier in scheme.tiers()])

    if dry == "bias" and wet is not None:
        flux = _shifted_flux(q, u, position, rank, scheme, wet)
    elif edges == "bias" and wet is None:
        flux = _shifted_flux(q, u, position, rank, scheme, None)
    else:
        flux = face_flux(
            q,
            u,
            axis=position,
            scheme=scheme,
            rank=rank,
            hierarchy=tiers,
            wet=wet,
        )
    if edges == "none":
        flux = _without_overhang(flux, position, scheme)
    if mask is not None and dry in ("fill", "none", "bias"):
        # these read across the coast, or leave a value on it
        on_faces = mask.wet(at=_FACES[axis])
        if around is not None:
            on_faces = extend_wet(on_faces, grid, around, [axis], extra)
        flux = array_namespace(flux).where(on_faces, flux, 0.0)
    if around is None:
        return flux

    def faces(start: int, stop: int) -> tuple[Any, ...]:
        return (..., slice(start, stop), *(slice(None),) * (-position - 1))

    # the last index has no face of the grid
    kept = flux[faces(extra, extra + size - 1)]
    xp = array_namespace(flux)
    return xp.concat([kept, xp.zeros_like(flux[faces(0, 1)])], axis=position)


def _shifted_flux(
    q: Array, u: Array, position: int, rank: int, scheme: Any, wet: Array | None
) -> Array:
    """The flux with the stencil shifted into the array, or onto wet cells."""
    others = [a for a in range(-rank, 0) if a != position]
    cells: Scheme = scheme
    if scheme.data is None:
        cells = dataclasses.replace(scheme, data="cell")
    if wet is None:
        face = along(q, position, "fwd", cells, derivative=0)
    else:
        face = along_wet(q, wet, position, "fwd", cells, derivative=0)
    inner = crop(q.ndim, range(-rank, 0))
    return interior(face[crop(q.ndim, others)] * u[inner], q)


def _without_overhang(flux: Array, position: int, scheme: Scheme) -> Array:
    """Zero on the faces whose widest stencil would leave the array."""
    lo, hi = scheme.footprint(staggered=True)
    n = flux.shape[position]
    for i in range(1, n - 1):
        if i + lo < 0 or i + hi > n - 1:
            flux = set_at(flux, (..., i, *(slice(None),) * (-position - 1)), 0.0)
    return flux


def check_scheme(scheme: Any) -> None:
    """Raise unless ``scheme`` is a scheme."""
    if not isinstance(scheme, Scheme):
        raise TypeError(
            f"scheme must be one of gridax.schemes, e.g. WENO(5); got {scheme!r}"
        )


def hierarchy_of(mask: Any, axis: str, scheme: Scheme) -> Hierarchy | None:
    """The cells of each tier of an upwinded scheme (``mask.tiers``)."""
    if not scheme.upwinded:
        return None
    return mask.tiers(axis, sizes=[tier.size for tier in scheme.tiers()])


def face_flux(
    q: Float[Array, "*batch *spatial"],
    u: Float[Array, "*batch *spatial"],
    *,
    axis: int,
    scheme: Scheme,
    rank: int | None = None,
    hierarchy: Hierarchy | None = None,
    wet: Float[Array, "*spatial"] | None = None,
) -> Float[Array, "*batch *spatial"]:
    """The advective flux ``q_face * u`` on the faces along ``axis``, on arrays.

        F[i+1/2] = u[i+1/2] * R+(q[i+lo .. i+hi])        u[i+1/2] >= 0
                 = u[i+1/2] * R-(q[i+1-hi .. i+1-lo])    u[i+1/2] <  0

    Parameters
    ----------
    q : Float[Array, "*batch *spatial"]
        Cell averages at T points, with one ghost ring on every spatial axis.
    u : Float[Array, "*batch *spatial"]
        The velocity normal to the faces along ``axis``, on ``q``'s shape.
    axis : int
        The axis of the faces, of the array.
    scheme : Scheme
        Any of `gridax.schemes`.
    rank : int, optional
        The number of spatial axes, the last ones; all of them by default.
    hierarchy : dict, optional
        The adaptive masks of the tiers of an upwinded scheme
        (`hierarchy_of`).
    wet : Float[Array, "*spatial"], optional
        The wet cells, for a scheme that is not upwinded.

    Returns
    -------
    Float[Array, "*batch *spatial"]
        The flux on the faces along ``axis``, with a zero ghost ring.
    """
    rank = q.ndim if rank is None else rank
    position = axis % q.ndim
    if scheme.upwinded and hierarchy is not None:
        return _masked_upwind(q, u, position, scheme, rank, hierarchy)
    if not scheme.upwinded and wet is not None:
        return _masked_fixed(q, u, position, scheme, rank, wet)
    xp = array_namespace(q, u)
    last = xp.moveaxis(q, position, -1)
    velocity = xp.moveaxis(u, position, -1)
    # the other spatial axes keep their interior; the faces along `axis` are 1 .. N-2
    crop = (..., *([slice(1, -1)] * (rank - 1)), slice(None))
    last, velocity = last[crop], velocity[crop][..., 1:-1]
    tiers = scheme.tiers()
    positive = _faces(xp, last, tiers, upwind="left")
    if scheme.upwinded:
        negative = _faces(xp, last, tiers, upwind="right")
        face = xp.where(velocity >= 0.0, positive, negative)
    else:
        face = positive
    return interior(xp.moveaxis(face * velocity, -1, position), q)


def _faces(xp: Any, q: Array, tiers: tuple[Tier, ...], *, upwind: str) -> Array:
    """Face values ``i + 1/2``, ``i = 1 .. N-2``, along the last axis.

    Each face takes the first (widest) tier whose cells are in ``[0, N-1]``;
    faces with the same tier form a run, computed with one call on slices.
    """
    size = q.shape[-1]
    runs: list[tuple[int, int, int]] = []  # (tier, first face, last face + 1)
    for i in range(1, size - 1):
        fits = [
            i + lo >= 0 and i + hi <= size - 1
            for lo, hi in (_reads(tier, upwind) for tier in tiers)
        ]
        if not any(fits):
            raise ValueError(f"no tier of the scheme fits an axis of length {size}")
        k = fits.index(True)
        if runs and runs[-1][0] == k:
            runs[-1] = (k, runs[-1][1], i + 1)
        else:
            runs.append((k, i, i + 1))
    pieces = []
    for k, start, stop in runs:
        tier = tiers[k]
        lo, hi = _reads(tier, upwind)
        fn = tier.left if upwind == "left" else tier.right
        cells = [q[..., start + o : stop + o] for o in range(lo, hi + 1)]
        pieces.append(fn(*cells))
    return pieces[0] if len(pieces) == 1 else xp.concat(pieces, axis=-1)


def _reads(tier: Tier, upwind: str) -> tuple[int, int]:
    """The cells the face ``i + 1/2`` reads, relative to ``i``."""
    if upwind == "left":
        return tier.lo, tier.hi
    return 1 - tier.hi, 1 - tier.lo


def _masked_upwind(
    q: Array, u: Array, position: int, scheme: Scheme, rank: int, hierarchy: Hierarchy
) -> Array:
    """Blend the tiers by the adaptive masks at the upwind cell."""
    tiers = scheme.tiers()
    # each size reconstructs with that rung and the narrower ones below it
    rec_funcs = {}
    for k, tier in enumerate(tiers):
        rung = Ladder(tiers[k:], scheme)

        def rec(field: Array, vel: Array, _rung: Scheme = rung) -> Array:
            return face_flux(field, vel, axis=position, scheme=_rung, rank=rank)

        rec_funcs.setdefault(tier.size, rec)
    return upwind_flux(
        q, u, dim=position, rec_funcs=rec_funcs, mask_hierarchy=hierarchy
    )


def _masked_fixed(
    q: Array, u: Array, position: int, scheme: Scheme, rank: int, wet: Array
) -> Array:
    """The widest tier whose cells are all wet, for a scheme that is not upwinded.

    Such a scheme reads the same cells for either sign of the velocity, so a
    tier fits a face if every cell it reads is wet, and a face with a dry cell
    on either side has no flux.
    """
    xp = array_namespace(q, u)
    tiers = scheme.tiers()
    along = position - q.ndim
    last = xp.moveaxis(xp.asarray(wet, dtype=bool), along, -1)
    reach = max(max(-tier.lo, tier.hi) for tier in tiers)
    land = xp.zeros((*last.shape[:-1], reach), dtype=bool)
    padded = xp.concat([land, last, land], axis=-1)
    size = q.shape[position]

    def cells(offset: int) -> Array:
        """Whether cell ``i + offset`` is wet, per face ``i + 1/2``."""
        return padded[..., reach + offset : reach + offset + size]

    flux = xp.zeros_like(xp.moveaxis(q, position, -1))
    for k in range(len(tiers) - 1, -1, -1):
        fits = cells(0) & cells(1)
        for offset in range(tiers[k].lo, tiers[k].hi + 1):
            fits = fits & cells(offset)
        rung = Ladder(tiers[k:], scheme)
        rung_flux = face_flux(q, u, axis=position, scheme=rung, rank=rank)
        flux = xp.where(fits, xp.moveaxis(rung_flux, position, -1), flux)
    return xp.moveaxis(flux, -1, position)
