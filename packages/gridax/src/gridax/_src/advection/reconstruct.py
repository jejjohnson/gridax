"""One axis-generic upwind reconstruction (Layer 2), for every scheme and rank.

``reconstruct(q, u, axis=..., scheme=...)`` returns the advective flux
``F[i+1/2] = q[i+1/2] u[i+1/2]`` on the faces along ``axis`` of a C-grid
array with one ghost ring. The face value comes from the widest tier of
``scheme`` that fits: inside the array, and with ``mask`` on wet cells too.
Upwind reconstructions narrow at a wall rather than shift, so they never
read a cell downwind of the face. Every ``Reconstruction1D/2D/3D`` method is
a call of this function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.advection.flux import upwind_flux
from gridax._src.operators._ghost import interior
from gridax._src.pytree import array_namespace
from gridax._src.schemes.base import Ladder, Scheme, Tier

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["reconstruct"]

_DIRECTIONS = {-1: "x", -2: "y", -3: "z"}


def reconstruct(
    q: Float[Array, "*spatial"],
    u: Float[Array, "*spatial"],
    *,
    axis: int,
    scheme: Scheme,
    mask: Any = None,
) -> Float[Array, "*spatial"]:
    """The upwind advective flux ``q_face * u`` on the faces along ``axis``.

    For flow towards ``+axis`` the face ``i + 1/2`` takes the cell averages
    ``q[i + lo] … q[i + hi]`` of the widest tier of ``scheme`` that fits; for
    flow towards ``-axis`` it takes their mirror ``q[i + 1 - hi] … q[i + 1 -
    lo]``. Each face takes the widest tier whose cells are inside the array
    and, when ``mask`` is given, wet (``mask.get_adaptive_masks``, as
    ``Advection*D`` uses it); the last tier, first-order upwind, always fits.
    A scheme that is not upwinded (``Centred``, ``Biased``) reads the same
    cells for either sign, and a face between two wet cells always has a flux.

        F[i+1/2] = u[i+1/2] * R+(q[i+lo .. i+hi])        u[i+1/2] >= 0
                 = u[i+1/2] * R-(q[i+1-hi .. i+1-lo])    u[i+1/2] <  0

    Parameters
    ----------
    q : Float[Array, "*spatial"]
        Cell averages at T points, with one ghost ring on every axis.
    u : Float[Array, "*spatial"]
        The velocity normal to the faces along ``axis``, on ``q``'s shape
        (U points for x, V points for y).
    axis : int
        The axis of the faces: ``-1`` (x), ``-2`` (y) or ``-3`` (z), or the
        same as a non-negative index.
    scheme : Scheme
        One of ``gridax.schemes``: ``Upwind(order)``, ``Centred(order)``,
        ``Biased(order, side)``, ``WENO(order, variant)`` or ``TVD(limiter)``.
    mask : Mask1D, Mask2D or Mask3D, optional
        A wet mask of ``q``'s rank; the tiers then also stop at the coast.

    Returns
    -------
    Float[Array, "*spatial"]
        The flux on the faces along ``axis``, ``q``'s shape, with a zero
        ghost ring.

    Examples
    --------
    >>> import numpy as np
    >>> import gridax.advection as gadv
    >>> from gridax.schemes import WENO
    >>> q, u = np.ones((6, 8)), np.ones((6, 8))
    >>> flux = gadv.reconstruct(q, u, axis=-1, scheme=WENO(5))
    >>> flux.shape, float(flux[2, 3])
    ((6, 8), 1.0)
    """
    if not isinstance(scheme, Scheme):
        raise TypeError(
            f"scheme must be one of gridax.schemes, e.g. WENO(5); got {scheme!r}"
        )
    position = axis % q.ndim
    if mask is not None:
        return _masked(q, u, position, scheme, mask)
    xp = array_namespace(q, u)
    last = xp.moveaxis(q, position, -1)
    velocity = xp.moveaxis(u, position, -1)
    # the other axes keep their interior only; the faces along `axis` are 1 .. N-2
    crop = (*([slice(1, -1)] * (q.ndim - 1)), slice(None))
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


def _masked(q: Array, u: Array, position: int, scheme: Scheme, mask: Any) -> Array:
    """Blend the tiers by the mask's adaptive hierarchy, as ``Advection*D`` does."""
    if not scheme.upwinded:
        return _masked_fixed(q, u, position, scheme, mask)
    tiers = scheme.tiers()
    sizes = tuple(sorted({tier.size for tier in tiers}))
    if q.ndim == 1:
        hierarchy = mask.get_adaptive_masks(stencil_sizes=sizes)
    else:
        hierarchy = mask.get_adaptive_masks(
            direction=_DIRECTIONS[position - q.ndim], stencil_sizes=sizes
        )
    # each size reconstructs with that rung and the narrower ones below it
    rec_funcs = {}
    for k, tier in enumerate(tiers):
        rung = Ladder(tiers[k:], scheme)

        def rec(field: Array, vel: Array, _rung: Scheme = rung) -> Array:
            return reconstruct(field, vel, axis=position, scheme=_rung)

        rec_funcs.setdefault(tier.size, rec)
    return upwind_flux(
        q, u, dim=position, rec_funcs=rec_funcs, mask_hierarchy=hierarchy
    )


def _masked_fixed(
    q: Array, u: Array, position: int, scheme: Scheme, mask: Any
) -> Array:
    """The widest tier whose cells are all wet, for a scheme that is not upwinded.

    Such a scheme reads the same cells for either sign of the velocity, so a
    tier fits a face if every cell it reads is wet, and a face with a dry cell
    on either side has no flux.
    """
    xp = array_namespace(q, u)
    tiers = scheme.tiers()
    wet = xp.moveaxis(xp.asarray(mask.wet(), dtype=bool), position, -1)
    reach = max(max(-tier.lo, tier.hi) for tier in tiers)
    land = xp.zeros((*wet.shape[:-1], reach), dtype=bool)
    padded = xp.concat([land, wet, land], axis=-1)
    size = q.shape[position]

    def cells(offset: int) -> Array:
        """Whether cell ``i + offset`` is wet, per face ``i + 1/2``."""
        return padded[..., reach + offset : reach + offset + size]

    flux = xp.zeros_like(xp.moveaxis(q, position, -1))
    for k in range(len(tiers) - 1, -1, -1):
        fits = cells(0) & cells(1)
        for offset in range(tiers[k].lo, tiers[k].hi + 1):
            fits = fits & cells(offset)
        rung = reconstruct(q, u, axis=position, scheme=Ladder(tiers[k:], scheme))
        flux = xp.where(fits, xp.moveaxis(rung, position, -1), flux)
    return xp.moveaxis(flux, -1, position)
