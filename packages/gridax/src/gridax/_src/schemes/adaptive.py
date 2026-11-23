"""`Adaptive`: what a scheme does where its widest stencil does not fit."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, get_args

import numpy as np

from gridax._src.schemes.base import Scheme, Tier
from gridax._src.stencils.coefficients import Data

__all__ = ["Adaptive", "policies"]

#: What happens at the edges of the array.
Edges = Literal["narrow", "bias", "extend", "periodic", "none"]
#: What happens next to dry points.
Dry = Literal["narrow", "bias", "fill", "none"]


@dataclass(frozen=True, repr=False)
class Adaptive(Scheme):
    """A scheme, and what it does at the edges of the array and at a coast.

    Every operation has its own answer, which a bare scheme takes: the moves
    shift a stencil inwards at an edge, the fluxes narrow it, and both narrow
    it next to dry points. ``Adaptive`` says it instead.

    Parameters
    ----------
    scheme : Scheme
        Any of `gridax.schemes`.
    edges : {"narrow", "bias", "extend", "periodic", "none"}, optional
        At the edges of the array: the widest narrower stencil that fits; as
        many points, shifted inwards; the same stencil, with what is outside
        filled from the operation's ``bc=``, or wrapped around; or zero (NaN on
        a collocated grid) where it does not fit. ``None`` leaves it to the
        operation: ``"extend"`` if it has a ``bc=``.
    mask : {"narrow", "bias", "fill", "none"}, optional
        Next to dry points, for an operation with a mask: the widest narrower
        stencil on wet points; a stencil of the same order shifted onto wet
        points if one fits, else a narrower one; the same stencil, with the dry points behind
        the coast filled from ``bc=``; or the same stencil, reading what the
        dry points hold. ``None`` leaves it to the operation.

    Raises
    ------
    TypeError
        If ``scheme`` is not a scheme, or is an ``Adaptive``.
    ValueError
        If a policy is unknown, or is ``"bias"`` for an upwinded scheme, which
        would then read downwind cells.

    Examples
    --------
    >>> from gridax.schemes import Adaptive, Centred
    >>> Adaptive(Centred(4), edges="periodic", mask="narrow")
    Adaptive(Centred(order=4), edges='periodic', mask='narrow')
    """

    scheme: Scheme
    edges: Edges | None = None
    mask: Dry | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scheme, Scheme) or isinstance(self.scheme, Adaptive):
            raise TypeError(
                f"Adaptive wraps one of gridax.schemes, e.g. Centred(4); got"
                f" {self.scheme!r}"
            )
        for name, value, known in (
            ("edges", self.edges, get_args(Edges)),
            ("mask", self.mask, get_args(Dry)),
        ):
            if value is not None and value not in known:
                raise ValueError(f"{name} must be one of {known}; got {value!r}")
            if value == "bias" and self.scheme.upwinded:
                raise ValueError(
                    f'{name}="bias" would make {type(self.scheme).__name__} read'
                    ' downwind cells; use "narrow"'
                )

    @property
    def order(self) -> int:
        return self.scheme.order

    @property
    def linear(self) -> bool:
        return self.scheme.linear

    @property
    def upwinded(self) -> bool:
        return self.scheme.upwinded

    @property
    def _data(self) -> Data | None:
        return self.scheme._data

    def tiers(self) -> tuple[Tier, ...]:
        return self.scheme.tiers()

    def offsets(
        self, *, derivative: int = 1, staggered: bool
    ) -> tuple[int, ...] | tuple[float, ...]:
        return self.scheme.offsets(derivative=derivative, staggered=staggered)

    def weights(self, *, derivative: int, staggered: bool) -> tuple[Fraction, ...]:
        return self.scheme.weights(derivative=derivative, staggered=staggered)

    def degrades_on(self, grid: Any, mask: Any = None) -> bool:
        edges = self.edges in (None, "narrow", "none") and self.scheme.degrades_on(grid)
        coast = mask is not None and self.mask in (None, "narrow")
        return edges or (coast and reads_dry(self.scheme, grid, mask))

    def _ladder(self) -> tuple[Scheme, ...]:
        return self.scheme._ladder()

    def __repr__(self) -> str:
        inner = f"{type(self.scheme).__name__}({self.scheme._arguments()})"
        given = [f"{k}={v!r}" for k in ("edges", "mask") if (v := getattr(self, k))]
        return f"Adaptive({', '.join([inner, *given])})"


def policies(
    scheme: Scheme, bc: Any, *, edges: Edges, dry: Dry
) -> tuple[Scheme, Edges, Dry]:
    """The scheme without its policies, and the policies, given the defaults.

    Raises
    ------
    ValueError
        If a policy needs ``bc``, or none uses it.
    """
    at_edges: Edges | None = None
    at_coasts: Dry | None = None
    if isinstance(scheme, Adaptive):
        scheme, at_edges, at_coasts = scheme.scheme, scheme.edges, scheme.mask
    if at_edges is None:
        at_edges = edges if bc is None else "extend"
    at_coasts = at_coasts or dry
    if bc is None and at_edges == "extend":
        raise ValueError('edges="extend" fills what is outside from bc=: give one')
    if bc is None and at_coasts == "fill":
        raise ValueError('mask="fill" fills the dry points from bc=: give one')
    if bc is not None and at_edges != "extend" and at_coasts != "fill":
        raise ValueError(f"bc= is not used with edges={at_edges!r}: leave one out")
    return scheme, at_edges, at_coasts


def reads_dry(scheme: Scheme, grid: Any, mask: Any) -> bool:
    """Whether a wet output of the widest stencil, off the edges, reads a dry cell."""
    wet = np.asarray(mask.wet(), dtype=bool)
    staggered = not getattr(grid, "_collocated", False)
    lo, hi = scheme.footprint(staggered=staggered)
    inner = (0, 1) if staggered else (0, 0)
    reach = max(-lo, hi)
    for axis in ("x", "y"):
        if axis not in grid.axes:
            continue
        # outside the array counts as wet: that is the edge, not the coast
        position = grid.axes.index(axis)
        line = np.moveaxis(wet, position, -1)
        padded = np.pad(
            line, [(0, 0)] * (wet.ndim - 1) + [(reach, reach)], constant_values=True
        )
        n = line.shape[-1]

        def cells(first: int, last: int, padded: Any = padded, n: int = n) -> Any:
            views = [
                padded[..., reach + k : reach + k + n] for k in range(first, last + 1)
            ]
            return np.logical_and.reduce(views)

        if np.any(cells(*inner) & ~cells(lo, hi)):
            return True
    return False
