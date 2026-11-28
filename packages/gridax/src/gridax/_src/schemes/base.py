"""What a scheme is: a ladder of stencil widths, and what follows from it.

A scheme is the list of its tiers, widest first. Each tier is one Layer 1
face-value reconstruction and its mirror: for flow towards ``+axis`` the face
``i + 1/2`` reads the cells ``i + lo … i + hi`` (the upwind cell is ``i``), and
for flow towards ``-axis`` it reads ``i + 1 - hi … i + 1 - lo`` (the upwind
cell is ``i + 1``). :func:`~gridax.advection.reconstruct` gives each face the
widest tier that fits in the array and, with a mask, on wet cells.

A subclass writes ``tiers()`` and ``order``. The base class derives the rest:
the footprint and the halo from the widest tier, and the weights from that
tier's offsets with the exact solver of Layer 0 (``gridax.stencils``).
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gridax._src.stencils.coefficients import Data, _exact_coeffs

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["Scheme", "Tier"]


@dataclass(frozen=True)
class Tier:
    """One stencil width of a scheme, as a face-value reconstruction.

    Attributes
    ----------
    lo, hi : int
        For flow towards ``+axis`` the face ``i + 1/2`` reads cells
        ``i + lo … i + hi``; for flow towards ``-axis`` it reads
        ``i + 1 - hi … i + 1 - lo``.
    left : Callable
        ``hi - lo + 1`` cell values, in order → the face value, for flow
        towards ``+axis``; for either sign if the scheme is not upwinded.
    right : Callable
        The mirror, for flow towards ``-axis``; not called if the scheme is
        not upwinded.
    mirrored : bool, default False
        ``right(*cells)`` is ``left(*reversed(cells))`` up to rounding, so the
        faces of both signs may be computed by one call of ``left``.
    """

    lo: int
    hi: int
    left: Callable[..., Array]
    right: Callable[..., Array]
    mirrored: bool = False

    @property
    def halo(self) -> int:
        """The cells read on either side of the upwind cell."""
        return max(-self.lo, self.hi)

    @property
    def size(self) -> int:
        """The adaptive-mask size of the tier, ``2 halo + 2``."""
        return 2 * self.halo + 2


class Scheme(abc.ABC):
    """Which values an operation reads around its output, and how it weighs them.

    The schemes are :class:`Centred`, :class:`Biased`, :class:`Upwind`,
    :class:`WENO` and :class:`TVD`. Each defines ``order``, its formal order,
    and :meth:`tiers`; everything else is derived from those.
    """

    #: the formal order of accuracy
    order: int

    @abc.abstractmethod
    def tiers(self) -> tuple[Tier, ...]:
        """The face-value stencils, widest first; the last one always fits."""

    @property
    def linear(self) -> bool:
        """Whether the result is a fixed linear combination of the values read."""
        return True

    @property
    def upwinded(self) -> bool:
        """Whether the face value depends on the sign of the velocity."""
        return False

    def offsets(
        self, *, derivative: int = 1, staggered: bool
    ) -> tuple[int, ...] | tuple[float, ...]:
        """The offsets of the values read, in grid steps from the output.

        Parameters
        ----------
        derivative : int, default 1
            The derivative order; 0 is an interpolation.
        staggered : bool
            Whether the output sits half a cell from its inputs (T to U) or
            on one of them (T to T, on a collocated grid).

        Returns
        -------
        tuple
            Half-integers if ``staggered``, integers if not, ascending. For
            an upwinded scheme, those for flow towards ``+axis``.

        Raises
        ------
        TypeError
            If the scheme is upwinded, and the output is not a face value.
        """
        tier = self.tiers()[0]
        self._faces_only(derivative, staggered)
        return tuple(k - 0.5 for k in range(tier.lo, tier.hi + 1))

    def footprint(self, *, staggered: bool) -> tuple[int, int]:
        """The first and the last cell read around the output.

        Relative to the cell ``i`` of the face ``i + 1/2`` if ``staggered``,
        and to the output itself if not. An upwinded scheme reads a stencil or
        its mirror, and the footprint holds both. On a collocated grid it is
        the footprint of a first derivative.

        Parameters
        ----------
        staggered : bool
            Whether the output sits half a cell from its inputs.

        Returns
        -------
        tuple of int
            ``(lo, hi)``.

        Examples
        --------
        >>> from gridax.schemes import WENO, Centred
        >>> Centred(4).footprint(staggered=True)
        (-1, 2)
        >>> WENO(5).footprint(staggered=True)
        (-2, 3)
        >>> Centred(4).footprint(staggered=False)
        (-2, 2)
        """
        if not staggered:
            offsets = self.offsets(staggered=False)
            return int(offsets[0]), int(offsets[-1])
        tier = self.tiers()[0]
        lo, hi = tier.lo, tier.hi
        return (min(lo, 1 - hi), max(hi, 1 - lo)) if self.upwinded else (lo, hi)

    def halo(self, *, staggered: bool) -> int:
        """The ghost cells the widest stencil needs, ``max(-lo, hi)`` of the footprint.

        Examples
        --------
        >>> from gridax.schemes import Centred
        >>> Centred(2).halo(staggered=True), Centred(4).halo(staggered=True)
        (1, 2)
        """
        lo, hi = self.footprint(staggered=staggered)
        return max(-lo, hi)

    def weights(self, *, derivative: int, staggered: bool) -> tuple[Fraction, ...]:
        """The exact weights of the widest stencil, one per offset.

        They solve the Taylor conditions of Layer 0 (``difference_coeffs``)
        in rational arithmetic. The values are point values unless ``data`` is
        ``"cell"``; :class:`Upwind` and :class:`WENO` read cell averages. For
        WENO they are the weights its smooth limit has.

        Parameters
        ----------
        derivative : int
            The derivative order; 0 is an interpolation.
        staggered : bool
            Whether the output sits half a cell from its inputs.

        Returns
        -------
        tuple of Fraction
            The weights, in the order of :meth:`offsets`.

        Raises
        ------
        TypeError
            If the scheme has no such stencil: an upwinded one only has
            face values, and TVD has no linear weights.

        Examples
        --------
        >>> from gridax.schemes import Centred, Upwind
        >>> [str(c) for c in Centred(4).weights(derivative=1, staggered=True)]
        ['1/24', '-9/8', '9/8', '-1/24']
        >>> [str(c) for c in Upwind(3).weights(derivative=0, staggered=True)]
        ['-1/6', '5/6', '1/3']
        """
        offsets = self.offsets(derivative=derivative, staggered=staggered)
        return _exact_coeffs(offsets, derivative, self._data or "point")

    def degrades_on(self, grid: Any, mask: Any = None) -> bool:
        """Whether the widest stencil does not fit everywhere on ``grid``.

        True if the halo is wider than the grid's ghost ring, so the points
        next to an edge take a narrower or a shifted stencil; and, with a
        mask, if a wet output reads a dry cell, so it takes a narrower one. A
        collocated grid that is periodic on every axis has no edge. For a model
        to warn at setup: it reads the values of the mask.

        Parameters
        ----------
        grid : CGrid or collocated grid
            The grid.
        mask : Mask1D, Mask2D or Mask3D, optional
            The wet cells; not traced.

        Returns
        -------
        bool

        Examples
        --------
        >>> import gridax
        >>> from gridax.schemes import Centred
        >>> grid = gridax.CartesianCGrid2D.from_interior(8, 6, 1.0, 1.0)
        >>> Centred(2).degrades_on(grid), Centred(4).degrades_on(grid)
        (False, True)
        """
        if mask is not None:
            from gridax._src.schemes.adaptive import reads_dry

            if reads_dry(self, grid, mask):
                return True
        collocated = getattr(grid, "_collocated", False)
        if collocated and set(grid.axes) <= set(grid.periodic):
            return False
        return self.halo(staggered=not collocated) > grid.ghost

    def _ladder(self) -> tuple[Scheme, ...]:
        """The schemes a move falls back through next to a dry point, widest first."""
        return (self,)

    @property
    def _data(self) -> Data | None:
        """What a value is; ``None`` leaves it to the operation."""
        return "cell" if self.upwinded else getattr(self, "data", None)

    def _faces_only(self, derivative: int, staggered: bool) -> None:
        if not staggered or derivative != 0:
            raise TypeError(
                f"{type(self).__name__} is upwinded and has face values only"
                " (derivative=0, staggered=True); use gridax.advection.reconstruct"
            )

    def _stencil(self) -> str:
        """The stencil in the repr."""
        return _named(self.weights(derivative=0, staggered=True))

    def _arguments(self) -> str:
        """The arguments in the repr, without the ones left to the operation."""
        fields: dict[str, Any] = vars(self)
        return ", ".join(f"{k}={v!r}" for k, v in fields.items() if v is not None)

    def __repr__(self) -> str:
        """The call that builds the scheme, and its face-value stencil."""
        return f"{type(self).__name__}({self._arguments()}) # {self._stencil()}"


def _named(weights: tuple[Fraction, ...]) -> str:
    """Weights over their common denominator, ``(-1, 9, 9, -1)/16``."""
    denominator = math.lcm(*(w.denominator for w in weights))
    numerators = ", ".join(str(int(w * denominator)) for w in weights)
    return f"({numerators})" + (f"/{denominator}" if denominator > 1 else "")


def weighted(coeffs: tuple[float, ...]) -> Callable[..., Array]:
    """The kernel ``sum_k c_k q_k`` of given weights."""

    def kernel(*values: Array) -> Array:
        total = coeffs[0] * values[0]
        for c, value in zip(coeffs[1:], values[1:], strict=True):
            total = total + c * value
        return total

    return kernel


def check_data(data: Any) -> None:
    """Raise unless ``data`` is ``"point"``, ``"cell"`` or ``None``."""
    if data not in ("point", "cell", None):
        raise ValueError(f"data must be 'point', 'cell' or None; got {data!r}")


Side = Literal["left", "right"]


@dataclass(frozen=True, repr=False)
class Ladder(Scheme):
    """A scheme cut down to some of its tiers (the narrower rungs of a mask hierarchy)."""

    rungs: tuple[Tier, ...]
    of: Scheme

    @property
    def order(self) -> int:
        return self.of.order

    @property
    def upwinded(self) -> bool:
        return self.of.upwinded

    def tiers(self) -> tuple[Tier, ...]:
        return self.rungs
