"""The linear schemes: centred, biased to a fixed side, and upwind."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING

import numpy as np

from gridax._src.advection import linear
from gridax._src.schemes.base import Scheme, Side, Tier, check_data, weighted
from gridax._src.stencils.coefficients import Data, difference_coeffs

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["Biased", "Centred", "Upwind"]


def _first(q0: Array) -> Array:
    return q0


def _upwind2_left(qm: Array, q0: Array) -> Array:
    # q[i+1/2] = 3/2 q[i] - 1/2 q[i-1]
    return 1.5 * q0 - 0.5 * qm


def _upwind2_right(qp: Array, qpp: Array) -> Array:
    # q[i+1/2] = 3/2 q[i+1] - 1/2 q[i+2]
    return 1.5 * qp - 0.5 * qpp


def _centred(q0: Array, qp: Array) -> Array:
    # q[i+1/2] = 1/2 (q[i] + q[i+1]), for either sign of the velocity
    return 0.5 * (q0 + qp)


UPWIND1 = Tier(0, 0, _first, _first)


@functools.lru_cache(maxsize=64)
def _face_value(lo: int, hi: int, data: Data) -> Callable[..., Array]:
    """The kernel of the face value from the cells ``lo … hi`` around it."""
    offsets = tuple(k - 0.5 for k in range(lo, hi + 1))
    return weighted(difference_coeffs(offsets, 0, data=data))


def _leaning(points: int, side: Side) -> tuple[int, int]:
    """The cells of a staggered stencil of ``points``, one or two more on ``side``."""
    lo, hi = -(points // 2), (points - 1) // 2
    return (lo, hi) if side == "left" else (1 - hi, 1 - lo)


@dataclass(frozen=True, repr=False)
class Centred(Scheme):
    """Symmetric stencils, of any even ``order``.

    Face values of cell averages (``data="cell"``, what ``reconstruct`` takes)
    and of point values (``"point"``, what ``difference`` and ``interpolate``
    take) differ from order 4 on:

        order 2            q[i+1/2] = (q[i] + q[i+1]) / 2
        order 4, "point"   q[i+1/2] = (-q[i-1] + 9 q[i] + 9 q[i+1] - q[i+2]) / 16
        order 4, "cell"    q[i+1/2] = (-q[i-1] + 7 q[i] + 7 q[i+1] - q[i+2]) / 12

    Parameters
    ----------
    order : int, default 2
        The formal order, even.
    data : {"point", "cell"}, optional
        What the values are. ``None`` leaves it to the operation.

    Examples
    --------
    >>> from gridax.schemes import Centred
    >>> Centred(4, data="point")
    Centred(order=4, data='point') # (-1, 9, 9, -1)/16
    """

    order: int = 2
    data: Data | None = None

    def __post_init__(self) -> None:
        if self.order < 2 or self.order % 2:
            raise ValueError(
                f"Centred order must be even and at least 2; got {self.order!r}"
            )
        check_data(self.data)

    def offsets(
        self, *, derivative: int = 1, staggered: bool
    ) -> tuple[int, ...] | tuple[float, ...]:
        # 2 h (staggered) or 2 h + 1 points have order `order` for the derivative:
        # symmetry gives one order to the odd derivatives, or to the even ones
        if staggered:
            half = self.order // 2 + derivative // 2
            return tuple(k + 0.5 for k in range(-half, half))
        half = _collocated(derivative, self.order // 2 + (derivative - 1) // 2)
        return tuple(range(-half, half + 1))

    def tiers(self) -> tuple[Tier, ...]:
        return tuple(_centred_tier(k, self.data) for k in range(self.order, 0, -2))

    def _ladder(self) -> tuple[Scheme, ...]:
        return tuple(Centred(k, self.data) for k in range(self.order, 0, -2))


@functools.lru_cache(maxsize=64)
def _centred_tier(order: int, data: Data | None) -> Tier:
    if order == 2:
        kernel: Callable[..., Array] = _centred
    elif data != "point" and order == 4:
        kernel = linear.linear_4pts
    elif data != "point" and order == 6:
        kernel = linear.linear_6pts
    else:
        kernel = _face_value(1 - order // 2, order // 2, data or "cell")
    return Tier(1 - order // 2, order // 2, kernel, kernel)


def _collocated(derivative: int, half: int) -> int:
    if derivative < 1:
        raise ValueError(
            "an interpolation onto the points themselves is the identity: use"
            " staggered=True, or a derivative of at least 1"
        )
    return half


@dataclass(frozen=True, repr=False)
class Biased(Scheme):
    """Stencils that lean to one ``side``, whatever the velocity.

    On a collocated grid they are the forward (``"right"``) and backward
    (``"left"``) differences, with the output and every other value on that
    side. On a staggered grid they have one more point on ``side`` than on the
    other if their number is odd, and two more if it is even, as
    :class:`Upwind` has on the upwind side.

    Parameters
    ----------
    order : int, default 2
        The formal order, at least 1.
    side : {"left", "right"}
        The side with more points, required.
    data : {"point", "cell"}, optional
        What the values are. ``None`` leaves it to the operation.

    Examples
    --------
    >>> from gridax.schemes import Biased
    >>> [str(c) for c in Biased(2, "right").weights(derivative=1, staggered=False)]
    ['-3/2', '2', '-1/2']
    >>> Biased(3, side="left", data="cell")
    Biased(order=3, side='left', data='cell') # (-1, 5, 2)/6
    """

    order: int = 2
    side: Side | None = None
    data: Data | None = None

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError(f"Biased order must be at least 1; got {self.order!r}")
        if self.side not in ("left", "right"):
            raise ValueError(
                f"Biased side must be 'left' or 'right', and is required; got"
                f" {self.side!r}"
            )
        check_data(self.data)

    def offsets(
        self, *, derivative: int = 1, staggered: bool
    ) -> tuple[int, ...] | tuple[float, ...]:
        side: Side = "left" if self.side == "left" else "right"
        points = derivative + self.order
        if staggered:
            lo, hi = _leaning(points, side)
            return tuple(k - 0.5 for k in range(lo, hi + 1))
        last = _collocated(derivative, points - 1)
        return tuple(range(-last, 1) if side == "left" else range(last + 1))

    def tiers(self) -> tuple[Tier, ...]:
        side: Side = "left" if self.side == "left" else "right"
        data = self.data or "cell"
        return tuple(_biased_tier(k, side, data) for k in range(self.order, 0, -1))

    def _ladder(self) -> tuple[Scheme, ...]:
        # the two-point stencil fits wherever the output has a wet cell on either side
        narrower = (Biased(k, self.side, self.data) for k in range(self.order, 0, -1))
        return (*narrower, Centred(2, self.data))


@functools.lru_cache(maxsize=64)
def _biased_tier(points: int, side: Side, data: Data) -> Tier:
    lo, hi = _leaning(points, side)
    kernel = _first if points == 1 else _face_value(lo, hi, data)
    return Tier(lo, hi, kernel, kernel)


@dataclass(frozen=True, repr=False)
class Upwind(Scheme):
    """Linear upwind-biased face values of cell averages, of any odd ``order``.

    ``order=1`` reads the upwind cell; ``3`` and ``5`` have the weights WENO-3
    and WENO-5 have on smooth data. Near a wall the stencil narrows, down to
    the upwind cell. ``order=2``, kept for compatibility, reads the upwind cell
    and the one behind it, ``3/2 q[i] - 1/2 q[i-1]``, and is dispersive.

    Parameters
    ----------
    order : int, default 1
        The formal order: odd, or 2.

    Examples
    --------
    >>> from gridax.schemes import Upwind
    >>> Upwind(3)
    Upwind(order=3) # (-1, 5, 2)/6
    """

    order: int = 1

    def __post_init__(self) -> None:
        if self.order < 1 or (self.order % 2 == 0 and self.order != 2):
            raise ValueError(
                f"Upwind order must be odd and at least 1, or 2; got {self.order!r}"
            )

    @property
    def upwinded(self) -> bool:
        return True

    def tiers(self) -> tuple[Tier, ...]:
        if self.order == 2:
            return (_upwind_tier(2), UPWIND1)
        return tuple(_upwind_tier(k) for k in range(self.order, 0, -2))


@functools.lru_cache(maxsize=64)
def _upwind_tier(order: int) -> Tier:
    kernels = {
        2: (_upwind2_left, _upwind2_right),
        3: (linear.linear_3pts_left, linear.linear_3pts_right),
        5: (linear.linear_5pts_left, linear.linear_5pts_right),
    }
    if order == 1:
        return UPWIND1
    lo, hi = _leaning(order, "left")
    if order in kernels:
        left, right = kernels[order]
        return Tier(lo, hi, left, right)
    mirror = _leaning(order, "right")
    return Tier(lo, hi, _face_value(lo, hi, "cell"), _face_value(*mirror, "cell"))
