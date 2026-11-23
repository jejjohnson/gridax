"""The nonlinear upwind schemes: WENO, and TVD with a flux limiter.

The WENO nonlinear weights and the TVD limiters are not stencils: they are what
these schemes do with the Layer 1 reconstructions, whose linear parts come from
Layer 0 (``gridax.stencils``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gridax._src.advection import weno
from gridax._src.advection.limiters import mc, minmod, superbee, van_leer
from gridax._src.pytree import array_namespace
from gridax._src.schemes.base import Scheme, Tier
from gridax._src.schemes.linear import UPWIND1

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["TVD", "WENO"]

# Relative TVD floor (#294): ``r = num / (den + copysign(eps, den))`` with
# ``eps = eps_rel · |num| + tiny``, a floor relative to the stencil's own
# slopes with the sign of ``den`` kept.  ``r`` therefore scales exactly with
# the field (an absolute floor made every limiter first-order upwind at small
# amplitude), is within a relative ``eps_rel`` of the exact ratio whenever
# ``|r| ≪ 1 / eps_rel``, and a tiny downwind slope cannot flip its sign.  A
# flat stencil gives ``r = 0`` (the floor is the square root of the dtype's
# smallest normal, so the gradient stays finite there).
_TVD_EPS_REL: float = 1e-6


def _slope_ratio(num: Array, den: Array) -> Array:
    """TVD slope ratio ``num / (den + copysign(eps_rel |num| + tiny, den))``."""
    xp = array_namespace(num, den)
    # sqrt(tiny): its square (in the gradient's 1 / (den + eps)²) stays normal,
    # so a flat stencil has a finite (zero) gradient, float32 included
    floor = float(np.sqrt(xp.finfo(xp.result_type(den)).tiny))
    return num / (den + xp.copysign(_TVD_EPS_REL * xp.abs(num) + floor, den))


_WENO_JS = {
    3: Tier(-1, 1, weno.weno_3pts, weno.weno_3pts_right),
    5: Tier(-2, 2, weno.weno_5pts, weno.weno_5pts_right),
    7: Tier(-3, 3, weno.weno_7pts, weno.weno_7pts_right),
    9: Tier(-4, 4, weno.weno_9pts, weno.weno_9pts_right),
}
_WENO_Z = {
    3: Tier(-1, 1, weno.weno_3pts_improved, weno.weno_3pts_improved_right),
    5: Tier(-2, 2, weno.weno_5pts_improved, weno.weno_5pts_improved_right),
}
_LIMITERS: dict[str, Callable[[Any], Any]] = {
    "minmod": minmod,
    "van_leer": van_leer,
    "superbee": superbee,
    "mc": mc,
}


@dataclass(frozen=True, repr=False)
class WENO(Scheme):
    """WENO face values of cell averages, of ``order`` 3, 5, 7 or 9.

    Near a wall the face falls back through the narrower WENO stencils to
    first-order upwind, so it never reads a downwind cell.

    Parameters
    ----------
    order : int, default 5
        The formal order on smooth data.
    variant : {"js", "z"}, default "js"
        Jiang–Shu, or WENO-Z (orders 3 and 5).

    Examples
    --------
    >>> from gridax.schemes import WENO
    >>> WENO(5)
    WENO(order=5, variant='js') # (2, -13, 47, 27, -3)/60 on smooth data
    """

    order: Literal[3, 5, 7, 9] = 5
    variant: Literal["js", "z"] = "js"

    def __post_init__(self) -> None:
        family = _WENO_JS if self.variant == "js" else _WENO_Z
        if self.variant not in ("js", "z") or self.order not in family:
            raise ValueError(
                f"WENO supports orders {sorted(_WENO_JS)} (js) and {sorted(_WENO_Z)}"
                f" (z); got order={self.order!r}, variant={self.variant!r}"
            )

    @property
    def linear(self) -> bool:
        return False

    @property
    def upwinded(self) -> bool:
        return True

    def tiers(self) -> tuple[Tier, ...]:
        family = _WENO_JS if self.variant == "js" else _WENO_Z
        return (
            *(family[k] for k in sorted(family, reverse=True) if k <= self.order),
            UPWIND1,
        )

    def _stencil(self) -> str:
        return f"{super()._stencil()} on smooth data"


@dataclass(frozen=True, repr=False)
class TVD(Scheme):
    """Second-order TVD: first-order upwind plus a limited anti-diffusive correction.

    ``q[i+1/2] = q[i] + 1/2 phi(r) (q[i+1] - q[i])`` with
    ``r = (q[i] - q[i-1]) / (q[i+1] - q[i])`` for flow towards ``+axis``,
    and the mirror for flow towards ``-axis``.

    Parameters
    ----------
    limiter : {"minmod", "van_leer", "superbee", "mc"}, default "minmod"
        The flux limiter ``phi``.

    Examples
    --------
    >>> from gridax.schemes import TVD
    >>> TVD("van_leer")
    TVD(limiter='van_leer') # q[i] + phi(r) (q[i+1] - q[i])/2
    """

    limiter: str = "minmod"

    def __post_init__(self) -> None:
        if self.limiter not in _LIMITERS:
            raise ValueError(
                f"Unknown flux limiter {self.limiter!r}. Choose one of {list(_LIMITERS)!r}."
            )

    @property
    def order(self) -> int:
        """2, away from the extrema where the limiter is 0."""
        return 2

    @property
    def linear(self) -> bool:
        return False

    @property
    def upwinded(self) -> bool:
        return True

    def weights(self, *, derivative: int, staggered: bool) -> tuple[Fraction, ...]:
        raise TypeError(
            "TVD has no linear weights: its correction is limited cell by cell"
        )

    def tiers(self) -> tuple[Tier, ...]:
        phi = _LIMITERS[self.limiter]

        def left(qm: Array, q0: Array, qp: Array) -> Array:
            # r = (q[i] - q[i-1]) / (q[i+1] - q[i])
            diff_fwd = qp - q0
            diff_bwd = q0 - qm
            return q0 + 0.5 * phi(_slope_ratio(diff_bwd, diff_fwd)) * diff_fwd

        def right(q0: Array, qp: Array, qpp: Array) -> Array:
            # r = (q[i+1] - q[i+2]) / (q[i] - q[i+1])
            diff_neg = q0 - qp
            diff_neg2 = qp - qpp
            return qp + 0.5 * phi(_slope_ratio(diff_neg2, diff_neg)) * diff_neg

        return (Tier(-1, 1, left, right), UPWIND1)

    def _stencil(self) -> str:
        return "q[i] + phi(r) (q[i+1] - q[i])/2"
