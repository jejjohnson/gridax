"""Layer 1: staggered differences and averages of any even width, on the C-grid.

``staggered_difference`` and ``staggered_average`` apply the staggered stencils
of a scheme of ``gridax.schemes`` to a C-grid array with one ghost ring, along
one axis, and return the interior of every spatial axis, exactly as the named
stencils ``<op>_<axis>_<side>_<d>d`` do. ``Centred(2)`` is the named two-point
stencil itself. The samples are point values (C-grid velocities, pressures,
Coriolis terms) unless the scheme says ``data="cell"``; the face values of a
finite-volume tracer, which are cell averages, come from ``reconstruct``.

The wall rule. A centred stencil of K points reads K/2 cells on each side of
its face; with one ghost cell it would read past the array on the faces next
to a wall (the last face of a forward stencil, the first of a backward one,
and more for K = 6). On those faces the stencil shifts inwards by as many
cells as it would overhang, keeping K points: order K - n there (n the
derivative order) instead of K. It never reads beyond the ghost ring, so the
ghost ring and the boundary fillers stay as they are. This is the rule for
centred operators; an upwind reconstruction narrows instead, so that it never
reads downwind cells.

Example:
    >>> import numpy as np
    >>> from gridax.schemes import Centred
    >>> h = np.arange(8.0) ** 2  # h = x^2 on cell centres x_i = i
    >>> staggered_difference(h, axis="x", side="fwd", ndim=1, scheme=Centred(4))
    array([ 3.,  5.,  7.,  9., 11., 13.])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from jaxtyping import Float
import numpy as np

from gridax._src.operators import stencils as _named
from gridax._src.pytree import array_namespace
from gridax._src.schemes.base import Scheme
from gridax._src.schemes.linear import Centred
from gridax._src.stencils.coefficients import difference_coeffs

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["CENTRED2", "check_linear", "staggered_average", "staggered_difference"]

#: The default scheme: the named two-point stencils.
CENTRED2 = Centred(2)

Axis = Literal["x", "y", "z"]
Side = Literal["fwd", "bwd"]

_AXIS_INDEX = {"x": -1, "y": -2, "z": -3}


def staggered_difference(
    h: Float[Array, "*batch *spatial"],
    *,
    axis: Axis,
    side: Side,
    ndim: Literal[1, 2, 3],
    scheme: Scheme = CENTRED2,
) -> Float[Array, "*batch *interior"]:
    """The staggered first difference of ``h`` along ``axis``, no metric applied.

    The result sits half a cell ahead of ``h`` (``side="fwd"``, e.g. T → U) or
    behind it (``"bwd"``, e.g. U → T), on the interior of all ``ndim`` spatial
    axes, as ``diff_<axis>_<side>_<ndim>d``. Divide by the step for the
    derivative. With :math:`c_k` the Layer 0 coefficients of the staggered
    offsets :math:`s_k`, the forward face value is

        D h[i+1/2] = sum_k c_k h[i + 1/2 + s_k]

    (``Centred(4)``: ``(h[i-1] - 27 h[i] + 27 h[i+1] - h[i+2]) / 24``).

    Parameters
    ----------
    h : Float[Array, "*batch *spatial"]
        Field with one ghost ring on each of its trailing ``ndim`` axes;
        leading axes are batch axes.
    axis : {"x", "y", "z"}
        The axis to difference: the last, second-to-last or third-to-last.
    side : {"fwd", "bwd"}
        Where the result sits: half a cell ahead (``"fwd"``) or behind.
    ndim : {1, 2, 3}
        The spatial rank.
    scheme : Scheme, default Centred(2)
        ``Centred(order)`` or ``Biased(order, side)``; its formal order is the
        order of the derivative. ``Centred(2)`` is the named two-point stencil.

    Returns
    -------
    Float[Array, "*batch *interior"]
        The raw difference on the interior of all ``ndim`` spatial axes.

    Raises
    ------
    ValueError
        ``axis`` outside ``ndim`` spatial axes, or an axis shorter than the
        stencil.
    TypeError
        A scheme that is nonlinear or upwinded.

    Examples
    --------
    >>> import numpy as np
    >>> h = np.sin(np.linspace(0.0, 1.0, 10))
    >>> staggered_difference(h, axis="x", side="fwd", ndim=1).shape
    (8,)
    """
    return _staggered(h, axis, side, ndim, scheme, derivative=1)


def staggered_average(
    h: Float[Array, "*batch *spatial"],
    *,
    axis: Axis,
    side: Side,
    ndim: Literal[1, 2, 3],
    scheme: Scheme = CENTRED2,
) -> Float[Array, "*batch *interior"]:
    """The staggered average (point-value interpolation) of ``h`` along ``axis``.

    As :func:`staggered_difference`, for the value instead of the first
    difference: ``Centred(2)`` is ``avg_<axis>_<side>_<ndim>d``, and
    ``Centred(4)`` is the fourth-order interpolation of point values

        h[i+1/2] = (-h[i-1] + 9 h[i] + 9 h[i+1] - h[i+2]) / 16.

    Cell averages need the cell-average reconstruction (-1, 7, 7, -1) / 12
    instead (``Centred(4, data="cell")``).

    Parameters
    ----------
    h : Float[Array, "*batch *spatial"]
        Field with one ghost ring on each of its trailing ``ndim`` axes.
    axis : {"x", "y", "z"}
        The axis to interpolate along.
    side : {"fwd", "bwd"}
        Where the result sits: half a cell ahead (``"fwd"``) or behind.
    ndim : {1, 2, 3}
        The spatial rank.
    scheme : Scheme, default Centred(2)
        ``Centred(order)`` or ``Biased(order, side)``.

    Returns
    -------
    Float[Array, "*batch *interior"]
        The interpolated values on the interior of all ``ndim`` spatial axes.

    Examples
    --------
    >>> import numpy as np
    >>> from gridax.schemes import Centred
    >>> h = np.arange(6.0) ** 3  # a cubic is interpolated exactly at order 4
    >>> staggered_average(h, axis="x", side="fwd", ndim=1, scheme=Centred(4))
    array([ 3.375, 15.625, 42.875, 91.125])
    """
    return _staggered(h, axis, side, ndim, scheme, derivative=0)


def check_linear(scheme: Scheme) -> None:
    """Raise unless ``scheme`` is linear and the same for either flow direction."""
    if not isinstance(scheme, Scheme):
        raise TypeError(
            f"scheme must be one of gridax.schemes, e.g. Centred(4); got {scheme!r}"
        )
    if scheme.linear and not scheme.upwinded:
        return
    what = [
        word
        for word, holds in (
            ("nonlinear", not scheme.linear),
            ("upwinded", scheme.upwinded),
        )
        if holds
    ]
    raise TypeError(
        f"{type(scheme).__name__} is {' and '.join(what)}; use gridax.reconstruct"
    )


def is_two_point(scheme: Scheme) -> bool:
    """Whether the scheme is the named two-point stencil."""
    return isinstance(scheme, Centred) and scheme.order == 2


def _staggered(
    h: Array, axis: Axis, side: Side, ndim: int, scheme: Scheme, *, derivative: int
) -> Array:
    if axis not in _AXIS_INDEX or -_AXIS_INDEX[axis] > ndim:
        raise ValueError(f"axis {axis!r} is not one of the {ndim} spatial axes")
    if side not in ("fwd", "bwd"):
        raise ValueError(f"side must be 'fwd' or 'bwd'; got {side!r}")
    if h.ndim < ndim:
        raise ValueError(f"h has {h.ndim} axes; ndim={ndim} spatial axes expected")
    check_linear(scheme)
    if is_two_point(scheme):
        # the named two-point stencil, bit for bit
        op = "diff" if derivative else "avg"
        return getattr(_named, f"{op}_{axis}_{side}_{ndim}d")(h)
    out = along(h, _AXIS_INDEX[axis], side, scheme, derivative=derivative)
    # the other spatial axes lose their ghost cells, as in the named stencils
    return out[crop(h.ndim, [a for a in range(-ndim, 0) if a != _AXIS_INDEX[axis]])]


def crop(ndim: int, axes: Sequence[int]) -> tuple[slice, ...]:
    """The interior of ``axes`` (negative), and all of the others."""
    return tuple(
        slice(1, -1) if axis in axes else slice(None) for axis in range(-ndim, 0)
    )


def along(
    h: Array, position: int, side: str, scheme: Scheme, *, derivative: int
) -> Array:
    """The stencil of ``scheme`` along one axis, on the interior of that axis only.

    Where the stencil would leave the array it shifts inwards, with the
    weights of its shifted offsets.
    """
    offsets = scheme.offsets(derivative=derivative, staggered=True)
    xp = array_namespace(h)
    size = h.shape[position]
    if size < len(offsets):
        raise ValueError(
            f"an axis of length {size} is shorter than the {len(offsets)}-point "
            "staggered stencil"
        )
    # output index i = 1 .. N-2 writes the face i + delta; sample k is the cell
    # i + delta + s_k + t, with t the inward shift that keeps it in the array
    delta = 0.5 if side == "fwd" else -0.5
    lowest, highest = int(delta + offsets[0]), int(delta + offsets[-1])
    data: Any = scheme._data or "point"
    pieces = []
    for shift, start, stop in _shift_runs(size, lowest, highest):
        coeffs = difference_coeffs(
            tuple(s + shift for s in offsets), derivative, data=data
        )
        total = None
        for s, c in zip(offsets, coeffs, strict=True):
            first = start + int(delta + s) + shift
            index = [slice(None)] * h.ndim
            index[position] = slice(first, first + stop - start)
            term = c * h[tuple(index)]
            total = term if total is None else total + term
        pieces.append(total)
    return pieces[0] if len(pieces) == 1 else xp.concat(pieces, axis=position)


def _shift_runs(size: int, lowest: int, highest: int) -> list[tuple[int, int, int]]:
    """``(shift, start, stop)`` runs of output indices 1 .. size-2 with one shift.

    Output ``i`` reads cells ``i + lowest + t`` to ``i + highest + t``; ``t`` is
    0 where that fits in ``[0, size - 1]`` and the smallest inward shift where
    it does not.
    """
    runs: list[tuple[int, int, int]] = []
    for i in range(1, size - 1):
        shift = max(-(i + lowest), min(0, size - 1 - (i + highest)))
        if runs and runs[-1][0] == shift:
            runs[-1] = (shift, runs[-1][1], i + 1)
        else:
            runs.append((shift, i, i + 1))
    return runs
