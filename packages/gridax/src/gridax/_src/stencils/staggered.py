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

from gridax._src.mask.utils import shifted
from gridax._src.operators import stencils as _named
from gridax._src.pytree import array_namespace
from gridax._src.schemes.adaptive import Adaptive, policies
from gridax._src.schemes.base import Scheme
from gridax._src.schemes.linear import Centred
from gridax._src.stencils.coefficients import difference_coeffs, nonuniform_coeffs

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
    h: Float[Array, "*field"],
    *,
    axis: Axis,
    side: Side,
    ndim: Literal[1, 2, 3],
    scheme: Scheme = CENTRED2,
) -> Float[Array, "*interior"]:
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
    h : Float[Array, "*field"]
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
    Float[Array, "*interior"]
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
    h: Float[Array, "*field"],
    *,
    axis: Axis,
    side: Side,
    ndim: Literal[1, 2, 3],
    scheme: Scheme = CENTRED2,
) -> Float[Array, "*interior"]:
    """The staggered average (point-value interpolation) of ``h`` along ``axis``.

    As :func:`staggered_difference`, for the value instead of the first
    difference: ``Centred(2)`` is ``avg_<axis>_<side>_<ndim>d``, and
    ``Centred(4)`` is the fourth-order interpolation of point values

        h[i+1/2] = (-h[i-1] + 9 h[i] + 9 h[i+1] - h[i+2]) / 16.

    Cell averages need the cell-average reconstruction (-1, 7, 7, -1) / 12
    instead (``Centred(4, data="cell")``).

    Parameters
    ----------
    h : Float[Array, "*field"]
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
    Float[Array, "*interior"]
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
    if isinstance(scheme, Adaptive):
        scheme = scheme.scheme
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
    scheme, edges, _ = policies(scheme, None, edges="bias", dry="none")
    if edges == "periodic":
        raise ValueError('edges="periodic" needs a grid: use gridax.difference')
    if is_two_point(scheme):
        # the named two-point stencil, bit for bit
        op = "diff" if derivative else "avg"
        return getattr(_named, f"{op}_{axis}_{side}_{ndim}d")(h)
    out = along(h, _AXIS_INDEX[axis], side, scheme, derivative=derivative, edges=edges)
    # the other spatial axes lose their ghost cells, as in the named stencils
    return out[crop(h.ndim, [a for a in range(-ndim, 0) if a != _AXIS_INDEX[axis]])]


def crop(ndim: int, axes: Sequence[int]) -> tuple[slice, ...]:
    """The interior of ``axes`` (negative), and all of the others."""
    return tuple(
        slice(1, -1) if axis in axes else slice(None) for axis in range(-ndim, 0)
    )


def along(
    h: Array,
    position: int,
    side: str,
    scheme: Scheme,
    *,
    derivative: int,
    edges: str = "bias",
    lines: tuple[Array, Array] | None = None,
) -> Array:
    """The stencil of ``scheme`` along one axis, on the interior of that axis only.

    Where the stencil would leave the array it shifts inwards, with the
    weights of its shifted offsets (``edges="bias"``), the next rung of its
    ladder takes over (``"narrow"``), or the result is zero (``"none"``).
    ``lines`` are the coordinates of the points read and of those written
    along an irregular axis: the weights are then solved for the distances,
    and a difference is per unit length.
    """
    xp = array_namespace(h)
    size = h.shape[position]
    pieces = []
    for rung, shift, start, stop in _runs(size, side, scheme, derivative, edges):
        if rung is None:
            shape = list(h.shape)
            shape[position] = stop - start
            pieces.append(xp.zeros(tuple(shape), dtype=h.dtype))
            continue
        offsets, cells = reads(rung, side, derivative)
        data: Any = rung._data or "point"
        if lines is None:
            coeffs: Any = difference_coeffs(
                tuple(s + shift for s in offsets), derivative, data=data
            )
        else:
            coeffs = _uneven(h, position, lines, cells, shift, start, stop, derivative)
        total = None
        for k, c in zip(cells, coeffs, strict=True):
            first = start + k + shift
            index = [slice(None)] * h.ndim
            index[position] = slice(first, first + stop - start)
            term = c * h[tuple(index)]
            total = term if total is None else total + term
        pieces.append(total)
    return pieces[0] if len(pieces) == 1 else xp.concat(pieces, axis=position)


def _uneven(
    h: Array,
    position: int,
    lines: tuple[Array, Array],
    cells: list[int],
    shift: int,
    start: int,
    stop: int,
    derivative: int,
) -> list[Array]:
    """The weights of one run of outputs of an irregular axis, one array per cell."""
    xp = array_namespace(h, *lines)
    source, output = lines
    distances = (
        xp.stack([source[start + k + shift : stop + k + shift] for k in cells], axis=-1)
        - output[start:stop][:, None]
    )
    weights = xp.asarray(nonuniform_coeffs(distances, derivative), dtype=h.dtype)
    shape = [1] * (-position)
    shape[0] = stop - start
    return [weights[:, i].reshape(shape) for i in range(len(cells))]


def reads(
    scheme: Scheme, side: str, derivative: int
) -> tuple[tuple[float, ...], list[int]]:
    """The offsets of a staggered stencil, and the cells it reads.

    Output ``i`` writes the face ``i + delta``, so sample ``k`` is the cell
    ``i + delta + s_k``.
    """
    offsets = scheme.offsets(derivative=derivative, staggered=True)
    delta = 0.5 if side == "fwd" else -0.5
    return offsets, [int(delta + s) for s in offsets]


def along_wet(
    h: Array,
    wet: Array,
    position: int,
    side: str,
    scheme: Scheme,
    *,
    derivative: int,
) -> Array:
    """As :func:`along`, with the stencil shifted onto wet points.

    Each output takes the stencil of ``scheme`` if its points are wet; then the
    least shifted one of the same order that has wet points only and still
    holds the two on either side of the output (a difference needs one point
    more than the symmetric stencil for that); then those of the next rung of
    its ladder; and at last the two points, wet or dry. ``wet`` holds the
    points of the last axes of ``h``.
    """
    xp = array_namespace(h, wet)
    core = (0, 1) if side == "fwd" else (-1, 0)
    delta = 0.5 if side == "fwd" else -0.5

    def inner(x: Array) -> Array:
        index = [slice(None)] * x.ndim
        index[position] = slice(1, -1)
        return x[tuple(index)]

    def stencil(cells: Sequence[int], data: Any) -> Any:
        offsets = tuple(k - delta for k in cells)
        coeffs = difference_coeffs(offsets, derivative, data=data)
        total = None
        for k, c in zip(cells, coeffs, strict=True):
            term = c * inner(shifted(h, position, k))
            total = term if total is None else total + term
        return total

    out = stencil(core, "point")
    for rung in reversed(scheme._ladder()):
        # the last one that fits is kept: the least shifted, and at last its own
        for cells in wet_windows(rung, side, derivative):
            fits = shifted(wet, position, cells[0])
            for k in cells[1:]:
                fits = fits & shifted(wet, position, k)
            out = xp.where(inner(fits), stencil(cells, rung._data or "point"), out)
    return out


def wet_windows(scheme: Scheme, side: str, derivative: int) -> list[list[int]]:
    """The stencils of a scheme on wet points, the most shifted first."""
    core = (0, 1) if side == "fwd" else (-1, 0)
    _, own = reads(scheme, side, derivative)
    n = scheme.order + derivative
    starts = range(core[1] - n + 1, core[0] + 1)
    windows = [list(range(first, first + n)) for first in starts]
    centre = sum(core) / 2
    windows.sort(key=lambda cells: -abs(sum(cells) / n - centre))
    return [*(w for w in windows if w != own), own]


def _runs(
    size: int, side: str, scheme: Scheme, derivative: int, edges: str
) -> list[tuple[Scheme | None, int, int, int]]:
    """``(rung, shift, start, stop)`` runs of the output indices 1 .. size-2."""
    _, cells = reads(scheme, side, derivative)
    if edges == "bias":
        if size < len(cells):
            raise ValueError(
                f"an axis of length {size} is shorter than the {len(cells)}-point "
                "staggered stencil"
            )
        shifts = _shift_runs(size, cells[0], cells[-1])
        return [(scheme, shift, start, stop) for shift, start, stop in shifts]
    rungs: list[Scheme] = [scheme]
    if edges == "narrow":
        rungs = list(scheme._ladder())
    spans = [reads(rung, side, derivative)[1] for rung in rungs]
    runs: list[tuple[int | None, int, int]] = []
    for i in range(1, size - 1):
        fits = [i + cells[0] >= 0 and i + cells[-1] <= size - 1 for cells in spans]
        k = fits.index(True) if any(fits) else None
        if runs and runs[-1][0] == k:
            runs[-1] = (k, runs[-1][1], i + 1)
        else:
            runs.append((k, i, i + 1))
    return [
        (None if k is None else rungs[k], 0, start, stop) for k, start, stop in runs
    ]


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
