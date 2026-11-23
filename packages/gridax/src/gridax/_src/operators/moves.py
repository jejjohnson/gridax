"""The canonical moves between staggered points: `difference` and `interpolate`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.analysis.differences import (
    adaptive_difference,
    difference as _collocated_difference,
)
from gridax._src.api import grid_dispatch
from gridax._src.boundary.fill import BC, extend, extend_wet
from gridax._src.fields.located import located
from gridax._src.grid import (
    CartesianAGrid1D,
    CartesianAGrid2D,
    CartesianAGrid3D,
    CartesianCGrid1D,
    CartesianCGrid2D,
    CartesianCGrid3D,
    SphericalAGrid2D,
    SphericalCGrid2D,
    SphericalCGrid3D,
)
from gridax._src.grid.points import Stagger, point_stagger
from gridax._src.mask.utils import shifted
from gridax._src.operators._ghost import interior
from gridax._src.operators._utils import _metric_like, _safe_div_cos, _spacing
from gridax._src.operators.triples import (
    PAIRS,
    TRIPLES,
    choose,
    moved_along,
    point_name,
)
from gridax._src.pytree import array_namespace
from gridax._src.schemes.base import Scheme
from gridax._src.schemes.linear import Biased, Centred
from gridax._src.stencils.staggered import (
    CENTRED2,
    along,
    check_linear,
    crop,
    is_two_point,
)

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


@grid_dispatch
@located("dst", f="src")
def difference(
    f: Float[Array, "..."],
    grid: Any,
    *,
    axis: str,
    src: str | Stagger | None = None,
    dst: str | Stagger | None = None,
    scheme: Scheme = CENTRED2,
    mask: Any = None,
    bc: BC | None = None,
) -> Float[Array, "..."]:
    """The difference of a field along an axis, at ``dst``.

    ``δₓf|_Q = 1/Δx_Q ∑ₖ cₖ f_{Q + sₖ}``, with ``sₖ`` the offsets of
    ``scheme`` in steps along the axis, ``cₖ`` its weights and ``Δx_Q`` the
    spacing at ``Q`` (``R cos(lat) dlon`` along x on a sphere). ``Centred(2)``
    reads the two points of ``src`` on either side of ``Q``, ``(f_{P+} -
    f_{P-}) / Δx_Q``. A wider stencil shifts inwards at the edge of the array,
    unless ``bc`` says what is outside it.

    On a collocated grid both points are T: a centred or one-sided difference,
    one-sided on the edges that are not periodic, with weights solved for the
    distances on an irregular axis.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        A field at ``src``, with its ghost ring.
    grid : grid
        A Cartesian or spherical grid, 1-D, 2-D or 3-D, staggered or
        collocated.
    axis : str
        One of ``grid.axes``.
    src, dst : str or Stagger
        The points: ``"T"``, ``"U"``, ``"V"``, ``"X"``, ``"W"``, or their
        staggers. They differ by half a step along ``axis``. A `CField` has
        its own ``src``. T, and left out, on a collocated grid.
    scheme : Scheme, default Centred(2)
        ``Centred(order)`` or ``Biased(order, side)`` of `gridax.schemes`.
    mask : Mask1D, Mask2D or Mask3D, optional
        Zeroes the dry points of ``dst`` (at X, those without four wet
        cells). A point that a wider stencil than two points would take from a
        dry one takes the next narrower stencil. On a collocated grid a dry
        cell is NaN, and a wet one next to it takes the widest stencil that
        fits on wet cells.
    bc : BC, optional
        The condition of the faces normal to ``axis``. The ghost ring of ``f``
        is filled from it, and as many layers beyond as the stencil reads, so
        the order is the scheme's up to the wall. Staggered grids only.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At ``dst``, with a zero ghost ring. ``NaN`` next to a pole, where a
        zonal difference is not defined.

    Raises
    ------
    ValueError
        If there is no such move, or a face ``bc`` is needed on has no condition.
    TypeError
        If the scheme is nonlinear or upwinded.
    NotImplementedError
        If the grid is of a type the difference is not defined on, or ``bc``
        is given for a collocated one.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 2.0, 2.0)
    >>> h = np.broadcast_to(grid.coords("x"), grid.shape)
    >>> gridax.difference(h, grid, axis="x", src="T", dst="U")[2].tolist()
    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    """
    check_linear(scheme)
    return difference._impl(
        grid, f, axis=axis, src=src, dst=dst, scheme=scheme, mask=mask, bc=bc
    )


@grid_dispatch
@located("dst", f="src")
def interpolate(
    f: Float[Array, "..."],
    grid: Any,
    *,
    src: str | Stagger | None = None,
    dst: str | Stagger | None = None,
    scheme: Scheme = CENTRED2,
    mask: Any = None,
    bc: BC | None = None,
) -> Float[Array, "..."]:
    """The value at each point of ``dst``, from the points of ``src`` around it.

    ``f|_Q = ∑ₖ cₖ f_{Q + sₖ}`` along an axis, with ``sₖ`` the offsets of
    ``scheme`` and ``cₖ`` its weights, and the product of the weights for a
    move along two axes (T to X, U to V), which is the same as two moves in
    turn. ``Centred(2)`` is the mean of two points, or of four. On a
    collocated grid there is only T, and the field is returned.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        A field at ``src``, with its ghost ring.
    grid : grid
        A Cartesian or spherical grid, 1-D, 2-D or 3-D, staggered or
        collocated.
    src, dst : str or Stagger
        The points: ``"T"``, ``"U"``, ``"V"``, ``"X"``, ``"W"``, or their
        staggers. A `CField` has its own ``src``.
    scheme : Scheme, default Centred(2)
        ``Centred(order)`` or ``Biased(order, side)`` of `gridax.schemes`.
    mask : Mask1D, Mask2D or Mask3D, optional
        Zeroes the dry points of ``dst`` (at X, those without four wet
        cells). A point that a wider stencil would take from a dry one takes
        the next narrower stencil; the two points (or four) of ``Centred(2)``
        are all read, wet or dry.
    bc : BC, optional
        The condition of the faces normal to the axes of the move, as in
        :func:`difference`.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At ``dst``, with a zero ghost ring.

    Raises
    ------
    ValueError
        If there is no such move, or a face ``bc`` is needed on has no condition.
    TypeError
        If the scheme is nonlinear or upwinded.
    NotImplementedError
        If the grid is of a type the interpolation is not defined on, or
        ``bc`` is given for a collocated one.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid1D.from_interior(4, 4.0)
    >>> gridax.interpolate(np.arange(6.0), grid, src="T", dst="U").tolist()
    [0.0, 1.5, 2.5, 3.5, 4.5, 0.0]
    """
    check_linear(scheme)
    return interpolate._impl(grid, f, src=src, dst=dst, scheme=scheme, mask=mask, bc=bc)


def _masked(out: Any, mask: Any, dst: str) -> Any:
    """``out``, zero on the dry points of ``dst``."""
    if mask is None:
        return out
    return out * _wet(mask, dst)


def _wet(mask: Any, point: str) -> Any:
    """The points of a mask that a move reads or writes."""
    return mask.wet(at=point, kind="strict" if point == "X" else "default")


def _move(
    f: Any, axes: tuple[str, ...], src: str, dst: str, scheme: Scheme, derivative: int
) -> Any:
    """The wide stencil of one move, on the interior of every grid axis."""
    out = f
    for axis, side in moved_along(src, dst):
        position = axes.index(axis) - len(axes)
        out = along(out, position, side, scheme, derivative=derivative)
    others = [a for a in axes if a not in {axis for axis, _ in moved_along(src, dst)}]
    return out[crop(f.ndim, [axes.index(a) - len(axes) for a in others])]


def _cells(side: str, scheme: Scheme, derivative: int) -> list[int]:
    """The points a move reads along one axis, relative to the one it writes."""
    delta = 0.5 if side == "fwd" else -0.5
    offsets = scheme.offsets(derivative=derivative, staggered=True)
    return [int(delta + s) for s in offsets]


def _reads_wet(
    wet: Any, axes: tuple[str, ...], src: str, dst: str, scheme: Scheme, derivative: int
) -> Any:
    """Whether every point a move reads, unshifted, is wet; on the interior."""
    xp = array_namespace(wet)
    fits = xp.asarray(wet, dtype=bool)
    for axis, side in moved_along(src, dst):
        position = axes.index(axis) - len(axes)
        cells = _cells(side, scheme, derivative)
        ok = shifted(fits, position, cells[0])
        for k in cells[1:]:
            ok = ok & shifted(fits, position, k)
        fits = ok
    return fits[crop(fits.ndim, range(-len(axes), 0))]


def _wide(
    f: Any,
    axes: tuple[str, ...],
    src: str,
    dst: str,
    scheme: Scheme,
    wet: Any,
    *,
    derivative: int,
    two_point: Any,
) -> Any:
    """A move with more than two points, narrowing next to dry points."""
    if wet is None:
        return _move(f, axes, src, dst, scheme, derivative)
    xp = array_namespace(f)
    *wider, _ = scheme._ladder()
    out = two_point(f)
    for rung in reversed(wider):
        fits = _reads_wet(wet, axes, src, dst, rung, derivative)
        out = xp.where(fits, _move(f, axes, src, dst, rung, derivative), out)
    return out


def _staggered(
    grid: Any,
    f: Any,
    src: str,
    dst: str,
    scheme: Scheme,
    mask: Any,
    bc: BC | None,
    *,
    derivative: int,
    two_point: Any,
) -> Any:
    """A move without its metric, on the interior of every grid axis."""
    axes = grid.axes
    along_axes = [axis for axis, _ in moved_along(src, dst)]
    two = is_two_point(scheme)
    wet = None if mask is None or two else _wet(mask, src)
    extra = 0
    if bc is not None:
        # as many layers as the stencil reads, the ring among them from bc
        reach = max(
            abs(k)
            for _, side in moved_along(src, dst)
            for k in _cells(side, scheme, derivative)
        )
        extra = max(reach - grid.ghost, 0)
        f = extend(f, grid, bc, point_stagger(src), along_axes, extra)
        wet = None if wet is None else extend_wet(wet, grid, bc, along_axes, extra)
    if two:
        raw = two_point(f)
    else:
        raw = _wide(
            f, axes, src, dst, scheme, wet, derivative=derivative, two_point=two_point
        )
    if extra == 0:
        return raw
    kept = slice(extra, -extra)
    return raw[(..., *(kept if axis in along_axes else slice(None) for axis in axes))]


def _raw_difference(
    grid: Any,
    f: Any,
    axis: str,
    src: str,
    dst: str,
    scheme: Scheme,
    mask: Any,
    bc: BC | None,
) -> Any:
    """The difference without its metric, on the interior."""
    stencil = choose("difference", (axis, src, dst), TRIPLES[grid.axes])
    return _staggered(
        grid, f, src, dst, scheme, mask, bc, derivative=1, two_point=stencil
    )


@difference.register(CartesianCGrid1D)
@difference.register(CartesianCGrid2D)
@difference.register(CartesianCGrid3D)
def _difference_cartesian(
    grid: Any,
    f: Any,
    *,
    axis: str,
    src: Any,
    dst: Any,
    scheme: Scheme,
    mask: Any,
    bc: BC | None,
) -> Any:
    src, dst = point_name(src), point_name(dst)
    raw = _raw_difference(grid, f, axis, src, dst, scheme, mask, bc)
    # δf[q] = (f[p+] - f[p-]) / Δ_q
    out = interior(raw / _spacing(grid, axis, f, at=dst), f)
    return _masked(out, mask, dst)


@difference.register(SphericalCGrid2D)
@difference.register(SphericalCGrid3D)
def _difference_spherical(
    grid: Any,
    f: Any,
    *,
    axis: str,
    src: Any,
    dst: Any,
    scheme: Scheme,
    mask: Any,
    bc: BC | None,
) -> Any:
    src, dst = point_name(src), point_name(dst)
    raw = _raw_difference(grid, f, axis, src, dst, scheme, mask, bc)
    if axis == "x":
        # δf[q] = (f[p+] - f[p-]) / (R cos(lat_q) dlon), NaN next to a pole
        cos = _metric_like(grid._cos_lat(dst), f)[1:-1, 1:-1]
        values = _safe_div_cos(raw, cos, _metric_like(grid.R * grid.dlon, f))
    else:
        # Δy = R dlat and Δz are uniform
        values = raw / _spacing(grid, axis, f, at=dst)
    return _masked(interior(values, f), mask, dst)


@interpolate.register(CartesianCGrid1D)
@interpolate.register(CartesianCGrid2D)
@interpolate.register(CartesianCGrid3D)
@interpolate.register(SphericalCGrid2D)
@interpolate.register(SphericalCGrid3D)
def _interpolate(
    grid: Any, f: Any, *, src: Any, dst: Any, scheme: Scheme, mask: Any, bc: BC | None
) -> Any:
    src, dst = point_name(src), point_name(dst)
    # f̄[q] = 1/2 (f[p-] + f[p+]), or the mean of four
    stencil = choose("interpolation", (src, dst), PAIRS[grid.axes])
    raw = _staggered(
        grid, f, src, dst, scheme, mask, bc, derivative=0, two_point=stencil
    )
    return _masked(interior(raw, f), mask, dst)


# ----------------------------------------------------------------------
# Collocated grids
# ----------------------------------------------------------------------


def _at_t(name: str, src: Any, dst: Any, bc: BC | None) -> None:
    if bc is not None:
        raise NotImplementedError(
            f"{name}: a collocated grid has no ghost ring for bc="
        )
    for point in (src, dst):
        if point is not None and point_name(point) != "T":
            raise ValueError(
                f"{name}: a collocated grid has T points only, not {point}"
            )


def _collocated_stencil(scheme: Scheme) -> tuple[Any, int]:
    """The ``method`` and the ``accuracy`` of the collocated kernel."""
    if scheme._data == "cell" and not is_two_point(scheme):
        raise NotImplementedError(
            'the collocated differences read point values; data="cell" is not available'
        )
    if isinstance(scheme, Biased):
        return ("forward" if scheme.side == "right" else "backward"), scheme.order
    if isinstance(scheme, Centred):
        # the central stencil of order p has one-sided ones of order p - 1 at the edges
        return "central", scheme.order - 1
    raise TypeError(f"{type(scheme).__name__} has no collocated difference")


@difference.register(CartesianAGrid1D)
@difference.register(CartesianAGrid2D)
@difference.register(CartesianAGrid3D)
@difference.register(SphericalAGrid2D)
def _difference_collocated(
    grid: Any,
    f: Any,
    *,
    axis: str,
    src: Any,
    dst: Any,
    scheme: Scheme,
    mask: Any,
    bc: BC | None,
) -> Any:
    _at_t("difference", src, dst, bc)
    method, accuracy = _collocated_stencil(scheme)
    kwargs: dict[str, Any] = {
        "axis": grid.axes.index(axis) - len(grid.axes),
        "accuracy": accuracy,
    }
    scale: Any = None
    if grid._step(axis) is not None:
        kwargs["step"] = grid.spacing(axis)
        kwargs["periodic"] = axis in grid.periodic
    else:
        kwargs["coords"] = grid.coords(axis)
        if isinstance(grid, SphericalAGrid2D):
            # d/dx = 1 / (R cos(lat)) d/dlon, d/dy = 1/R d/dlat
            scale = grid.R * (grid._cos_lat() if axis == "x" else 1.0)
    if mask is None:
        out = _collocated_difference(f, method=method, **kwargs)
    else:
        dropped = ~mask.wet()
        out = adaptive_difference(f, mask=dropped, **kwargs)
        if method != "central":
            # the side of the scheme wherever it fits on wet cells
            xp = array_namespace(f)
            field = xp.where(dropped, xp.nan, f)
            biased = _collocated_difference(field, method=method, **kwargs)
            out = xp.where(xp.isfinite(biased), biased, out)
    return out if scale is None else out / _metric_like(scale, out)


@interpolate.register(CartesianAGrid1D)
@interpolate.register(CartesianAGrid2D)
@interpolate.register(CartesianAGrid3D)
@interpolate.register(SphericalAGrid2D)
def _interpolate_collocated(
    grid: Any, f: Any, *, src: Any, dst: Any, scheme: Scheme, mask: Any, bc: BC | None
) -> Any:
    _at_t("interpolate", src, dst, bc)
    return _masked(f, mask, "T")
