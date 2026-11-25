"""The canonical moves between staggered points: `difference` and `interpolate`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np
from pipekit_array.arrays import namespace_of, set_at

from gridax._src.analysis.differences import (
    adaptive_difference,
    difference as _collocated_difference,
)
from gridax._src.api import grid_dispatch
from gridax._src.boundary.fill import (
    BC,
    extend,
    extend_wet,
    fill_coasts,
    wrapping,
)
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
    StretchedCGrid1D,
    StretchedCGrid2D,
    StretchedCGrid3D,
)
from gridax._src.grid.points import POINT_AXES, Stagger, point_stagger
from gridax._src.mask.utils import shifted
from gridax._src.operators._ghost import interior
from gridax._src.operators._utils import (
    _metric_like,
    _safe_div_cos,
    _spacing,
    without_dry,
)
from gridax._src.operators.triples import (
    PAIRS,
    TRIPLES,
    choose,
    moved_along,
    point_name,
)
from gridax._src.schemes.adaptive import policies
from gridax._src.schemes.base import Scheme
from gridax._src.schemes.linear import Biased, Centred
from gridax._src.stencils.staggered import (
    CENTRED2,
    along,
    along_wet,
    check_linear,
    crop,
    is_two_point,
    reads,
    wet_windows,
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
    wet = _wet(mask, dst)
    return namespace_of(out, wet).where(wet, out, 0.0)


def _wet(mask: Any, point: str) -> Any:
    """The points of a mask that a move reads or writes."""
    return mask.wet(at=point, kind="strict" if point == "X" else "default")


def _others(f: Any, axes: tuple[str, ...], moved: list[str]) -> Any:
    """``f`` without the ghost ring of the axes that are not moved along."""
    return f[crop(f.ndim, [axes.index(a) - len(axes) for a in axes if a not in moved])]


def _move(
    f: Any,
    axes: tuple[str, ...],
    src: str,
    dst: str,
    scheme: Scheme,
    derivative: int,
    edges: str = "bias",
    lines: Any = None,
) -> Any:
    """The wide stencil of one move, on the interior of every grid axis.

    ``lines(axis)`` gives the coordinates of an irregular axis (`along`).
    """
    out = f
    for axis, side in moved_along(src, dst):
        out = along(
            out,
            axes.index(axis) - len(axes),
            side,
            scheme,
            derivative=derivative,
            edges=edges,
            lines=None if lines is None else lines(axis),
        )
    return _others(out, axes, [axis for axis, _ in moved_along(src, dst)])


def _reads_wet(
    wet: Any, axes: tuple[str, ...], src: str, dst: str, scheme: Scheme, derivative: int
) -> Any:
    """Whether every point a move reads, unshifted, is wet; on the interior."""
    xp = namespace_of(wet)
    fits = xp.asarray(wet, dtype=bool)
    for axis, side in moved_along(src, dst):
        position = axes.index(axis) - len(axes)
        _, cells = reads(scheme, side, derivative)
        ok = shifted(fits, position, cells[0])
        for k in cells[1:]:
            ok = ok & shifted(fits, position, k)
        fits = ok
    return fits[crop(fits.ndim, range(-len(axes), 0))]


def _narrowing(
    f: Any,
    axes: tuple[str, ...],
    src: str,
    dst: str,
    scheme: Scheme,
    wet: Any,
    *,
    derivative: int,
    two_point: Any,
    lines: Any = None,
) -> Any:
    """A move with more than two points, narrowing next to dry points."""
    xp = namespace_of(f)
    *wider, _ = scheme._ladder()
    out = two_point(f)
    for rung in reversed(wider):
        fits = _reads_wet(wet, axes, src, dst, rung, derivative)
        value = _move(f, axes, src, dst, rung, derivative, lines=lines)
        out = xp.where(fits, value, out)
    return out


def _shifting(
    f: Any,
    axes: tuple[str, ...],
    src: str,
    dst: str,
    scheme: Scheme,
    wet: Any,
    *,
    derivative: int,
) -> Any:
    """A move with its stencil shifted onto wet points, one axis at a time.

    ``wet`` gives the wet points of a point.
    """
    out, point = f, src
    moved: list[str] = []
    for axis, side in moved_along(src, dst):
        position = axes.index(axis) - len(axes)
        # the axes moved along before have lost their ring
        here = wet(point)
        here = here[tuple(slice(1, -1) if a in moved else slice(None) for a in axes)]
        out = along_wet(out, here, position, side, scheme, derivative=derivative)
        along_axes = set(POINT_AXES[point]) ^ {axis}
        point = next(p for p, on in POINT_AXES.items() if set(on) == along_axes)
        moved.append(axis)
    return _others(out, axes, moved)


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
    moves = moved_along(src, dst)
    along_axes = [axis for axis, _ in moves]
    scheme, edges, dry = policies(scheme, bc, edges="bias", dry="narrow")
    two = is_two_point(scheme)
    if mask is not None and dry != "none":
        f = without_dry(f, mask, src, len(axes))
    if mask is None or (two and dry == "narrow"):
        dry = "none"
    irregular = isinstance(grid, _STRETCHED)
    if irregular:
        _check_stretched(scheme, edges, dry)
    reach = max(
        abs(k)
        for _, side in moves
        for cells in (
            wet_windows(scheme, side, derivative)
            if dry == "bias"
            else [reads(scheme, side, derivative)[1]]
        )
        for k in cells
    )
    if dry == "fill":
        assert bc is not None  # policies() asked for it
        f = fill_coasts(
            f, grid, bc, mask, point_stagger(src), width=grid.ghost, layers=reach
        )

    extra = 0
    around: BC | None = None
    if edges in ("extend", "periodic"):
        # as many layers as the stencil reads, the ring among them
        around = wrapping(along_axes) if edges == "periodic" else bc
        assert around is not None  # policies() asked for it
        extra = max(reach - grid.ghost, 0)
        f = extend(f, grid, around, point_stagger(src), along_axes, extra)

    lines: Any = None
    if irregular:

        def lines(axis: str) -> tuple[Any, Any]:
            at_src, at_dst = (grid._line(axis, p, extra) for p in (src, dst))
            return _metric_like(at_src, f), _metric_like(at_dst, f)

        def two_point(f: Any) -> Any:
            return _move(f, axes, src, dst, CENTRED2, derivative, lines=lines)

    def wet(point: str) -> Any:
        out = _wet(mask, point)
        if around is None:
            return out
        return extend_wet(out, grid, around, along_axes, extra)

    if dry == "bias":
        raw = _shifting(f, axes, src, dst, scheme, wet, derivative=derivative)
    elif two:
        raw = two_point(f)
    elif dry == "narrow":
        raw = _narrowing(
            f,
            axes,
            src,
            dst,
            scheme,
            wet(src),
            derivative=derivative,
            two_point=two_point,
            lines=lines,
        )
    else:
        at = edges if edges in ("bias", "narrow", "none") else "bias"
        raw = _move(f, axes, src, dst, scheme, derivative, at, lines)
    if edges == "none" and dry in ("narrow", "bias"):
        raw = _without_overhang(raw, axes, moves, scheme, derivative)
    if extra == 0:
        return raw
    kept = slice(extra, -extra)
    return raw[(..., *(kept if axis in along_axes else slice(None) for axis in axes))]


_STRETCHED = (StretchedCGrid1D, StretchedCGrid2D, StretchedCGrid3D)


def _check_stretched(scheme: Scheme, edges: str, dry: str) -> None:
    if scheme._data == "cell" and not is_two_point(scheme):
        raise NotImplementedError(
            'the moves of a stretched grid read point values; data="cell" is not'
            " available"
        )
    if dry == "bias":
        raise NotImplementedError('mask="bias" is not available on a stretched grid')
    if edges == "periodic":
        raise NotImplementedError("a stretched axis is not periodic")


def _without_overhang(
    raw: Any, axes: tuple[str, ...], moves: Any, scheme: Scheme, derivative: int
) -> Any:
    """Zero where the widest stencil, unshifted, would leave the array."""
    for axis, side in moves:
        position = axes.index(axis) - len(axes)
        _, cells = reads(scheme, side, derivative)
        rest = (slice(None),) * (-position - 1)
        # raw[r] is the output i = r + 1, which reads i + cells
        low, high = max(-cells[0] - 1, 0), max(cells[-1] - 1, 0)
        if low:
            raw = set_at(raw, (..., slice(0, low), *rest), 0.0)
        if high:
            raw = set_at(raw, (..., slice(-high, None), *rest), 0.0)
    return raw


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


@difference.register(StretchedCGrid1D)
@difference.register(StretchedCGrid2D)
@difference.register(StretchedCGrid3D)
def _difference_stretched(
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
    # the weights are solved for the distances: already per unit length
    raw = _raw_difference(grid, f, axis, src, dst, scheme, mask, bc)
    return _masked(interior(raw, f), mask, dst)


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
@interpolate.register(StretchedCGrid1D)
@interpolate.register(StretchedCGrid2D)
@interpolate.register(StretchedCGrid3D)
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
    scheme, edges, dry = policies(scheme, bc, edges="bias", dry="narrow")
    method, accuracy = _collocated_stencil(scheme)
    uniform = grid._step(axis) is not None
    kwargs: dict[str, Any] = {"axis": grid.axes.index(axis) - len(grid.axes)}
    scale: Any = None
    if uniform:
        kwargs["step"] = grid.spacing(axis)
        kwargs["periodic"] = axis in grid.periodic or edges == "periodic"
    else:
        if edges == "periodic":
            raise ValueError(f'edges="periodic" needs a uniform {axis} axis')
        kwargs["coords"] = grid.coords(axis)
        if isinstance(grid, SphericalAGrid2D):
            # d/dx = 1 / (R cos(lat)) d/dlon, d/dy = 1/R d/dlat
            scale = grid.R * (grid._cos_lat() if axis == "x" else 1.0)

    def plain(f: Any) -> Any:
        if edges not in ("narrow", "none") or kwargs.get("periodic"):
            return _collocated_difference(f, method=method, accuracy=accuracy, **kwargs)
        return _at_bounds(f, scheme, method, accuracy, kwargs, edges)

    xp = namespace_of(f)
    if mask is None:
        out = plain(f)
    elif dry == "none":
        out = xp.where(mask.wet(), plain(f), xp.nan)
    else:
        dropped = ~mask.wet()
        out = adaptive_difference(f, mask=dropped, accuracy=accuracy, **kwargs)
        if method != "central" or edges in ("narrow", "none"):
            # the side of the scheme, or what it does at the edges, wherever
            # it fits on wet cells
            field = xp.where(dropped, xp.nan, f)
            own = plain(field)
            out = xp.where(xp.isfinite(own), own, out)
            if edges == "none":
                out = xp.where(_fits(f, scheme, kwargs), out, xp.nan)
    return out if scale is None else out / _metric_like(scale, out)


def _fits(f: Any, scheme: Scheme, kwargs: dict[str, Any]) -> Any:
    """Where the collocated stencil of ``scheme`` is inside the array."""
    position = kwargs["axis"]
    if kwargs.get("periodic"):
        return np.ones((1,) * f.ndim, dtype=bool)
    n = f.shape[position]
    offsets = scheme.offsets(derivative=1, staggered=False)
    index = np.arange(n)
    inside = (index + offsets[0] >= 0) & (index + offsets[-1] <= n - 1)
    return inside.reshape((n, *(1,) * (-position - 1)))


def _at_bounds(
    f: Any, scheme: Scheme, method: Any, accuracy: int, kwargs: Any, edges: str
) -> Any:
    """The collocated difference, NaN or narrower where its stencil does not fit."""
    xp = namespace_of(f)
    out = _collocated_difference(f, method=method, accuracy=accuracy, **kwargs)
    if edges == "none":
        return xp.where(_fits(f, scheme, kwargs), out, xp.nan)
    for rung in _narrower(scheme):
        _, narrow = _collocated_stencil(rung)
        value = _collocated_difference(f, method=method, accuracy=narrow, **kwargs)
        out = xp.where(_fits(f, _wider_than(scheme, rung), kwargs), out, value)
    return out


def _narrower(scheme: Scheme) -> list[Scheme]:
    """The collocated schemes under ``scheme``, the widest first."""
    if isinstance(scheme, Biased):
        return [
            Biased(k, scheme.side, scheme.data) for k in range(scheme.order - 1, 0, -1)
        ]
    return [Centred(k, scheme._data) for k in range(scheme.order - 2, 0, -2)]


def _wider_than(scheme: Scheme, rung: Scheme) -> Scheme:
    """The scheme one rung above ``rung``, on the way down from ``scheme``."""
    ladder = [scheme, *_narrower(scheme)]
    return ladder[ladder.index(rung) - 1]


@interpolate.register(CartesianAGrid1D)
@interpolate.register(CartesianAGrid2D)
@interpolate.register(CartesianAGrid3D)
@interpolate.register(SphericalAGrid2D)
def _interpolate_collocated(
    grid: Any, f: Any, *, src: Any, dst: Any, scheme: Scheme, mask: Any, bc: BC | None
) -> Any:
    _at_t("interpolate", src, dst, bc)
    policies(scheme, bc, edges="bias", dry="narrow")
    return _masked(f, mask, "T")
