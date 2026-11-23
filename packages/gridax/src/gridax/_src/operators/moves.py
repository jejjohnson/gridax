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
from gridax._src.grid.points import Stagger
from gridax._src.operators._ghost import interior
from gridax._src.operators._utils import _metric_like, _safe_div_cos, _spacing
from gridax._src.operators.triples import PAIRS, TRIPLES, choose, point_name

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
    mask: Any = None,
) -> Float[Array, "..."]:
    """The centred difference of a staggered field along an axis, at ``dst``.

    ``δₓf|_Q = (f_{P+} - f_{P-}) / Δx_Q``, with ``P-`` and ``P+`` the two
    points of ``src`` on either side of the point ``Q`` of ``dst`` along the
    axis, and ``Δx_Q`` the spacing at ``Q`` (``R cos(lat) dlon`` along x on a
    sphere).

    On a collocated grid both points are T: the second-order centred
    difference, one-sided on the edges that are not periodic, with weights
    solved for the distances on an irregular axis.

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
    mask : Mask1D, Mask2D or Mask3D, optional
        Zeroes the dry points of ``dst`` (at X, those without four wet
        cells). On a collocated grid a dry cell is NaN, and a wet one next to
        it takes the widest stencil that fits on wet cells.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At ``dst``, with a zero ghost ring. ``NaN`` next to a pole, where a
        zonal difference is not defined.

    Raises
    ------
    ValueError
        If there is no such move.
    NotImplementedError
        If the grid is of a type the difference is not defined on.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 2.0, 2.0)
    >>> h = np.broadcast_to(grid.coords("x"), grid.shape)
    >>> gridax.difference(h, grid, axis="x", src="T", dst="U")[2].tolist()
    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    """
    return difference._impl(grid, f, axis=axis, src=src, dst=dst, mask=mask)


@grid_dispatch
@located("dst", f="src")
def interpolate(
    f: Float[Array, "..."],
    grid: Any,
    *,
    src: str | Stagger | None = None,
    dst: str | Stagger | None = None,
    mask: Any = None,
) -> Float[Array, "..."]:
    """The mean of the points of ``src`` around each point of ``dst``.

    Two points along an axis, ``1/2 (f_{P-} + f_{P+})``, or four for a move
    along two (T to X, U to V), which is the same as two moves in turn. On a
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
    mask : Mask1D, Mask2D or Mask3D, optional
        Zeroes the dry points of ``dst`` (at X, those without four wet
        cells). The mean is over all the neighbours, wet or dry.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At ``dst``, with a zero ghost ring.

    Raises
    ------
    ValueError
        If there is no such move.
    NotImplementedError
        If the grid is of a type the interpolation is not defined on.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid1D.from_interior(4, 4.0)
    >>> gridax.interpolate(np.arange(6.0), grid, src="T", dst="U").tolist()
    [0.0, 1.5, 2.5, 3.5, 4.5, 0.0]
    """
    return interpolate._impl(grid, f, src=src, dst=dst, mask=mask)


def _masked(out: Any, mask: Any, dst: str) -> Any:
    """``out``, zero on the dry points of ``dst``."""
    if mask is None:
        return out
    return out * mask.wet(at=dst, kind="strict" if dst == "X" else "default")


@difference.register(CartesianCGrid1D)
@difference.register(CartesianCGrid2D)
@difference.register(CartesianCGrid3D)
def _difference_cartesian(
    grid: Any, f: Any, *, axis: str, src: Any, dst: Any, mask: Any
) -> Any:
    src, dst = point_name(src), point_name(dst)
    stencil = choose("difference", (axis, src, dst), TRIPLES[grid.axes])
    # δf[q] = (f[p+] - f[p-]) / Δ_q
    out = interior(stencil(f) / _spacing(grid, axis, f, at=dst), f)
    return _masked(out, mask, dst)


@difference.register(SphericalCGrid2D)
@difference.register(SphericalCGrid3D)
def _difference_spherical(
    grid: Any, f: Any, *, axis: str, src: Any, dst: Any, mask: Any
) -> Any:
    src, dst = point_name(src), point_name(dst)
    raw = choose("difference", (axis, src, dst), TRIPLES[grid.axes])(f)
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
def _interpolate(grid: Any, f: Any, *, src: Any, dst: Any, mask: Any) -> Any:
    src, dst = point_name(src), point_name(dst)
    stencil = choose("interpolation", (src, dst), PAIRS[grid.axes])
    # f̄[q] = 1/2 (f[p-] + f[p+]), or the mean of four
    return _masked(interior(stencil(f), f), mask, dst)


# ----------------------------------------------------------------------
# Collocated grids
# ----------------------------------------------------------------------


def _at_t(name: str, src: Any, dst: Any) -> None:
    for point in (src, dst):
        if point is not None and point_name(point) != "T":
            raise ValueError(
                f"{name}: a collocated grid has T points only, not {point}"
            )


@difference.register(CartesianAGrid1D)
@difference.register(CartesianAGrid2D)
@difference.register(CartesianAGrid3D)
@difference.register(SphericalAGrid2D)
def _difference_collocated(
    grid: Any, f: Any, *, axis: str, src: Any, dst: Any, mask: Any
) -> Any:
    _at_t("difference", src, dst)
    kwargs: dict[str, Any] = {"axis": grid.axes.index(axis) - len(grid.axes)}
    scale: Any = None
    if grid._step(axis) is not None:
        kwargs["step"] = grid.spacing(axis)
        kwargs["periodic"] = axis in grid.periodic
    else:
        kwargs["coords"] = grid.coords(axis)
        if isinstance(grid, SphericalAGrid2D):
            # d/dx = 1 / (R cos(lat)) d/dlon, d/dy = 1/R d/dlat
            scale = grid.R * (grid._cos_lat() if axis == "x" else 1.0)
    kernel: Any = _collocated_difference
    if mask is not None:
        kernel = adaptive_difference
        kwargs["mask"] = ~mask.wet()
    out = kernel(f, **kwargs)
    return out if scale is None else out / _metric_like(scale, out)


@interpolate.register(CartesianAGrid1D)
@interpolate.register(CartesianAGrid2D)
@interpolate.register(CartesianAGrid3D)
@interpolate.register(SphericalAGrid2D)
def _interpolate_collocated(grid: Any, f: Any, *, src: Any, dst: Any, mask: Any) -> Any:
    _at_t("interpolate", src, dst)
    return _masked(f, mask, "T")
