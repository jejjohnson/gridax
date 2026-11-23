"""The canonical moves between staggered points: `difference` and `interpolate`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.api import grid_dispatch
from gridax._src.grid import (
    CartesianCGrid1D,
    CartesianCGrid2D,
    CartesianCGrid3D,
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

    Parameters
    ----------
    f : Float[Array, "*batch *grid"]
        A field at ``src``, with its ghost ring.
    grid : grid
        A Cartesian or spherical C-grid, 1-D, 2-D or 3-D.
    axis : str
        One of ``grid.axes``.
    src, dst : str or Stagger
        The points: ``"T"``, ``"U"``, ``"V"``, ``"X"``, ``"W"``, or their
        staggers. They differ by half a step along ``axis``.
    mask : Mask1D, Mask2D or Mask3D, optional
        Zeroes the dry points of ``dst`` (at X, those without four wet
        cells).

    Returns
    -------
    Float[Array, "*batch *grid"]
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
    along two (T to X, U to V), which is the same as two moves in turn.

    Parameters
    ----------
    f : Float[Array, "*batch *grid"]
        A field at ``src``, with its ghost ring.
    grid : grid
        A Cartesian or spherical C-grid, 1-D, 2-D or 3-D.
    src, dst : str or Stagger
        The points: ``"T"``, ``"U"``, ``"V"``, ``"X"``, ``"W"``, or their
        staggers.
    mask : Mask1D, Mask2D or Mask3D, optional
        Zeroes the dry points of ``dst`` (at X, those without four wet
        cells). The mean is over all the neighbours, wet or dry.

    Returns
    -------
    Float[Array, "*batch *grid"]
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
