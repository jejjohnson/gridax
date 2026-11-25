"""Shared utilities for the operators: the grid metric in a field's dtype, pole guards."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np
from pipekit_array.functions import constants_like, namespace_of
from pipekit_array.hooks import set_at

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

_COS_EPS = 1e-12  # guard against division by cos(lat) ≈ 0 near poles


def _metric_like(metric: Any, like: Any) -> Float[Array, "..."]:
    """A grid metric in the floating dtype of the field it acts on.

    Grid metrics are float64 NumPy arrays (0-d ones included): multiplying one into a
    float32 field would promote the result to float64 (under x64 on JAX).
    The metric is a property of the grid, not a precision request, so it
    takes the field's dtype and namespace instead. An integer ``like``
    leaves the metric's dtype alone.

    Parameters
    ----------
    metric : array
        A grid metric, e.g. ``grid.spacing("x")``.
    like : array
        The field the metric scales.

    Returns
    -------
    array
        ``metric`` as an array of ``like``'s namespace and floating dtype.
    """
    xp = namespace_of(metric, like)
    return constants_like(metric, xp.asarray(like), xp)


def _spacing(
    grid: Any, axis: str, like: Float[Array, "..."], *, at: str = "T"
) -> Float[Array, "..."]:
    """``grid.spacing(axis, at=at)`` in the dtype and namespace of ``like``."""
    return _metric_like(grid.spacing(axis, at=at), like)


def _safe_div_cos(
    numerator: Float[Array, "..."],
    cos_val: Float[Array, "..."],
    scale: float | Float[Array, "..."],
) -> Float[Array, "..."]:
    """Compute ``numerator / (scale * cos_val)`` with pole guard.

    Returns NaN where ``|cos_val| < eps`` instead of Inf.

    Parameters
    ----------
    numerator : array
        Numerator of the division.
    cos_val : array
        cos(latitude) values; near-zero values trigger the pole guard.
    scale : float or array
        Additional scale factor in the denominator.

    Returns
    -------
    array
        Result with NaN at pole-adjacent points.
    """
    xp = namespace_of(numerator, cos_val)
    denom = scale * cos_val
    safe_denom = xp.where(xp.abs(cos_val) < _COS_EPS, 1.0, denom)
    result = numerator / safe_denom
    return xp.where(xp.abs(cos_val) < _COS_EPS, xp.nan, result)


def _spherical_d2_lon(
    c: Float[Array, "Ny-2 Nx-2"],
    e: Float[Array, "Ny-2 Nx-2"],
    w: Float[Array, "Ny-2 Nx-2"],
    cos_T: Float[Array, "Ny-2 Nx-2"],
    dlon: Float[Array, ""],
    R: Float[Array, ""],
) -> Float[Array, "Ny-2 Nx-2"]:
    """Zonal second-derivative term of the spherical Laplacian, interior only.

    d²h/dx²[j, i] = 1/(R² cos²φ) · (h[j, i+1] − 2h[j, i] + h[j, i−1]) / dlon²

    ``c``, ``e`` and ``w`` are the centre, east and west views of ``h`` (see
    `_Neighbours2D`); the arithmetic is that of
    ``(diff_x_fwd_2d(h) - diff_x_bwd_2d(h)) / dlon**2``.
    """
    d2h = ((e - c) - (c - w)) / _metric_like(dlon**2, c)
    return _safe_div_cos(d2h, cos_T, _metric_like(R**2, cos_T) * cos_T)


def _spherical_laplacian_merid(
    c: Float[Array, "Ny-2 Nx-2"],
    n: Float[Array, "Ny-2 Nx-2"],
    s: Float[Array, "Ny-2 Nx-2"],
    cos_N: Float[Array, "Ny-2 Nx-2"],
    cos_S: Float[Array, "Ny-2 Nx-2"],
    cos_T: Float[Array, "Ny-2 Nx-2"],
    dlat: Float[Array, ""],
    R: Float[Array, ""],
) -> Float[Array, "Ny-2 Nx-2"]:
    """Meridional term of the spherical Laplacian, interior only.

    1/(R² cosφ) · d/dφ(cosφ · dh/dφ), with

    dh_N[j, i] = (h[j+1, i] − h[j, i]) / dlat   (north V-face of cell j)
    dh_S[j, i] = (h[j, i] − h[j−1, i]) / dlat   (south V-face of cell j)

    ``c``, ``n`` and ``s`` are the centre, north and south views of ``h``;
    ``cos_N`` and ``cos_S`` are cos(lat) at the north and south V-faces,
    read from the grid (rows ``j`` and ``j-1``).
    """
    dlat = _metric_like(dlat, c)
    dh_N = (n - c) / dlat
    dh_S = (c - s) / dlat
    d_cos_dh = (cos_N * dh_N - cos_S * dh_S) / dlat
    return _safe_div_cos(d_cos_dh, cos_T, _metric_like(R**2, cos_T))


def without_dry(f: Any, mask: Any, at: str, rank: int) -> Any:
    """``f`` with zeros on its dry points, so no stencil reads land.

    The ghost ring is the caller's (its boundary conditions) and stays. A corner
    is dry if all its cells are: the corners on a coast hold boundary values.
    """
    if mask is None:
        return f
    xp = namespace_of(f, mask.wet())
    return xp.where(readable(mask.wet(at=at), rank), f, 0.0)


def readable(wet: Any, rank: int) -> Any:
    """The wet points and the ghost ring, on the last ``rank`` axes."""
    xp = namespace_of(wet)
    wet = xp.asarray(wet, dtype=bool)
    inner = (..., *(slice(1, -1),) * rank)
    return set_at(xp.ones_like(wet), inner, wet[inner])


def _where(
    wet: Array | None, field: Float[Array, "... Ny Nx"]
) -> Float[Array, "... Ny Nx"]:
    """Zero ``field`` where ``wet`` is False; identity when ``wet`` is None.

    field[..., j, i] = field[..., j, i] if wet[j, i] else 0
    """
    xp = namespace_of(wet, field)
    if wet is None:
        return field
    return xp.where(wet, field, 0.0)


def _sanitize(
    wet: Array | None, field: Float[Array, "... Ny Nx"]
) -> Float[Array, "... Ny Nx"]:
    """Zero ``field`` on dry *interior* cells; pass the ghost ring through.

    field[..., j, i] = field[..., j, i]  on the ghost ring (BC-owned)
    field[..., j, i] = field[..., j, i] if wet[j, i] else 0
                                         for 1 <= j <= Ny-2, 1 <= i <= Nx-2
    """
    xp = namespace_of(wet, field)
    if wet is None:
        return field
    wet = xp.asarray(wet, dtype=bool)
    # keep[j, i] = True on the ghost ring, wet[j, i] in the interior
    keep = set_at(
        xp.ones_like(wet), (..., slice(1, -1), slice(1, -1)), wet[..., 1:-1, 1:-1]
    )
    return xp.where(keep, field, 0.0)
