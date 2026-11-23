"""
Area- and volume-weighted reduction helpers for Arakawa C-grids.

Provides scalar totals and means of T-point fields that account for
the correct per-cell metric on Cartesian and spherical grids:

    Cartesian T-cell area  = dx · dy
    Spherical T-cell area  = 2 R² · dlon · sin(dlat / 2) · cos(lat_T)
    3-D T-cell volume      = area · dz

Only physical interior cells contribute — the ghost ring is excluded
by construction.  When a mask is supplied, dry cells are excluded from
both the numerator and the denominator of ``*_mean``.

:func:`area_sum`, :func:`area_mean`, :func:`volume_sum` and
:func:`volume_mean` dispatch on the type of the grid.

Example
-------
>>> import jax.numpy as jnp
>>> from gridax import area_sum, SphericalCGrid2D
>>> grid = SphericalCGrid2D.from_interior(64, 32, (0.0, 360.0), (-80.0, 80.0))
>>> h = jnp.ones((grid.Ny, grid.Nx))
>>> total = area_sum(h, grid)  # integrated tracer over the basin
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from jaxtyping import Bool, Float
import numpy as np

from gridax._src.api import grid_dispatch
from gridax._src.fields.located import located
from gridax._src.grid import (
    CartesianAGrid2D,
    CartesianAGrid3D,
    CartesianCGrid2D,
    CartesianCGrid3D,
    CollocatedGrid2D,
    CollocatedGrid3D,
    CurvilinearGrid2D,
    CurvilinearGrid3D,
    SphericalAGrid2D,
    SphericalCGrid2D,
    SphericalCGrid3D,
    StretchedCGrid2D,
    StretchedCGrid3D,
)
from gridax._src.grid.points import Stagger
from gridax._src.mask import Mask2D, Mask3D
from gridax._src.operators._utils import _metric_like
from gridax._src.pytree import array_namespace

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


# ----------------------------------------------------------------------
# Cell-area / cell-volume metrics
# ----------------------------------------------------------------------


def area_weights(
    grid: CurvilinearGrid2D | CollocatedGrid2D | StretchedCGrid2D, *, xp: Any = None
) -> Float[Array, "Ny Nx"]:
    """The T-cell areas of a 2-D grid: ``grid.area()`` in a chosen namespace.

    Parameters
    ----------
    grid : CurvilinearGrid2D
        A Cartesian or spherical 2-D grid.
    xp : array namespace, optional
        The namespace to build the weights in. Default: the namespace of
        the grid's own arrays (NumPy for a grid built with ``from_interior``,
        JAX when the grid's leaves are traced). The reductions pass the
        field's namespace.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Per-T-cell area, exact for a lat-lon cell.

    Raises
    ------
    TypeError
        If the grid is not a 2-D one.
    """
    if isinstance(grid, CurvilinearGrid2D | CollocatedGrid2D | StretchedCGrid2D):
        return cast(Any, grid)._area(xp=xp)
    raise TypeError(
        f"area_weights: unsupported grid type {type(grid).__name__}. Use a 2-D grid."
    )


def volume_weights(
    grid: CurvilinearGrid3D | CollocatedGrid3D | StretchedCGrid3D, *, xp: Any = None
) -> Float[Array, "Nz Ny Nx"]:
    """The T-cell volumes of a 3-D grid, area times ``dz`` (``xp`` as in :func:`area_weights`)."""
    if isinstance(grid, CartesianCGrid3D):
        cell = grid.spacing("x") * grid.spacing("y") * grid.spacing("z")
        xp = array_namespace(cell) if xp is None else xp
        return xp.full(grid.shape, cell)
    if isinstance(grid, CurvilinearGrid3D | CollocatedGrid3D | StretchedCGrid3D):
        levels = cast(Any, grid)
        area = levels._area(xp=xp)
        xp = array_namespace(area) if xp is None else xp
        return xp.broadcast_to(area * abs(levels.spacing("z")), levels.shape)
    raise TypeError(
        f"volume_weights: unsupported grid type {type(grid).__name__}. Use a 3-D grid."
    )


# ----------------------------------------------------------------------
# Interior reductions
# ----------------------------------------------------------------------


def _weighted(
    x: Float[Array, "..."], weights: Float[Array, "..."], mask: Mask2D | Mask3D | None
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """The field and the weights, zero on the dry cells, in the field's dtype.

    Uses ``where(mask.wet(), x, 0.0)`` rather than a multiplicative
    float-cast mask, so that NaN/Inf values on dry T-cells, common for land
    cells in realistic data, cannot contaminate the wet-cell totals. The 0.0
    is weakly typed, so the dtype stays that of the field.
    """
    xp = array_namespace(x)
    weights = _metric_like(weights, x)
    if mask is None:
        return x, weights
    wet = mask.wet()
    return xp.where(wet, x, 0.0), xp.where(wet, weights, 0.0)


def _interior_axes(
    x: Any, weights: Any, ghost: int
) -> tuple[Any, tuple[int, ...] | None]:
    """The interior of the grid axes, which are the last, and those axes.

    ``None`` (all of them) when there are no others, where NumPy sums in
    another order than over a tuple of axes.
    """
    rank = weights.ndim
    axes = None if x.ndim == rank else tuple(range(-rank, 0))
    inner = slice(ghost, -ghost or None)
    return (..., *(inner,) * rank), axes


def _sum(x: Any, weights: Any, mask: Any, ghost: int) -> Any:
    """∑ w x over the interior."""
    xp = array_namespace(x)
    x, weights = _weighted(x, weights, mask)
    inner, axes = _interior_axes(x, weights, ghost)
    return xp.sum((weights * x)[inner], axis=axes)


def _mean(x: Any, weights: Any, mask: Any, ghost: int) -> Any:
    """∑ w x / ∑ w over the interior, NaN where there is no wet cell."""
    xp = array_namespace(x)
    x, weights = _weighted(x, weights, mask)
    inner, axes = _interior_axes(x, weights, ghost)
    num = xp.sum(weights[inner] * x[inner], axis=axes)
    den = xp.sum(weights[inner])
    return xp.where(den == 0.0, xp.nan, num / xp.where(den == 0.0, 1.0, den))


@grid_dispatch
@located(None, x="T")
def area_sum(
    x: Float[Array, "*batch Ny Nx"], grid: Any, *, mask: Mask2D | None = None
) -> Float[Array, "*batch"]:
    """Area-weighted sum over the interior T cells.

    ``∑ⱼᵢ A[j, i] x[j, i]`` over the interior, all the cells of a collocated
    grid. With a mask the dry cells
    are zeroed before the product, so a NaN or Inf on land cannot
    contaminate the result.

    Parameters
    ----------
    x : CField or Float[Array, "*batch Ny Nx"]
        T-point field.
    grid : grid
        Two-dimensional, staggered or collocated.
    mask : Mask2D, optional
        Dry cells contribute zero.

    Returns
    -------
    Float[Array, "*batch"]
        One value per leading index, in the dtype of ``x``.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 2.0, 3.0)
    >>> float(gridax.area_sum(np.ones(grid.shape), grid))
    6.0
    """
    return area_sum._impl(grid, x, mask=mask)


@grid_dispatch
@located(None, x="T")
def area_mean(
    x: Float[Array, "*batch Ny Nx"], grid: Any, *, mask: Mask2D | None = None
) -> Float[Array, "*batch"]:
    """Area-weighted mean over the interior T cells.

    ``(∑ A x) / (∑ A)`` over the wet T cells; a NaN or Inf on land does not
    contaminate it.

    Parameters
    ----------
    x : CField or Float[Array, "*batch Ny Nx"]
        T-point field.
    grid : grid
        Two-dimensional, staggered or collocated.
    mask : Mask2D, optional
        Dry cells are left out of both sums.

    Returns
    -------
    Float[Array, "*batch"]
        NaN if the total wet area is zero.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 2.0, 3.0)
    >>> float(gridax.area_mean(2.0 * np.ones(grid.shape), grid))
    2.0
    """
    return area_mean._impl(grid, x, mask=mask)


@grid_dispatch
@located(None, x="T")
def volume_sum(
    x: Float[Array, "*batch Nz Ny Nx"], grid: Any, *, mask: Mask3D | None = None
) -> Float[Array, "*batch"]:
    """Volume-weighted sum over the interior T cells.

    :func:`area_sum` with the volumes ``V[k, j, i] = A[j, i] dz`` and the
    interior ``[1:-1, 1:-1, 1:-1]``.

    Parameters
    ----------
    x : CField or Float[Array, "*batch Nz Ny Nx"]
        T-point field.
    grid : grid
        Three-dimensional, staggered or collocated.
    mask : Mask3D, optional
        Dry cells contribute zero.

    Returns
    -------
    Float[Array, ""]
    """
    return volume_sum._impl(grid, x, mask=mask)


@grid_dispatch
@located(None, x="T")
def volume_mean(
    x: Float[Array, "*batch Nz Ny Nx"], grid: Any, *, mask: Mask3D | None = None
) -> Float[Array, "*batch"]:
    """Volume-weighted mean over the interior T cells, ``(∑ V x) / (∑ V)``.

    Parameters
    ----------
    x : CField or Float[Array, "*batch Nz Ny Nx"]
        T-point field.
    grid : grid
        Three-dimensional, staggered or collocated.
    mask : Mask3D, optional
        Dry cells are left out of both sums.

    Returns
    -------
    Float[Array, "*batch"]
        NaN if the total wet volume is zero.
    """
    return volume_mean._impl(grid, x, mask=mask)


@area_sum.register(CartesianAGrid2D)
@area_sum.register(SphericalAGrid2D)
@area_sum.register(CartesianCGrid2D)
@area_sum.register(SphericalCGrid2D)
@area_sum.register(StretchedCGrid2D)
def _area_sum(grid: Any, x: Any, *, mask: Mask2D | None) -> Any:
    return _sum(x, area_weights(grid, xp=array_namespace(x)), mask, grid.ghost)


@area_mean.register(CartesianAGrid2D)
@area_mean.register(SphericalAGrid2D)
@area_mean.register(CartesianCGrid2D)
@area_mean.register(SphericalCGrid2D)
@area_mean.register(StretchedCGrid2D)
def _area_mean(grid: Any, x: Any, *, mask: Mask2D | None) -> Any:
    return _mean(x, area_weights(grid, xp=array_namespace(x)), mask, grid.ghost)


@volume_sum.register(CartesianAGrid3D)
@volume_sum.register(CartesianCGrid3D)
@volume_sum.register(SphericalCGrid3D)
@volume_sum.register(StretchedCGrid3D)
def _volume_sum(grid: Any, x: Any, *, mask: Mask3D | None) -> Any:
    return _sum(x, volume_weights(grid, xp=array_namespace(x)), mask, grid.ghost)


@volume_mean.register(CartesianAGrid3D)
@volume_mean.register(CartesianCGrid3D)
@volume_mean.register(SphericalCGrid3D)
@volume_mean.register(StretchedCGrid3D)
def _volume_mean(grid: Any, x: Any, *, mask: Mask3D | None) -> Any:
    return _mean(x, volume_weights(grid, xp=array_namespace(x)), mask, grid.ghost)


# ----------------------------------------------------------------------
# Masked sample statistics
# ----------------------------------------------------------------------


def _sample_counts(
    samples: Float[Array, "N ..."],
    mask: Mask2D | Mask3D | None,
    at: str | Stagger,
    axis: int | tuple[int, ...],
) -> tuple[Float[Array, "N ..."], Float[Array, "..."], Bool[Array, "..."] | None]:
    """Mask-safe samples, per-output sample counts, and the wet indicator.

    Returns ``(safe_samples, count, wet)`` where ``safe_samples`` has
    dry entries replaced by ``0.0`` (so NaN/Inf land sentinels cannot
    contaminate the reduction), ``count`` is the number of contributing
    entries per output element, and ``wet`` is the broadcast boolean
    mask (``None`` when no mask was supplied).
    """
    xp = array_namespace(samples)
    dtype = _accumulation_dtype(samples)
    if mask is None:
        # The count is the same everywhere, so take it from the shape
        # rather than reducing an indicator array: materialising
        # ``ones`` the size of an ``[N, Nz, Ny, Nx]`` stack can cost
        # gigabytes of device memory for a number known statically.
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        count = 1
        for reduced in axes:
            count *= samples.shape[reduced]
        return samples.astype(dtype), xp.asarray(count, dtype=dtype), None

    m = mask.wet(at=at)
    wet = xp.broadcast_to(m, samples.shape)
    safe = xp.where(wet, samples, 0.0).astype(dtype)
    count = xp.sum(wet.astype(dtype), axis=axis)
    return safe, count, wet


def _accumulation_dtype(samples: Float[Array, "N ..."]) -> Any:
    """Floating dtype wide enough to accumulate ``samples`` safely.

    At least float32.  A half-precision input is promoted rather than
    accumulated in place: ``float16`` saturates at 65504, so a wet-cell
    count over a 256x256 field overflows to ``inf`` and every mean it
    divides silently collapses to zero.  Its ``eps**2`` variance floor
    underflows to zero for the same reason, taking the documented
    safe-to-divide guarantee with it.

    Promoting is deliberate rather than a cast-back: at ``eps=1e-8``
    the floor itself is below ``float16``'s smallest subnormal, so
    there is no half-precision value that could carry the guarantee.
    ``float32`` and ``float64`` inputs are returned unchanged.
    """
    xp = array_namespace(samples)
    return xp.promote_types(xp.asarray(samples).dtype, xp.float32)


def masked_mean(
    samples: Float[Array, "N ..."],
    mask: Mask2D | Mask3D | None = None,
    *,
    at: str | Stagger = "T",
    axis: int | tuple[int, ...] = 0,
) -> Float[Array, "..."]:
    """Mean over ``axis``, excluding dry cells.

    With the default ``axis=0`` and a leading sample axis (time or
    ensemble member) this is the per-gridpoint mean — the ``loc`` of a
    standardising affine transform.  Pass ``axis=(0, -2, -1)`` for a
    single scalar per field, in which case only wet cells enter both
    the sum and the divisor.

    Dry cells are zeroed via ``jnp.where(mask, x, 0.0)`` *before* the
    reduction, so NaN/Inf sentinels stored on land cannot contaminate
    wet-cell statistics — the same guarantee :func:`area_mean` gives.

    Parameters
    ----------
    samples : Float[Array, "N ..."]
        Stacked samples; the reduced axes are given by ``axis``.
    mask : Mask2D or Mask3D or None, optional
        Land/ocean mask, broadcast against the trailing axes of
        ``samples``.  ``None`` means every cell is wet.
    at : str or Stagger, optional
        The point of the samples, whose wet points to use: ``"T"`` (default),
        ``"U"``, ``"V"``, ``"X"``, ``"W"`` (3-D only) or a field's stagger.
    axis : int or tuple of int, optional
        Axis or axes to reduce over.  Default ``0``.

    Returns
    -------
    Float[Array, "..."]
        The mean, in the input dtype or ``float32``, whichever is
        wider — see :func:`_accumulation_dtype`.  ``0.0`` wherever no
        wet sample contributed.
        Zero — rather than NaN — so that the paired
        ``(x - loc) / scale`` leaves a dry cell at zero and the inverse
        map stays well defined.
    """
    xp = array_namespace(samples)
    safe, count, _ = _sample_counts(samples, mask, at, axis)
    total = xp.sum(safe, axis=axis)
    empty = count == 0
    return xp.where(empty, 0.0, total / xp.where(empty, 1.0, count))


def masked_std(
    samples: Float[Array, "N ..."],
    mask: Mask2D | Mask3D | None = None,
    *,
    at: str | Stagger = "T",
    axis: int | tuple[int, ...] = 0,
    eps: float = 1e-8,
    ddof: int = 0,
) -> Float[Array, "..."]:
    """Standard deviation over ``axis``, excluding dry cells, floored at ``eps``.

    Companion to :func:`masked_mean`: together they give the ``loc``
    and ``scale`` of a per-gridpoint standardising transform.

    The floor matters because a constant field — a spun-up layer
    thickness, or any cell the dynamics never touch — has exactly zero
    sample variance, and dividing by it would produce inf/NaN.  Dry
    cells return ``1.0`` (an identity scale) rather than ``eps``, so
    that a masked field round-trips unchanged through
    ``(x - loc) / scale``.

    Parameters
    ----------
    samples : Float[Array, "N ..."]
    mask : Mask2D or Mask3D or None, optional
    at : str or Stagger, optional
        See :func:`masked_mean`.
    axis : int or tuple of int, optional
    eps : float, optional
        Lower bound on the returned standard deviation of a wet cell.
        Default ``1e-8``. Applied to the standard deviation itself, so
        any positive value is honoured exactly — including one whose
        square would underflow the accumulation dtype.
    ddof : int, optional
        Delta degrees of freedom; the divisor is ``count - ddof``.
        Default ``0``, matching :func:`jax.numpy.std`.

    Returns
    -------
    Float[Array, "..."]
        Standard deviation, at least ``eps`` on wet cells and exactly
        ``1.0`` on dry cells or where too few samples contributed to
        form the estimate.  In the input dtype or ``float32``,
        whichever is wider — see :func:`_accumulation_dtype`.
    """
    xp = array_namespace(samples)
    safe, count, wet = _sample_counts(samples, mask, at, axis)
    empty = count == 0

    mean = xp.where(empty, 0.0, xp.sum(safe, axis=axis) / xp.where(empty, 1.0, count))
    # ``xp.expand_dims`` accepts a tuple, re-inserting every reduced
    # axis as a size-1 dimension so the mean broadcasts back.
    dev = safe - xp.expand_dims(mean, axis)
    # Re-zero dry entries: ``safe - mean`` is ``-mean`` on land, which
    # would otherwise contribute to the variance of a spatial reduction.
    if wet is not None:
        dev = xp.where(wet, dev, 0.0)

    denom = count - ddof
    degenerate = denom <= 0
    var = xp.sum(dev**2, axis=axis) / xp.where(degenerate, 1.0, denom)

    # Two separate jobs, done in two steps rather than one.
    #
    # ``sqrt`` has an infinite derivative at zero, so a constant field
    # — exactly the case ``eps`` exists for — would come back with a
    # NaN gradient. The doubled ``where`` is the standard remedy: the
    # inner one keeps the zero away from ``sqrt`` on the backward pass,
    # the outer one restores the exact zero on the forward pass. No
    # value is lifted, so nothing is perturbed.
    #
    # Clamping the variance instead would be simpler but wrong twice
    # over. ``eps**2`` underflows for a small floor — ``1e-30`` squares
    # to ``1e-60``, which is zero in float32 — and the dtype's smallest
    # normal imposes a second, invisible floor of ``sqrt(tiny)``, about
    # ``1.1e-19``, so a requested ``eps`` below it could never be
    # returned and real deviations beneath it would be inflated.
    #
    # The requested floor is applied to the standard deviation itself,
    # where it is representable whatever its size. Where it binds, its
    # gradient is zero.
    positive = var > 0.0
    std = xp.where(positive, xp.sqrt(xp.where(positive, var, 1.0)), 0.0)
    std = xp.maximum(std, eps)
    # ``empty`` as well as ``degenerate``: with a negative ddof a dry
    # cell has ``count == 0`` but ``denom > 0``, so it is not
    # degenerate and would be handed ``eps`` instead of the identity.
    return xp.where(degenerate | empty, 1.0, std)


def masked_moments(
    samples: Float[Array, "N ..."],
    mask: Mask2D | Mask3D | None = None,
    *,
    at: str | Stagger = "T",
    axis: int | tuple[int, ...] = 0,
    eps: float = 1e-8,
    ddof: int = 0,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """``(mean, std)`` in one call — see :func:`masked_mean`, :func:`masked_std`.

    Returns
    -------
    tuple of Float[Array, "..."]
        The mask-aware mean and the ``eps``-floored standard deviation,
        which are exactly what :class:`StateAffine`-style standardising
        transforms need as ``loc`` and ``scale``.
    """
    return (
        masked_mean(samples, mask, at=at, axis=axis),
        masked_std(samples, mask, at=at, axis=axis, eps=eps, ddof=ddof),
    )
