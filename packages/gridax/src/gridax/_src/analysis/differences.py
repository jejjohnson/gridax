"""Finite differences on collocated grids, on any array backend.

The collocated part of Layer 2 of gridax's derivative stack (metric-scaled and
mask-free), moved from ``geoops.geom`` (#423): a
derivative along one axis (:func:`difference`, NaN propagating) and its
mask-aware twin (:func:`adaptive_difference`, the widest stencil that fits on
valid data), which the canonical functions on collocated grids call. The
stencils come from :mod:`gridax.stencils` (Layer 0).

Every function computes in the namespace of its input (NumPy, JAX, CuPy, or
anything ``array-api-compat`` knows), so the result has the input's dtype and
backend: a float64 NumPy cube stays float64 NumPy, and a JAX array runs
under ``jax.jit``. Fields are collocated (all on the same points); a
staggered C-grid field goes through the Layer 1 stencils of
:mod:`gridax.stencils` or :func:`gridax.difference` instead.
"""

from __future__ import annotations

from jaxtyping import Bool, Float
import numpy as np

from gridax._src.pytree import array_namespace
from gridax._src.stencils.coefficients import (
    METHODS,
    Method,
)


def difference(
    f: Float[np.ndarray, "*shape"],
    *,
    axis: int = -1,
    step: Float[np.ndarray, "*shape"] | float = 1.0,
    coords: Float[np.ndarray, " n"] | None = None,
    derivative: int = 1,
    accuracy: int = 1,
    method: Method = "central",
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    r"""Finite-difference derivative along one axis; a NaN propagates.

    The plain stencil, finitediffx's ``difference`` on any backend: with
    ``method="central"`` the centred stencil of order ``accuracy + 1`` inside
    and forward / backward stencils of order ``accuracy`` on the edges,
    nothing wrapping around (``accuracy=1`` is ``numpy.gradient``'s
    scheme). ``"forward"`` / ``"backward"`` use that stencil everywhere it
    fits and the other one-sided stencil where it does not.

    A stencil that reads a NaN returns NaN, so a masked cell costs a halo of
    about ``accuracy`` cells; :func:`adaptive_difference` recovers them.

    .. math::

        \partial^n_s f(x_i) \approx h^{-n} \sum_k c_k\, f(x_{i + s_k})

    Args:
        f: The field; any shape.
        axis: The axis to differentiate.
        step: The signed step :math:`h` between centres along ``axis``: a
            scalar, or an array broadcasting against ``f``.
        coords: The positions of the centres along ``axis``, one per entry,
            for an unevenly spaced axis (depth or pressure levels, irregular
            times); replaces ``step``. Each point's stencil is then solved for
            its actual distances (`nonuniform_coeffs`), exact for polynomials
            of the stencil's degree.
        derivative: The derivative order :math:`n`.
        accuracy: The accuracy order of the one-sided stencils.
        method: ``"central"``, ``"forward"`` or ``"backward"``.
        periodic: The axis wraps (a global longitude): every point uses the
            ``method`` stencil, reading across the seam. Not with ``coords``.

    Returns:
        :math:`\partial^n f / \partial s^n`, ``f``'s shape.

    Raises:
        ValueError: The axis is shorter than the stencil needs, ``coords`` has
            the wrong length, or ``coords`` meets ``periodic``.

    Examples:
        >>> f = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x², x = 0…4
        >>> difference(f)
        array([1., 2., 4., 6., 7.])
        >>> difference(f, accuracy=2)
        array([0., 2., 4., 6., 8.])
        >>> difference(f, derivative=2)
        array([2., 2., 2., 2., 2.])
        >>> z = np.array([0.0, 1.0, 3.0, 7.0, 15.0])  # uneven levels
        >>> difference(z**2, coords=z, derivative=2)  # exact for a quadratic
        array([2., 2., 2., 2., 2.])
    """
    if method == "staggered":
        raise ValueError(
            "difference applies collocated stencils, whose result sits on the "
            "input points; a staggered stencil writes half a cell away, so apply "
            "it with the C-grid stencils of gridax.stencils"
        )
    xp = array_namespace(f)
    axis = axis % f.ndim
    if coords is not None:
        if periodic:
            raise ValueError("difference: coords= does not combine with periodic=True")
        if coords.shape != (f.shape[axis],):
            raise ValueError(
                f"difference: coords has shape {tuple(coords.shape)}; the axis has "
                f"{f.shape[axis]} entries"
            )
        return _bounded(xp, f, axis, derivative, accuracy, method, coords)
    if periodic:
        out = _periodic(xp, f, axis, derivative, accuracy, method)
    else:
        out = _bounded(xp, f, axis, derivative, accuracy, method)
    if hasattr(step, "dtype") and xp.isdtype(out.dtype, "real floating"):
        # a step array is geometry: it takes the field's precision (#421)
        step = xp.astype(xp.asarray(step), out.dtype)
    return out / step**derivative


def adaptive_difference(
    f: Float[np.ndarray, "*shape"],
    *,
    mask: Bool[np.ndarray, "*shape"] | None = None,
    axis: int = -1,
    step: Float[np.ndarray, "*shape"] | float = 1.0,
    coords: Float[np.ndarray, " n"] | None = None,
    derivative: int = 1,
    accuracy: int = 1,
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    r"""Mask-aware finite difference: the widest stencil that fits on valid data.

    Near a mask (land, a swath edge, a cloud) the centred stencil reads a
    masked cell and :func:`difference` loses the valid cells next to it.
    This keeps, per point, the first finite result among the central,
    forward and backward stencils at ``accuracy``, then at each lower
    accuracy down to 1, so a narrow channel still gets a first-order
    estimate. Recovered cells carry one order less than the interior.

    NaN is its own support detector: a stencil that reads a masked cell
    returns NaN (``0 * nan`` is ``nan``), so the selection needs no mask
    arithmetic (xrtoolz ``docs/design/xrgrad-nan-masks.md``, where it beat
    fill-then-differentiate by ~90x at the coast). Every candidate is
    computed, so the call costs ``3 * accuracy`` stencils and runs under
    ``jax.jit``. Masked cells are ``NaN`` in the result.

    Args:
        f: The field; any shape.
        mask: ``True`` drops the cell (the analysis mask convention: ``True`` = dropped), on ``f``'s
            shape; ``None`` drops only non-finite cells.
        axis: The axis to differentiate.
        step: The signed step between centres along ``axis``.
        coords: The centres' positions along ``axis``, for an unevenly spaced
            axis (see :func:`difference`); replaces ``step``.
        derivative: The derivative order.
        accuracy: The highest accuracy order to try.
        periodic: The axis wraps.

    Returns:
        The derivative, ``f``'s shape, ``NaN`` where ``f`` is masked or no
        stencil fits on valid data.

    Example:
        >>> f = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0])  # f = x
        >>> difference(f)
        array([ 1.,  1., nan, nan, nan,  1.,  1.])
        >>> adaptive_difference(f)
        array([ 1.,  1.,  1., nan,  1.,  1.,  1.])
    """
    xp = array_namespace(f)
    dropped = ~xp.isfinite(f)
    if mask is not None:
        dropped = dropped | mask
    field = xp.where(dropped, xp.nan, f)
    out = None
    for acc in range(accuracy, 0, -1):
        for method in METHODS:
            try:
                candidate = difference(
                    field,
                    axis=axis,
                    step=step,
                    coords=coords,
                    derivative=derivative,
                    accuracy=acc,
                    method=method,
                    periodic=periodic,
                )
            except ValueError:
                continue  # a stencil wider than the axis: try a narrower one
            out = (
                candidate if out is None else xp.where(xp.isfinite(out), out, candidate)
            )
    if out is None:
        raise ValueError(
            f"axis of length {f.shape[axis]} is too short for derivative {derivative}"
        )
    return xp.where(dropped, xp.nan, out)


def _bounded(
    xp: Any,
    f: Any,
    axis: int,
    derivative: int,
    accuracy: int,
    method: Method,
    coords: Any = None,
) -> Any:
    """The stencils of ``method`` with one-sided ones where it would leave the axis."""
    size = f.shape[axis]
    forward = _collocated_offsets(
        derivative=derivative, accuracy=accuracy, method="forward"
    )
    backward = _collocated_offsets(
        derivative=derivative, accuracy=accuracy, method="backward"
    )
    if method == "central":
        centre = _collocated_offsets(
            derivative=derivative, accuracy=accuracy, method="central"
        )
        if size < len(centre):
            raise ValueError(
                f"an axis of length {size} is shorter than the {len(centre)}-point "
                f"central stencil (derivative {derivative}, accuracy {accuracy})"
            )
        regions = [
            (0, -centre[0], forward),
            (-centre[0], size - centre[-1], centre),
            (size - centre[-1], size, backward),
        ]
    elif method in ("forward", "backward"):
        if size < len(forward):
            raise ValueError(
                f"an axis of length {size} is shorter than the {len(forward)}-point "
                f"{method} stencil (derivative {derivative}, accuracy {accuracy})"
            )
        if method == "forward":
            regions = [
                (0, size - forward[-1], forward),
                (size - forward[-1], size, backward),
            ]
        else:
            regions = [(0, -backward[0], forward), (-backward[0], size, backward)]
    else:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    regions = [
        (start, stop, offsets) for start, stop, offsets in regions if stop > start
    ]
    for start, stop, offsets in regions:
        # every point of a region must read inside the axis
        if start + offsets[0] < 0 or stop - 1 + offsets[-1] > size - 1:
            raise ValueError(
                f"an axis of length {size} is too short for the {len(offsets)}-point "
                f"boundary stencils of method {method!r} (derivative {derivative}, "
                f"accuracy {accuracy}); lower the accuracy"
            )
    parts = [
        _apply_stencil(f, axis, start, stop, offsets, derivative, coords)
        for start, stop, offsets in regions
    ]
    return xp.concat(parts, axis=axis)


def _periodic(
    xp: Any, f: Any, axis: int, derivative: int, accuracy: int, method: Method
) -> Any:
    """The ``method`` stencil at every point, reading across the seam."""
    offsets = _collocated_offsets(
        derivative=derivative, accuracy=accuracy, method=method
    )
    if f.shape[axis] < len(offsets):
        raise ValueError(
            f"a periodic axis of length {f.shape[axis]} is shorter than the "
            f"{len(offsets)}-point {method} stencil"
        )
    coeffs = difference_coeffs(offsets, derivative)
    out = None
    for offset, coeff in zip(offsets, coeffs, strict=True):
        term = coeff * xp.roll(f, -offset, axis=axis)
        out = term if out is None else out + term
    return out
