"""Fields between a collocated grid and its C-grid: `stagger` and `collocate`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
import warnings

from jaxtyping import Float
import numpy as np
from pipekit_array.hooks import pad

from gridax._src.boundary.fill import BC, FACES, _fill
from gridax._src.boundary.regions import interior
from gridax._src.fields.cfield import CField, StaggerMismatchError
from gridax._src.grid.points import Stagger, point_stagger
from gridax._src.operators._utils import _metric_like
from gridax._src.operators.moves import _STRETCHED, interpolate
from gridax._src.operators.triples import PAIRS, choose, moved_along, point_name
from gridax._src.schemes.adaptive import Adaptive
from gridax._src.schemes.base import Scheme
from gridax._src.stencils.staggered import CENTRED2, along, check_linear, reads

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["LossyConversionWarning", "collocate", "stagger"]


class LossyConversionWarning(UserWarning):
    """A conversion that averages, so the way back does not give the field again."""


def _warn(what: str) -> None:
    warnings.warn(
        f"{what} averages neighbours: it damps short waves, and the way back does not"
        " restore them",
        LossyConversionWarning,
        stacklevel=3,
    )


def stagger(
    f: Float[Array, "..."],
    grid: Any,
    *,
    dst: str | Stagger,
    bc: BC,
    scheme: Scheme = CENTRED2,
) -> Float[Array, "..."]:
    """A field of a collocated grid, on its C-grid (``grid.staggered()``).

    ``T`` to ``T`` moves the values to the interior and fills the ghost ring
    from ``bc``: nothing is lost. To another point, the values are then
    interpolated, ``u[j, i+1/2] = (f[j, i] + f[j, i+1]) / 2`` with the default
    scheme, on the walls and on the ring too, from as many ghost layers of
    ``f`` as the scheme reads. The ring holds these interpolated values; the
    condition of the new field, such as no flow through a wall, is for
    `fill_ghosts` to write.

    There and back, U to T to U, is the filter ``1/4 (1, 2, 1)`` with
    ``Centred(2)``, whose response ``cos²(kΔx/2)`` removes the shortest wave.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        A field on the collocated grid.
    grid : CartesianAGrid* or SphericalAGrid2D
        The collocated grid; its axes increase.
    dst : str or Stagger
        The point: ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.
    bc : BC
        The condition of ``f`` on the faces. Faces of a periodic axis of the
        grid need none; any other face normal to the move does.
    scheme : Scheme, default Centred(2)
        ``Centred(order)`` or ``Biased(order, side)`` of `gridax.schemes`.

    Returns
    -------
    CField or Float[Array, "*batch *cgrid"]
        The field at ``dst``, with its ghost ring.

    Raises
    ------
    TypeError
        If the grid is not collocated, or the scheme is nonlinear or upwinded.
    ValueError
        If there is no such move, a face has no condition, or an axis of the
        grid decreases.

    Warns
    -----
    LossyConversionWarning
        Unless ``dst`` is ``"T"``; once per trace under ``jit``.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianAGrid1D.from_interior(4, 4.0)
    >>> flat = gridax.BC.no_flux("x")
    >>> gridax.stagger(np.arange(4.0), grid, dst="T", bc=flat).tolist()
    [0.0, 0.0, 1.0, 2.0, 3.0, 3.0]
    """
    if not getattr(grid, "_collocated", False):
        raise TypeError(
            f"stagger: {type(grid).__name__} is staggered already; the moves between"
            " its points are gridax.interpolate"
        )
    check_linear(scheme)
    name = point_name(dst)
    if isinstance(f, CField) and f.stagger:
        raise StaggerMismatchError(f"stagger: {f!r} is not at T")
    if name != "T":
        choose("interpolation", ("T", name), PAIRS[grid.axes])
        _warn(f"stagger to {name}")
    if not isinstance(f, CField):
        return _stagger(f, grid, name, bc, scheme)
    canonical = _canonical(f, grid)
    data = _stagger(canonical.data, grid, name, bc, scheme)
    return cast(Any, _field(data, canonical, grid, name).transpose(f.dims))


def collocate(
    f: Float[Array, "..."],
    grid: Any,
    *,
    src: str | Stagger | None = None,
    scheme: Scheme = CENTRED2,
    bc: BC | None = None,
) -> Float[Array, "..."]:
    """A field of a C-grid, on its collocated grid (``grid.collocated()``).

    The values are interpolated to T, ``f[j, i] = (u[j, i-1/2] + u[j, i+1/2])
    / 2`` with the default scheme, and the ghost ring is dropped. A field at T
    only loses its ring.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        A field at ``src``, with its ghost ring.
    grid : grid
        A C-grid.
    src : str or Stagger
        The point of the field. A `CField` has its own.
    scheme : Scheme, default Centred(2)
        ``Centred(order)`` or ``Biased(order, side)`` of `gridax.schemes`.
    bc : BC, optional
        The condition of ``f``, for a scheme that reads more than the ring, as
        in `interpolate`.

    Returns
    -------
    CField or Float[Array, "*batch *agrid"]
        The field at T, without a ghost ring.

    Raises
    ------
    TypeError
        If the grid is collocated, or the scheme is nonlinear or upwinded.
    ValueError
        If there is no such move, or the point of a plain array is not given.

    Warns
    -----
    LossyConversionWarning
        Unless ``src`` is ``"T"``; once per trace under ``jit``.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid1D.from_interior(4, 4.0)
    >>> gridax.collocate(np.arange(6.0), grid, src="T").tolist()
    [1.0, 2.0, 3.0, 4.0]
    """
    if getattr(grid, "_collocated", False):
        raise TypeError(f"collocate: {type(grid).__name__} is collocated already")
    check_linear(scheme)
    if isinstance(f, CField):
        if src is not None and point_name(src) != point_name(f.stagger):
            raise StaggerMismatchError(f"collocate: {f!r} is not at {src!r}")
        src = f.stagger
    name = point_name(src)
    at_t = f
    if name != "T":
        _warn(f"collocate from {name}")
        at_t = interpolate(f, grid, src=name, dst="T", scheme=scheme, bc=bc)
    return interior(at_t, grid)


# ----------------------------------------------------------------------


def _canonical(f: CField, grid: Any) -> CField:
    missing = [axis for axis in grid.axes if axis not in f.dims]
    if missing:
        raise ValueError(f"{f!r} has no dims {missing}")
    others = tuple(dim for dim in f.dims if dim not in grid.axes)
    return f.transpose((*others, *grid.axes))


def _field(data: Any, like: CField, grid: Any, point: str) -> CField:
    # the labels of a grid axis are those of its points, which have changed
    labels = {k: v for k, v in like.labels.items() if k not in grid.axes}
    return CField(data, like.dims, labels, point_stagger(point))


def _stagger(f: Any, grid: Any, dst: str, bc: BC, scheme: Scheme) -> Any:
    axes = grid.axes
    rank = len(axes)
    if tuple(f.shape[-rank:]) != tuple(grid.shape):
        raise ValueError(f"a field of shape {f.shape} is not on {grid.shape}")
    cgrid = grid.staggered()
    periodic = [axis for axis in grid.periodic if axis not in bc.faces]
    if periodic:
        bc = bc | BC.periodic(*periodic)
    moves = moved_along("T", dst)
    at_t = point_stagger("T")
    for name, (axis, side) in FACES.items():
        needed = axis in {a for a, _ in moves} or (dst == "T" and axis in bc.faces)
        if needed and axis in axes and bc.at_face(axis, side, at_t, axes) is None:
            raise ValueError(f"bc has no condition for the {name} face")
    scheme = scheme.scheme if isinstance(scheme, Adaptive) else scheme
    # one ghost layer more than the stencil reads from the faces on the ring
    reach = max(
        (abs(k) for _, side in moves for k in reads(scheme, side, 0)[1]), default=0
    )
    width = 1 + reach
    lead = ((0, 0),) * (f.ndim - rank)
    out = pad(f, (*lead, *((width, width),) * rank))
    out = _fill(out, cgrid, bc, width, at_t)
    for axis, side in moves:
        lines = None
        if isinstance(cgrid, _STRETCHED):
            read, written = (
                _metric_like(cgrid._line(axis, point, reach), f) for point in ("T", dst)
            )
            lines = (read, written)
        position = axes.index(axis) - rank
        out = along(out, position, side, scheme, derivative=0, lines=lines)
    moved = {axis for axis, _ in moves}
    kept = []
    for axis, n in zip(axes, cgrid.shape, strict=True):
        # a moved axis has lost one point on each side already
        first = reach - 1 if axis in moved else reach
        kept.append(slice(first, first + n))
    return out[(..., *kept)]
