"""The regions of a field: its interior, its ghost ring, its faces and its corners."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from pipekit_array.functions import namespace_of

from gridax._src.backend import pad, set_at
from gridax._src.boundary.fill import FACES
from gridax._src.fields.cfield import CField, StaggerMismatchError
from gridax._src.grid.points import Stagger, as_stagger, point_axes, point_stagger

__all__ = ["apply_mask", "embed", "interior", "select"]

#: A side of an axis, as a face name's pair.
_SIDES = {"low": 0, "high": 1}


def interior(f: Any, grid: Any) -> Any:
    """A field without its ghost ring.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        A field on the grid, at any point.
    grid : grid
        A grid of gridax. A collocated one has no ring.

    Returns
    -------
    CField or Float[Array, "*interior"]
        ``[*batch, Ny, Nx]`` for a field ``[*batch, Ny + 2, Nx + 2]``.

    Raises
    ------
    TypeError
        If ``grid`` is not a grid.
    ValueError
        If the field is not on the grid.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid1D.from_interior(3, 3.0)
    >>> gridax.interior(np.arange(5.0), grid).tolist()
    [1.0, 2.0, 3.0]
    """
    return _on_array(f, grid, lambda x: x[_inner(grid)])


def embed(values: Any, grid: Any, fill: Any = 0) -> Any:
    """A field from its interior, with ``fill`` on the ghost ring.

    Parameters
    ----------
    values : CField or Float[Array, "*interior"]
        The interior values, as :func:`interior` gives them.
    grid : grid
        A grid of gridax.
    fill : float, default 0
        What the ghost ring holds.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        The field.

    Raises
    ------
    TypeError
        If ``grid`` is not a grid.
    ValueError
        If the values do not have the shape of the interior.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid1D.from_interior(3, 3.0)
    >>> gridax.embed(np.ones(3), grid, fill=-1.0).tolist()
    [-1.0, 1.0, 1.0, 1.0, -1.0]
    """
    ghost = _ghost(grid)
    inside = tuple(n - 2 * ghost for n in grid.shape)

    def put(x: Any) -> Any:
        rank = len(inside)
        if np.ndim(x) < rank or tuple(x.shape[-rank:]) != inside:
            raise ValueError(f"values of shape {x.shape}: the interior is {inside}")
        if ghost == 0:
            return x
        lead = ((0, 0),) * (x.ndim - rank)
        widths = (*lead, *((ghost, ghost),) * rank)
        if isinstance(fill, int | float) and fill == 0:
            return pad(x, widths)
        xp = namespace_of(x)
        shape = (*x.shape[:-rank], *grid.shape)
        ring = xp.zeros(shape, dtype=x.dtype) + xp.asarray(fill, dtype=x.dtype)
        return set_at(ring, _inner(grid), x)

    return _on_array(values, grid, put, on_grid=False)


def select(
    f: Any, grid: Any, region: str | tuple[str, str], *, stagger: Any = None
) -> Any:
    """One region of a field, by name, read-only.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        A field with its ghost ring.
    grid : grid
        A grid with a ghost ring.
    region : str or tuple of str
        - ``"interior"``: as :func:`interior`;
        - ``"south"``, ``"north"``, ``"west"``, ``"east"`` (``"bottom"``,
          ``"top"``), or ``(axis, "low" or "high")``: the ghost layer next to
          the interior, corners included, without the axis normal to it;
        - ``"wall_south"``, …: the first layer inside that face (the nodes on
          the wall, where the field has some), without the corners;
        - ``"ghost"``: the ghost ring, flattened: the low and the high layers
          of each axis in turn, without those of the axes before it;
        - ``"corners"``: the ghost corners next to the interior, the first axis
          varying slowest.
    stagger : str or Stagger, optional
        The point of a plain array, for a wall (T by default). A `CField` has
        its own.

    Returns
    -------
    Float[Array, "*batch region"]
        A plain array: what is selected is no longer on the grid.

    Raises
    ------
    TypeError
        If ``grid`` is not a grid.
    ValueError
        If the grid has no such region, or the field is not on the grid.
    StaggerMismatchError
        If ``stagger`` is not the point of the `CField`.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(2, 2, 1.0, 1.0)
    >>> f = np.arange(16.0).reshape(4, 4)
    >>> gridax.select(f, grid, "north").tolist()
    [12.0, 13.0, 14.0, 15.0]
    >>> gridax.select(f, grid, "wall_east", stagger="U").tolist()
    [6.0, 10.0]
    >>> gridax.select(f, grid, "corners").tolist()
    [0.0, 3.0, 12.0, 15.0]
    """
    ghost = _ghost(grid)
    point = _point(f, stagger)
    data = _array(f, grid)
    if region == "interior":
        return data[_inner(grid)]
    if ghost == 0:
        raise ValueError(f"{type(grid).__name__} has no ghost ring: no {region!r}")
    axes = grid.axes
    inner = slice(ghost, -ghost)

    def at(where: dict[str, Any]) -> Any:
        return data[(..., *(where.get(axis, slice(None)) for axis in axes))]

    if region == "ghost":
        pieces = []
        for k, axis in enumerate(axes):
            for layers in (slice(0, ghost), slice(-ghost, None)):
                piece = at({**dict.fromkeys(axes[:k], inner), axis: layers})
                pieces.append(piece.reshape(*piece.shape[: -len(axes)], -1))
        return namespace_of(data).concat(pieces, axis=-1)
    if region == "corners":
        sides = (slice(ghost - 1, ghost), slice(-ghost, 1 - ghost or None))
        pieces = []
        for corner in itertools.product(sides, repeat=len(axes)):
            piece = at(dict(zip(axes, corner, strict=True)))
            pieces.append(piece.reshape(*piece.shape[: -len(axes)], 1))
        return namespace_of(data).concat(pieces, axis=-1)

    wall = isinstance(region, str) and region.startswith("wall_")
    face = region.removeprefix("wall_") if isinstance(region, str) else region
    axis, side = _face(face, axes)
    n = grid.shape[axes.index(axis)]
    if not wall:
        layer = ghost - 1 if side == 0 else n - ghost
        return at({axis: layer})
    on_wall = axis in point_axes(point, axes)
    if side == 0:
        layer = ghost - 1 if on_wall else ghost
    else:
        layer = n - 1 - ghost
    return at({**dict.fromkeys(axes, inner), axis: layer})


def apply_mask(f: Any, mask: Any, *, at: Any = None, fill: Any = 0) -> Any:
    """A field with ``fill`` on the dry points of its point.

    The wet points are those the operators write: at X, the corners with four
    wet cells. The dry values are not read, so they may be NaN, and their
    gradient is zero.

    Parameters
    ----------
    f : CField or Float[Array, "*batch *grid"]
        The field.
    mask : Mask1D, Mask2D or Mask3D
        The wet cells.
    at : str or Stagger, optional
        The point of a plain array, required for one. A `CField` has its own.
    fill : float, default 0
        What the dry points hold.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        The field.

    Raises
    ------
    TypeError
        If ``f`` is a plain array and ``at`` is not given.
    StaggerMismatchError
        If ``at`` is not the point of the `CField`.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> mask = gridax.Mask1D.from_mask([0, 1, 1, 0])
    >>> gridax.apply_mask(np.ones(4), mask, at="U").tolist()
    [0.0, 1.0, 0.0, 0.0]
    """
    if at is None and not isinstance(f, CField):
        raise TypeError("apply_mask: at= is the point of a plain array")
    point = _point(f, at)
    axes = mask._axes
    kind = "strict" if len(point_axes(point, axes)) == 2 else "default"
    wet = mask.wet(at=point, kind=kind)

    def put(x: Any) -> Any:
        xp = namespace_of(x)
        return xp.where(wet, x, xp.asarray(fill, dtype=x.dtype))

    if not isinstance(f, CField):
        return put(f)
    others = tuple(dim for dim in f.dims if dim not in axes)
    canonical = f.transpose((*others, *axes))
    return canonical.with_data(put(canonical.data)).transpose(f.dims)


# ----------------------------------------------------------------------


def _ghost(grid: Any) -> int:
    if not hasattr(grid, "ghost"):
        raise TypeError(
            f"grid must be a grid of gridax, got {type(grid).__name__}; the Layer 1"
            " padding of a stencil's values is gridax.stencils.interior"
        )
    return grid.ghost


def _inner(grid: Any) -> tuple[Any, ...]:
    ghost = _ghost(grid)
    kept = slice(ghost, -ghost) if ghost else slice(None)
    return (..., *(kept for _ in grid.axes))


def _on_grid(x: Any, grid: Any) -> None:
    rank = len(grid.axes)
    if np.ndim(x) < rank or tuple(x.shape[-rank:]) != tuple(grid.shape):
        raise ValueError(f"a field of shape {np.shape(x)} is not on {grid.shape}")


def _canonical(f: CField, grid: Any) -> CField:
    missing = [axis for axis in grid.axes if axis not in f.dims]
    if missing:
        raise ValueError(f"{f!r} has no dims {missing}")
    others = tuple(dim for dim in f.dims if dim not in grid.axes)
    return f.transpose((*others, *grid.axes))


def _array(f: Any, grid: Any) -> Any:
    """The array of a field, with the axes of the grid last."""
    data = _canonical(f, grid).data if isinstance(f, CField) else f
    _on_grid(data, grid)
    return data


def _on_array(f: Any, grid: Any, fn: Any, *, on_grid: bool = True) -> Any:
    """``fn`` of the array of a field, with the dims and the point of the field."""
    _ghost(grid)
    if not isinstance(f, CField):
        if on_grid:
            _on_grid(f, grid)
        return fn(f)
    canonical = _canonical(f, grid)
    if on_grid:
        _on_grid(canonical.data, grid)
    # the labels of a grid axis are those of its points, which have changed
    labels = {k: v for k, v in f.labels.items() if k not in grid.axes}
    out = CField(fn(canonical.data), canonical.dims, labels, f.stagger)
    return out.transpose(f.dims)


def _point(f: Any, given: Any) -> Stagger:
    stagger = None
    if given is not None:
        stagger = point_stagger(given) if isinstance(given, str) else as_stagger(given)
    if not isinstance(f, CField):
        return point_stagger("T") if stagger is None else stagger
    if stagger is not None and stagger != f.stagger:
        raise StaggerMismatchError(f"{f!r} is not at {given!r}")
    return f.stagger


def _face(region: Any, axes: tuple[str, ...]) -> tuple[str, int]:
    """The axis and the side of a face, by its name or as ``(axis, side)``."""
    if isinstance(region, tuple) and len(region) == 2 and region[1] in _SIDES:
        axis, side = region[0], _SIDES[region[1]]
    elif region in FACES:
        axis, side = FACES[region]
    else:
        raise ValueError(f"unknown region {region!r}")
    if axis not in axes:
        raise ValueError(f"region {region!r}: the grid has the axes {axes}")
    return axis, side
