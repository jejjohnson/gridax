"""A `CField` to and from other carriers, and to the right faces."""

from __future__ import annotations

from typing import Any

import numpy as np

from gridax._src.backend import set_at
from gridax._src.boundary.fill import BC, fill_ghosts
from gridax._src.fields.cfield import CField
from gridax._src.grid.points import GRID_AXES, POINT_AXES
from gridax._src.pytree import array_namespace

#: A point staggered along each axis, whose coordinate is the right faces'.
_FACE = {along[0]: point for point, along in POINT_AXES.items() if len(along) == 1}


def to_xarray(field: CField, grid: Any, name: str = "field") -> Any:
    """`CField.to_xarray`."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("CField.to_xarray needs xarray: pip install xarray") from exc

    positions = dict(field.stagger)
    unsupported = {a: p for a, p in positions.items() if p != "right"}
    if unsupported:
        raise ValueError(
            f"{field!r} is at {unsupported}: to_xarray writes centred and right"
            " axes; convert left ones with gridax.to_right first"
        )
    dims, coords = [], {}
    for dim in field.dims:
        if dim not in grid.axes:
            dims.append(dim)
            if dim in field.labels:
                coords[dim] = (dim, list(field.labels[dim]))
            continue
        centre, face = f"{dim}_c", f"{dim}_g"
        axis = {"axis": dim.upper()}
        # xgcm reads c_grid_axis_shift = 0.5 as the right faces of the cells
        coords[centre] = (centre, np.asarray(grid.coords(dim)), axis)
        coords[face] = (
            face,
            np.asarray(grid.coords(dim, at=_FACE[dim])),
            {**axis, "c_grid_axis_shift": 0.5},
        )
        dims.append(face if dim in positions else centre)
    return xr.Dataset({name: (dims, np.asarray(field.data))}, coords=coords)


def from_xarray(array: Any) -> CField:
    """`CField.from_xarray`."""
    dims, stagger, labels = [], {}, {}
    for dim in array.dims:
        coord = array.coords.get(dim)
        attrs = {} if coord is None else coord.attrs
        axis = str(attrs.get("axis", "")).lower()
        if axis not in GRID_AXES:
            dims.append(str(dim))
            if coord is not None:
                labels[str(dim)] = tuple(coord.values.tolist())
            continue
        dims.append(axis)
        shift = float(attrs.get("c_grid_axis_shift", 0.0))
        if shift not in (0.0, -0.5, 0.5):
            raise ValueError(f"{dim}: c_grid_axis_shift is {shift}, not 0, -0.5 or 0.5")
        if shift:
            stagger[axis] = "right" if shift > 0 else "left"
    return CField(array.data, dims, labels, stagger)


def dimarray(data: Any, dims: Any, labels: Any) -> Any:
    """A ``pipekit_array.DimArray``, or an error that names the extra."""
    try:
        from pipekit_array import DimArray
    except ImportError as exc:
        raise ImportError(
            'a DimArray needs pipekit-array: pip install "gridax[pipekit]"'
        ) from exc

    return DimArray(data, dims, labels)


def to_dimarray(field: CField) -> Any:
    """The data, dims and labels of a field as a ``pipekit_array.DimArray``.

    A ``DimArray`` has no stagger: the point is dropped.

    Parameters
    ----------
    field : CField

    Returns
    -------
    pipekit_array.DimArray

    Raises
    ------
    ImportError
        If pipekit-array is not installed (the ``pipekit`` extra).

    Examples
    --------
    >>> import numpy as np
    >>> from gridax import CField, to_dimarray
    >>> to_dimarray(CField(np.zeros((2, 3)), ("y", "x"), loc="U")).dims
    ('y', 'x')
    """
    return dimarray(field.data, field.dims, field.labels)


def to_right(field: CField, grid: Any, bc: BC) -> CField:
    """A field on left faces (MITgcm's U, on the west one) as one on right faces.

    The value on the left face of cell ``i`` is the value on the right face of
    cell ``i - 1``: each ``"left"`` axis is shifted by one index. That leaves
    the last index, the face outside the domain, which takes the value
    `fill_ghosts` gives it from ``bc``. Fields already on right faces (NEMO's)
    are returned as they are.

    Parameters
    ----------
    field : CField
        A field with its ghost ring, staggered ``"left"`` along some axes.
    grid : grid
        The grid of the field.
    bc : BC
        The condition of the faces, for the last index of each shifted axis.

    Returns
    -------
    CField
        The same values on ``"right"`` axes.

    Raises
    ------
    ValueError
        If an axis is at another position (``"outer"``, ``"inner"``), or
        ``bc`` has no condition for the high face of a shifted axis.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 2, 1.0, 1.0)
    >>> west = gridax.CField(
    ...     np.broadcast_to(np.arange(6.0), (4, 6)), ("y", "x"), stagger={"x": "left"}
    ... )
    >>> bc = gridax.BC(x=gridax.Neumann())
    >>> gridax.to_right(west, grid, bc).data[1].tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0, 4.0]
    """
    positions = dict(field.stagger)
    left = [axis for axis, position in positions.items() if position == "left"]
    other = {a: p for a, p in positions.items() if p not in ("left", "right")}
    if other:
        raise ValueError(f"{field!r} is at {other}: to_right converts left axes")
    if not left:
        return field
    for axis in left:
        if bc.faces.get(axis, (None, None))[1] is None:
            raise ValueError(f"bc has no condition for the high {axis} face")
    xp = array_namespace(field.data)
    data = field.data
    for axis in left:
        # right[i] = left[i + 1]; the last index is filled below
        data = xp.roll(data, -1, axis=field.dims.index(axis))
    shifted = CField(data, field.dims, field.labels, {a: "right" for a in positions})
    filled = fill_ghosts(shifted, grid, bc).data
    for axis in left:
        last = tuple(-1 if dim == axis else slice(None) for dim in field.dims)
        data = set_at(data, last, filled[last])
    return shifted.with_data(data)
