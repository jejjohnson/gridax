"""`BC` says which condition each face has; `fill_ghosts` writes the ghost ring."""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np

from gridax._src.boundary.atoms import Atom, Periodic, _Atom
from gridax._src.fields.cfield import CField, StaggerMismatchError
from gridax._src.grid.points import GRID_AXES, Stagger, as_stagger, point_stagger
from gridax._src.pytree import pytree_dataclass, static_field

#: Face names, as the axis and the side (0 low, 1 high).
FACES: dict[str, tuple[str, int]] = {
    "west": ("x", 0),
    "east": ("x", 1),
    "south": ("y", 0),
    "north": ("y", 1),
    "bottom": ("z", 0),
    "top": ("z", 1),
}

type Corners = Literal["average", "priority"]


@pytree_dataclass
class BC:
    """The boundary condition of each face of a grid.

    Faces are named by grid axis: one condition for both sides of the axis,
    or a ``(low, high)`` pair (``None`` leaves a side alone). ``west`` /
    ``east``, ``south`` / ``north`` and ``bottom`` / ``top`` name one side.
    A condition knows neither the point nor the grid: `fill_ghosts` gives
    both, so one `BC` serves every field with the same physics.

    Parameters
    ----------
    corners : {"average", "priority"}
        What a ghost corner holds, where two faces meet. ``"average"``: the
        mean over the orders the faces can be applied in. ``"priority"``: the
        later axis of the grid wins (x over y over z).
    **faces
        ``x=``, ``y=``, ``z=``: a condition or a ``(low, high)`` pair; or a
        face name with one condition.

    Raises
    ------
    ValueError
        If a face is given twice or unknown, one side of an axis is periodic
        and the other is not, or ``corners`` is unknown.
    TypeError
        If a condition is not one of gridax's.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 1.0, 1.0)
    >>> bc = gridax.BC(x=gridax.Periodic(), y=(gridax.Dirichlet(0.0), gridax.Neumann()))
    >>> gridax.fill_ghosts(np.ones(grid.shape), grid, bc)[:, 2].tolist()
    [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    """

    faces: dict[str, tuple[Atom | None, Atom | None]]
    corners: Corners = static_field()

    def __init__(self, *, corners: Corners = "average", **faces: Any) -> None:
        if corners not in ("average", "priority"):
            raise ValueError(
                f"corners must be 'average' or 'priority', got {corners!r}"
            )
        sides: dict[str, list[Any]] = {}

        def put(axis: str, side: int, atom: Any) -> None:
            if atom is not None and not isinstance(atom, _Atom):
                raise TypeError(f"{atom!r} is not a boundary condition of gridax")
            pair = sides.setdefault(axis, [None, None])
            if pair[side] is not None:
                raise ValueError(f"the {'low high'.split()[side]} {axis} face twice")
            pair[side] = atom

        for name, given in faces.items():
            if name in FACES:
                put(*FACES[name], given)
            elif name in GRID_AXES:
                pair = given if isinstance(given, tuple | list) else (given, given)
                if len(pair) != 2:
                    raise ValueError(f"{name}= takes one condition or (low, high)")
                for side, atom in enumerate(pair):
                    put(name, side, atom)
            else:
                raise ValueError(
                    f"unknown face {name!r}: an axis of {GRID_AXES} or one of"
                    f" {sorted(FACES)}"
                )
        for axis, pair in sides.items():
            if sum(isinstance(atom, Periodic) for atom in pair) == 1:
                raise ValueError(f"both {axis} faces must be periodic, or neither")
        self.faces = {axis: (low, high) for axis, (low, high) in sides.items()}
        self.corners = corners


def fill_ghosts(
    f: Any,
    grid: Any,
    bc: BC,
    *,
    width: int | None = None,
    stagger: str | Stagger | None = None,
) -> Any:
    """Fill the ghost ring of a field from its boundary conditions.

    The point of the field says where the wall is (`gridax.BC` conditions
    hold at the wall) and the grid gives the spacing on each face.

    Parameters
    ----------
    f : CField or array
        A field with its ghost ring. A plain array has its grid axes last, in
        the order of ``grid.axes``.
    grid : grid
        A Cartesian or spherical C-grid, 1-D, 2-D or 3-D.
    bc : BC
        The condition of each face. Faces it leaves out are not written.
    width : int, optional
        Number of ghost layers of ``f`` on each side: the grid's by default.
    stagger : str or Stagger, optional
        The point of a plain array, e.g. ``"U"`` (T by default). A `CField`
        has its own.

    Returns
    -------
    CField or array
        ``f`` with its ghost layers (and the nodes on a wall) written.

    Raises
    ------
    StaggerMismatchError
        If ``stagger`` is not the point of the `CField`.
    ValueError
        If ``bc`` has an axis the grid or the field has not, or the field is
        too small for ``width`` ghost layers.
    """
    width = grid.ghost if width is None else width
    unknown = [axis for axis in bc.faces if axis not in grid.axes]
    if unknown:
        raise ValueError(f"bc has the axes {unknown}, not in grid axes {grid.axes}")
    if isinstance(f, CField):
        if stagger is not None and _as_stagger(stagger) != f.stagger:
            raise StaggerMismatchError(f"{f!r} is not at {stagger!r}")
        missing = [axis for axis in grid.axes if axis not in f.dims]
        if missing:
            raise ValueError(f"{f!r} has no dims {missing}")
        others = tuple(dim for dim in f.dims if dim not in grid.axes)
        canonical = f.transpose((*others, *grid.axes))
        data = _fill(canonical.data, grid, bc, width, f.stagger)
        return canonical.with_data(data).transpose(f.dims)
    return _fill(f, grid, bc, width, _as_stagger("T" if stagger is None else stagger))


def _as_stagger(stagger: str | Stagger) -> Stagger:
    return point_stagger(stagger) if isinstance(stagger, str) else as_stagger(stagger)


def _fill(f: Any, grid: Any, bc: BC, width: int, stagger: Stagger) -> Any:
    n_axes = len(grid.axes)
    if np.ndim(f) < n_axes:
        raise ValueError(f"array of shape {f.shape} on a grid with axes {grid.axes}")
    sizes = tuple(f.shape[-n_axes:])
    if width < 1 or any(size <= 2 * width for size in sizes):
        raise ValueError(f"no room for {width} ghost layers in shape {sizes}")
    axes = [axis for axis in grid.axes if axis in bc.faces]

    def apply(order: Any) -> Any:
        out = f
        for axis in order:
            for side, atom in enumerate(bc.faces[axis]):
                if atom is not None:
                    out = atom.fill(out, grid, axis, side, stagger, width)
        return out

    if bc.corners == "priority" or len(axes) < 2:
        return apply(axes)
    # The mean over the orders, as the first plus the mean difference: off the
    # corners every order agrees, and the value stays exact.
    first, *rest = (apply(order) for order in itertools.permutations(axes))
    return first + sum(out - first for out in rest) / (len(rest) + 1)
