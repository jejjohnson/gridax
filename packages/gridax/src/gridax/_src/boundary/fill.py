"""`BC` says which condition each face has; `fill_ghosts` writes the ghost ring."""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np
from pipekit_array.functions import namespace_of

from gridax._src.boundary.atoms import Atom, Dirichlet, Neumann, Periodic, Wall, _Atom
from gridax._src.fields.cfield import CField, StaggerMismatchError
from gridax._src.grid.points import (
    GRID_AXES,
    Stagger,
    as_stagger,
    point_axes,
    point_stagger,
)
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
    coast : BC, optional
        The conditions at the coasts inside the domain, where they differ from
        those of the faces (`fill_ghosts` with ``mask=``). A coast on the east
        side of the water takes the condition of the east face by default, and
        none if that face is periodic or left alone.
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
    coast: BC | None
    corners: Corners = static_field()

    def __init__(
        self,
        *,
        corners: Corners = "average",
        coast: BC | None = None,
        **faces: Any,
    ) -> None:
        if not (coast is None or isinstance(coast, BC)):
            raise TypeError(f"coast must be a BC, got {type(coast).__name__}")
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
        self.coast = coast
        self.corners = corners

    def at_face(
        self, axis: str, side: int, point: Stagger, axes: tuple[str, ...]
    ) -> Atom | None:
        """The condition of a face for a field at ``point``, ``None`` for none."""
        atom = self.faces.get(axis, (None, None))[side]
        return None if atom is None else atom.at(axis, point, axes)

    def at_coast(
        self, axis: str, side: int, point: Stagger, axes: tuple[str, ...]
    ) -> Atom | None:
        """The condition of the coasts on a side of the water, ``None`` for none."""
        for bc in (self.coast, self):
            atom = None if bc is None else bc.at_face(axis, side, point, axes)
            if atom is not None and not isinstance(atom, Periodic):
                return atom
        return None

    # ── presets, by physical role ────────────────────────────────────────────

    @classmethod
    def no_flux(cls, *axes: str) -> BC:
        """No flux through the walls, ``∂φ/∂n = 0``: tracers, pressure.

        Parameters
        ----------
        *axes : str
            The axes with walls: x and y by default.
        """
        return cls._same(Neumann(0.0), axes)

    @classmethod
    def zero(cls, *axes: str) -> BC:
        """Zero at the walls, ``φ = 0``: a streamfunction.

        Parameters
        ----------
        *axes : str
            The axes with walls: x and y by default.
        """
        return cls._same(Dirichlet(0.0), axes)

    @classmethod
    def periodic(cls, *axes: str) -> BC:
        """Periodic axes.

        Parameters
        ----------
        *axes : str
            The periodic axes: x and y by default.
        """
        return cls._same(Periodic(), axes)

    @classmethod
    def wall_normal(cls, *axes: str) -> BC:
        """No flow through the walls, ``u·n = 0``, for the velocity normal to them.

        U across x, V across y, W across z. Faces along a component are left
        alone: ``BC.wall_normal() | BC.wall_tangential(slip="no")`` is a solid
        wall for every component.

        Parameters
        ----------
        *axes : str
            The axes with walls: x and y by default.
        """
        return cls._same(Wall(normal=Dirichlet(0.0)), axes)

    @classmethod
    def wall_tangential(cls, *axes: str, slip: Literal["no", "free"]) -> BC:
        """The velocity along the walls: zero (no slip), or of zero gradient.

        Faces normal to a component are left alone.

        Parameters
        ----------
        *axes : str
            The axes with walls: x and y by default.
        slip : {"no", "free"}
            ``"no"``: ``u_t = 0`` at the wall. ``"free"``: ``∂u_t/∂n = 0``.

        Raises
        ------
        ValueError
            If ``slip`` is neither.
        """
        if slip not in ("no", "free"):
            raise ValueError(f"slip must be 'no' or 'free', got {slip!r}")
        atom = Dirichlet(0.0) if slip == "no" else Neumann(0.0)
        return cls._same(Wall(tangential=atom), axes)

    @classmethod
    def _same(cls, atom: Atom, axes: tuple[str, ...]) -> BC:
        faces: dict[str, Any] = dict.fromkeys(axes or ("x", "y"), atom)
        return cls(**faces)

    def __or__(self, other: BC) -> BC:
        """The conditions of both, those of ``self`` where they both have one.

        Two walls combine: the normal condition of one and the tangential of
        the other.
        """
        if not isinstance(other, BC):
            return NotImplemented

        def join(a: Any, b: Any) -> Any:
            if isinstance(a, Wall) and isinstance(b, Wall):
                return Wall(
                    normal=b.normal if a.normal is None else a.normal,
                    tangential=b.tangential if a.tangential is None else a.tangential,
                )
            return b if a is None else a

        none = (None, None)
        faces: dict[str, Any] = {
            axis: tuple(
                join(a, b)
                for a, b in zip(
                    self.faces.get(axis, none), other.faces.get(axis, none), strict=True
                )
            )
            for axis in dict.fromkeys((*self.faces, *other.faces))
        }
        coast = self.coast
        if other.coast is not None:
            coast = other.coast if coast is None else coast | other.coast
        return BC(corners=self.corners, coast=coast, **faces)


def fill_ghosts(
    f: Any,
    grid: Any,
    bc: Any,
    *,
    mask: Any = None,
    width: int | None = None,
    stagger: Any = None,
) -> Any:
    """Fill the ghost ring of a field, or of every field of a state.

    The point of the field says where the wall is (`gridax.BC` conditions
    hold at the wall) and the grid gives the spacing on each face. With a
    mask, the dry points behind the coasts are filled by the same rules, so a
    coast and a face with the same condition give the same values.

    Parameters
    ----------
    f : CField, array or pytree of them
        A field with its ghost ring. A plain array has its grid axes last, in
        the order of ``grid.axes``. A state (a dict, a NamedTuple, an
        ``eqx.Module``, …) is filled leaf by leaf, which needs JAX.
    grid : grid
        A Cartesian or spherical C-grid, 1-D, 2-D or 3-D.
    bc : BC or pytree of BC
        The condition of each face. Faces it leaves out are not written. For
        a state, a pytree of its structure with one `BC` per field, or
        ``None`` to leave the field alone.
    mask : Mask1D, Mask2D or Mask3D, optional
        The wet cells. ``width`` layers of dry points behind each coast are
        written too (and the nodes on it), with the wet points they mirror
        (``mask.coast_ghosts``). A dry point with water on several sides holds
        the mean over them. No wet point is written.
    width : int, optional
        Number of ghost layers on each side: the grid's by default.
    stagger : str or Stagger, optional
        The point of a plain array, e.g. ``"U"`` (T by default). A `CField`
        has its own. For a state, a pytree of its structure with the point of
        each field (``None`` for the fields that have their own).

    Returns
    -------
    CField, array or pytree of them
        ``f`` with its ghost layers (and the nodes on a wall) written.

    Raises
    ------
    StaggerMismatchError
        If ``stagger`` is not the point of the `CField`.
    ValueError
        If ``bc`` has an axis the grid or the field has not, the field is
        too small for ``width`` ghost layers, or the structure of a state and
        of its ``bc`` differ.
    TypeError
        If ``bc`` holds something other than a `BC` for a field.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 1.0, 1.0)
    >>> state = {
    ...     "h": gridax.CField(np.ones(grid.shape), ("y", "x"), loc="T"),
    ...     "u": gridax.CField(np.ones(grid.shape), ("y", "x"), loc="U"),
    ... }
    >>> wall, flat = gridax.Dirichlet(0.0), gridax.Neumann()
    >>> bcs = {"h": gridax.BC(x=flat, y=flat), "u": gridax.BC(x=wall, y=flat)}
    >>> out = gridax.fill_ghosts(state, grid, bcs)
    >>> float(out["u"].data[2, 0]), float(out["h"].data[2, 0])
    (0.0, 1.0)
    """
    if isinstance(bc, BC):
        return _fill_field(f, grid, bc, mask, width, stagger)
    if isinstance(f, CField) or hasattr(f, "shape"):
        raise TypeError(f"bc for one field must be a BC, got {type(bc).__name__}")
    import jax

    leaves, structure = jax.tree.flatten(f, is_leaf=lambda x: isinstance(x, CField))
    conditions = structure.flatten_up_to(bc)
    points = [None] * len(leaves)
    if stagger is not None:
        points = structure.flatten_up_to(stagger)
    filled = []
    for leaf, condition, point in zip(leaves, conditions, points, strict=True):
        if condition is None:
            filled.append(leaf)
        elif isinstance(condition, BC):
            filled.append(_fill_field(leaf, grid, condition, mask, width, point))
        else:
            raise TypeError(
                f"each field takes a BC or None, got {type(condition).__name__}"
            )
    return structure.unflatten(filled)


def _fill_field(
    f: Any, grid: Any, bc: BC, mask: Any, width: int | None, stagger: Any
) -> Any:
    """`fill_ghosts` of one field."""
    width = grid.ghost if width is None else width
    coast = {} if bc.coast is None else bc.coast.faces
    unknown = [axis for axis in (*bc.faces, *coast) if axis not in grid.axes]
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
        data = _filled(canonical.data, grid, bc, mask, width, f.stagger)
        return canonical.with_data(data).transpose(f.dims)
    point = _as_stagger("T" if stagger is None else stagger)
    return _filled(f, grid, bc, mask, width, point)


def _as_stagger(stagger: str | Stagger) -> Stagger:
    return point_stagger(stagger) if isinstance(stagger, str) else as_stagger(stagger)


def _filled(f: Any, grid: Any, bc: BC, mask: Any, width: int, stagger: Stagger) -> Any:
    """The faces and, with a mask, the coasts."""
    if mask is None:
        return _fill(f, grid, bc, width, stagger)
    return fill_coasts(f, grid, bc, mask, stagger, width=width, layers=width)


def fill_coasts(
    f: Any,
    grid: Any,
    bc: BC,
    mask: Any,
    stagger: Stagger,
    *,
    width: int,
    layers: int,
) -> Any:
    """Fill ``width`` ghost layers, and ``layers`` dry points behind the coasts."""
    # the coasts read the ring (a periodic image of the water), and the ring
    # then wraps or mirrors the coasts
    out = _fill(f, grid, bc, width, stagger)
    out = _fill_coasts(out, grid, bc, mask, stagger, layers=layers)
    return _fill(out, grid, bc, width, stagger)


def _fill_coasts(
    f: Any,
    grid: Any,
    bc: BC,
    mask: Any,
    stagger: Stagger,
    *,
    layers: int,
) -> Any:
    """Write ``layers`` dry points behind the coasts, the nearest first.

    Nearest by distance from the water: the nodes on a coast come before any
    mirror, so a wall keeps its value where it ends next to another one.
    """
    xp = namespace_of(f)
    on_wall = point_axes(stagger, grid.axes)
    kind = "strict" if len(on_wall) == 2 else "default"
    coasts = [
        (axis, side, atom)
        for axis in grid.axes
        for side in (0, 1)
        if (atom := bc.at_coast(axis, side, stagger, grid.axes)) is not None
    ]
    written: Any = None
    # in half cells from the coast: the nodes and the layers behind them are at
    # 0, 2, 4, … and the layers of a cell-centred direction at 1, 3, 5, …
    for distance in range(2 * layers + 1):
        total: Any = 0.0
        count: Any = 0
        for axis, side, atom in coasts:
            if (axis in on_wall) != (distance % 2 == 0):
                continue
            dry = distance // 2 + 1
            points = mask.coast_ghosts(
                axis, ("low", "high")[side], at=stagger, kind=kind, layer=dry
            ).points
            values = atom.coast(f, grid, axis, side, stagger, dry)
            total = total + xp.where(points, values, xp.zeros_like(values))
            count = count + xp.asarray(points, dtype=xp.int8)
        if isinstance(count, int):
            continue  # no coast at this distance
        here = count > 0
        mean = total / xp.asarray(xp.where(here, count, 1), dtype=f.dtype)
        if written is not None:
            here = here & ~written
        f = xp.where(here, mean, f)
        written = here if written is None else written | here
    return f


def _fill(f: Any, grid: Any, bc: BC, width: int, stagger: Stagger) -> Any:
    n_axes = len(grid.axes)
    if np.ndim(f) < n_axes:
        raise ValueError(f"array of shape {f.shape} on a grid with axes {grid.axes}")
    sizes = tuple(f.shape[-n_axes:])
    if width < 1 or any(size <= 2 * width for size in sizes):
        raise ValueError(f"no room for {width} ghost layers in shape {sizes}")
    axes = [axis for axis in grid.axes if axis in bc.faces]
    return _apply(f, grid, bc, axes, width, stagger)


def _apply(
    f: Any, grid: Any, bc: BC, axes: list[str], width: int, stagger: Stagger
) -> Any:
    """Fill the faces of ``axes``, and the corners between them by their rule."""

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


def wrapping(axes: list[str]) -> BC:
    """Periodic faces along ``axes``."""
    faces: dict[str, Any] = {axis: Periodic() for axis in axes}
    return BC(**faces)


def extend(
    f: Any, grid: Any, bc: BC, stagger: Stagger, axes: list[str], extra: int
) -> Any:
    """``f`` with ``extra`` more ghost layers on ``axes``, its ring among them filled.

    What an operator reads when its stencil is wider than the ghost ring; a
    field that is returned keeps the ring of the grid. The faces of ``axes``
    must all have a condition.
    """
    unknown = [axis for axis in bc.faces if axis not in grid.axes]
    if unknown:
        raise ValueError(f"bc has the axes {unknown}, not in grid axes {grid.axes}")
    for name, (axis, side) in FACES.items():
        if axis in axes and bc.at_face(axis, side, stagger, grid.axes) is None:
            raise ValueError(f"bc has no condition for the {name} face")
    xp = namespace_of(f)
    for axis in axes:
        position = grid.axes.index(axis) - len(grid.axes)
        f = _widened(xp, f, position, extra, "zero")
    return _apply(f, grid, bc, axes, grid.ghost + extra, stagger)


def extend_wet(wet: Any, grid: Any, bc: BC, axes: list[str], extra: int) -> Any:
    """The wet points of an extended field.

    Wrapped where the faces are periodic, and those of the ghost ring elsewhere.
    """
    xp = namespace_of(wet)
    for axis in axes:
        position = grid.axes.index(axis) - len(grid.axes)
        periodic = isinstance(bc.faces[axis][0], Periodic)
        wet = _widened(xp, wet, position, extra, "wrap" if periodic else "ring", grid)
    return wet


def _widened(
    xp: Any, x: Any, position: int, extra: int, how: str, grid: Any = None
) -> Any:
    """``x`` with ``extra`` layers on both sides of an axis."""
    if extra == 0:
        return x
    n = x.shape[position]

    def layers(start: int, stop: int) -> Any:
        return x[(..., slice(start, stop), *(slice(None),) * (-position - 1))]

    if how == "zero":
        low = high = xp.zeros_like(layers(0, 1))
        sides = [low] * extra, [high] * extra
    elif how == "ring":
        sides = [layers(0, 1)] * extra, [layers(n - 1, n)] * extra
    else:
        period = n - 2 * grid.ghost
        sides = (
            [layers(period - extra, period)],
            [layers(n - period, n - period + extra)],
        )
    return xp.concat([*sides[0], x, *sides[1]], axis=position)
