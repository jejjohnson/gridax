"""The points of the Arakawa C-grid, as the axes each one is staggered along.

A point has a one-letter name (``"U"``) and a *stagger*: the position of its
values along each axis where they are off centre, in xgcm's vocabulary. The
names are shorthands for the staggers the operators compute on; a stagger
also describes the others (``"left"`` faces of MITgcm-style output, the
``"outer"`` and ``"inner"`` faces that include or exclude both walls).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

#: The axis names of the grids, in array order.
GRID_AXES: tuple[str, ...] = ("z", "y", "x")

#: ``((axis, position), ...)``, sorted by axis; a centred axis is left out.
Stagger = tuple[tuple[str, str], ...]

#: xgcm's axis positions. The operators compute on ``center`` and ``right``.
POSITIONS: tuple[str, ...] = ("center", "left", "right", "outer", "inner")

#: ``point -> axes`` it sits half a cell along (the ``+`` face of its T cell).
POINT_AXES: dict[str, tuple[str, ...]] = {
    "T": (),
    "U": ("x",),
    "V": ("y",),
    "X": ("x", "y"),
    "W": ("z",),
}


def as_stagger(stagger: Mapping[str, str] | Iterable[tuple[str, str]]) -> Stagger:
    """A stagger in normal form: sorted by axis, with no centred axis.

    Parameters
    ----------
    stagger : mapping or iterable of (axis, position)
        Position per axis, e.g. ``{"x": "right"}``.

    Returns
    -------
    Stagger

    Raises
    ------
    ValueError
        If an axis repeats or a position is not one of xgcm's.

    Examples
    --------
    >>> as_stagger({"y": "right", "x": "right", "z": "center"})
    (('x', 'right'), ('y', 'right'))
    """
    pairs = list(stagger.items() if isinstance(stagger, Mapping) else stagger)
    axes = [axis for axis, _ in pairs]
    if len(set(axes)) != len(axes):
        raise ValueError(f"stagger repeats an axis: {pairs}")
    for axis, position in pairs:
        if position not in POSITIONS:
            raise ValueError(
                f"position {position!r} on axis {axis!r} is not one of {POSITIONS}"
            )
    return tuple(sorted((a, p) for a, p in pairs if p != "center"))


def point_stagger(point: str) -> Stagger:
    """The stagger of a point name.

    Raises
    ------
    ValueError
        If ``point`` is not ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.

    Examples
    --------
    >>> point_stagger("X")
    (('x', 'right'), ('y', 'right'))
    """
    if point not in POINT_AXES:
        raise ValueError(f"{point!r} is not a point: {sorted(POINT_AXES)}")
    return tuple((axis, "right") for axis in POINT_AXES[point])


def stagger_point(stagger: Stagger) -> str | None:
    """The name of the point with this stagger, or None if it has none.

    Examples
    --------
    >>> stagger_point((("x", "right"),)), stagger_point((("x", "left"),))
    ('U', None)
    """
    for point in POINT_AXES:
        if point_stagger(point) == stagger:
            return point
    return None


def point_axes(at: str | Stagger, axes: tuple[str, ...]) -> tuple[str, ...]:
    """The axes the point ``at`` is staggered along, on a grid with ``axes``.

    Parameters
    ----------
    at : str or Stagger
        ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``, or such a point's
        stagger (``CField.stagger``).
    axes : tuple of str
        The grid's axis names.

    Returns
    -------
    tuple of str
        The staggered axes, empty at T.

    Raises
    ------
    ValueError
        If ``at`` is not a point of a grid with those axes.

    Examples
    --------
    >>> point_axes("X", ("y", "x")), point_axes((("x", "right"),), ("y", "x"))
    (('x', 'y'), ('x',))
    """
    point = at if isinstance(at, str) else stagger_point(as_stagger(at))
    staggered = None if point is None else POINT_AXES.get(point)
    if staggered is None or not set(staggered) <= set(axes):
        valid = [p for p, along in POINT_AXES.items() if set(along) <= set(axes)]
        raise ValueError(f"at={at!r} is not one of {valid} on a grid with axes {axes}")
    return staggered
