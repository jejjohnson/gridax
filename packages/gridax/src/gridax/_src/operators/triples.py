"""The staggered moves: which Layer 1 stencil takes one point to another.

A move along an axis is a forward stencil when the destination gains the
axis's half step (T to U along x) and a backward one when it loses it (U to
T), in the same-index convention (AGENTS.md contract 1). The tables hold every
move that has a raw stencil, for each rank.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from gridax._src.grid.points import POINT_AXES, Stagger, as_stagger, stagger_point
from gridax._src.operators import stencils

type Stencil = Callable[[Any], Any]

#: The axes of each rank, in array order.
RANK_AXES: tuple[tuple[str, ...], ...] = (("x",), ("y", "x"), ("z", "y", "x"))


def _points(axes: tuple[str, ...]) -> list[str]:
    return [p for p, along in POINT_AXES.items() if set(along) <= set(axes)]


def _side(axis: str, dst: str) -> str:
    return "fwd" if axis in POINT_AXES[dst] else "bwd"


def _moves(axes: tuple[str, ...]) -> list[tuple[tuple[str, ...], str, str]]:
    """``(axes moved along, src, dst)`` for every two points."""
    return [
        (tuple(sorted(set(POINT_AXES[src]) ^ set(POINT_AXES[dst]))), src, dst)
        for src in _points(axes)
        for dst in _points(axes)
        if src != dst
    ]


def _differences(axes: tuple[str, ...]) -> dict[tuple[str, str, str], Stencil]:
    table = {}
    for along, src, dst in _moves(axes):
        if len(along) == 1:
            (axis,) = along
            name = f"diff_{axis}_{_side(axis, dst)}_{len(axes)}d"
            table[axis, src, dst] = getattr(stencils, name)
    return table


def _averages(axes: tuple[str, ...]) -> dict[tuple[str, str], Stencil]:
    table = {}
    for along, src, dst in _moves(axes):
        if len(along) == 1:
            sides = f"{along[0]}_{_side(along[0], dst)}"
        else:
            sides = "_".join(f"{axis}{_side(axis, dst)}" for axis in along)
        # a diagonal move across z (U to W) has no raw stencil
        stencil = getattr(stencils, f"avg_{sides}_{len(axes)}d", None)
        if stencil is not None:
            table[src, dst] = stencil
    return table


#: ``grid axes -> (axis, src, dst) -> Layer 1 difference``.
TRIPLES: Mapping[tuple[str, ...], Mapping[tuple[str, str, str], Stencil]] = {
    axes: _differences(axes) for axes in RANK_AXES
}

#: ``grid axes -> (src, dst) -> Layer 1 average``.
PAIRS: Mapping[tuple[str, ...], Mapping[tuple[str, str], Stencil]] = {
    axes: _averages(axes) for axes in RANK_AXES
}


def point_name(point: str | Stagger | None) -> str:
    """The name of a point given by name or by stagger.

    Raises
    ------
    ValueError
        If the stagger has no name, or no point is given.
    """
    if isinstance(point, str):
        return point
    if point is None:
        raise ValueError("a staggered move needs src= and dst=")
    name = stagger_point(as_stagger(point))
    if name is None:
        raise ValueError(f"gridax does not compute on the stagger {point}")
    return name


def choose[K: tuple[str, ...]](
    what: str, key: K, table: Mapping[K, Stencil]
) -> Stencil:
    """The stencil of a move, or an error that lists the moves there are.

    Raises
    ------
    ValueError
        If the table has no such move.
    """
    if key in table:
        return table[key]
    options = ", ".join("/".join(k) for k in sorted(table))
    raise ValueError(f"{what} {'/'.join(key)} is not available; valid: {options}")
