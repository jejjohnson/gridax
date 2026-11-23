"""`located`: canonical functions on located fields.

A canonical function is written on arrays. ``@located`` declares the point of
each of its field arguments and of its result, and makes it accept `CField`
operands: it checks their points, aligns their dims, calls the function on
their arrays and puts the result on its point. Calls on plain arrays go
straight through.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
import inspect
from typing import Any, cast

from gridax._src.fields.cfield import CField, StaggerMismatchError
from gridax._src.fields.combine import _check_plain
from gridax._src.grid.points import (
    POINT_AXES,
    Stagger,
    as_stagger,
    point_stagger,
)

#: What a point of the declaration may be: a point name, or the name of the
#: keyword argument that gives it (``"src"``, ``"dst"``, ``"at"``), or that
#: gives the axis of the faces it is (``"axis"``).
type Point = str


#: The faces along each axis, for a keyword that is an axis (``axis="x"``).
_FACES = {"x": "U", "y": "V", "z": "W"}


def _as_stagger(point: str | Stagger) -> Stagger:
    if isinstance(point, str):
        return point_stagger(_FACES.get(point, point))
    return as_stagger(point)


def _where(stagger: Stagger) -> Any:
    from gridax._src.grid.points import stagger_point

    return stagger_point(stagger) or dict(stagger)


def located[F: Callable[..., Any]](
    out: Point | tuple[Point, ...] | None, **points: Point
) -> Callable[[F], F]:
    """Declare the points of a canonical function, so it accepts `CField`.

    Parameters
    ----------
    out : str, tuple of str or None
        The point of the result, one per result of a function that returns
        several, or ``None`` for a reduction over the grid axes.
    **points : str
        The point of each field argument, by its name. A point is a name
        (``"U"``) or the keyword argument that holds it (``"src"``); a
        ``CField`` then supplies that keyword, and one that contradicts it
        raises.

    Returns
    -------
    callable
        A decorator. The function it returns has ``__located__``, the
        declaration.

    Notes
    -----
    Located operands must have the same dims, whatever their order; the
    result has the order of the first. A later operand may lack the leading
    ones (a Coriolis parameter without the levels of the velocity), and
    broadcasts over them. A plain array next to a field may only
    broadcast, as in `gridax.combine`. A reduction returns a plain array, or a
    ``pipekit_array.DimArray`` if other dims are left.
    """

    def decorate(fn: F) -> F:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not any(isinstance(x, CField) for x in (*args, *kwargs.values())):
                return fn(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return _call(fn, bound.arguments, out, points)

        located_fn = cast(Any, wrapper)
        located_fn.__located__ = (out, dict(points))
        return located_fn

    return decorate


def _call(
    fn: Any,
    arguments: dict[str, Any],
    out: Point | tuple[Point, ...] | None,
    points: dict[str, Point],
) -> Any:
    grid = arguments["grid"]
    fields = {name: x for name, x in arguments.items() if isinstance(x, CField)}
    undeclared = sorted(set(fields) - set(points))
    if undeclared:
        raise TypeError(f"{fn.__name__} takes no CField for {undeclared}")
    # the field with the most dims, the first of those
    first = max(fields.values(), key=lambda x: x.ndim)
    missing = [axis for axis in grid.axes if axis not in first.dims]
    if missing:
        raise ValueError(f"{first!r} has no dims {missing} of the grid")
    others = tuple(dim for dim in first.dims if dim not in grid.axes)
    canonical = (*others, *grid.axes)
    if getattr(grid, "_collocated", False):
        # a collocated grid has one point
        points = dict.fromkeys(points, "T")
        out = ("T",) * len(out) if isinstance(out, tuple) else out and "T"

    labels: dict[str, Any] = {}
    values = dict(arguments)
    for i, (name, point) in enumerate(points.items()):
        x = arguments[name]
        if x is None:
            continue
        if not isinstance(x, CField):
            _check_plain(x, first.transpose(canonical), i)
            continue
        _check_point(fn, name, x, point, values)
        own = canonical[len(canonical) - x.ndim :]
        if set(x.dims) != set(own):
            raise ValueError(f"{name} has dims {x.dims}, not {first.dims}")
        for dim in others:
            if (
                dim in x.labels
                and labels.setdefault(dim, x.labels[dim]) != x.labels[dim]
            ):
                raise ValueError(f"{name} has other labels on {dim!r}")
        values[name] = x.transpose(own).data

    result = fn(**values)
    if out is None:
        return _reduced(result, others, labels)
    if isinstance(out, tuple):
        return tuple(
            _wrap(r, p, values, canonical, first.dims, labels)
            for r, p in zip(result, out, strict=True)
        )
    return _wrap(result, out, values, canonical, first.dims, labels)


def _check_point(
    fn: Any, name: str, x: CField, point: Point, values: dict[str, Any]
) -> None:
    """Check the point of ``x``, or supply the keyword that names it."""
    if point in POINT_AXES:
        wanted = point_stagger(point)
    elif values[point] is None:
        values[point] = x.stagger
        return
    else:
        wanted = _as_stagger(values[point])
    if x.stagger != wanted:
        raise StaggerMismatchError(
            f"{fn.__name__}: {name} sits at {_where(x.stagger)}, not at"
            f" {_where(wanted)}"
        )


def _wrap(
    data: Any,
    point: Point,
    values: dict[str, Any],
    canonical: tuple[str, ...],
    dims: tuple[str, ...],
    labels: dict[str, Any],
) -> CField:
    given = point if point in POINT_AXES else values[point]
    return CField(data, canonical, labels, _as_stagger(given)).transpose(dims)


def _reduced(data: Any, dims: tuple[str, ...], labels: dict[str, Any]) -> Any:
    """What a reduction over the grid axes leaves: no point, maybe other dims."""
    if not dims:
        return data
    from gridax._src.fields.convert import dimarray

    return dimarray(data, dims, labels)
