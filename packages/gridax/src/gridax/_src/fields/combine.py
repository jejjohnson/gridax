"""Element-wise maths on located fields: `combine` and the functions built on it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gridax._src.fields.cfield import CField, StaggerMismatchError
from gridax._src.grid.points import GRID_AXES
from gridax._src.pytree import array_namespace


def combine(fn: Callable[..., Any], *operands: Any) -> CField:
    """Apply an element-wise function to fields on the same point.

    The fields are aligned by dim name, in the order of the first one, and
    ``fn`` is called on their arrays: a compiled program is the one ``fn``
    gives on bare arrays.

    A plain array has no point, so it may only broadcast: it must have size 1
    along every axis the fields are staggered along, and along at least one
    grid axis. A scalar, a per-layer column ``[nl, 1, 1]`` and, next to a
    U field, a profile ``[Ny, 1]`` are fine. Wrap a full array with
    ``f.like(array)``, which asserts that it sits on ``f``'s point.

    Parameters
    ----------
    fn : callable
        An element-wise function of the operands' arrays.
    *operands : CField, array or number
        At least one field. A plain array's axes are aligned with the last
        dims of the first field, as in NumPy broadcasting.

    Returns
    -------
    CField
        On the point, dims and labels of the first field.

    Raises
    ------
    StaggerMismatchError
        If two fields sit on different points.
    ValueError
        If two fields differ in dims, sizes or labels, a plain array does
        not broadcast as above, or ``fn`` is not element-wise.
    TypeError
        If no operand is a field.

    Examples
    --------
    >>> import numpy as np
    >>> from gridax import CField, combine
    >>> u = CField(np.ones((4, 4)), ("y", "x"), loc="U")
    >>> h = CField(np.ones((4, 4)), ("y", "x"), loc="T")
    >>> combine(lambda a, b: a - 2.0 * b, u, u).loc
    'U'
    >>> combine(np.multiply, u, h)
    Traceback (most recent call last):
        ...
    gridax._src.fields.cfield.StaggerMismatchError: operand 1 sits at T, not at U
    """
    fields = [x for x in operands if isinstance(x, CField)]
    if not fields:
        raise TypeError("combine needs at least one CField")
    first = fields[0]
    arrays = []
    labels = dict(first.labels)
    for i, x in enumerate(operands):
        if isinstance(x, CField):
            arrays.append(_aligned(x, first, labels, i).data)
        else:
            _check_plain(x, first, i)
            arrays.append(x)
    out = fn(*arrays)
    if tuple(out.shape) != first.shape:
        raise ValueError(
            f"fn is not element-wise: it maps {first!r} to shape {tuple(out.shape)}"
        )
    return CField(out, first.dims, labels, first.stagger)


def _aligned(x: CField, first: CField, labels: dict[str, Any], i: int) -> CField:
    """``x`` in the dim order of ``first``, checked; its labels join ``labels``."""
    if x.stagger != first.stagger:
        here = x.loc or dict(x.stagger)
        there = first.loc or dict(first.stagger)
        raise StaggerMismatchError(f"operand {i} sits at {here}, not at {there}")
    if set(x.dims) != set(first.dims):
        raise ValueError(f"operand {i} has dims {x.dims}, not {first.dims}")
    x = x.transpose(first.dims)
    if x.shape != first.shape:
        raise ValueError(f"operand {i} has sizes {x.sizes}, not {first.sizes}")
    for dim, values in x.labels.items():
        if labels.setdefault(dim, values) != values:
            raise ValueError(
                f"operand {i} has labels {values} on {dim!r}, not {labels[dim]}"
            )
    return x


def _check_plain(x: Any, first: CField, i: int) -> None:
    """Raise unless the plain operand only broadcasts over ``first``."""
    shape = tuple(getattr(x, "shape", ()))
    if not shape:
        return
    if len(shape) > first.ndim:
        raise ValueError(f"operand {i} of shape {shape} has more axes than {first!r}")
    padded = (1,) * (first.ndim - len(shape)) + shape
    sizes = dict(zip(first.dims, padded, strict=True))
    for dim, size in sizes.items():
        if size not in (1, first.sizes[dim]):
            raise ValueError(
                f"operand {i} of shape {shape} does not broadcast over {first!r}"
            )
    grid_axes = [dim for dim in first.dims if dim in GRID_AXES]
    staggered = [axis for axis, _ in first.stagger]
    unlocated = [axis for axis in staggered if sizes[axis] != 1]
    if unlocated or (grid_axes and all(sizes[axis] != 1 for axis in grid_axes)):
        along = unlocated or grid_axes
        raise ValueError(
            f"operand {i} is a plain array of shape {shape}: it varies along {along}"
            " but has no point. Wrap it with f.like(array), where f is a field on"
            " the point it sits on."
        )


def _elementwise(name: str) -> Callable[..., CField]:
    def function(*operands: Any, **kwargs: Any) -> CField:
        field = next((x for x in operands if isinstance(x, CField)), None)
        if field is None:
            raise TypeError(f"gridax.fields.{name} needs at least one CField")
        xp = array_namespace(*(getattr(x, "data", x) for x in operands))
        fn = getattr(xp, name)
        return combine(lambda *arrays: fn(*arrays, **kwargs), *operands)

    function.__name__ = function.__qualname__ = name
    function.__module__ = "gridax.fields"
    function.__doc__ = (
        f"``{name}`` of the array namespace, element by element (`gridax.combine`)."
    )
    return function


#: The element-wise functions of `gridax.fields`, by their array-API names.
ELEMENTWISE: tuple[str, ...] = (
    "abs",
    "clip",
    "cos",
    "exp",
    "log",
    "maximum",
    "minimum",
    "sign",
    "sin",
    "sqrt",
    "tanh",
    "where",
)

FUNCTIONS: dict[str, Callable[..., CField]] = {
    name: _elementwise(name) for name in ELEMENTWISE
}
