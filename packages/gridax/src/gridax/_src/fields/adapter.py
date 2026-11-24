"""`CFieldAdapter`: pipekit's dim functions on located fields.

Carrier-agnostic code (``pipekit_array``'s dim functions and ``KernelOp``,
``Sequential.summary``) reaches a `CField` through this adapter, which
``import gridax`` registers.

A field stays a field while any grid axis remains, with its stagger trimmed
to the axes left: a V field averaged over ``x`` keeps ``y: "right"``. It
becomes a ``pipekit_array.DimArray``, which has no stagger, when no grid axis
is left, and when its values no longer sit on its point: ``diff`` along a grid
axis, and ``coarsen`` along a staggered one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pipekit_array import DimArrayAdapter, coarsen, diff
from pipekit_ops import Signature, register_adapter

from gridax._src.fields.cfield import CField, StaggerMismatchError
from gridax._src.fields.combine import combine
from gridax._src.fields.convert import to_dimarray
from gridax._src.grid.points import GRID_AXES

_REFERENCE = DimArrayAdapter()


def _located(x: CField, like: Any) -> Any:
    """The ``DimArray`` as a field with ``x``'s stagger, if a grid axis is left."""
    if not any(dim in GRID_AXES for dim in like.dims):
        return like
    stagger = tuple((a, p) for a, p in x.stagger if a in like.dims)
    return CField(like.data, like.dims, like.labels, stagger)


class CFieldAdapter:
    """The ``pipekit_ops.NamedArrayAdapter`` for `gridax.CField`.

    Examples
    --------
    >>> import numpy as np
    >>> import pipekit_array as pa
    >>> from gridax import CField
    >>> v = CField(np.ones((3, 6, 8)), ("time", "y", "x"), loc="V")
    >>> pa.reduce(v, "mean", "x").loc
    'V'
    >>> type(pa.reduce(v, "mean", ("y", "x"))).__name__
    'DimArray'
    """

    def dims(self, x: CField) -> tuple[str, ...]:
        """``x.dims``."""
        return x.dims

    def data(self, x: CField) -> Any:
        """The raw array."""
        return x.data

    def labels(self, x: Any, dim: str) -> tuple[Any, ...] | None:
        """The labels of ``dim``, or None (also of the ``DimArray`` `rewrap` gives)."""
        return x.labels.get(dim)

    def isel(self, x: CField, indexers: dict[str, Any]) -> Any:
        """Select by position; a sliced axis keeps its position, an int drops it."""
        return _located(x, _REFERENCE.isel(to_dimarray(x), indexers))

    def rewrap(self, x: CField, data: Any, dims: tuple[str, ...]) -> Any:
        """A field around new data while a grid axis is left, else a ``DimArray``."""
        return _located(x, _REFERENCE.rewrap(to_dimarray(x), data, dims))

    def relabel(
        self, x: CField, labels: Mapping[str, tuple[Any, ...] | None]
    ) -> CField:
        """``x`` with the labels of the named dims set; None removes them."""
        return _located(x, _REFERENCE.relabel(to_dimarray(x), labels))

    def check_compatible(self, *xs: Any) -> tuple[Any, ...]:
        """The fields back, if they share one point, sizes and labels.

        Raises
        ------
        StaggerMismatchError
            If two fields sit on different points.
        ValueError
            If a dim has two sizes or two sets of labels.
        """
        fields = [x for x in xs if isinstance(x, CField)]
        for i, x in enumerate(fields[1:], start=1):
            if x.stagger != fields[0].stagger:
                raise StaggerMismatchError(
                    f"input {i} sits at {x.loc or dict(x.stagger)}, not at"
                    f" {fields[0].loc or dict(fields[0].stagger)}"
                )
        _REFERENCE.check_compatible(*xs)
        return xs

    def signature(self, x: CField) -> Signature:
        """Dims with their sizes, and the dtype's name."""
        dims = tuple(zip(x.dims, x.shape, strict=True))
        return Signature(dims=dims, dtype=str(x.dtype))

    # -- fast paths ------------------------------------------------------

    def diff(self, x: CField, dim: str, *, n: int, label: str) -> Any:
        """``pipekit_array.diff``; a ``DimArray`` along a grid axis.

        The difference of two neighbours sits half a cell from both, with one
        entry less and no ghost ring, so it fits no grid.
        `gridax.difference` is the grid's own.
        """
        out = diff(to_dimarray(x), dim, n, label=label)
        return out if n and dim in GRID_AXES else _located(x, out)

    def coarsen(
        self,
        x: CField,
        fn: str | Callable[..., Any],
        windows: dict[str, int],
        *,
        boundary: str,
    ) -> Any:
        """``pipekit_array.coarsen``; a ``DimArray`` along a staggered axis.

        A block of cell centres is centred on the block, so a centred axis
        stays one. A block of faces is centred on no face of the coarse grid.
        """
        out = coarsen(to_dimarray(x), fn, boundary=boundary, **windows)
        staggered = any(axis in windows for axis, _ in x.stagger)
        return out if staggered else _located(x, out)

    def combine(self, fn: Callable[..., Any], *xs: Any) -> CField:
        """`gridax.combine`."""
        return combine(fn, *xs)


register_adapter(CField, CFieldAdapter())
