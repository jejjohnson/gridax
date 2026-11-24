"""`CField` — an array with named dims and the point of the grid it sits on.

The third carrier of the family of ``pipekit_array.DimArray`` and geoarray's
``GeoArray``. It holds its array (no subclassing), the array is its only
pytree leaf, and its dims are named. What it adds is a *stagger*: the
position of its values along each axis where they are off the cell centre.
It never holds the grid.

A `CField` has no arithmetic operators, because ``u * h`` multiplies values
half a cell apart. Element-wise maths goes through
`gridax.combine`, whose check is that its operands sit on the same point.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import dataclasses
from typing import Any

from pipekit_array.dimarray import core_dims, vmapped_dims
from pipekit_array.functions import namespace_of
from pipekit_ops import on_import

from gridax._src.grid.points import (
    Stagger,
    as_stagger,
    point_stagger,
    stagger_point,
)


class StaggerMismatchError(ValueError):
    """Fields on different points of the grid met in one operation."""


@dataclasses.dataclass(frozen=True, eq=False, init=False, repr=False)
class CField:
    """An array with named dims, static labels and a stagger per axis.

    The array is the only pytree leaf; ``dims``, ``labels`` and ``stagger``
    are static, so new values never recompile and a compiled program equals
    the one on bare arrays. Under a bare ``jax.vmap`` the mapped dim is the
    first one: it is dropped inside and comes back as ``"batch"``. A stack of
    fields along ``"batch"`` has the tree structure of one of them, as a stack
    of arrays has.

    Parameters
    ----------
    data : array
        The values, on any backend.
    dims : iterable of str
        One unique name per axis, e.g. ``("member", "y", "x")``. The grid
        axes are called ``"x"``, ``"y"`` and ``"z"``.
    labels : mapping, optional
        Static labels per dim (``{"member": (0, 1, 2)}``), one per index.
        Only on dims the field is not staggered along.
    stagger : mapping or iterable of (axis, position), optional
        Position per axis in xgcm's vocabulary (``"left"``, ``"right"``,
        ``"outer"``, ``"inner"``); an axis left out is centred. The
        operators compute on centred and ``"right"`` axes.
    loc : {"T", "U", "V", "X", "W"}, optional
        Shorthand for ``stagger``: the point's name.

    Attributes
    ----------
    stagger : tuple of (axis, position)
        In normal form: sorted by axis, with no centred axis. Empty at T.

    Raises
    ------
    ValueError
        If the dims do not match the data's rank or repeat, a stagger axis
        is not a dim, a labelled dim is staggered or its labels do not match
        its size, or ``loc`` and ``stagger`` are both given.

    Examples
    --------
    >>> import numpy as np
    >>> from gridax import CField
    >>> u = CField(np.zeros((3, 10, 12)), dims=("member", "y", "x"), loc="U")
    >>> u.loc, u.stagger, u.sizes
    ('U', (('x', 'right'),), {'member': 3, 'y': 10, 'x': 12})
    """

    data: Any
    dims: tuple[str, ...]
    labels: Mapping[str, tuple[Any, ...]]
    stagger: Stagger

    # NumPy must not broadcast over a field element by element: an ``ndarray``
    # times a `CField` raises TypeError instead.
    __array_ufunc__ = None

    def __init__(
        self,
        data: Any,
        dims: Iterable[str],
        labels: Mapping[str, Iterable[Any]] | None = None,
        stagger: Mapping[str, str] | Iterable[tuple[str, str]] = (),
        *,
        loc: str | None = None,
    ) -> None:
        dims = tuple(dims)
        stagger = as_stagger(stagger)
        if loc is not None:
            if stagger:
                raise ValueError("give loc= or stagger=, not both")
            stagger = point_stagger(loc)
        shape = tuple(data.shape)
        if len(dims) != len(shape):
            raise ValueError(f"dims {dims} for data of shape {shape}")
        if len(set(dims)) != len(dims):
            raise ValueError(f"dims must be unique, got {dims}")
        missing = [axis for axis, _ in stagger if axis not in dims]
        if missing:
            raise ValueError(f"staggered along {missing}, not in dims {dims}")
        checked: dict[str, tuple[Any, ...]] = {}
        for dim, values in dict(labels or {}).items():
            entries = tuple(values)
            if dim not in dims:
                raise ValueError(f"labels for {dim!r}, which is not in dims {dims}")
            if dim in dict(stagger):
                raise ValueError(f"labels for {dim!r}, a staggered axis")
            if len(entries) != shape[dims.index(dim)]:
                raise ValueError(
                    f"{len(entries)} labels for dim {dim!r} of size"
                    f" {shape[dims.index(dim)]}"
                )
            checked[dim] = entries
        _set(self, data, dims, checked, stagger)

    @property
    def loc(self) -> str | None:
        """The point's name, or None for a stagger that has none."""
        return stagger_point(self.stagger)

    @property
    def shape(self) -> tuple[int, ...]:
        """The data's shape."""
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        """The data's rank."""
        return len(self.dims)

    @property
    def dtype(self) -> Any:
        """The data's dtype."""
        return self.data.dtype

    @property
    def sizes(self) -> dict[str, int]:
        """The size of each dim."""
        return dict(zip(self.dims, self.shape, strict=True))

    def like(self, data: Any) -> CField:
        """A full array asserted to sit on this field's point.

        Parameters
        ----------
        data : array
            Values of this field's shape, in its dim order.

        Returns
        -------
        CField
            ``data`` with this field's dims, labels and stagger.

        Raises
        ------
        ValueError
            If the shape differs, or if ``data`` is a field (it already has
            a point).
        """
        if isinstance(data, CField):
            raise ValueError(f"like() takes a plain array, got {data!r}")
        if tuple(data.shape) != self.shape:
            raise ValueError(f"array of shape {tuple(data.shape)} is not like {self!r}")
        return self.with_data(data)

    def with_data(self, data: Any) -> CField:
        """This field with new data: same dims, labels and stagger.

        Unlike :meth:`like` the sizes of unlabelled dims may change.
        """
        return CField(data, self.dims, self.labels, self.stagger)

    def transpose(self, dims: Iterable[str]) -> CField:
        """This field with its axes in the order ``dims``.

        Raises
        ------
        ValueError
            If ``dims`` is not a permutation of the field's dims.
        """
        dims = tuple(dims)
        if sorted(dims) != sorted(self.dims):
            raise ValueError(f"cannot order dims {self.dims} as {dims}")
        if dims == self.dims:
            return self
        order = tuple(self.dims.index(dim) for dim in dims)
        data = namespace_of(self.data).permute_dims(self.data, order)
        return CField(data, dims, self.labels, self.stagger)

    def to_xarray(self, grid: Any, name: str = "field") -> Any:
        """The field as an ``xr.Dataset`` that ``xgcm.Grid`` understands.

        Each grid axis gets two coordinates from ``grid``, the cell centres
        (``x_c``) and the right faces (``x_g``), with xgcm's ``axis`` and
        ``c_grid_axis_shift`` attributes, and the variable uses the one it
        sits on. Labels become coordinates.

        Parameters
        ----------
        grid : grid
            The grid of the field; its ``coords`` include the ghost ring.
        name : str
            The variable's name.

        Returns
        -------
        xr.Dataset

        Raises
        ------
        ImportError
            If xarray is not installed.
        ValueError
            If an axis is at another position than centred or right.

        Examples
        --------
        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianCGrid2D.from_interior(4, 2, 1.0, 1.0)
        >>> u = gridax.CField(np.zeros(grid.shape), ("y", "x"), loc="U")
        >>> u.to_xarray(grid, "u")["u"].dims
        ('y_c', 'x_g')
        """
        from gridax._src.fields.convert import to_xarray

        return to_xarray(self, grid, name)

    @classmethod
    def from_xarray(cls, array: Any) -> CField:
        """The field of an ``xr.DataArray`` whose coordinates carry xgcm's attributes.

        A dim whose coordinate has an ``axis`` attribute (``"X"``, ``"Y"`` or
        ``"Z"``) becomes that grid axis, on the right faces where
        ``c_grid_axis_shift`` is 0.5 and on the left ones where it is -0.5.
        The other dims keep their names and their coordinates become labels.

        Parameters
        ----------
        array : xr.DataArray
            With its ghost ring.

        Returns
        -------
        CField

        Raises
        ------
        ValueError
            If a shift is not 0, 0.5 or -0.5.
        """
        from gridax._src.fields.convert import from_xarray

        return from_xarray(array)

    def __repr__(self) -> str:
        where = self.loc or dict(self.stagger)
        labelled = f", labels={sorted(self.labels)}" if self.labels else ""
        shape = getattr(self.data, "shape", None)
        return f"CField[{' '.join(self.dims)}]{shape} at {where}{labelled}"


def _set(field: CField, data: Any, dims: Any, labels: Any, stagger: Any) -> CField:
    for name, value in (
        ("data", data),
        ("dims", dims),
        ("labels", labels),
        ("stagger", stagger),
    ):
        object.__setattr__(field, name, value)
    return field


def _register_with_jax(jax: Any) -> None:
    """Register `CField` as a JAX pytree."""
    key = jax.tree_util.GetAttrKey("data")

    def aux(f: CField) -> tuple[Any, ...]:
        labels = tuple((dim, f.labels[dim]) for dim in f.dims if dim in f.labels)
        return core_dims(f.dims), labels, f.stagger

    def flatten_with_keys(f: CField) -> tuple[Any, Any]:
        return ((key, f.data),), aux(f)

    def flatten(f: CField) -> tuple[Any, Any]:
        return (f.data,), aux(f)

    def unflatten(meta: Any, leaves: Any) -> CField:
        # Never validate: JAX unflattens with placeholder leaves while tracing.
        dims, labels, stagger = meta
        (data,) = leaves
        dims = vmapped_dims(dims, getattr(data, "shape", None))
        return _set(
            object.__new__(CField),
            data,
            dims,
            {dim: values for dim, values in labels if dim in dims},
            tuple((axis, position) for axis, position in stagger if axis in dims),
        )

    jax.tree_util.register_pytree_with_keys(
        CField, flatten_with_keys, unflatten, flatten
    )


# Without importing JAX: the class becomes a pytree when something imports it.
on_import("jax", _register_with_jax)
