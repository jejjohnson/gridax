"""Cartesian C-grids with irregular axes, from an irregular collocated grid.

Every axis holds the coordinates of its T points, ghost ring included. The
faces sit halfway between them, so the two-point average onto a face and the
difference across it are exact for a linear field, and the T point of a cell is
not in its middle.

Only the operations that are defined on these grids (``SUPPORT`` in
``gridax/_src/api.py``) take them: the moves, the first-order vector calculus
and the reductions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np
from pipekit_array.functions import namespace_of

from gridax._src.grid.base import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from gridax._src.grid.points import point_axes
from gridax._src.pytree import pytree_dataclass

if TYPE_CHECKING:
    from jaxtyping import Array

    from gridax._src.grid.collocated import (
        CartesianAGrid1D,
        CartesianAGrid2D,
        CartesianAGrid3D,
    )
else:
    Array = np.ndarray

__all__ = ["StretchedCGrid1D", "StretchedCGrid2D", "StretchedCGrid3D"]


def _midpoints(x: Any) -> Any:
    """Halfway between neighbours; past the last one, half of the last gap."""
    xp = namespace_of(x)
    inner = 0.5 * (x[:-1] + x[1:])
    return xp.concat([inner, x[-1:] + 0.5 * (x[-1:] - x[-2:-1])])


class _Stretched:
    """The metric accessor of a grid with the coordinates of each axis."""

    # Fields and properties of the dataclasses this mixes into.
    axes: tuple[str, ...]
    ghost: int
    periodic: tuple[str, ...]

    if TYPE_CHECKING:

        @property
        def shape(self) -> tuple[int, ...]: ...

    def _size(self, axis: str) -> int:
        """Number of points along ``axis``, ghost ring included."""
        if axis not in self.axes:
            raise ValueError(
                f"{type(self).__name__} has the axes {self.axes}, not {axis!r}"
            )
        return self.shape[self.axes.index(axis)]

    def _line(self, axis: str, at: str, extra: int = 0) -> Float[Array, "N"]:
        """The coordinates along ``axis``, with ``extra`` more ghost points."""
        self._size(axis)
        x = getattr(self, f"_{axis}")
        if extra:
            xp = namespace_of(x)
            # mirrors of the interior points about the walls
            n, ghost = x.shape[0], self.ghost
            low = x[ghost - 1] + x[ghost]  # twice the coordinate of the wall
            high = x[n - ghost - 1] + x[n - ghost]
            first = low - x[2 * ghost : 2 * ghost + extra][::-1]
            last = high - x[n - 2 * ghost - extra : n - 2 * ghost][::-1]
            x = xp.concat([first, x, last])
        return _midpoints(x) if axis in point_axes(at, self.axes) else x

    def coords(self, axis: str, *, at: str = "T") -> Float[Array, "N"]:
        """The coordinate of the points ``at`` along ``axis``, ghost ring included.

        Parameters
        ----------
        axis : str
            One of ``grid.axes``.
        at : str
            The point: ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.

        Returns
        -------
        Float[Array, "N"]
            One coordinate per index along the axis; a face is halfway between
            its cells.
        """
        return self._line(axis, at)

    def _gaps(self, axis: str, at: str) -> Float[Array, "N"]:
        """The spacing around each point, along its axis."""
        x = self._line(axis, "T")
        xp = namespace_of(x)
        if axis in point_axes(at, self.axes):
            # Δ[i+1/2] = x[i+1] - x[i]; the last face repeats the last gap
            gaps = x[1:] - x[:-1]
            return xp.concat([gaps, gaps[-1:]])
        faces = _midpoints(x)
        # Δ[i] = x[i+1/2] - x[i-1/2]; the first cell repeats the half gap
        inner = faces[1:] - faces[:-1]
        return xp.concat([x[1:2] - x[:1], inner])

    def spacing(self, axis: str, *, at: str = "T") -> Float[Array, "..."]:
        """The grid spacing along ``axis`` at the points ``at``.

        Parameters
        ----------
        axis : str
            One of ``grid.axes``.
        at : str
            The point: ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.

        Returns
        -------
        Float[Array, "..."]
            The distance between the faces of a cell, or the cells of a face,
            shaped to broadcast against a field.
        """
        shape = [1] * len(self.axes)
        shape[self.axes.index(axis)] = self._size(axis)
        return self._gaps(axis, at).reshape(shape)

    def _wall_step(self, axis: str, side: int, at: str) -> Float[Array, ""]:
        """The distance between the two nodes next to a wall, across or on it."""
        x = self._line(axis, at)
        ghost, n = self.ghost, x.shape[0]
        if side == 0:
            return x[ghost] - x[ghost - 1]
        if axis in point_axes(at, self.axes):
            return x[n - ghost - 1] - x[n - ghost - 2]
        return x[n - ghost] - x[n - ghost - 1]

    def area(self, *, at: str = "T") -> Float[Array, "..."]:
        """The horizontal cell measure around the points ``at``.

        Parameters
        ----------
        at : str
            The point: ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.

        Returns
        -------
        Float[Array, "..."]
            One area per horizontal cell (its length in 1-D), ghost ring
            included.
        """
        return self._area(at=at)

    def _area(self, *, at: str = "T", xp: Any = None) -> Float[Array, "..."]:
        """:meth:`area` in the namespace ``xp`` (the grid's own by default)."""
        point_axes(at, self.axes)
        horizontal = [axis for axis in self.axes if axis != "z"]
        cell: Any = self._gaps(horizontal[-1], at)
        for axis in reversed(horizontal[:-1]):
            cell = self._gaps(axis, at)[:, None] * cell
        return cell if xp is None else xp.asarray(cell)

    def _to_collocated(self, cls: Any) -> Any:
        fields: dict[str, Any] = {}
        for axis in self.axes:
            fields[f"N{axis}"] = self._size(axis) - 2 * self.ghost
            fields[f"_{axis}"] = getattr(self, f"_{axis}")[self.ghost : -self.ghost]
            fields[f"_d{axis}"] = getattr(self, f"_d{axis}")
        return cls(**fields, periodic=self.periodic)


@pytree_dataclass
class StretchedCGrid1D(_Stretched, ArakawaCGrid1D):
    """1-D Cartesian C-grid with irregular axes.

    Built by ``CartesianAGrid1D.staggered()``. The faces sit halfway between
    the T points.
    """

    _x: Float[Array, "Nx"]
    _dx: Float[Array, ""] | None

    def collocated(self) -> CartesianAGrid1D:
        """The collocated grid of the interior T points.

        ``agrid.staggered().collocated()`` is ``agrid``.

        Returns
        -------
        CartesianAGrid1D
        """
        from gridax._src.grid.collocated import CartesianAGrid1D

        return self._to_collocated(CartesianAGrid1D)


@pytree_dataclass
class StretchedCGrid2D(_Stretched, ArakawaCGrid2D):
    """2-D Cartesian C-grid with irregular axes.

    Built by ``CartesianAGrid2D.staggered()``. The faces sit halfway between
    the T points.
    """

    _x: Float[Array, "Nx"]
    _dx: Float[Array, ""] | None
    _y: Float[Array, "Ny"]
    _dy: Float[Array, ""] | None

    def collocated(self) -> CartesianAGrid2D:
        """The collocated grid of the interior T points.

        ``agrid.staggered().collocated()`` is ``agrid``.

        Returns
        -------
        CartesianAGrid2D
        """
        from gridax._src.grid.collocated import CartesianAGrid2D

        return self._to_collocated(CartesianAGrid2D)


@pytree_dataclass
class StretchedCGrid3D(_Stretched, ArakawaCGrid3D):
    """3-D Cartesian C-grid with irregular axes.

    Built by ``CartesianAGrid3D.staggered()``. The faces sit halfway between
    the T points.
    """

    _x: Float[Array, "Nx"]
    _dx: Float[Array, ""] | None
    _y: Float[Array, "Ny"]
    _dy: Float[Array, ""] | None
    _z: Float[Array, "Nz"]
    _dz: Float[Array, ""] | None

    def collocated(self) -> CartesianAGrid3D:
        """The collocated grid of the interior T points.

        ``agrid.staggered().collocated()`` is ``agrid``.

        Returns
        -------
        CartesianAGrid3D
        """
        from gridax._src.grid.collocated import CartesianAGrid3D

        return self._to_collocated(CartesianAGrid3D)
