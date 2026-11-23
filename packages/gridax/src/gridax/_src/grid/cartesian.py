"""Cartesian Arakawa C-grid containers (1-D, 2-D, 3-D).

These are the standard uniform-spacing Cartesian grids used by the
core finite-volume operators in finitevolX.  They are concrete
subclasses of :class:`CurvilinearGrid1D` / ``2D`` / ``3D`` and add an
origin, the constructor :meth:`from_interior`, the conversion
:meth:`collocated` and (in 3-D) the :meth:`horizontal_grid` helper.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.grid.base import (
    CurvilinearGrid1D,
    CurvilinearGrid2D,
    CurvilinearGrid3D,
    _as_leaf,
)
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


def _origin() -> Any:
    """The coordinate of the first interior point, ``None`` for the default."""
    return dataclasses.field(default=None, kw_only=True)


def _collocated(grid: Any, cls: Any) -> Any:
    """The interior T points of a uniform C-grid."""
    fields: dict[str, Any] = {}
    ghost = grid.ghost
    for axis in grid.axes:
        coords = grid.coords(axis)
        fields[f"N{axis}"] = grid._size(axis) - 2 * ghost
        fields[f"_{axis}"] = coords[ghost:-ghost]
        fields[f"_d{axis}"] = _as_leaf(grid._step(axis))
    return cls(**fields, periodic=grid.periodic)


@pytree_dataclass
class CartesianCGrid1D(CurvilinearGrid1D):
    """1-D Cartesian Arakawa C-grid (uniform spacing).

    All field arrays have total shape ``[Nx]``.  The physical interior
    is ``Nx-2`` cells (slice ``[1:-1]``); one ghost cell on each end is
    reserved for boundary conditions.

    Notes
    -----
    Colocation convention (same-index rule)::

        T[i]  cell centre  at   i        * dx
        U[i]  east face    at  (i + 1/2) * dx

    The 1-D grid has only T- and U-points (no V/X corners).  Array
    index ``[i]`` encodes the **west** edge of the stencil
    neighbourhood, so ``U[i]`` is the face immediately east of ``T[i]``.
    """

    x0: Float[Array, ""] | None = _origin()

    @classmethod
    def from_interior(cls, nx_interior: int, Lx: float) -> CartesianCGrid1D:
        """Construct grid from interior cell count.

        Parameters
        ----------
        nx_interior : int
            Number of interior (physical) cells.
        Lx : float
            Physical domain length.

        Returns
        -------
        CartesianCGrid1D
        """
        return cls(Nx=nx_interior + 2, Lx=_as_leaf(Lx))

    def collocated(self) -> CartesianAGrid1D:
        """The collocated grid of the interior T points.

        ``agrid.staggered().collocated()`` is ``agrid``.

        Returns
        -------
        CartesianAGrid1D
        """
        from gridax._src.grid.collocated import CartesianAGrid1D

        return _collocated(self, CartesianAGrid1D)


@pytree_dataclass
class CartesianCGrid2D(CurvilinearGrid2D):
    """2-D Cartesian Arakawa C-grid (uniform spacing).

    All field arrays have total shape ``[Ny, Nx]``.  The physical
    interior is ``(Ny-2) x (Nx-2)`` (slice ``[1:-1, 1:-1]``); one
    ghost-cell ring on each side is reserved for boundary conditions.

    Notes
    -----
    Colocation convention (same-index rule)::

        T[j, i]  cell centre  at  ( i        * dx,  j        * dy )
        U[j, i]  east face    at  ((i + 1/2) * dx,  j        * dy )
        V[j, i]  north face   at  ( i        * dx, (j + 1/2) * dy )
        X[j, i]  NE corner    at  ((i + 1/2) * dx, (j + 1/2) * dy )

    The "same-index" rule means array index ``[j, i]`` encodes the
    **south-west** corner of the stencil neighbourhood::

           X[j,i] ---V[j,i]--- X[j,i+1]
             |           |           |
           U[j,i]     T[j,i]    U[j,i+1]
             |           |           |
          X[j-1,i]--V[j-1,i]--X[j-1,i+1]
    """

    x0: Float[Array, ""] | None = _origin()
    y0: Float[Array, ""] | None = _origin()

    @classmethod
    def from_interior(
        cls, nx_interior: int, ny_interior: int, Lx: float, Ly: float
    ) -> CartesianCGrid2D:
        """Construct grid from interior cell counts.

        Parameters
        ----------
        nx_interior : int
            Number of interior (physical) cells in x.
        ny_interior : int
            Number of interior (physical) cells in y.
        Lx : float
            Physical domain length in x.
        Ly : float
            Physical domain length in y.

        Returns
        -------
        CartesianCGrid2D
        """
        return cls(
            Nx=nx_interior + 2, Ny=ny_interior + 2, Lx=_as_leaf(Lx), Ly=_as_leaf(Ly)
        )

    def collocated(self) -> CartesianAGrid2D:
        """The collocated grid of the interior T points.

        ``agrid.staggered().collocated()`` is ``agrid``.

        Returns
        -------
        CartesianAGrid2D
        """
        from gridax._src.grid.collocated import CartesianAGrid2D

        return _collocated(self, CartesianAGrid2D)


@pytree_dataclass
class CartesianCGrid3D(CurvilinearGrid3D):
    """3-D Cartesian Arakawa C-grid (uniform spacing).

    All field arrays have total shape ``[Nz, Ny, Nx]``.  The physical
    interior is ``(Nz-2) x (Ny-2) x (Nx-2)`` (slice
    ``[1:-1, 1:-1, 1:-1]``); one ghost-cell ring on each side of every
    axis is reserved for boundary conditions.

    The horizontal staggering (T, U, V, X) follows the 2-D convention
    applied independently at each z-level — i.e. ``Difference3D`` and
    related operators act on the ``(y, x)`` plane for fixed ``k``.  The
    vertical axis is **not** staggered within this same-shape array
    convention.  W-point fields (vertical velocity at z-interfaces) use the same
    ``[Nz, Ny, Nx]`` shape, ``w[k]`` on the top face of ``T[k]`` (as
    ``Mask3D.wet(at="W")`` and ``vertical_velocity``).

    Notes
    -----
    Colocation convention (same-index rule, horizontal staggering only)::

        T[k, j, i]  cell centre  at  ( i        * dx,  j        * dy,  k * dz )
        U[k, j, i]  east face    at  ((i + 1/2) * dx,  j        * dy,  k * dz )
        V[k, j, i]  north face   at  ( i        * dx, (j + 1/2) * dy,  k * dz )
        X[k, j, i]  NE corner    at  ((i + 1/2) * dx, (j + 1/2) * dy,  k * dz )

    All four point types share the same z-coordinate ``k * dz``;
    W-points at ``(k + 1/2) * dz`` (the top face of ``T[k]``) are stored
    in the same ``[Nz, Ny, Nx]`` shape.
    """

    x0: Float[Array, ""] | None = _origin()
    y0: Float[Array, ""] | None = _origin()
    z0: Float[Array, ""] | None = _origin()

    @classmethod
    def from_interior(
        cls,
        nx_interior: int,
        ny_interior: int,
        nz_interior: int,
        Lx: float,
        Ly: float,
        Lz: float,
    ) -> CartesianCGrid3D:
        """Construct grid from interior cell counts.

        Parameters
        ----------
        nx_interior : int
            Number of interior cells in x.
        ny_interior : int
            Number of interior cells in y.
        nz_interior : int
            Number of interior cells in z.
        Lx : float
            Physical domain length in x.
        Ly : float
            Physical domain length in y.
        Lz : float
            Physical domain length in z.

        Returns
        -------
        CartesianCGrid3D
        """
        return cls(
            Nx=nx_interior + 2,
            Ny=ny_interior + 2,
            Nz=nz_interior + 2,
            Lx=_as_leaf(Lx),
            Ly=_as_leaf(Ly),
            Lz=_as_leaf(Lz),
        )

    def collocated(self) -> CartesianAGrid3D:
        """The collocated grid of the interior T points.

        ``agrid.staggered().collocated()`` is ``agrid``.

        Returns
        -------
        CartesianAGrid3D
        """
        from gridax._src.grid.collocated import CartesianAGrid3D

        return _collocated(self, CartesianAGrid3D)

    def horizontal_grid(self) -> CartesianCGrid2D:
        """Extract the horizontal 2-D grid from this 3-D grid.

        Returns
        -------
        CartesianCGrid2D
            A 2-D grid with the same Nx, Ny, Lx, Ly, origin and ghost ring.
        """
        return CartesianCGrid2D(
            Nx=self.Nx,
            Ny=self.Ny,
            Lx=self.Lx,
            Ly=self.Ly,
            x0=self.x0,
            y0=self.y0,
            ghost=self.ghost,
            periodic=tuple(axis for axis in self.periodic if axis != "z"),
        )
