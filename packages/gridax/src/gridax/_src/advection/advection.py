"""The advection operators: :func:`gridax.advect` with a fixed grid and mask.

The scheme of a call is one of `gridax.schemes`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np

from gridax._src.advection.advect import Wall, advect
from gridax._src.grid import CartesianCGrid1D, CartesianCGrid2D, CartesianCGrid3D
from gridax._src.mask import Mask1D, Mask2D, Mask3D
from gridax._src.pytree import pytree_dataclass
from gridax._src.schemes.base import Scheme
from gridax._src.schemes.linear import Upwind

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

#: The default scheme of the classes.
UPWIND = Upwind(1)


@pytree_dataclass
class Advection1D:
    """Advection of a tracer on a 1-D Cartesian C-grid.

    `gridax.advect` with a fixed grid and mask.

    Parameters
    ----------
    grid : CartesianCGrid1D
        The grid.
    mask : Mask1D, optional
        The wet cells: dry values are not read, no flux crosses a dry face,
        and the tendency is zero on dry cells.
    """

    grid: CartesianCGrid1D
    mask: Mask1D | None = None

    def __call__(
        self,
        h: Float[Array, "*batch Nx"],
        u: Float[Array, "*batch Nx"],
        scheme: Scheme = UPWIND,
        wall: Wall = "closed",
    ) -> Float[Array, "*batch Nx"]:
        """The tendency ``-∇·(h u)`` at T (:func:`gridax.advect`).

        Parameters
        ----------
        h : Float[Array, "*batch Nx"]
            Cell averages at T.
        u : Float[Array, "*batch Nx"]
            The velocity along x, at U.
        scheme : Scheme, default Upwind(1)
            Any of `gridax.schemes`.
        wall : {"closed", "open"}, default "closed"
            As in `gridax.advect`.

        Returns
        -------
        Float[Array, "*batch Nx"]
            The tendency at T.
        """
        return advect(h, u, None, self.grid, scheme=scheme, mask=self.mask, wall=wall)


@pytree_dataclass
class Advection2D:
    """Advection of a tracer on a 2-D Cartesian C-grid.

    `gridax.advect` with a fixed grid and mask.

    Parameters
    ----------
    grid : CartesianCGrid2D
        The grid.
    mask : Mask2D, optional
        The wet cells: dry values are not read, no flux crosses a dry face,
        and the tendency is zero on dry cells.
    """

    grid: CartesianCGrid2D
    mask: Mask2D | None = None

    def __call__(
        self,
        h: Float[Array, "*batch Ny Nx"],
        u: Float[Array, "*batch Ny Nx"],
        v: Float[Array, "*batch Ny Nx"],
        scheme: Scheme = UPWIND,
        wall: Wall = "closed",
    ) -> Float[Array, "*batch Ny Nx"]:
        """The tendency ``-∇·(h u)`` at T (:func:`gridax.advect`).

        Parameters
        ----------
        h : Float[Array, "*batch Ny Nx"]
            Cell averages at T.
        u : Float[Array, "*batch Ny Nx"]
            The velocity along x, at U.
        v : Float[Array, "*batch Ny Nx"]
            The velocity along y, at V.
        scheme : Scheme, default Upwind(1)
            Any of `gridax.schemes`.
        wall : {"closed", "open"}, default "closed"
            As in `gridax.advect`.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            The tendency at T.
        """
        return advect(h, u, v, self.grid, scheme=scheme, mask=self.mask, wall=wall)


@pytree_dataclass
class Advection3D:
    """Horizontal advection of a tracer on a 3-D Cartesian C-grid, level by level.

    `gridax.advect` with a fixed grid and mask.

    Parameters
    ----------
    grid : CartesianCGrid3D
        The grid.
    mask : Mask3D, optional
        The wet cells: dry values are not read, no flux crosses a dry face,
        and the tendency is zero on dry cells.
    """

    grid: CartesianCGrid3D
    mask: Mask3D | None = None

    def __call__(
        self,
        h: Float[Array, "*batch Nz Ny Nx"],
        u: Float[Array, "*batch Nz Ny Nx"],
        v: Float[Array, "*batch Nz Ny Nx"],
        scheme: Scheme = UPWIND,
        wall: Wall = "closed",
    ) -> Float[Array, "*batch Nz Ny Nx"]:
        """The tendency ``-∇·(h u)`` at T (:func:`gridax.advect`).

        Parameters
        ----------
        h : Float[Array, "*batch Nz Ny Nx"]
            Cell averages at T.
        u : Float[Array, "*batch Nz Ny Nx"]
            The velocity along x, at U.
        v : Float[Array, "*batch Nz Ny Nx"]
            The velocity along y, at V.
        scheme : Scheme, default Upwind(1)
            Any of `gridax.schemes`.
        wall : {"closed", "open"}, default "closed"
            As in `gridax.advect`.

        Returns
        -------
        Float[Array, "*batch Nz Ny Nx"]
            The tendency at T.
        """
        return advect(h, u, v, self.grid, scheme=scheme, mask=self.mask, wall=wall)
