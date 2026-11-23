"""The advection operators: :func:`gridax.advect` with a fixed grid and mask.

The classes cache the adaptive masks of the grid's axes, so a call with a mask
only selects tiers. The scheme of a call is one of `gridax.schemes`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Bool, Float
import numpy as np

from gridax._src.advection.advect import Wall, advect, cache, check_wall
from gridax._src.advection.reconstruct import check_scheme
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


def tendency(
    op: Any, h: Any, u: Any, v: Any, scheme: Scheme, wall: str
) -> Float[Array, "..."]:
    """:func:`gridax.advect` on the grid and the mask of ``op``, with its caches."""
    check_scheme(scheme)
    check_wall(wall, op.mask)
    return advect._impl(
        op.grid, h, u, v, scheme=scheme, mask=op.mask, wall=wall, cached=op._hierarchies
    )


@pytree_dataclass
class Advection1D:
    """Advection of a tracer on a 1-D Cartesian C-grid.

    `gridax.advect` with a fixed grid and mask. With a mask, the adaptive
    masks of the schemes up to order 9 are computed once, here.

    Parameters
    ----------
    grid : CartesianCGrid1D
        The grid.
    mask : Mask1D, optional
        The wet cells: dry values are not read, no flux crosses a dry face,
        and the tendency is zero on dry cells.
    """

    grid: CartesianCGrid1D
    mask: Mask1D | None
    _hierarchies: dict[str, dict[int, Bool[Array, "Nx"]]] | None

    def __init__(self, grid: CartesianCGrid1D, mask: Mask1D | None = None) -> None:
        self.grid = grid
        self.mask = mask
        self._hierarchies = cache(mask, grid.axes)

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
        return tendency(self, h, u, None, scheme, wall)


@pytree_dataclass
class Advection2D:
    """Advection of a tracer on a 2-D Cartesian C-grid.

    `gridax.advect` with a fixed grid and mask. With a mask, the adaptive
    masks of the schemes up to order 9 are computed once, here.

    Parameters
    ----------
    grid : CartesianCGrid2D
        The grid.
    mask : Mask2D, optional
        The wet cells: dry values are not read, no flux crosses a dry face,
        and the tendency is zero on dry cells.
    """

    grid: CartesianCGrid2D
    mask: Mask2D | None
    _hierarchies: dict[str, dict[int, Bool[Array, "Ny Nx"]]] | None

    def __init__(self, grid: CartesianCGrid2D, mask: Mask2D | None = None) -> None:
        self.grid = grid
        self.mask = mask
        self._hierarchies = cache(mask, grid.axes)

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
        return tendency(self, h, u, v, scheme, wall)


@pytree_dataclass
class Advection3D:
    """Horizontal advection of a tracer on a 3-D Cartesian C-grid, level by level.

    `gridax.advect` with a fixed grid and mask. With a mask, the adaptive
    masks of the schemes up to order 9 are computed once, here.

    Parameters
    ----------
    grid : CartesianCGrid3D
        The grid.
    mask : Mask3D, optional
        The wet cells: dry values are not read, no flux crosses a dry face,
        and the tendency is zero on dry cells.
    """

    grid: CartesianCGrid3D
    mask: Mask3D | None
    _hierarchies: dict[str, dict[int, Bool[Array, "Nz Ny Nx"]]] | None

    def __init__(self, grid: CartesianCGrid3D, mask: Mask3D | None = None) -> None:
        self.grid = grid
        self.mask = mask
        self._hierarchies = cache(mask, grid.axes)

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
        return tendency(self, h, u, v, scheme, wall)
