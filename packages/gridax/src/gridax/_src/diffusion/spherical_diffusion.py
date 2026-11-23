"""The spherical diffusion operators: :func:`gridax.diffuse` and
:func:`gridax.biharmonic` with a fixed grid and mask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np

from gridax._src.diffusion.diffuse import biharmonic, diffuse, face_fluxes
from gridax._src.grid import SphericalCGrid2D, SphericalCGrid3D
from gridax._src.mask import Mask2D, Mask3D
from gridax._src.pytree import pytree_dataclass

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


@pytree_dataclass
class SphericalDiffusion2D:
    """Harmonic diffusion on a 2-D spherical C-grid, in flux form.

    `gridax.diffuse` with a fixed grid and mask.

    Parameters
    ----------
    grid : SphericalCGrid2D
        The grid.
    mask : Mask2D, optional
        The wet cells: no flux crosses a dry face, and the tendency is zero on
        dry cells.
    """

    grid: SphericalCGrid2D
    mask: Mask2D | None = None

    def __call__(
        self,
        h: Float[Array, "*batch Ny Nx"],
        kappa: float | Float[Array, "..."],
    ) -> Float[Array, "*batch Ny Nx"]:
        """The tendency ``∇·(κ∇h)`` at T (:func:`gridax.diffuse`).

        Parameters
        ----------
        h : Float[Array, "*batch Ny Nx"]
            The tracer, at T.
        kappa : float or Float[Array, "..."]
            The diffusivity: a number, or a field at T.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            The tendency, at T.
        """
        return diffuse(h, self.grid, kappa=kappa, mask=self.mask, wall="closed")

    def fluxes(
        self,
        h: Float[Array, "*batch Ny Nx"],
        kappa: float | Float[Array, "..."],
    ) -> tuple[Float[Array, "*batch Ny Nx"], Float[Array, "*batch Ny Nx"]]:
        """The face fluxes ``κ δₓh`` at U and ``κ δᵧh`` at V, before the divergence.

        Parameters
        ----------
        h : Float[Array, "*batch Ny Nx"]
            The tracer, at T.
        kappa : float or Float[Array, "..."]
            The diffusivity: a number, or a field at T.

        Returns
        -------
        tuple[Float[Array, "*batch Ny Nx"], Float[Array, "*batch Ny Nx"]]
            ``(flux_x, flux_y)``, at U and at V, zero on dry faces.
        """
        return face_fluxes(h, self.grid, kappa=kappa, mask=self.mask, wall="closed")


@pytree_dataclass
class SphericalDiffusion3D:
    """Horizontal harmonic diffusion on a 3-D spherical C-grid, level by level.

    `gridax.diffuse` with a fixed grid and mask.

    Parameters
    ----------
    grid : SphericalCGrid3D
        The grid.
    mask : Mask3D, optional
        The wet cells: no flux crosses a dry face, and the tendency is zero on
        dry cells.
    """

    grid: SphericalCGrid3D
    mask: Mask3D | None = None

    def __call__(
        self,
        h: Float[Array, "*batch Nz Ny Nx"],
        kappa: float | Float[Array, "..."],
    ) -> Float[Array, "*batch Nz Ny Nx"]:
        """The tendency ``∇·(κ∇h)`` at T (:func:`gridax.diffuse`).

        Parameters
        ----------
        h : Float[Array, "*batch Nz Ny Nx"]
            The tracer, at T.
        kappa : float or Float[Array, "..."]
            The diffusivity: a number, or a field at T.

        Returns
        -------
        Float[Array, "*batch Nz Ny Nx"]
            The tendency, at T.
        """
        return diffuse(h, self.grid, kappa=kappa, mask=self.mask, wall="closed")

    def fluxes(
        self,
        h: Float[Array, "*batch Nz Ny Nx"],
        kappa: float | Float[Array, "..."],
    ) -> tuple[Float[Array, "*batch Nz Ny Nx"], Float[Array, "*batch Nz Ny Nx"]]:
        """The face fluxes ``κ δₓh`` at U and ``κ δᵧh`` at V, before the divergence.

        Parameters
        ----------
        h : Float[Array, "*batch Nz Ny Nx"]
            The tracer, at T.
        kappa : float or Float[Array, "..."]
            The diffusivity: a number, or a field at T.

        Returns
        -------
        tuple[Float[Array, "*batch Nz Ny Nx"], Float[Array, "*batch Nz Ny Nx"]]
            ``(flux_x, flux_y)``, at U and at V, zero on dry faces.
        """
        return face_fluxes(h, self.grid, kappa=kappa, mask=self.mask, wall="closed")


@pytree_dataclass
class SphericalBiharmonicDiffusion2D:
    """Biharmonic diffusion on a 2-D spherical C-grid.

    `gridax.biharmonic` with a fixed grid and mask.

    Parameters
    ----------
    grid : SphericalCGrid2D
        The grid.
    mask : Mask2D, optional
        The wet cells: the coast is no-flux for ``h`` and for ``∇²h``, and dry
        values are not read.
    """

    grid: SphericalCGrid2D
    mask: Mask2D | None = None

    def __call__(
        self, h: Float[Array, "*batch Ny Nx"], kappa: float | Float[Array, "..."]
    ) -> Float[Array, "*batch Ny Nx"]:
        """The tendency ``-κ ∇⁴h`` at T (:func:`gridax.biharmonic`).

        Parameters
        ----------
        h : Float[Array, "*batch Ny Nx"]
            The tracer, at T.
        kappa : float or Float[Array, "..."]
            The biharmonic diffusivity.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            The tendency, at T.
        """
        return biharmonic(h, self.grid, kappa4=kappa, mask=self.mask)


@pytree_dataclass
class SphericalBiharmonicDiffusion3D:
    """Horizontal biharmonic diffusion on a 3-D spherical C-grid, level by level.

    `gridax.biharmonic` with a fixed grid and mask.

    Parameters
    ----------
    grid : SphericalCGrid3D
        The grid.
    mask : Mask3D, optional
        The wet cells: the coast is no-flux for ``h`` and for ``∇²h``, and dry
        values are not read.
    """

    grid: SphericalCGrid3D
    mask: Mask3D | None = None

    def __call__(
        self, h: Float[Array, "*batch Nz Ny Nx"], kappa: float | Float[Array, "..."]
    ) -> Float[Array, "*batch Nz Ny Nx"]:
        """The tendency ``-κ ∇⁴h`` at T (:func:`gridax.biharmonic`).

        Parameters
        ----------
        h : Float[Array, "*batch Nz Ny Nx"]
            The tracer, at T.
        kappa : float or Float[Array, "..."]
            The biharmonic diffusivity.

        Returns
        -------
        Float[Array, "*batch Nz Ny Nx"]
            The tendency, at T.
        """
        return biharmonic(h, self.grid, kappa4=kappa, mask=self.mask)
