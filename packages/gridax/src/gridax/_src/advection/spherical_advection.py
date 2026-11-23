"""
Spherical-coordinate advection operators for Arakawa C-grids.

Computes ``∂h/∂t = −∇·(h·u_vec)`` on the sphere using face-value
reconstruction (which is coordinate-independent) followed by a
spherical flux-divergence::

    ∂h/∂t = − 1/(R·cosφ) · [ ∂(h·u)/∂λ + ∂(h·v·cosφ)/∂φ ]

Only the final flux-divergence step differs from the Cartesian
:class:`~gridax.Advection2D`: the face values are the same, and the divergence
has the cos(lat) metric weights. Both classes are :func:`gridax.advect` with a
fixed grid and mask.

At the equator (``cos(lat) = 1``) the spherical operator reduces to
the Cartesian advection with ``dx = R·dlon`` and ``dy = R·dlat``.

Both classes take the Cartesian ``wall="closed" | "open"`` argument with the
same semantics.  The flux divergence telescopes against
:func:`~gridax.area_weights` (``R² cos(lat_T) dlon dlat``, the
cell area the divergence divides by), so ``Σ A · tendency`` is the net flux
through the faces bounding the written cells, and is zero to rounding when
those cancel:

* ``wall="closed"`` writes ``[2:-2, 2:-2]``, so the sum over the domain
  closes only if the velocity vanishes on the faces bounding that region
  (``i = 1, Nx-3``; ``j = 1, Ny-3``) — not on a zonally periodic grid, where
  the seam-adjacent physical columns ``i = 1, Nx-2`` get no tendency at all
  (#247).
* ``wall="open"`` writes ``[1:-1, 1:-1]``: with zero velocity on the
  boundary faces, or a zonally periodic ghost ring (``h``, ``u`` and ``v``
  wrapped by the caller), mass is conserved exactly — on a one-cell ghost
  ring the seam face is then first-order upwind, and equal on both sides.

Periodic halo.  For the tendency of the physical cells to be the
*periodic* scheme's (full-order stencils across the seam, not the boundary
fallbacks), build the grid with ``H - 1`` extra columns on each side, fill
the outer ``H`` columns of every field by wrapping, and keep the physical
columns ``[H:-H]``.  Measured minimum ``H``:

=================================================================  ===
scheme                                                             H
=================================================================  ===
``Upwind(1)``                                                      1
``Centred(2)``, ``Upwind(2)``, ``Upwind(3)``, ``WENO(3)``, TVD     2
``WENO(5)``, ``WENO(5, "z")``                                      3
``WENO(7)``                                                        4
``WENO(9)``                                                        5
=================================================================  ===

with ``wall="open"``; ``wall="closed"`` needs ``max(H, 2)``.  ``Σ A ·
tendency`` over ``[H:-H]`` is then zero to rounding, the two seam faces
carrying the same flux.  A halo *between* 1 and that ``H`` is not exactly
conservative: the two seam faces then use different boundary fallbacks
(measured ``sum(A t) / sum(A |t|) ~ 5e-7`` for ``WENO(5)`` with ``H = 2``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np

from gridax._src.advection.advect import Wall, advect
from gridax._src.advection.advection import UPWIND
from gridax._src.grid import SphericalCGrid2D, SphericalCGrid3D
from gridax._src.mask import Mask2D, Mask3D
from gridax._src.pytree import pytree_dataclass
from gridax._src.schemes.base import Scheme

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


@pytree_dataclass
class SphericalAdvection2D:
    """Advection of a tracer on a 2-D spherical C-grid.

    `gridax.advect` with a fixed grid and mask.

    Parameters
    ----------
    grid : SphericalCGrid2D
        The grid.
    mask : Mask2D, optional
        The wet cells: dry values are not read, no flux crosses a dry face,
        and the tendency is zero on dry cells.
    """

    grid: SphericalCGrid2D
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
class SphericalAdvection3D:
    """Horizontal advection of a tracer on a 3-D spherical C-grid, level by level.

    `gridax.advect` with a fixed grid and mask.

    Parameters
    ----------
    grid : SphericalCGrid3D
        The grid.
    mask : Mask3D, optional
        The wet cells: dry values are not read, no flux crosses a dry face,
        and the tendency is zero on dry cells.
    """

    grid: SphericalCGrid3D
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
