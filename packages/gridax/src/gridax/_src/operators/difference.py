"""
Finite-difference operators on Arakawa C-grids.

All operators follow the interior-point idiom:
  * Output array has the same shape as the input.
  * Only interior cells [1:-1, 1:-1] are written; the ghost ring is zero.
  * The caller is responsible for boundary conditions.

Half-index notation
-------------------
Storage index [j, i] encodes:
  T[j, i]  at cell centre    (j,     i    )
  U[j, i]  at east face      (j,     i+1/2)
  V[j, i]  at north face     (j+1/2, i    )
  X[j, i]  at NE corner      (j+1/2, i+1/2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np

from gridax._src.grid import CartesianCGrid1D, CartesianCGrid2D, CartesianCGrid3D
from gridax._src.mask import Mask1D, Mask2D, Mask3D
from gridax._src.operators._ghost import interior
from gridax._src.operators.calculus import curl, divergence, grad_perp, laplacian
from gridax._src.operators.moves import difference
from gridax._src.operators.stencils import (
    _neighbours,
    diff_x_fwd_2d,
    diff_y_fwd_2d,
)
from gridax._src.pytree import pytree_dataclass

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

# ======================================================================
# Shared primitive implementations (used by both class and functional APIs)
# ======================================================================


def _curl_2d(
    u: Float[Array, "*batch Ny Nx"],
    v: Float[Array, "*batch Ny Nx"],
    dx: float | Float[Array, ""],
    dy: float | Float[Array, ""],
) -> Float[Array, "*batch Ny Nx"]:
    """Curl (relative vorticity) at X-points.  Shared implementation.

    zeta[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
                        - (u[j+1, i+1/2] - u[j, i+1/2]) / dy
    """
    out = interior(diff_x_fwd_2d(v) / dx - diff_y_fwd_2d(u) / dy, u)
    return out


def _laplacian_2d_interior(
    h: Float[Array, "*batch Ny Nx"],
    dx: float | Float[Array, ""],
    dy: float | Float[Array, ""],
) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Laplacian at the interior T-points.  Shared implementation.

    nabla2_h[j, i] = (h[j, i+1] - 2*h[j, i] + h[j, i-1]) / dx^2
                   + (h[j+1, i] - 2*h[j, i] + h[j-1, i]) / dy^2

    Each offset is sliced once (see `_Neighbours2D`); the arithmetic is
    that of ``(diff_x_fwd_2d(h) - diff_x_bwd_2d(h)) / dx**2`` plus the same
    in y.  Returns the interior only, so a caller can combine it with other
    interior terms before padding.
    """
    nb = _neighbours(h)  # c, e, w, n, s views, each [Ny-2, Nx-2]
    d2x = ((nb.e - nb.c) - (nb.c - nb.w)) / dx**2
    d2y = ((nb.n - nb.c) - (nb.c - nb.s)) / dy**2
    return d2x + d2y


@pytree_dataclass
class Difference1D:
    """Finite-difference operators on a 1-D Arakawa C-grid.

    Parameters
    ----------
    grid : CartesianCGrid1D
        The underlying 1-D grid.
    mask : Mask1D or None, optional
        Optional land/ocean mask.  When provided, every method
        post-multiplies its output by the mask field matching the
        output stagger:

        * T-output → ``mask.wet()``
        * U-output → ``mask.wet(at="U")``

        ``None`` (default) leaves every output untouched.
    """

    grid: CartesianCGrid1D
    mask: Mask1D | None = None

    def diff_x_T_to_U(self, h: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
        """Forward difference in x: T-point -> U-point.

        dh_dx[i+1/2] = (h[i+1] - h[i]) / dx

        Parameters
        ----------
        h : Float[Array, "Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Nx"]
            Forward x-difference at U-points, same shape as input.
            When ``self.mask`` is set, the output is zeroed at dry
            U-faces via ``* self.mask.wet(at="U")``.
        """
        return difference(h, self.grid, axis="x", src="T", dst="U", mask=self.mask)

    def diff_x_U_to_T(self, u: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
        """Backward difference in x: U-point -> T-point.

        du_dx[i] = (u[i+1/2] - u[i-1/2]) / dx

        Parameters
        ----------
        u : Float[Array, "Nx"]
            Velocity field at U-points.

        Returns
        -------
        Float[Array, "Nx"]
            Backward x-difference at T-points, same shape as input.
            When ``self.mask`` is set, the output is zeroed at dry
            T-cells via ``* self.mask.wet()``.
        """
        return difference(u, self.grid, axis="x", src="U", dst="T", mask=self.mask)

    def laplacian(self, h: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
        """Laplacian at T-points.

        nabla2_h[i] = (h[i+1] - 2*h[i] + h[i-1]) / dx^2

        Parameters
        ----------
        h : Float[Array, "Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Nx"]
            Laplacian at T-points, same shape as input.  When
            ``self.mask`` is set, the output is zeroed at dry T-cells
            via ``* self.mask.wet()``.
        """
        return laplacian(h, self.grid, mask=self.mask)


@pytree_dataclass
class Difference2D:
    """Finite-difference operators on a 2-D Arakawa C-grid.

    Parameters
    ----------
    grid : CartesianCGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, every method
        post-multiplies its output by the mask field matching the
        output stagger:

        * T-output → ``mask.wet()`` (divergence, laplacian, U→T, V→T)
        * U-output → ``mask.wet(at="U")`` (T→U, X→U, grad_perp's u component)
        * V-output → ``mask.wet(at="V")`` (T→V, X→V, grad_perp's v component)
        * X-output → ``mask.wet(at="X", kind="strict")`` (curl, U→X, V→X)

        The strict corner mask is used because an X-point output is
        trusted only when all four surrounding T-cells are wet.
        ``None`` (default) leaves every output untouched.
    """

    grid: CartesianCGrid2D
    mask: Mask2D | None = None

    # ------------------------------------------------------------------
    # Forward differences
    # ------------------------------------------------------------------

    def diff_x_T_to_U(
        self, h: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """Forward x-difference: T-point -> U-point.

        dh_dx[j, i+1/2] = (h[j, i+1] - h[j, i]) / dx

        Parameters
        ----------
        h : Float[Array, "*batch Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Forward x-difference at U-points.  When ``self.mask`` is
            set, the output is zeroed at dry U-faces via ``* mask.wet(at="U")``.
        """
        return difference(h, self.grid, axis="x", src="T", dst="U", mask=self.mask)

    def diff_y_T_to_V(
        self, h: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """Forward y-difference: T-point -> V-point.

        dh_dy[j+1/2, i] = (h[j+1, i] - h[j, i]) / dy

        Parameters
        ----------
        h : Float[Array, "*batch Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Forward y-difference at V-points.  When ``self.mask`` is
            set, the output is zeroed at dry V-faces via ``* mask.wet(at="V")``.
        """
        return difference(h, self.grid, axis="y", src="T", dst="V", mask=self.mask)

    def diff_y_U_to_X(
        self, u: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """Forward y-difference: U-point -> X-point (corner).

        du_dy[j+1/2, i+1/2] = (u[j+1, i+1/2] - u[j, i+1/2]) / dy

        Parameters
        ----------
        u : Float[Array, "*batch Ny Nx"]
            Velocity field at U-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Forward y-difference at X-points.  When ``self.mask`` is
            set, the output is zeroed at dry X-corners via
            ``* mask.wet(at="X", kind="strict")``.
        """
        return difference(u, self.grid, axis="y", src="U", dst="X", mask=self.mask)

    def diff_x_V_to_X(
        self, v: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """Forward x-difference: V-point -> X-point (corner).

        dv_dx[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx

        Parameters
        ----------
        v : Float[Array, "*batch Ny Nx"]
            Velocity field at V-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Forward x-difference at X-points.  When ``self.mask`` is
            set, the output is zeroed at dry X-corners via
            ``* mask.wet(at="X", kind="strict")``.
        """
        return difference(v, self.grid, axis="x", src="V", dst="X", mask=self.mask)

    # ------------------------------------------------------------------
    # Backward differences
    # ------------------------------------------------------------------

    def diff_y_X_to_U(
        self, q: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """Backward y-difference: X-point -> U-point.

        dq_dy[j, i+1/2] = (q[j+1/2, i+1/2] - q[j-1/2, i+1/2]) / dy

        Parameters
        ----------
        q : Float[Array, "*batch Ny Nx"]
            Scalar field at X-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Backward y-difference at U-points.  When ``self.mask`` is
            set, the output is zeroed at dry U-faces via ``* mask.wet(at="U")``.
        """
        return difference(q, self.grid, axis="y", src="X", dst="U", mask=self.mask)

    def diff_x_X_to_V(
        self, q: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """Backward x-difference: X-point -> V-point.

        dq_dx[j+1/2, i] = (q[j+1/2, i+1/2] - q[j+1/2, i-1/2]) / dx

        Parameters
        ----------
        q : Float[Array, "*batch Ny Nx"]
            Scalar field at X-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Backward x-difference at V-points.  When ``self.mask`` is
            set, the output is zeroed at dry V-faces via ``* mask.wet(at="V")``.
        """
        return difference(q, self.grid, axis="x", src="X", dst="V", mask=self.mask)

    def diff_x_U_to_T(
        self, u: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """Backward x-difference: U-point -> T-point.

        du_dx[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx

        Parameters
        ----------
        u : Float[Array, "*batch Ny Nx"]
            Velocity field at U-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Backward x-difference at T-points.  When ``self.mask`` is
            set, the output is zeroed at dry T-cells via ``* mask.wet()``.
        """
        return difference(u, self.grid, axis="x", src="U", dst="T", mask=self.mask)

    def diff_y_V_to_T(
        self, v: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """Backward y-difference: V-point -> T-point.

        dv_dy[j, i] = (v[j+1/2, i] - v[j-1/2, i]) / dy

        Parameters
        ----------
        v : Float[Array, "*batch Ny Nx"]
            Velocity field at V-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Backward y-difference at T-points.  When ``self.mask`` is
            set, the output is zeroed at dry T-cells via ``* mask.wet()``.
        """
        return difference(v, self.grid, axis="y", src="V", dst="T", mask=self.mask)

    # ------------------------------------------------------------------
    # Compound operators
    # ------------------------------------------------------------------

    def divergence(
        self,
        u: Float[Array, "*batch Ny Nx"],
        v: Float[Array, "*batch Ny Nx"],
    ) -> Float[Array, "*batch Ny Nx"]:
        """Divergence of (u, v) at T-points.

        delta[j, i] = du_dx[j, i] + dv_dy[j, i]
                    = (u[j, i+1/2] - u[j, i-1/2]) / dx
                    + (v[j+1/2, i] - v[j-1/2, i]) / dy

        Parameters
        ----------
        u : Float[Array, "*batch Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "*batch Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Divergence at T-points.  When ``self.mask`` is set, the
            output is zeroed at dry T-cells via ``* mask.wet()``.
        """
        return divergence(u, v, self.grid, mask=self.mask)

    def curl(
        self,
        u: Float[Array, "*batch Ny Nx"],
        v: Float[Array, "*batch Ny Nx"],
    ) -> Float[Array, "*batch Ny Nx"]:
        """Curl (relative vorticity) of (u, v) at X-points (corners).

        zeta[j+1/2, i+1/2] = dv_dx[j+1/2, i+1/2] - du_dy[j+1/2, i+1/2]
                            = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
                            - (u[j+1, i+1/2] - u[j, i+1/2]) / dy

        Parameters
        ----------
        u : Float[Array, "*batch Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "*batch Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Relative vorticity at X-points (corners).  When
            ``self.mask`` is set, the output is zeroed at dry X-corners
            via ``* mask.wet(at="X", kind="strict")``.
        """
        return curl(u, v, self.grid, mask=self.mask)

    def laplacian(
        self, h: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """Laplacian at T-points.

        nabla2_h[j, i] = (h[j, i+1] - 2*h[j, i] + h[j, i-1]) / dx^2
                       + (h[j+1, i] - 2*h[j, i] + h[j-1, i]) / dy^2

        Parameters
        ----------
        h : Float[Array, "*batch Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "*batch Ny Nx"]
            Laplacian at T-points.  When ``self.mask`` is set, the
            output is zeroed at dry T-cells via ``* mask.wet()``.
        """
        return laplacian(h, self.grid, mask=self.mask)

    def grad_perp(
        self,
        psi: Float[Array, "*batch Ny Nx"],
    ) -> tuple[Float[Array, "*batch Ny Nx"], Float[Array, "*batch Ny Nx"]]:
        """Perpendicular gradient: T-point streamfunction to geostrophic velocity.

        Maps a streamfunction ψ at T-points to face-centred geostrophic
        velocities (u, v) = (-∂ψ/∂y, ∂ψ/∂x) on an Arakawa C-grid.

        Expanding the T→X bilinear interpolation into the backward X→U / X→V
        differences gives a compact stencil that reads the T-point ghost cells
        of ψ directly:

        u[j, i+1/2] = -(ψ[j+1,i] + ψ[j+1,i+1] - ψ[j-1,i] - ψ[j-1,i+1]) / (4·dy)
        v[j+1/2, i] =  (ψ[j,i+1] + ψ[j+1,i+1] - ψ[j,i-1] - ψ[j+1,i-1]) / (4·dx)

        The resulting (unmasked) velocity field is discretely non-divergent:
        div(u, v) = 0 at T-points. When ``self.mask`` is set, ``u`` is
        multiplied by ``self.mask.wet(at="U")`` and ``v`` by ``self.mask.wet(at="V")``; the
        divergence-free property no longer holds in general in that case.

        Parameters
        ----------
        psi : Float[Array, "*batch Ny Nx"]
            Streamfunction at T-points.

        Returns
        -------
        tuple[Float[Array, "*batch Ny Nx"], Float[Array, "*batch Ny Nx"]]
            (u, v) — zonal velocity at U-points and meridional velocity at
            V-points.

        References
        ----------
        .. [1] louity/MQGeometry ``fd.py`` — ``grad_perp`` function.
           https://github.com/louity/MQGeometry/blob/main/fd.py
        .. [2] louity/qgsw-pytorch ``finite_diff.py`` — ``grad_perp`` function.
           https://github.com/louity/qgsw-pytorch/blob/main/src/finite_diff.py

        Examples
        --------
        >>> from gridax import CartesianCGrid2D, Difference2D
        >>> grid = CartesianCGrid2D.from_interior(8, 8, 1e3, 1e3)
        >>> diff = Difference2D(grid=grid)
        >>> psi = jnp.ones((grid.Ny, grid.Nx))
        >>> u, v = diff.grad_perp(psi)
        """
        return grad_perp(psi, self.grid, at="T", mask=self.mask)


@pytree_dataclass
class Difference3D:
    """Finite-difference operators on a 3-D Arakawa C-grid.

    Operates on the horizontal (y, x) plane for each z-level.
    The 3-D array shape is [Nz, Ny, Nx].

    Parameters
    ----------
    grid : CartesianCGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional land/ocean mask.  When provided, every method
        post-multiplies its output by the mask field matching the
        output stagger (``mask.wet()``, ``mask.wet(at="U")``, or ``mask.wet(at="V")``).
        ``None`` (default) leaves every output untouched.
    """

    grid: CartesianCGrid3D
    mask: Mask3D | None = None

    def diff_x_T_to_U(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Forward x-difference over all z-levels: T -> U.

        dh_dx[k, j, i+1/2] = (h[k, j, i+1] - h[k, j, i]) / dx

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Forward x-difference at U-points.  When ``self.mask`` is
            set, the output is zeroed at dry U-faces via ``* mask.wet(at="U")``.
        """
        return difference(h, self.grid, axis="x", src="T", dst="U", mask=self.mask)

    def diff_y_T_to_V(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Forward y-difference over all z-levels: T -> V.

        dh_dy[k, j+1/2, i] = (h[k, j+1, i] - h[k, j, i]) / dy

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Forward y-difference at V-points.  When ``self.mask`` is
            set, the output is zeroed at dry V-faces via ``* mask.wet(at="V")``.
        """
        return difference(h, self.grid, axis="y", src="T", dst="V", mask=self.mask)

    def diff_x_U_to_T(self, u: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Backward x-difference over all z-levels: U -> T.

        du_dx[k, j, i] = (u[k, j, i+1/2] - u[k, j, i-1/2]) / dx

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Backward x-difference at T-points.  When ``self.mask`` is
            set, the output is zeroed at dry T-cells via ``* mask.wet()``.
        """
        return difference(u, self.grid, axis="x", src="U", dst="T", mask=self.mask)

    def diff_y_V_to_T(self, v: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Backward y-difference over all z-levels: V -> T.

        dv_dy[k, j, i] = (v[k, j+1/2, i] - v[k, j-1/2, i]) / dy

        Parameters
        ----------
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Backward y-difference at T-points.  When ``self.mask`` is
            set, the output is zeroed at dry T-cells via ``* mask.wet()``.
        """
        return difference(v, self.grid, axis="y", src="V", dst="T", mask=self.mask)

    def divergence(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Horizontal divergence at T-points over all z-levels.

        delta[k, j, i] = du_dx[k, j, i] + dv_dy[k, j, i]

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Divergence at T-points.  When ``self.mask`` is set, the
            output is zeroed at dry T-cells via ``* mask.wet()`` (applied
            inside the ``diff_*`` sub-calls — both sub-methods share
            the same T-point output stagger, so the mask is already
            baked into each addend).
        """
        return divergence(u, v, self.grid, mask=self.mask)

    def laplacian(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Horizontal Laplacian at T-points over all z-levels.

        nabla2_h[k, j, i] = (h[k, j, i+1] - 2*h[k, j, i] + h[k, j, i-1]) / dx^2
                           + (h[k, j+1, i] - 2*h[k, j, i] + h[k, j-1, i]) / dy^2

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Laplacian at T-points.  When ``self.mask`` is set, the
            output is zeroed at dry T-cells via ``* mask.wet()``.
        """
        return laplacian(h, self.grid, mask=self.mask)
