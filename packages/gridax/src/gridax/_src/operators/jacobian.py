"""
Arakawa (1966) Jacobian operator for energy- and enstrophy-conserving advection.

The Arakawa discretization of J(f, g) = ∂f/∂x·∂g/∂y − ∂f/∂y·∂g/∂x is the
standard advection operator for quasi-geostrophic models.  Unlike simple
centred-difference advection, the three-term average (J⁺⁺ + J⁺× + J×⁺)/3
conserves energy, enstrophy, and satisfies J(f,f) = 0 and ∫J(f,g)dA = 0
exactly at the discrete level.

Reference
---------
Arakawa, A. (1966). Computational design for long-term numerical integration
of the equations of fluid motion: Two-dimensional incompressible flow.
Part I. *Journal of Computational Physics*, 1(1), 119–143.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.api import grid_dispatch
from gridax._src.fields.located import located
from gridax._src.grid import CartesianCGrid2D
from gridax._src.mask import Mask2D
from gridax._src.operators._ghost import interior
from gridax._src.operators._utils import _sanitize, _spacing
from gridax._src.operators.stencils import _neighbours
from gridax._src.pytree import array_namespace, pytree_dataclass

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


@grid_dispatch
@located("T", a="T", b="T")
def arakawa_jacobian(
    a: Float[Array, "..."], b: Float[Array, "..."], grid: Any, *, mask: Any = None
) -> Float[Array, "..."]:
    """The Arakawa (1966) Jacobian ``J(a, b) = ∂ₓa ∂ᵧb - ∂ᵧa ∂ₓb``, at T.

    Parameters
    ----------
    a, b : CField or Float[Array, "*batch *grid"]
        Two fields at T, with one ghost ring. (Both may sit on X instead, as
        arrays: the stencil is collocated.)
    grid : grid
        A 2-D Cartesian C-grid.
    mask : Mask2D, optional
        The wet cells: dry interior values are read as zero, and the Jacobian
        is zero on dry cells. The discrete sums below then hold for the
        unmasked operator only.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At T, with a zero ghost ring.

    Notes
    -----
    The Arakawa scheme averages three discrete forms:

    * J⁺⁺ (standard centred form):

      .. code-block:: text

         Jpp[j,i] = ( (a[j,i+1] - a[j,i-1]) * (b[j+1,i] - b[j-1,i])
                    - (a[j+1,i] - a[j-1,i]) * (b[j,i+1] - b[j,i-1]) ) / (4 dx dy)

    * J⁺× (advective form):

      .. code-block:: text

         Jpx[j,i] = ( a[j,i+1] * (b[j+1,i+1] - b[j-1,i+1])
                    - a[j,i-1] * (b[j+1,i-1] - b[j-1,i-1])
                    - a[j+1,i] * (b[j+1,i+1] - b[j+1,i-1])
                    + a[j-1,i] * (b[j-1,i+1] - b[j-1,i-1]) ) / (4 dx dy)

    * J×⁺ (divergence form):

      .. code-block:: text

         Jxp[j,i] = ( b[j+1,i] * (a[j+1,i+1] - a[j+1,i-1])
                    - b[j-1,i] * (a[j-1,i+1] - a[j-1,i-1])
                    - b[j,i+1] * (a[j+1,i+1] - a[j-1,i+1])
                    + b[j,i-1] * (a[j+1,i-1] - a[j-1,i-1]) ) / (4 dx dy)

    Together: ``J = (Jpp + Jpx + Jxp) / 3``

    This triple average conserves energy (∫∫ a·J dA = 0), enstrophy
    (∫∫ b·J dA = 0), satisfies J(a, a) = 0, and ∫∫ J(a, b) dA = 0 at the
    discrete level.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
    >>> x = np.broadcast_to(grid.coords("x"), grid.shape)
    >>> y = np.broadcast_to(grid.coords("y")[:, None], grid.shape)
    >>> gridax.arakawa_jacobian(x, y, grid)[2].tolist()
    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    """
    return arakawa_jacobian._impl(grid, a, b, mask=mask)


@arakawa_jacobian.register(CartesianCGrid2D)
def _arakawa_jacobian_cartesian(grid: Any, a: Any, b: Any, *, mask: Any) -> Any:
    xp = array_namespace(a, b)
    wet = None if mask is None else mask.wet()
    # zero on dry interior cells; the ghost ring belongs to the caller
    a, b = _sanitize(wet, a), _sanitize(wet, b)
    values = _jacobian(a, b, _spacing(grid, "x", a), _spacing(grid, "y", a))
    shape = xp.broadcast_shapes(a.shape, b.shape)
    out = interior(values, xp.zeros(shape, dtype=values.dtype))
    return out if wet is None else xp.where(wet, out, 0.0)


def _jacobian(f: Any, g: Any, dx: Any, dy: Any) -> Any:
    """J(f, g) on the interior of the last two axes."""
    # Named 3x3 views, one static slice per offset (see `_Neighbours2D`):
    #   F.e = f[..., 1:-1, 2:]  (f[j, i+1])     F.ne = f[..., 2:, 2:]    (f[j+1, i+1])
    #   F.w = f[..., 1:-1, :-2] (f[j, i-1])     F.nw = f[..., 2:, :-2]   (f[j+1, i-1])
    #   F.n = f[..., 2:, 1:-1]  (f[j+1, i])     F.se = f[..., :-2, 2:]   (f[j-1, i+1])
    #   F.s = f[..., :-2, 1:-1] (f[j-1, i])     F.sw = f[..., :-2, :-2]  (f[j-1, i-1])
    # The three forms below are the same expressions as the original
    # slice-per-term code (bitwise-identical forward); slicing each offset
    # once makes the VJP ~3-4x faster on CPU (one pad per offset).
    F, G = _neighbours(f), _neighbours(g)  # each view [..., Ny-2, Nx-2]

    # J++ (standard centred form)
    # Jpp[j,i] = (df/dx * dg/dy - df/dy * dg/dx)
    # where df/dx ~ (f[j, i+1] - f[j, i-1]) / (2dx), etc.
    Jpp = (F.e - F.w) * (G.n - G.s) - (F.n - F.s) * (G.e - G.w)

    # J+x (advective form)
    # f evaluated at off-centre x-neighbours, g differenced in y at those neighbours
    Jpx = (
        F.e * (G.ne - G.se)
        - F.w * (G.nw - G.sw)
        - F.n * (G.ne - G.nw)
        + F.s * (G.se - G.sw)
    )

    # Jx+ (divergence form)
    # g evaluated at off-centre y-neighbours, f differenced in x at those neighbours
    Jxp = (
        G.n * (F.ne - F.nw)
        - G.s * (F.se - F.sw)
        - G.e * (F.ne - F.se)
        + G.w * (F.nw - F.sw)
    )

    return (Jpp + Jpx + Jxp) / (12.0 * dx * dy)


@pytree_dataclass
class ArakawaJacobian2D:
    """The Arakawa Jacobian on a 2-D Cartesian C-grid.

    `gridax.arakawa_jacobian` with a fixed grid and mask.

    Parameters
    ----------
    grid : CartesianCGrid2D
        The grid.
    mask : Mask2D, optional
        The wet cells.
    """

    grid: CartesianCGrid2D
    mask: Mask2D | None = None

    def __call__(
        self, f: Float[Array, "*batch Ny Nx"], g: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """``J(f, g)`` at T (:func:`gridax.arakawa_jacobian`)."""
        return arakawa_jacobian(f, g, self.grid, mask=self.mask)
