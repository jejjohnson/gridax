"""Coriolis force operator for Arakawa C-grids.

Applies the Coriolis force to staggered velocity fields (u, v) on a C-grid,
producing tendencies du_cor and dv_cor at velocity points.

For the horizontal momentum equations on a C-grid the Coriolis force is
the vortex force with the planetary vorticity f, in the energy-conserving
form of Sadourny (1975): f is taken to the X-points (corners), multiplied
there by the cross velocity averaged to the corner, and the product is
averaged back to the faces:

    du/dt|cor[j, i+1/2] = +avg_y( f_X * avg_x(v) )[j, i+1/2]
    dv/dt|cor[j+1/2, i] = -avg_x( f_X * avg_y(u) )[j+1/2, i]

with ``f_X[j+1/2, i+1/2]`` the 4-point average of f (given at T-points).
Because both tendencies are built from the same corner products,
``sum(u * du) + sum(v * dv) = 0`` exactly for any f (beta-plane included),
on a periodic domain or a closed basin.  For uniform f it reduces to the
familiar ``f`` times the mean of ``v`` at U and ``-f`` times the mean of ``u`` at V.

References
----------
.. [1] Veros ocean model, ``veros/core/momentum.py``, function
       ``tend_coriolis``.
.. [2] Sadourny (1975) "The dynamics of finite-difference models of the
       shallow-water equations", J. Atmos. Sci., 32, 680–689.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.api import grid_dispatch
from gridax._src.fields.located import located
from gridax._src.grid import CartesianCGrid2D, CartesianCGrid3D
from gridax._src.operators._ghost import interior
from gridax._src.operators.calculus import _levels
from gridax._src.pytree import array_namespace

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


@grid_dispatch
@located(("U", "V"), u="U", v="V", f="T")
def coriolis(
    u: Float[Array, "..."],
    v: Float[Array, "..."],
    grid: Any,
    *,
    f: float | Float[Array, "..."],
    mask: Any = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """The Coriolis tendencies ``(+f v, -f u)``, at U and V.

    The energy-conserving form of Sadourny (1975), with ``f`` taken to the
    corners:

        du[j, i+1/2] = +1/2 (P[j+1/2, i+1/2] + P[j-1/2, i+1/2])
        dv[j+1/2, i] = -1/2 (Q[j+1/2, i+1/2] + Q[j+1/2, i-1/2])

        P = f_X avg_x(v),   Q = f_X avg_y(u)   at X-points

    so ``∑ u du + ∑ v dv = 0`` to rounding for any ``f``. For a uniform ``f``
    it is ``f`` times the four-point mean of the other component.

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity, at U and V. The corners on the first interior row and
        column read the ghost ring.
    grid : grid
        A 2-D or 3-D Cartesian C-grid.
    f : float, CField or Float[Array, "*batch *grid"]
        The Coriolis parameter: a number, or a field at T with its ghost ring
        filled, which broadcasts over the levels of a 3-D grid.
    mask : Mask2D or Mask3D, optional
        The wet cells. Dry velocities are not read, and the tendencies are
        zero on dry faces.

    Returns
    -------
    tuple
        ``(du, dv)``, at U and V, with a zero ghost ring.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
    >>> u, v = np.zeros(grid.shape), np.ones(grid.shape)
    >>> du, dv = gridax.coriolis(u, v, grid, f=0.5)
    >>> du[2].tolist()
    [0.0, 0.5, 0.5, 0.5, 0.5, 0.0]
    """
    return coriolis._impl(grid, u, v, f=f, mask=mask)


@coriolis.register(CartesianCGrid2D)
@coriolis.register(CartesianCGrid3D)
def _coriolis_cartesian(grid: Any, u: Any, v: Any, *, f: Any, mask: Any) -> Any:
    xp = array_namespace(u, v, f)
    if mask is not None:
        # zero dry faces before any stencil reads them (NaN-filled land)
        u = xp.where(mask.wet(at="U"), u, 0.0)
        v = xp.where(mask.wet(at="V"), v, 0.0)
    like_u, like_v = u, v
    u, v = _levels(grid, u), _levels(grid, v)
    f = xp.asarray(f)
    if f.ndim >= 3 and len(grid.axes) == 3:
        f = _levels(grid, f)
    # a number, or a profile along y, on the whole plane
    f = xp.broadcast_to(f, (*f.shape[:-2], *u.shape[-2:]))
    # f_X[j+1/2, i+1/2] = 1/4 (f[j, i] + f[j+1, i] + f[j, i+1] + f[j+1, i+1])
    #   X rows / cols 0 .. N-2 (the south / west ghost corners included)
    f_x = 0.25 * (f[..., :-1, :-1] + f[..., 1:, :-1] + f[..., :-1, 1:] + f[..., 1:, 1:])
    # v_X[j+1/2, i+1/2] = 1/2 (v[j+1/2, i] + v[j+1/2, i+1])
    v_x = 0.5 * (v[..., :-1, :-1] + v[..., :-1, 1:])
    # u_X[j+1/2, i+1/2] = 1/2 (u[j, i+1/2] + u[j+1, i+1/2])
    u_x = 0.5 * (u[..., :-1, :-1] + u[..., 1:, :-1])
    # corner products P = f_X v_X, Q = f_X u_X
    p, q = f_x * v_x, f_x * u_x
    # du[j, i+1/2] = +1/2 (P[j+1/2, i+1/2] + P[j-1/2, i+1/2])
    du = interior(0.5 * (p[..., 1:, 1:] + p[..., :-1, 1:]), like_u)
    # dv[j+1/2, i] = -1/2 (Q[j+1/2, i+1/2] + Q[j+1/2, i-1/2])
    dv = interior(-0.5 * (q[..., 1:, 1:] + q[..., 1:, :-1]), like_v)
    if mask is not None:
        du = xp.where(mask.wet(at="U"), du, 0.0)
        dv = xp.where(mask.wet(at="V"), dv, 0.0)
    return du, dv
