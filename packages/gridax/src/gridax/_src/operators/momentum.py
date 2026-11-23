"""The vortex force: `pv_flux` and `advect_momentum`.

The vector-invariant form of momentum advection:

    du/dt|adv[j, i+1/2] = +(ζ v)[j, i+1/2] - ∂K/∂x[j, i+1/2]
    dv/dt|adv[j+1/2, i] = -(ζ u)[j+1/2, i] - ∂K/∂y[j+1/2, i]

with ``ζ = ∂v/∂x - ∂u/∂y`` at X and ``K = ½(ū² + v̄²)`` at T. The products of
the vorticity and the velocity have two discrete forms, after Sadourny (1975),
"The dynamics of finite-difference models of the shallow-water equations", J.
Atmos. Sci., 32, 680–689: multiplied at the corners and averaged to the faces
(``"energy"``), or averaged to the faces and multiplied there
(``"enstrophy"``). For non-divergent flow each conserves what it is named after.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from jaxtyping import Float
import numpy as np

from gridax._src.api import grid_dispatch
from gridax._src.backend import set_at
from gridax._src.fields.located import located
from gridax._src.grid import (
    CartesianCGrid2D,
    CartesianCGrid3D,
    SphericalCGrid2D,
    SphericalCGrid3D,
)
from gridax._src.operators._ghost import interior
from gridax._src.operators.calculus import curl
from gridax._src.operators.moves import difference, interpolate
from gridax._src.pytree import array_namespace

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["advect_momentum", "pv_flux"]

Form = Literal["energy", "enstrophy", "blend"]

#: The weight of the energy form in the blend.
BLEND = 1.0 / 3.0


def _check_form(form: str) -> None:
    if form not in ("energy", "enstrophy", "blend"):
        raise ValueError(f"form must be 'energy', 'enstrophy' or 'blend', got {form!r}")


@grid_dispatch
@located(("U", "V"), u="U", v="V")
def advect_momentum(
    u: Float[Array, "..."],
    v: Float[Array, "..."],
    grid: Any,
    *,
    form: Form = "energy",
    mask: Any = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """The advection of momentum in vector-invariant form, at U and V.

    ``du = (ζ v) - δₓK`` and ``dv = -(ζ u) - δᵧK``, with the vorticity ``ζ`` at
    X (:func:`curl`), ``K = ½(ū² + v̄²)`` at T and the products of ``form``. A
    3-D grid is a stack of levels.

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity, at U and V.
    grid : grid
        A 2-D or 3-D C-grid.
    form : {"energy", "enstrophy", "blend"}
        Multiply the vorticity and the velocity at the corners, or on the
        faces; ``"blend"`` is a third of the first and two thirds of the
        second, and conserves neither.
    mask : Mask2D or Mask3D, optional
        The wet cells. The tendencies are zero on dry faces. On a 2-D grid the
        intermediate fields are too.

    Returns
    -------
    tuple
        ``(du, dv)``, at U and V, zero on the two outer rings.

    Raises
    ------
    ValueError
        If ``form`` is unknown.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(6, 6, 6.0, 6.0)
    >>> u = np.broadcast_to(grid.coords("x", at="U"), grid.shape)
    >>> du, dv = gridax.advect_momentum(u, 0.0 * u, grid)
    >>> (-du[3, 2:-2]).tolist()  # u ∂u/∂x = x
    [2.5, 3.5, 4.5, 5.5]
    """
    _check_form(form)
    return advect_momentum._impl(grid, u, v, form=form, mask=mask)


@grid_dispatch
@located(("U", "V"), q="X", u="U", v="V")
def pv_flux(
    q: Float[Array, "..."],
    u: Float[Array, "..."],
    v: Float[Array, "..."],
    grid: Any,
    *,
    form: Form = "energy",
    alpha: float = BLEND,
    mask: Any = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """The flux of a vorticity by a velocity, ``(q u, q v)``, at U and V.

    ``form="energy"`` multiplies at the corners and averages to the faces::

        qu[j, i+1/2] = 1/2 (q ū[j+1/2, i+1/2] + q ū[j-1/2, i+1/2])

    ``"enstrophy"`` averages and then multiplies, ``qu = q̄ u``, and ``"blend"``
    is ``alpha`` of the first and ``1 - alpha`` of the second. The names are for
    the vortex force, where the transports are the cross components averaged to
    each face (:func:`advect_momentum`). The blend is not the scheme of Arakawa
    and Lamb (1981).

    Parameters
    ----------
    q : CField or Float[Array, "*batch *grid"]
        The (potential) vorticity, at X. The corners of the ghost ring on the
        south and west are read.
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity or the mass flux, at U and V.
    grid : grid
        A 2-D C-grid. The metric does not enter.
    form : {"energy", "enstrophy", "blend"}
        See above.
    alpha : float, default 1/3
        The weight of the energy form in the blend.
    mask : Mask2D, optional
        The wet cells. Dry interior corners carry no product, and the fluxes
        are zero on dry faces.

    Returns
    -------
    tuple
        ``(qu, qv)``, at U and V, with a zero ghost ring.

    Raises
    ------
    ValueError
        If ``form`` is unknown.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
    >>> q, u = 2.0 * np.ones(grid.shape), 3.0 * np.ones(grid.shape)
    >>> gridax.pv_flux(q, u, u, grid)[0][2].tolist()
    [0.0, 6.0, 6.0, 6.0, 6.0, 0.0]
    """
    _check_form(form)
    return pv_flux._impl(grid, q, u, v, form=form, alpha=alpha, mask=mask)


# ----------------------------------------------------------------------

_C = (..., slice(1, -1), slice(1, -1))
_CC = (..., slice(2, -2), slice(2, -2))


@pv_flux.register(CartesianCGrid2D)
@pv_flux.register(SphericalCGrid2D)
def _pv_flux(
    grid: Any, q: Any, u: Any, v: Any, *, form: str, alpha: float, mask: Any
) -> tuple[Any, Any]:
    if form != "enstrophy":
        energy = _corner_product(q, u, v, mask)
        if form == "energy":
            return energy
    q_on_u = interpolate(q, grid, src="X", dst="U", mask=mask)
    q_on_v = interpolate(q, grid, src="X", dst="V", mask=mask)
    # qu[j, i+1/2] = q̄ʸ u,  qv[j+1/2, i] = q̄ˣ v
    enstrophy = (interior(q_on_u[_C] * u[_C], u), interior(q_on_v[_C] * v[_C], v))
    if form == "enstrophy":
        return enstrophy
    qu, qv = (
        alpha * e + (1.0 - alpha) * s for e, s in zip(energy, enstrophy, strict=True)
    )
    return qu, qv


def _corner_product(q: Any, u: Any, v: Any, mask: Any) -> tuple[Any, Any]:
    """The energy form: multiply at X, then average to the faces.

    The products are formed on the full arrays, so the first interior row and
    column average with the south and west ghost corners of the caller.
    """
    xp = array_namespace(q, u, v)
    # pu[j+1/2, i+1/2] = q 1/2 (u[j, i+1/2] + u[j+1, i+1/2]),   [Ny-1, Nx]
    pu = q[..., :-1, :] * (0.5 * (u[..., :-1, :] + u[..., 1:, :]))
    # pv[j+1/2, i+1/2] = q 1/2 (v[j+1/2, i] + v[j+1/2, i+1]),   [Ny, Nx-1]
    pv = q[..., :, :-1] * (0.5 * (v[..., :, :-1] + v[..., :, 1:]))
    if mask is not None:
        wet = xp.asarray(mask.wet(at="X", kind="strict"), dtype=bool)
        # the ghost corners are the caller's, dry or not
        keep = set_at(xp.ones_like(wet), _C, wet[_C])
        pu = xp.where(keep[..., :-1, :], pu, 0.0)
        pv = xp.where(keep[..., :, :-1], pv, 0.0)
    # qu[j, i+1/2] = 1/2 (pu[j+1/2, i+1/2] + pu[j-1/2, i+1/2])
    qu = interior(0.5 * (pu[..., 1:, 1:-1] + pu[..., :-1, 1:-1]), u)
    # qv[j+1/2, i] = 1/2 (pv[j+1/2, i+1/2] + pv[j+1/2, i-1/2])
    qv = interior(0.5 * (pv[..., 1:-1, 1:] + pv[..., 1:-1, :-1]), v)
    if mask is None:
        return qu, qv
    return qu * mask.wet(at="U"), qv * mask.wet(at="V")


@advect_momentum.register(CartesianCGrid2D)
@advect_momentum.register(SphericalCGrid2D)
@advect_momentum.register(CartesianCGrid3D)
@advect_momentum.register(SphericalCGrid3D)
def _advect_momentum(
    grid: Any, u: Any, v: Any, *, form: str, mask: Any
) -> tuple[Any, Any]:
    # on a 3-D grid the mask acts on the tendencies only
    inner = mask if len(grid.axes) == 2 else None

    def to(f: Any, src: str, dst: str) -> Any:
        return interpolate(f, grid, src=src, dst=dst, mask=inner)

    zeta = curl(u, v, grid, mask=inner)
    # K[j, i] = 1/2 (u_T² + v_T²)
    kinetic = interior(0.5 * (to(u, "U", "T")[_C] ** 2 + to(v, "V", "T")[_C] ** 2), u)
    dk_dx = difference(kinetic, grid, axis="x", src="T", dst="U", mask=inner)
    dk_dy = difference(kinetic, grid, axis="y", src="T", dst="V", mask=inner)

    def energy() -> tuple[Any, Any]:
        # the product at X, averaged to U and V
        zv = interior(zeta[_C] * to(v, "V", "X")[_C], u)
        zu = interior(zeta[_C] * to(u, "U", "X")[_C], v)
        return to(zv, "X", "U"), to(zu, "X", "V")

    def enstrophy() -> tuple[Any, Any]:
        zv = interior(to(zeta, "X", "U")[_C] * to(v, "V", "U")[_C], u)
        zu = interior(to(zeta, "X", "V")[_C] * to(u, "U", "V")[_C], v)
        return zv, zu

    if form == "energy":
        zv, zu = energy()
    elif form == "enstrophy":
        zv, zu = enstrophy()
    else:
        (zv_e, zu_e), (zv_s, zu_s) = energy(), enstrophy()
        zv = interior(BLEND * zv_e[_C] + (1.0 - BLEND) * zv_s[_C], u)
        zu = interior(BLEND * zu_e[_C] + (1.0 - BLEND) * zu_s[_C], v)
    # du = +(ζ v)_U - ∂K/∂x,  dv = -(ζ u)_V - ∂K/∂y
    du = interior(zv[_CC] - dk_dx[_CC], u, ghost=2)
    dv = interior(-zu[_CC] - dk_dy[_CC], v, ghost=2)
    if mask is None:
        return du, dv
    return du * mask.wet(at="U"), dv * mask.wet(at="V")
