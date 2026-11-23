"""Dissipative upwind vortex-force fluxes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np
from pipekit_array.functions import namespace_of

from gridax._src.operators._ghost import interior
from gridax._src.operators.stencils import avg_xbwd_yfwd_2d, avg_xfwd_ybwd_2d

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

# Linear (optimal-polynomial) one-sided reconstruction weights, divided by their
# common denominator and indexed from the most-upwind cell.  The ``*_POS``
# stencil is left-biased (used where the advecting velocity is >= 0); ``*_NEG``
# is right-biased (used where it is < 0).
_UP3_DENOM = 6.0
_UP3_POS = (-1.0, 5.0, 2.0)
_UP3_NEG = (2.0, 5.0, -1.0)
_UP5_DENOM = 60.0
_UP5_POS = (2.0, -13.0, 47.0, 27.0, -3.0)
_UP5_NEG = (-3.0, 27.0, 47.0, -13.0, 2.0)


def _upwind_face_bwd_last(
    omega: Float[Array, "..."],
    vel: Float[Array, "..."],
    order: int,
) -> Float[Array, "..."]:
    """Upwind-reconstruct ``omega`` to the interior faces along the last axis.

    Faces follow the backward-average stagger: face ``i`` lies between cells
    ``i-1`` (minus side) and ``i`` (plus side), so the centred limit matches
    :func:`avg_x_bwd_2d` / :func:`avg_y_bwd_2d`.  The upwind bias is chosen by the
    sign of ``vel`` at each face.  Returns the ``N - 2`` interior-face values
    along the last axis (all other axes untouched).  Higher orders fall back
    to a lower-order stencil on the outer face(s) where the wide stencil does
    not fit.
    """
    xp = namespace_of(omega, vel)
    v = vel[..., 1:-1]  # advecting velocity at interior faces i = 1 .. N-2

    # Order 1: pure upwind neighbour.
    f1 = xp.where(v >= 0.0, omega[..., 0:-2], omega[..., 1:-1])
    if order == 1:
        return f1

    n = omega.shape[-1]
    # 3rd-order stencil valid for faces i = 2 .. N-2 (needs cell i-2 .. i+1).
    o0, o1, o2, o3 = (
        omega[..., 0:-3],
        omega[..., 1:-2],
        omega[..., 2:-1],
        omega[..., 3:],
    )
    v3 = vel[..., 2:-1]
    f3_pos = (_UP3_POS[0] * o0 + _UP3_POS[1] * o1 + _UP3_POS[2] * o2) / _UP3_DENOM
    f3_neg = (_UP3_NEG[0] * o1 + _UP3_NEG[1] * o2 + _UP3_NEG[2] * o3) / _UP3_DENOM
    f3 = xp.where(v3 >= 0.0, f3_pos, f3_neg)
    # Outermost interior face (i = 1) falls back to 1st-order upwind.
    face3 = xp.concatenate([f1[..., 0:1], f3], axis=-1)
    if order == 3 or n < 7:
        return face3

    # 5th-order stencil valid for faces i = 3 .. N-3 (needs cell i-3 .. i+2).
    p0, p1, p2, p3, p4, p5 = (
        omega[..., 0:-5],
        omega[..., 1:-4],
        omega[..., 2:-3],
        omega[..., 3:-2],
        omega[..., 4:-1],
        omega[..., 5:],
    )
    v5 = vel[..., 3:-2]
    f5_pos = (
        _UP5_POS[0] * p0
        + _UP5_POS[1] * p1
        + _UP5_POS[2] * p2
        + _UP5_POS[3] * p3
        + _UP5_POS[4] * p4
    ) / _UP5_DENOM
    f5_neg = (
        _UP5_NEG[0] * p1
        + _UP5_NEG[1] * p2
        + _UP5_NEG[2] * p3
        + _UP5_NEG[3] * p4
        + _UP5_NEG[4] * p5
    ) / _UP5_DENOM
    f5 = xp.where(v5 >= 0.0, f5_pos, f5_neg)
    # The two leading and one trailing interior faces fall back to 3rd order.
    return xp.concatenate([face3[..., 0:2], f5, face3[..., -1:]], axis=-1)


def vorticity_flux_upwind(
    omega: Float[Array, "*batch Ny Nx"],
    U: Float[Array, "*batch Ny Nx"],
    V: Float[Array, "*batch Ny Nx"],
    order: int = 3,
) -> tuple[Float[Array, "*batch Ny Nx"], Float[Array, "*batch Ny Nx"]]:
    r"""Dissipative upwind vortex-force fluxes on a C-grid.

    The MASSH-parity alternative to the energy/enstrophy-conserving schemes
    (:func:`~gridax.pv_flux`): instead of averaging, the (absolute)
    vorticity at corners is reconstructed onto the velocity faces with an
    order-``{1, 3, 5}`` *upwind* stencil and multiplied by the cross mass
    flux.  Upwinding makes it monotone and dissipative — a robust, cheap
    baseline whose adjoint stays well-defined (the bias only kinks where the
    advecting flux changes sign).

    For the u-momentum equation the flux is the corner vorticity reconstructed
    in y (advected by the v mass flux ``V`` averaged to U-points) times that
    flux; for the v-momentum equation it is reconstructed in x (advected by
    ``U`` averaged to V-points).  The momentum tendencies are then
    ``dt_u += omega_V`` and ``dt_v -= omega_U`` (caller's responsibility).
    Mirrors ``MASSH sw.py::_omega_adv_upwind{3,5}``.

    Parameters
    ----------
    omega : Float[Array, "*batch Ny Nx"]
        Absolute vorticity ``zeta + f`` (or PV) at X-points (corners),
        including ghost ring.
    U : Float[Array, "*batch Ny Nx"]
        x mass flux ``h * u`` at U-points.
    V : Float[Array, "*batch Ny Nx"]
        y mass flux ``h * v`` at V-points.
    order : int, optional
        Upwind stencil width, one of ``{1, 3, 5}``.  Default ``3``.  Near the
        boundary, where the wide stencil does not fit, it falls back to a
        lower order.

    Returns
    -------
    tuple[Float[Array, "*batch Ny Nx"], Float[Array, "*batch Ny Nx"]]
        ``(omega_V, omega_U)`` — the vortex-force flux at U-points (for the
        u-equation) and V-points (for the v-equation).  Ghost ring is zero.

    Raises
    ------
    ValueError
        If ``order`` is not ``1``, ``3``, or ``5``.
    """
    xp = namespace_of(omega, U, V)
    if order not in (1, 3, 5):
        raise ValueError(f"order must be 1, 3, or 5, got {order!r}")

    # Cross mass flux advecting each velocity face.
    v_on_u = interior(avg_xfwd_ybwd_2d(V), V)  # V_to_U: north flux at U-points
    u_on_v = interior(avg_xbwd_yfwd_2d(U), U)  # U_to_V: east flux at V-points

    # omega reconstructed to U-points along y (axis -2): swap y to the last axis.
    omega_y = xp.swapaxes(omega, -2, -1)
    vel_y = xp.swapaxes(v_on_u, -2, -1)
    face_y = xp.swapaxes(_upwind_face_bwd_last(omega_y, vel_y, order), -2, -1)
    omega_V = interior(face_y[..., :, 1:-1] * v_on_u[..., 1:-1, 1:-1], omega)

    # omega reconstructed to V-points along x (axis -1, already last).
    face_x = _upwind_face_bwd_last(omega, u_on_v, order)
    omega_U = interior(face_x[..., 1:-1, :] * u_on_v[..., 1:-1, 1:-1], omega)

    return omega_V, omega_U
