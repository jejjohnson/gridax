"""
Diagnostic quantities on Arakawa C-grids, and on collocated grids.

Includes kinetic / potential energy, strain, enstrophy, Okubo-Weiss,
and domain-integrated conservation diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geophysix.constants import STANDARD
from jaxtyping import Float
import numpy as np
from pipekit_array.functions import namespace_of

from gridax._src.api import grid_dispatch
from gridax._src.backend import set_at
from gridax._src.fields.located import located
from gridax._src.grid import (
    CartesianAGrid2D,
    CartesianAGrid3D,
    CartesianCGrid2D,
    CartesianCGrid3D,
    SphericalAGrid2D,
    SphericalCGrid2D,
    SphericalCGrid3D,
)
from gridax._src.operators._ghost import interior
from gridax._src.operators._utils import (
    _metric_like,
    _safe_div_cos,
    _sanitize,
    _spacing,
    _where,
    readable,
    without_dry,
)
from gridax._src.operators.calculus import _levels, _sphere, curl, gradient
from gridax._src.operators.moves import difference, interpolate
from gridax._src.operators.stencils import (
    _neighbours,
    avg_x_bwd_2d,
    avg_y_bwd_2d,
    diff_x_bwd_2d,
    diff_x_bwd_3d,
    diff_x_fwd_2d,
    diff_y_bwd_2d,
    diff_y_bwd_3d,
    diff_y_fwd_2d,
)

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

# ======================================================================
# The canonical functions: they move values between points, or use the metric
# ======================================================================


@grid_dispatch
@located("T", u="U", v="V")
def kinetic_energy(
    u: Float[Array, "..."], v: Float[Array, "..."], grid: Any, *, mask: Any = None
) -> Float[Array, "..."]:
    """The kinetic energy per unit mass, at T.

    ``KE = 1/2 (avg_x(u²) + avg_y(v²))`` on a C-grid, the squares of the two
    faces of each cell averaged; ``1/2 (u² + v²)`` on a collocated grid.

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity, at U and V (at T on a collocated grid).
    grid : grid
        Two- or three-dimensional. The metric does not enter.
    mask : Mask2D or Mask3D, optional
        The wet cells. Dry interior faces are read as zero, and the result is
        zero on dry cells (NaN on a collocated grid).

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At T, with a zero ghost ring on a C-grid.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
    >>> u = np.ones(grid.shape)
    >>> gridax.kinetic_energy(u, u, grid)[2].tolist()
    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    """
    return kinetic_energy._impl(grid, u, v, mask=mask)


@grid_dispatch
@located("X", u="U", v="V")
def shear_strain(
    u: Float[Array, "..."], v: Float[Array, "..."], grid: Any, *, mask: Any = None
) -> Float[Array, "..."]:
    """The shear strain ``δₓv + δᵧu``, at X.

    On a collocated sphere ``δₓv + δᵧu + u tan φ / R``, with ``δ`` along arc length.

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity, at U and V (at T on a collocated grid).
    grid : grid
        A 2-D or 3-D Cartesian C-grid, or any 2-D or 3-D collocated grid.
    mask : Mask2D or Mask3D, optional
        The wet cells. Dry interior faces are read as zero, and the result is
        zero on the corners without four wet cells. On a collocated grid as in
        :func:`divergence`.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At X (T on a collocated grid), with a zero ghost ring on a C-grid. NaN
        on a row at a pole.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
    >>> v = np.broadcast_to(grid.coords("x", at="V"), grid.shape)
    >>> gridax.shear_strain(0.0 * v, v, grid)[2].tolist()
    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    """
    return shear_strain._impl(grid, u, v, mask=mask)


@grid_dispatch
@located("T", u="U", v="V")
def tensor_strain(
    u: Float[Array, "..."], v: Float[Array, "..."], grid: Any, *, mask: Any = None
) -> Float[Array, "..."]:
    """The normal (tensor) strain ``δₓu - δᵧv``, at T.

    On a collocated sphere ``δₓu - δᵧv - v tan φ / R``, with ``δ`` along arc length.

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity, at U and V (at T on a collocated grid).
    grid : grid
        A 2-D or 3-D Cartesian C-grid, or any 2-D or 3-D collocated grid.
    mask : Mask2D or Mask3D, optional
        The wet cells. Dry interior faces are read as zero, and the result is
        zero on dry cells. On a collocated grid as in :func:`divergence`.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At T, with a zero ghost ring on a C-grid. NaN on a row at a pole.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
    >>> u = np.broadcast_to(grid.coords("x", at="U"), grid.shape)
    >>> gridax.tensor_strain(u, 0.0 * u, grid)[2].tolist()
    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    """
    return tensor_strain._impl(grid, u, v, mask=mask)


@grid_dispatch
@located("T", h="T", u="U", v="V")
def bernoulli_potential(
    h: Float[Array, "..."],
    u: Float[Array, "..."],
    v: Float[Array, "..."],
    grid: Any,
    *,
    gravity: float = STANDARD.standard_gravity,
    mask: Any = None,
) -> Float[Array, "..."]:
    """The Bernoulli potential ``K + g h``, at T.

    ``K`` is :func:`kinetic_energy`.

    Parameters
    ----------
    h : CField or Float[Array, "*batch *grid"]
        The layer thickness, at T.
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity, at U and V.
    grid : grid
        A 2-D or 3-D C-grid. The metric does not enter.
    gravity : float
        The acceleration of gravity.
    mask : Mask2D or Mask3D, optional
        The wet cells. Dry interior values are read as zero, and the result is
        zero on dry cells.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At T, with a zero ghost ring.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(3, 3, 3.0, 3.0)
    >>> one = np.ones(grid.shape)
    >>> gridax.bernoulli_potential(one, one, one, grid, gravity=10.0)[2].tolist()
    [0.0, 11.0, 11.0, 11.0, 0.0]
    """
    return bernoulli_potential._impl(grid, h, u, v, gravity=gravity, mask=mask)


@grid_dispatch
@located("T", u="U", v="V")
def strain_rate_squared(
    u: Float[Array, "..."], v: Float[Array, "..."], grid: Any, *, mask: Any = None
) -> Float[Array, "..."]:
    """The squared strain rate ``σₙ² + σₛ²``, at T.

    The normal strain :func:`tensor_strain` is at T; the shear strain at X is
    averaged from the four corners, those of the south and west ghost ring
    among them, which come from the ghost velocities. On a collocated grid both
    are at T and nothing is averaged.

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity, at U and V, with its ghost ring filled (at T on a
        collocated grid).
    grid : CartesianCGrid2D, CartesianAGrid2D or SphericalAGrid2D
        The grid.
    mask : Mask2D, optional
        The wet cells. Dry interior velocities and corners are read as zero,
        and the result is zero on dry cells. On a collocated grid as in
        :func:`divergence`.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At T, with a zero ghost ring on a C-grid. NaN on a row at a pole.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(3, 3, 3.0, 3.0)
    >>> v = np.broadcast_to(grid.coords("x", at="V"), grid.shape)  # shear
    >>> gridax.strain_rate_squared(0.0 * v, v, grid)[2].tolist()
    [0.0, 1.0, 1.0, 1.0, 0.0]
    """
    return strain_rate_squared._impl(grid, u, v, mask=mask)


@grid_dispatch
@located("T", u="U", v="V")
def okubo_weiss_parameter(
    u: Float[Array, "..."], v: Float[Array, "..."], grid: Any, *, mask: Any = None
) -> Float[Array, "..."]:
    """The Okubo–Weiss parameter ``σₙ² + σₛ² - ζ²`` of a flow, at T.

    As :func:`strain_rate_squared`, minus the squared vorticity, averaged from
    the corners in the same way. Positive where strain dominates. On a
    collocated grid ``ζ`` is :func:`curl`, and nothing is averaged.

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity, at U and V, with its ghost ring filled (at T on a
        collocated grid).
    grid : CartesianCGrid2D, CartesianAGrid2D or SphericalAGrid2D
        The grid.
    mask : Mask2D, optional
        The wet cells, as in :func:`strain_rate_squared`.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At T, with a zero ghost ring on a C-grid. NaN on a row at a pole.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(3, 3, 3.0, 3.0)
    >>> v = np.broadcast_to(grid.coords("x", at="V"), grid.shape)
    >>> gridax.okubo_weiss_parameter(0.0 * v, v, grid)[2].tolist()  # 1 - 1
    [0.0, 0.0, 0.0, 0.0, 0.0]
    """
    return okubo_weiss_parameter._impl(grid, u, v, mask=mask)


@grid_dispatch
@located("X", u="U", v="V", h="T", f="T")
def sw_potential_vorticity(
    u: Float[Array, "..."],
    v: Float[Array, "..."],
    h: Float[Array, "..."],
    f: Float[Array, "..."],
    grid: Any,
    *,
    mask: Any = None,
) -> Float[Array, "..."]:
    """The shallow-water potential vorticity ``(ζ + f) / h``, at X.

    ``ζ`` is :func:`curl`, and ``f`` and ``h`` are averaged to the corners (on a
    collocated grid they are read as they are). Layers are leading axes.

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch *grid"]
        The velocity, at U and V (at T on a collocated grid).
    h : CField or Float[Array, "*batch *grid"]
        The layer thickness, at T.
    f : CField or Float[Array, "*batch *grid"]
        The Coriolis parameter, at T; it broadcasts over leading axes.
    grid : CartesianCGrid2D, SphericalCGrid2D, CartesianAGrid2D or SphericalAGrid2D
        The grid.
    mask : Mask2D, optional
        The wet cells. The result is zero on the corners without four of them.
        On a collocated grid as in :func:`divergence`.

    Returns
    -------
    CField or Float[Array, "*batch *grid"]
        At X (T on a collocated grid), with a zero ghost ring on a C-grid. NaN
        on a (wet) point of zero thickness.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(3, 3, 3.0, 3.0)
    >>> one = np.ones(grid.shape)
    >>> gridax.sw_potential_vorticity(one, one, 2 * one, 3 * one, grid)[2].tolist()
    [0.0, 1.5, 1.5, 1.5, 0.0]
    """
    return sw_potential_vorticity._impl(grid, u, v, h, f, mask=mask)


@grid_dispatch
@located("T", psi="T")
def qg_potential_vorticity(
    psi: Float[Array, "..."],
    grid: Any,
    *,
    f0: float | Float[Array, ""],
    beta: float | Float[Array, ""] = 0.0,
    y: Float[Array, "..."] | None = None,
    y0: float | Float[Array, ""] = 0.0,
    A: Float[Array, "nl nl"] | None = None,
    mask: Any = None,
) -> Float[Array, "..."]:
    """The quasi-geostrophic potential vorticity, at T.

    ``q = ∇²ψ / f₀ + β (y - y₀) / f₀``, and with the coupling matrix ``A`` of a
    stack of layers also ``- f₀ ∑ₘ A[k, m] ψ[m]`` (:func:`stretching_term`).

    Parameters
    ----------
    psi : CField or Float[Array, "*batch Ny Nx"]
        The streamfunction, at T; ``[nl, Ny, Nx]`` with ``A``.
    grid : CartesianCGrid2D
        The grid.
    f0, beta : float
        The Coriolis parameter and its gradient.
    y : Float[Array, "..."], optional
        The meridional coordinate at T; that of the grid by default.
    y0 : float
        Where ``β`` is centred.
    A : Float[Array, "nl nl"], optional
        The coupling matrix of the layers, as ``build_coupling_matrix`` of
        sesmax-core gives it (``f₀`` is applied here).
    mask : Mask2D, optional
        The wet cells. ``psi`` is read as zero on dry interior cells, and the
        result is zero on dry cells.

    Returns
    -------
    CField or Float[Array, "*batch Ny Nx"]
        At T, with a zero ghost ring.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(3, 3, 3.0, 3.0)
    >>> psi = np.broadcast_to(grid.coords("x") ** 2, grid.shape)
    >>> gridax.qg_potential_vorticity(psi, grid, f0=0.5)[2].tolist()
    [0.0, 4.0, 4.0, 4.0, 0.0]
    """
    return qg_potential_vorticity._impl(
        grid, psi, f0=f0, beta=beta, y=y, y0=y0, A=A, mask=mask
    )


@grid_dispatch
@located(("U", "V"), h="T", f="T")
def geostrophic_velocity(
    h: Float[Array, "..."],
    grid: Any,
    *,
    f: Float[Array, "..."],
    gravity: float = STANDARD.standard_gravity,
    mask: Any = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """The geostrophic velocity of a free surface, at U and V.

    ``u = -g / (f R) ∂h/∂φ`` and ``v = g / (f R cos φ) ∂h/∂λ``, each from the
    four cells around the face; on a collocated grid ``u = -g/f ∂h/∂y`` and
    ``v = g/f ∂h/∂x`` at T, from :func:`gradient`.

    Parameters
    ----------
    h : CField or Float[Array, "Ny Nx"]
        The height of the surface, at T.
    grid : SphericalCGrid2D, CartesianAGrid2D or SphericalAGrid2D
        The grid.
    f : CField or Float[Array, "Ny Nx"]
        The Coriolis parameter, at T. It may not be zero on a wet face of a
        C-grid.
    gravity : float
        The acceleration of gravity.
    mask : Mask2D, optional
        The wet cells. The velocity is zero on dry faces, and ``f`` is not read
        there. A wet face next to a coast reads the height of the land beside
        it, which must be finite. On a collocated grid as in :func:`divergence`.

    Returns
    -------
    tuple
        ``(u, v)``, at U and V, with a zero ghost ring (both at T on a
        collocated grid, NaN where ``f`` is zero and on a row at a pole).

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.SphericalCGrid2D.from_interior(4, 3, (0, 8), (40, 46), R=1.0)
    >>> h = np.broadcast_to(grid.coords("y")[:, None], grid.shape)
    >>> u, v = gridax.geostrophic_velocity(h, grid, f=1.0 + 0.0 * h, gravity=2.0)
    >>> u[2].round(6).tolist()
    [0.0, -2.0, -2.0, -2.0, -2.0, 0.0]
    """
    return geostrophic_velocity._impl(grid, h, f=f, gravity=gravity, mask=mask)


@grid_dispatch
@located("W", u="U", v="V")
def vertical_velocity(
    u: Float[Array, "..."], v: Float[Array, "..."], grid: Any, *, mask: Any = None
) -> Float[Array, "..."]:
    """The vertical velocity of a non-divergent flow, at W.

    The horizontal divergence of the interior levels, integrated upwards from a
    rigid floor (``w = 0``)::

        w[k+1/2] = w[k-1/2] - (δₓu + δᵧv)[k] Δz,   k = 1 .. Nz-2

    Parameters
    ----------
    u, v : CField or Float[Array, "*batch Nz Ny Nx"]
        The velocity, at U and V.
    grid : CartesianCGrid3D
        The grid.
    mask : Mask3D, optional
        The wet cells. Dry velocities are not read, dry cells have no
        divergence, and ``w`` is zero on dry faces.

    Returns
    -------
    CField or Float[Array, "*batch Nz Ny Nx"]
        ``w[k]`` on the top face of ``T[k]``, with a zero ghost ring (``w[0]``
        is the floor).

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid3D.from_interior(3, 3, 3, 3.0, 3.0, 3.0)
    >>> u = np.broadcast_to(grid.coords("x", at="U"), grid.shape)
    >>> gridax.vertical_velocity(u, 0.0 * u, grid)[:, 2, 2].tolist()
    [0.0, -1.0, -2.0, -3.0, 0.0]
    """
    return vertical_velocity._impl(grid, u, v, mask=mask)


def _velocities(mask: Any, u: Any, v: Any) -> tuple[Any, Any]:
    """``u`` and ``v``, zero on dry interior faces (no normal flow)."""
    if mask is None:
        return u, v
    return _sanitize(mask.wet(at="U"), u), _sanitize(mask.wet(at="V"), v)


@kinetic_energy.register(CartesianCGrid2D)
@kinetic_energy.register(CartesianCGrid3D)
@kinetic_energy.register(SphericalCGrid2D)
@kinetic_energy.register(SphericalCGrid3D)
def _kinetic_energy_staggered(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    u, v = _velocities(mask, u, v)
    out = interior(_kinetic_energy_interior(_levels(grid, u), _levels(grid, v)), u)
    return _where(None if mask is None else mask.wet(), out)


@kinetic_energy.register(CartesianAGrid2D)
@kinetic_energy.register(CartesianAGrid3D)
@kinetic_energy.register(SphericalAGrid2D)
def _kinetic_energy_collocated(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    # 1/2 (u² + v²)
    out = 0.5 * (u**2 + v**2)
    return out if mask is None else namespace_of(out).where(mask.wet(), out, np.nan)


@shear_strain.register(CartesianCGrid2D)
@shear_strain.register(CartesianCGrid3D)
def _shear_strain_cartesian(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    u, v = _velocities(mask, u, v)
    dx, dy = _spacing(grid, "x", v, at="X"), _spacing(grid, "y", u, at="X")
    like = u
    u, v = _levels(grid, u), _levels(grid, v)
    # (v[j+1/2, i+1] - v[j+1/2, i]) / dx + (u[j+1, i+1/2] - u[j, i+1/2]) / dy
    out = interior(diff_x_fwd_2d(v) / dx + diff_y_fwd_2d(u) / dy, like)
    return _where(None if mask is None else mask.wet(at="X", kind="strict"), out)


@tensor_strain.register(CartesianCGrid2D)
@tensor_strain.register(CartesianCGrid3D)
def _tensor_strain_cartesian(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    u, v = _velocities(mask, u, v)
    dx, dy = _spacing(grid, "x", u), _spacing(grid, "y", v)
    like = u
    u, v = _levels(grid, u), _levels(grid, v)
    # (u[j, i+1/2] - u[j, i-1/2]) / dx - (v[j+1/2, i] - v[j-1/2, i]) / dy
    out = interior(diff_x_bwd_2d(u) / dx - diff_y_bwd_2d(v) / dy, like)
    return _where(None if mask is None else mask.wet(), out)


@shear_strain.register(CartesianAGrid2D)
@shear_strain.register(CartesianAGrid3D)
def _shear_strain_collocated(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    # ∂v/∂x + ∂u/∂y
    dv_dx = difference(v, grid, axis="x", mask=mask)
    return dv_dx + difference(u, grid, axis="y", mask=mask)


@shear_strain.register(SphericalAGrid2D)
def _shear_strain_collocated_sphere(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    # the same, along arc length, + u tan φ / R
    return _shear_strain_collocated(grid, u, v, mask=mask) + _turning(grid, u, mask)


@tensor_strain.register(CartesianAGrid2D)
@tensor_strain.register(CartesianAGrid3D)
def _tensor_strain_collocated(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    # ∂u/∂x - ∂v/∂y
    du_dx = difference(u, grid, axis="x", mask=mask)
    return du_dx - difference(v, grid, axis="y", mask=mask)


@tensor_strain.register(SphericalAGrid2D)
def _tensor_strain_collocated_sphere(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    # the same, along arc length, - v tan φ / R
    return _tensor_strain_collocated(grid, u, v, mask=mask) - _turning(grid, v, mask)


def _turning(grid: Any, f: Any, mask: Any) -> Any:
    """``f tan φ / R`` on a collocated sphere, NaN at a pole; dry cells are not read.

    This form of the metric terms, not ``cos φ ∂(f / cos φ)/∂φ``: that one is
    exact for a rotation about the poles and worse for every other.
    """
    if mask is not None:
        f = namespace_of(f).where(mask.wet(), f, 0.0)
    cos, R = _sphere(grid, f)
    return _safe_div_cos(f * _metric_like(grid._sin_lat(), f), cos, R)


@bernoulli_potential.register(CartesianCGrid2D)
@bernoulli_potential.register(CartesianCGrid3D)
@bernoulli_potential.register(SphericalCGrid2D)
@bernoulli_potential.register(SphericalCGrid3D)
def _bernoulli_staggered(
    grid: Any, h: Any, u: Any, v: Any, *, gravity: float, mask: Any
) -> Any:
    xp = namespace_of(h, u, v)
    wet = None if mask is None else mask.wet()
    u, v = _velocities(mask, u, v)
    h = _sanitize(wet, xp.asarray(h, dtype=xp.result_type(h, u, v, 0.0)))
    # p[j, i] = K[j, i] + g h[j, i]
    ke = _kinetic_energy_interior(_levels(grid, u), _levels(grid, v))
    out = interior(ke + gravity * _levels(grid, h)[..., 1:-1, 1:-1], h)
    return _where(wet, out)


def _corners(grid: Any, u: Any, v: Any, sign: float, mask: Any) -> Any:
    """``δₓv + sign δᵧu`` at X, on the south and west ghost corners too.

    The shear strain (+1) or the vorticity (-1). Under a mask the dry interior
    corners are zero; the ghost corners are the caller's, dry or not.
    """
    xp = namespace_of(u, v)
    dv_dx = (v[..., :-1, 1:] - v[..., :-1, :-1]) / _spacing(grid, "x", v, at="X")
    du_dy = (u[..., 1:, :-1] - u[..., :-1, :-1]) / _spacing(grid, "y", u, at="X")
    x = dv_dx + sign * du_dy
    # out[j, i] = x[j, i] for 0 <= j <= Ny-2, 0 <= i <= Nx-2
    out = set_at(xp.zeros_like(u, dtype=x.dtype), (..., slice(-1), slice(-1)), x)
    return _sanitize(None if mask is None else mask.wet(at="X", kind="strict"), out)


def _at_t(grid: Any, u: Any, v: Any, sign: float, mask: Any) -> Any:
    return interpolate(
        _corners(grid, u, v, sign, mask), grid, src="X", dst="T", mask=mask
    )


@strain_rate_squared.register(CartesianCGrid2D)
def _strain_rate_squared(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    u, v = _velocities(mask, u, v)
    sn = tensor_strain(u, v, grid)
    out = strain_magnitude_squared(sn, _at_t(grid, u, v, 1.0, mask))
    return _where(None if mask is None else mask.wet(), out)


@okubo_weiss_parameter.register(CartesianCGrid2D)
def _okubo_weiss_parameter(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    u, v = _velocities(mask, u, v)
    sn = tensor_strain(u, v, grid)
    shear, omega = (_at_t(grid, u, v, sign, mask) for sign in (1.0, -1.0))
    out = okubo_weiss(sn, shear, omega)
    return _where(None if mask is None else mask.wet(), out)


@strain_rate_squared.register(CartesianAGrid2D)
@strain_rate_squared.register(SphericalAGrid2D)
def _strain_rate_squared_collocated(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    sn = tensor_strain(u, v, grid, mask=mask)
    return strain_magnitude_squared(sn, shear_strain(u, v, grid, mask=mask))


@okubo_weiss_parameter.register(CartesianAGrid2D)
@okubo_weiss_parameter.register(SphericalAGrid2D)
def _okubo_weiss_collocated(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    sn, ss, omega = (
        op(u, v, grid, mask=mask) for op in (tensor_strain, shear_strain, curl)
    )
    return okubo_weiss(sn, ss, omega)


@sw_potential_vorticity.register(CartesianCGrid2D)
@sw_potential_vorticity.register(SphericalCGrid2D)
def _sw_potential_vorticity(
    grid: Any, u: Any, v: Any, h: Any, f: Any, *, mask: Any
) -> Any:
    xp = namespace_of(u, v, h, f)
    inner = (..., slice(1, -1), slice(1, -1))
    zeta = curl(u, v, grid, mask=mask)
    f_on_x = interpolate(f, grid, src="T", dst="X", mask=mask)
    h_on_x = interpolate(h, grid, src="T", dst="X", mask=mask)
    # q = (ζ + f) / h
    num = zeta[inner] + f_on_x[inner]
    den = h_on_x[inner]
    if mask is not None:
        # h is 0 on dry corners: divide by 1 there, so that reverse mode has no
        # 0 * inf, and zero the result below; a wet corner keeps its NaN
        den = xp.where(mask.wet(at="X", kind="strict")[inner], den, 1.0)
    out = interior(xp.where(den == 0, xp.nan, num / den), zeta)
    if mask is None:
        return out
    return xp.where(mask.wet(at="X", kind="strict"), out, 0.0)


@sw_potential_vorticity.register(CartesianAGrid2D)
@sw_potential_vorticity.register(SphericalAGrid2D)
def _sw_potential_vorticity_collocated(
    grid: Any, u: Any, v: Any, h: Any, f: Any, *, mask: Any
) -> Any:
    # q = (ζ + f) / h
    num = curl(u, v, grid, mask=mask) + _wet_or_one(f, mask)
    return _divide(num, _wet_or_one(h, mask))


@qg_potential_vorticity.register(CartesianCGrid2D)
def _qg_potential_vorticity(
    grid: Any, psi: Any, *, f0: Any, beta: Any, y: Any, y0: Any, A: Any, mask: Any
) -> Any:
    wet = None if mask is None else mask.wet()
    # a masked elliptic solve returns zero on land; the ring is the caller's
    psi = _sanitize(wet, psi)
    dx, dy = _spacing(grid, "x", psi), _spacing(grid, "y", psi)
    if y is None:
        y = _metric_like(grid.coords("y"), psi)[:, None]
        y_inner = y[1:-1]
    else:
        y_inner = y[..., 1:-1, 1:-1]
    # q = ∇²ψ / f₀ + β (y - y₀) / f₀
    out = interior(
        _laplacian_interior(psi, dx, dy) / f0 + beta * (y_inner - y0) / f0, psi
    )
    if A is not None:
        out = out - f0 * stretching_term(A, psi)
    return _where(wet, out)


def _laplacian_interior(f: Any, dx: Any, dy: Any) -> Any:
    """∇²f on the interior of the last two axes."""
    nb = _neighbours(f)
    d2x = ((nb.e - nb.c) - (nb.c - nb.w)) / dx**2
    d2y = ((nb.n - nb.c) - (nb.c - nb.s)) / dy**2
    return d2x + d2y


@geostrophic_velocity.register(SphericalCGrid2D)
def _geostrophic_velocity(
    grid: Any, h: Any, *, f: Any, gravity: float, mask: Any
) -> tuple[Any, Any]:
    xp = namespace_of(h, f)
    R, dlon, dlat = (_metric_like(m, h) for m in (grid.R, grid.dlon, grid.dlat))
    c = slice(1, -1)
    f_on_u = 0.5 * (f[..., c, 1:-1] + f[..., c, 2:])
    f_on_v = 0.5 * (f[..., 1:-1, c] + f[..., 2:, c])
    h = without_dry(h, mask, "T", 2)
    if mask is not None:
        # a dry face does not divide by zero, forwards or backwards
        f_on_u = xp.where(mask.wet(at="U")[c, c], f_on_u, 1.0)
        f_on_v = xp.where(mask.wet(at="V")[c, c], f_on_v, 1.0)
    # u[j, i+1/2] = -g / (f R) (h[j+1,i] + h[j+1,i+1] - h[j-1,i] - h[j-1,i+1]) / (4 dφ)
    dlat_h = h[..., 2:, c] + h[..., 2:, 2:] - h[..., :-2, c] - h[..., :-2, 2:]
    dlon_h = h[..., c, 2:] + h[..., 2:, 2:] - h[..., c, :-2] - h[..., 2:, :-2]
    if mask is not None:
        dlat_h, dlon_h = _wet_slopes(h, mask, dlat_h, dlon_h)
    u = interior(-gravity / (f_on_u * R) * (dlat_h / (4.0 * dlat)), h)
    # v[j+1/2, i] = g / (f R cos φ) (h[j,i+1] + h[j+1,i+1] - h[j,i-1] - h[j+1,i-1]) / (4 dλ)
    cos = _metric_like(grid._cos_lat("V"), h)[c, c]
    v = interior(_safe_div_cos(gravity * (dlon_h / (4.0 * dlon)), cos, f_on_v * R), h)
    if mask is None:
        return u, v
    return xp.where(mask.wet(at="U"), u, 0.0), xp.where(mask.wet(at="V"), v, 0.0)


@geostrophic_velocity.register(CartesianAGrid2D)
@geostrophic_velocity.register(SphericalAGrid2D)
def _geostrophic_velocity_collocated(
    grid: Any, h: Any, *, f: Any, gravity: float, mask: Any
) -> tuple[Any, Any]:
    f = _wet_or_one(f, mask)
    dh_dx, dh_dy = gradient(h, grid, mask=mask)
    # u = -g/f ∂h/∂y, v = g/f ∂h/∂x
    return _divide(-gravity * dh_dy, f), _divide(gravity * dh_dx, f)


def _wet_or_one(f: Any, mask: Any) -> Any:
    """``f``, and 1 on the dry cells of a collocated grid, which are not read."""
    return f if mask is None else namespace_of(f, mask.wet()).where(mask.wet(), f, 1.0)


def _divide(num: Any, den: Any) -> Any:
    """``num / den``, NaN where ``den`` is zero, and no ``0 * inf`` in reverse mode."""
    xp = namespace_of(num, den)
    safe = xp.where(den == 0, 1.0, den)
    return xp.where(den == 0, xp.nan, num / safe)


def _wet_slopes(h: Any, mask: Any, dlat_h: Any, dlon_h: Any) -> tuple[Any, Any]:
    """Four times the mean of the wet differences around each face.

    ``dlat_h`` and ``dlon_h``, the sums of all four, where they are all wet.
    """
    xp = namespace_of(h)
    wet = readable(mask.wet(), 2)

    def mean(axis: int, plain: Any) -> Any:
        a, b = xp.moveaxis(h, axis, -2), xp.moveaxis(wet, axis, -2)
        step = a[..., 1:, :] - a[..., :-1, :]  # [N-1, M]
        ok = b[..., 1:, :] & b[..., :-1, :]
        step = xp.where(ok, step, 0.0)
        ok = xp.asarray(ok, dtype=step.dtype)

        def around(x: Any) -> Any:
            # the two faces on either side of the point, in two columns
            pair = x[..., 1:, :] + x[..., :-1, :]
            return pair[..., 1:-1] + pair[..., 2:]

        count = around(ok)
        value = 4.0 * around(step) / xp.where(count > 0, count, 1.0)
        return xp.where(
            xp.moveaxis(count, -2, axis) == 4, plain, xp.moveaxis(value, -2, axis)
        )

    return mean(-2, dlat_h), mean(-1, dlon_h)


@vertical_velocity.register(CartesianCGrid3D)
def _vertical_velocity(grid: Any, u: Any, v: Any, *, mask: Any) -> Any:
    xp = namespace_of(u, v)
    inner = (..., slice(1, -1), slice(1, -1), slice(1, -1))
    if mask is not None:
        # zero dry faces before the stencil reads them (NaN-filled land)
        u = xp.where(mask.wet(at="U"), u, 0.0)
        v = xp.where(mask.wet(at="V"), v, 0.0)
    dx, dy, dz = (_spacing(grid, axis, u) for axis in "xyz")
    # div[k, j, i] on the interior levels, k = 1 .. Nz-2
    div = diff_x_bwd_3d(u) / dx + diff_y_bwd_3d(v) / dy
    if mask is not None:
        div = xp.where(mask.wet()[inner], div, 0.0)
    # w[k+1/2] = -sum_{k'=1..k} div[k'] dz; w[0] = 0 is the floor
    w = interior(-xp.cumsum(div, axis=-3) * dz, u)
    return w if mask is None else xp.where(mask.wet(at="W"), w, 0.0)


def _kinetic_energy(
    u: Float[Array, "*batch Ny Nx"], v: Float[Array, "*batch Ny Nx"]
) -> Float[Array, "*batch Ny Nx"]:
    """The kinetic energy at T on the last two axes, with a zero ghost ring."""
    return interior(_kinetic_energy_interior(u, v), u)


def _kinetic_energy_interior(u: Any, v: Any) -> Any:
    """``1/2 (avg_x(u²) + avg_y(v²))`` on the interior of the last two axes.

    u²_T[j, i] = 1/2 (u[j, i+1/2]² + u[j, i-1/2]²)
    v²_T[j, i] = 1/2 (v[j+1/2, i]² + v[j-1/2, i]²)
    """
    xp = namespace_of(u, v)
    dtype = xp.result_type(u, v, 0.0)
    u, v = xp.asarray(u, dtype=dtype), xp.asarray(v, dtype=dtype)
    return 0.5 * (avg_x_bwd_2d(u**2) + avg_y_bwd_2d(v**2))


# ======================================================================
# Standalone vorticity functions  (Issue #73)
# ======================================================================


def potential_vorticity(
    omega: Float[Array, "*batch Ny Nx"],
    f: Float[Array, "*batch Ny Nx"],
    h: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Potential vorticity (pointwise).

    q = (omega + f) / h

    All inputs must live on the same grid points (typically X-points
    after interpolating ``f`` and ``h`` from T-points).

    Parameters
    ----------
    omega : Float[Array, "*batch Ny Nx"]
        Relative vorticity.
    f : Float[Array, "*batch Ny Nx"]
        Coriolis parameter.
    h : Float[Array, "*batch Ny Nx"]
        Layer thickness (or depth).

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Potential vorticity.  Where ``h == 0`` the result is NaN.
    """
    xp = namespace_of(omega, f, h)
    return xp.where(h == 0, xp.nan, (omega + f) / h)


# ======================================================================
# Strain operators  (Issue #2)
# ======================================================================


def strain_magnitude_squared(
    sn: Float[Array, "*batch Ny Nx"],
    ss: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Squared strain magnitude (pointwise).

    sigma^2 = Sn^2 + Ss^2

    Inputs must live on the same grid points.  If ``sn`` is at T-points
    and ``ss`` at X-points, interpolate one to the other before calling.

    Parameters
    ----------
    sn : Float[Array, "*batch Ny Nx"]
        Tensor (normal) strain.
    ss : Float[Array, "*batch Ny Nx"]
        Shear strain.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Squared strain magnitude.
    """
    return sn**2 + ss**2


def okubo_weiss(
    sn: Float[Array, "*batch Ny Nx"],
    ss: Float[Array, "*batch Ny Nx"],
    omega: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Okubo-Weiss parameter (pointwise).

    OW = Sn^2 + Ss^2 - omega^2

    All inputs must live on the same grid points.

    Parameters
    ----------
    sn : Float[Array, "*batch Ny Nx"]
        Tensor (normal) strain.
    ss : Float[Array, "*batch Ny Nx"]
        Shear strain.
    omega : Float[Array, "*batch Ny Nx"]
        Relative vorticity.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Okubo-Weiss parameter.  Positive = strain-dominated,
        negative = vorticity-dominated.
    """
    return sn**2 + ss**2 - omega**2


# ======================================================================
# Enstrophy  (Issues #2 and #73)
# ======================================================================


def enstrophy(
    omega: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Enstrophy (pointwise).

    Z = 0.5 * omega^2

    Mask-free Layer-2 functional helper per #209.  For masked enstrophy,
    multiply the result by your mask at the call site:
    ``z = enstrophy(omega) * mask.wet(at="X", kind="strict")``.

    Parameters
    ----------
    omega : Float[Array, "*batch Ny Nx"]
        Relative vorticity (typically at X-points).

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Enstrophy at the same grid points as *omega*.
    """
    return 0.5 * omega**2


def potential_enstrophy(
    q: Float[Array, "*batch Ny Nx"],
    h: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Potential enstrophy (pointwise).

    PE = 0.5 * q^2 * h

    Inputs must live on the same grid points.  Mask-free Layer-2
    functional helper per #209; apply masks at the call site.

    Parameters
    ----------
    q : Float[Array, "*batch Ny Nx"]
        Potential vorticity.
    h : Float[Array, "*batch Ny Nx"]
        Layer thickness (or depth).

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Potential enstrophy.
    """
    return 0.5 * q**2 * h


# ======================================================================
# Potential energy  (Issue #73)
# ======================================================================


def available_potential_energy(
    h: Float[Array, "*batch Ny Nx"],
    H: Float[Array, "*batch Ny Nx"],
    g_prime: float | Float[Array, ""],
) -> Float[Array, "*batch Ny Nx"]:
    """Available potential energy at T-points.

    APE = 0.5 * g' * (h - H)^2

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Layer thickness at T-points.
    H : Float[Array, "*batch Ny Nx"]
        Mean (reference) layer thickness at T-points.
    g_prime : float
        Reduced gravity.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Available potential energy at T-points.
    """
    eta = h - H
    return 0.5 * g_prime * eta**2


# ======================================================================
# QG stretching  (Issue #73)
# ======================================================================


def stretching_term(
    A: Float[Array, "nl nl"],
    psi: Float[Array, "nl Ny Nx"],
) -> Float[Array, "nl Ny Nx"]:
    """Cross-layer stretching term for multi-layer QG models.

    Computes ``A . psi`` — the vertical coupling that mixes information
    across layers.  This is a cross-layer operation and cannot be vmapped;
    it should be used alongside :func:`qg_potential_vorticity` (which
    handles the per-layer Laplacian + beta terms via
    ``jax.vmap``).

    Parameters
    ----------
    A : Float[Array, "nl nl"]
        Coupling (stretching) matrix.  Shape ``(nl, nl)`` where ``nl``
        is the number of layers.
    psi : Float[Array, "nl Ny Nx"]
        Streamfunction at T-points for all layers.

    Returns
    -------
    Float[Array, "nl Ny Nx"]
        Stretching contribution, same shape as ``psi``.
        Ghost ring is zero; interior is ``[:, 1:-1, 1:-1]``.
    """
    xp = namespace_of(A, psi)
    # (A . psi)_k = sum_j A[k,j] * psi_j  for each spatial point
    # Flatten spatial dims, matmul along layer axis, reshape back.
    nl = A.shape[0]
    Ny, Nx = psi.shape[-2], psi.shape[-1]
    psi_flat = psi.reshape(nl, -1)  # [nl, Ny*Nx]
    Ap_flat = A @ psi_flat  # [nl, Ny*Nx]
    Ap = Ap_flat.reshape(nl, Ny, Nx)  # [nl, Ny, Nx]
    out = xp.zeros_like(psi)
    out = set_at(
        out, (..., slice(None), slice(1, -1), slice(1, -1)), Ap[..., :, 1:-1, 1:-1]
    )
    return out


# ======================================================================
# Streamfunction ↔ SSH conversion  (Issue #153)
# ======================================================================


def streamfn_to_ssh(
    psi: Float[Array, "..."],
    f0: float,
    g: float = STANDARD.standard_gravity,
) -> Float[Array, "..."]:
    """Convert streamfunction to sea surface height.

    eta = (f0 / g) * psi

    Parameters
    ----------
    psi : Float[Array, "..."]
        Streamfunction.
    f0 : float
        Reference Coriolis parameter (1/s).
    g : float
        Gravitational acceleration (m/s²).  Default: standard gravity, 9.80665.

    Returns
    -------
    Float[Array, "..."]
        Sea surface height, same shape as *psi*.
    """
    return (f0 / g) * psi


def ssh_to_streamfn(
    ssh: Float[Array, "..."],
    f0: float,
    g: float = STANDARD.standard_gravity,
) -> Float[Array, "..."]:
    """Convert sea surface height to streamfunction.

    psi = (g / f0) * eta

    Parameters
    ----------
    ssh : Float[Array, "..."]
        Sea surface height.
    f0 : float
        Reference Coriolis parameter (1/s).
    g : float
        Gravitational acceleration (m/s²).  Default: standard gravity, 9.80665.

    Returns
    -------
    Float[Array, "..."]
        Streamfunction, same shape as *ssh*.
    """
    return (g / f0) * ssh
