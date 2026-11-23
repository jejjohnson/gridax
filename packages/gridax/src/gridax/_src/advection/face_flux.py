"""
Public functional API for raw face fluxes on Arakawa C-grids.

These functions expose the intermediate face-flux arrays ``(fe, fn)`` that
:class:`~gridax.Advection2D` computes internally but does not return.
The face fluxes are the building blocks for custom divergence operators,
momentum advection, and RHS assembly.

Typical usage::

    from gridax import uv_center_flux, CartesianCGrid2D

    grid = CartesianCGrid2D.from_interior(nx, ny, Lx, Ly)
    fe, fn = uv_center_flux(h, u, v, grid)

See Also
--------
:func:`uv_node_flux` : Analogous function for node-centred (q-point) fluxes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np

from gridax._src.advection.reconstruct import axis_flux, check_scheme
from gridax._src.grid import CartesianCGrid2D
from gridax._src.mask import Mask2D
from gridax._src.operators.differentiable import smooth_abs
from gridax._src.pytree import array_namespace
from gridax._src.schemes.base import Scheme
from gridax._src.schemes.linear import Upwind

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


#: The default scheme.
UPWIND = Upwind(1)


def uv_center_flux(
    h: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    grid: CartesianCGrid2D,
    scheme: Scheme = UPWIND,
    mask: Mask2D | None = None,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Compute raw face fluxes for a cell-centred scalar on a C-grid.

    Returns the east-face and north-face flux arrays ``(fe, fn)`` for the
    transport of scalar *h* by velocity ``(u, v)``.  These are the same
    intermediate quantities that :class:`~gridax.Advection2D` computes
    internally before taking the divergence.

    The advective tendency at T-points can be recovered as::

        dh[j, i] = -((fe[j, i] - fe[j, i - 1]) / dx + (fn[j, i] - fn[j - 1, i]) / dy)

    (using the interior indexing convention where ``fe[j, i]`` is the flux
    at the east face of cell ``[j, i]``).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Scalar at T-points (cell centres).  Includes ghost ring.
    u : Float[Array, "Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "Ny Nx"]
        y-velocity at V-points.
    grid : CartesianCGrid2D
        The grid; the metric does not enter.
    scheme : Scheme, default Upwind(1)
        Any of `gridax.schemes`.
    mask : Mask2D | None
        The wet cells: a face takes the widest tier of the scheme that reads
        wet cells only.

    Returns
    -------
    fe : Float[Array, "Ny Nx"]
        East-face flux (h * u reconstructed at east faces).
    fn : Float[Array, "Ny Nx"]
        North-face flux (h * v reconstructed at north faces).
    """
    return _face_fluxes(h, u, v, grid, scheme, mask)


def uv_node_flux(
    q: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    grid: CartesianCGrid2D,
    scheme: Scheme = UPWIND,
    mask: Mask2D | None = None,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Compute raw face fluxes for a node-centred tracer on a C-grid.

    Analogous to :func:`uv_center_flux` but for a tracer *q* that lives at
    grid nodes (vorticity / psi points) rather than cell centres.  The
    reconstruction is the one of ``Advection2D`` — along x for east faces and
    along y for north faces — applied to the q-point field.

    Parameters
    ----------
    q : Float[Array, "Ny Nx"]
        Tracer at node points (q/psi grid).  Includes ghost ring.
    u : Float[Array, "Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "Ny Nx"]
        y-velocity at V-points.
    grid : CartesianCGrid2D
        The grid; the metric does not enter.
    scheme : Scheme, default Upwind(1)
        Any of `gridax.schemes`.
    mask : Mask2D | None
        The wet cells, as in :func:`uv_center_flux`.

    Returns
    -------
    uq_flux : Float[Array, "Ny Nx"]
        East-face flux (q * u reconstructed at east faces).
    vq_flux : Float[Array, "Ny Nx"]
        North-face flux (q * v reconstructed at north faces).
    """
    return _face_fluxes(q, u, v, grid, scheme, mask)


def rusanov_flux(
    q: Float[Array, "..."],
    a: Float[Array, "..."],
    axis: int = -1,
    eps: float = 1e-8,
) -> Float[Array, "..."]:
    r"""Local Lax--Friedrichs (Rusanov) numerical flux at faces along ``axis``.

    For a conserved scalar ``q`` advected at face-normal velocity ``a``, the
    Rusanov flux at the face between two adjacent cells is

    .. math::

        F = \tfrac12\, a\, (q_L + q_R) - \tfrac12\, |a|_\varepsilon\, (q_R - q_L),
        \qquad |a|_\varepsilon = \sqrt{a^2 + \varepsilon^2},

    where ``q_L``/``q_R`` are the cell values on either side of the face.  The
    dissipation term :math:`\tfrac12 |a| (q_R - q_L)` makes the flux monotone
    (first-order, no reconstruction).  This is the standard robust, AD-friendly
    fallback for the continuity/height flux: unlike WENO it has no nonlinear
    smoothness weights, so its adjoint stays well-conditioned.

    The absolute value uses the smooth surrogate :func:`smooth_abs`
    (:math:`\sqrt{a^2 + \varepsilon^2}`) so the dissipation term is
    differentiable at ``a = 0``.  Pass ``eps=0`` to recover the exact textbook
    flux with a hard :func:`jax.numpy.abs` (non-smooth at ``a = 0``).

    Parameters
    ----------
    q : Float[Array, "..."]
        Cell-centred scalar (e.g. layer thickness on the h-grid), including
        any ghost ring.  Reconstructed left/right values are the adjacent
        cells along ``axis``.
    a : Float[Array, "..."]
        Face-normal advecting velocity sampled at the ``N - 1`` faces along
        ``axis`` (i.e. broadcastable to ``q`` with one fewer element along
        ``axis``).  Following the MASSH convention, ``a`` already lives on the
        faces; no interpolation is performed here.
    axis : int, optional
        Axis along which to take the flux.  Default ``-1`` (x / east faces);
        use ``-2`` for y / north faces.
    eps : float, optional
        Smoothing floor for ``|a|``.  ``eps > 0`` (default ``1e-8``) gives the
        AD-safe variant; ``eps == 0`` gives the exact non-smooth flux.

    Returns
    -------
    Float[Array, "..."]
        Numerical flux on the ``N - 1`` interior faces along ``axis``.  This is
        a reduced-shape array (one fewer element along ``axis``), **not** a full
        ghost-padded C-grid field, so :meth:`Difference2D.divergence` (which expects full
        ``[Ny, Nx]`` inputs) does not apply to it directly — take the divergence
        by differencing the faces along the same axis instead.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from gridax import rusanov_flux
    >>> h = jnp.ones((6, 6))
    >>> u = jnp.ones((6, 5))  # face-normal velocity on the 5 x-faces
    >>> fx = rusanov_flux(h, u, axis=-1)
    >>> fx.shape
    (6, 5)
    """
    xp = array_namespace(q, a)
    ndim = q.ndim
    ax = axis % ndim
    left = [slice(None)] * ndim
    right = [slice(None)] * ndim
    left[ax] = slice(None, -1)
    right[ax] = slice(1, None)
    q_l = q[tuple(left)]
    q_r = q[tuple(right)]
    speed = smooth_abs(a, eps) if eps > 0.0 else xp.abs(a)
    return 0.5 * a * (q_l + q_r) - 0.5 * speed * (q_r - q_l)


def _face_fluxes(
    q: Float[Array, "*batch Ny Nx"],
    u: Float[Array, "*batch Ny Nx"],
    v: Float[Array, "*batch Ny Nx"],
    grid: CartesianCGrid2D,
    scheme: Scheme,
    mask: Mask2D | None,
) -> tuple[Float[Array, "*batch Ny Nx"], Float[Array, "*batch Ny Nx"]]:
    """The flux along x and along y."""
    check_scheme(scheme)
    fe, fn = (
        axis_flux(grid, q, velocity, axis=axis, scheme=scheme, mask=mask, bc=None)
        for axis, velocity in (("x", u), ("y", v))
    )
    return fe, fn
