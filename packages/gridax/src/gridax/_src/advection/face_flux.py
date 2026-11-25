"""The Rusanov (local Lax–Friedrichs) face flux."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np
from pipekit_array.arrays import namespace_of

from gridax._src.operators.differentiable import smooth_abs

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


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
        ghost-padded C-grid field, so :func:`divergence` (which expects full
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
    xp = namespace_of(q, a)
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
