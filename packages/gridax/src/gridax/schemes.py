"""Schemes: which values an operation reads around its output, and their weights.

One object says which stencil an operation uses: :class:`Centred`,
:class:`Biased`, :class:`Upwind`, :class:`WENO` and :class:`TVD`. ``order`` is
the formal order of accuracy of each. A :class:`Scheme` is a ladder of
:class:`Tier` s, widest first, and answers what it reads:

- ``footprint(staggered=)`` and ``halo(staggered=)``: the cells it reads and
  the ghost cells it needs;
- ``weights(derivative=, staggered=)``: its exact weights, as fractions;
- ``degrades_on(grid)``: whether it must narrow or shift next to the edges;
- ``linear`` and ``upwinded``: ``difference`` and ``interpolate`` take the
  schemes that are linear and not upwinded, ``reconstruct`` takes any.

Everything here imports without JAX.

Example:
    >>> from gridax.schemes import WENO, Centred
    >>> Centred(4, data="point")
    Centred(order=4, data='point') # (-1, 9, 9, -1)/16
    >>> WENO(5).halo(staggered=True)
    3
"""

from gridax._src.schemes.base import Scheme, Tier
from gridax._src.schemes.linear import Biased, Centred, Upwind
from gridax._src.schemes.nonlinear import TVD, WENO

__all__ = ["TVD", "WENO", "Biased", "Centred", "Scheme", "Tier", "Upwind"]
