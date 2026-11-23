"""Advection: fluxes and tendencies with any scheme of :mod:`gridax.schemes`.

:func:`advect` is the finite-volume tendency of a tracer and :func:`reconstruct`
its flux on the faces along one axis, also at the top level of ``gridax``. Each
face takes the widest tier of the scheme that fits in the array and, with a
mask, on wet cells. The point stencils underneath are in
:mod:`gridax.advection.kernels`.

Example:
    >>> import gridax, numpy as np
    >>> from gridax.schemes import WENO
    >>> grid = gridax.CartesianCGrid2D.from_interior(10, 8, 1.0, 1.0)
    >>> q = np.random.default_rng(0).standard_normal(grid.shape)
    >>> u = np.ones(grid.shape)
    >>> gridax.advect(q, u, 0.0 * u, grid, scheme=WENO(5)).shape
    (10, 12)
"""

from gridax._src.advection.advect import advect
from gridax._src.advection.reconstruct import reconstruct

__all__ = ["advect", "reconstruct"]
