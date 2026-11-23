"""Advection: one axis-generic reconstruction, for every scheme of ``gridax.schemes``.

``reconstruct(q, u, axis=..., scheme=...)`` is the advective flux on the faces
along any axis of a C-grid array, for any scheme: ``Upwind``, ``Centred``,
``Biased``, ``WENO`` and ``TVD`` from :mod:`gridax.schemes`. Each face takes the
widest tier of the scheme that fits in the array and, with a mask, on wet
cells. ``Reconstruction1D/2D/3D`` and ``Advection1D/2D/3D`` are built on it.

Example:
    >>> import numpy as np
    >>> import gridax.advection as gadv
    >>> from gridax.schemes import WENO
    >>> q = np.random.default_rng(0).standard_normal((10, 12))
    >>> u = np.ones((10, 12))
    >>> gadv.reconstruct(q, u, axis=-1, scheme=WENO(5)).shape
    (10, 12)
"""

from gridax._src.advection.reconstruct import reconstruct

__all__ = ["reconstruct"]
