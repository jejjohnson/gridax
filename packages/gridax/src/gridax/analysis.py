"""Analysis kernels on bare collocated arrays: a finite difference and remapping.

Every function here computes in the namespace of its input, so it runs on
NumPy, JAX (under ``jit`` too), CuPy, or any backend ``array-api-compat``
knows, and returns the input's dtype. On a collocated grid the canonical
functions (:func:`gridax.difference`, :func:`gridax.gradient`,
:func:`gridax.divergence`, :func:`gridax.curl`, :func:`gridax.laplacian`) take
the steps from the grid, and a mask.

- :func:`difference`: the difference along one axis of an array, from a step or
  from coordinates (a NaN propagates). The coefficients are
  :mod:`gridax.stencils`.
- :func:`remap_axis` (one 1-D source coordinate) and
  :func:`remap_axis_columns` (one set of levels per column).
"""

from gridax._src.analysis.differences import difference
from gridax._src.analysis.remap import Extrapolate, remap_axis, remap_axis_columns

__all__ = ["Extrapolate", "difference", "remap_axis", "remap_axis_columns"]
