"""The point stencils of advection: cell values in, a face value out.

Each function takes the cell averages around a face, in order, and returns the
face value; ``*_right`` is the mirror, for flow towards ``-axis``. The schemes
of :mod:`gridax.schemes` are built on them, and every weight comes from Layer 0
(:mod:`gridax.stencils`).

- ``linear_*``: the optimal linear weights;
- ``weno_*``: WENO, Jiang–Shu, and WENO-Z (``*_improved``);
- ``upwind_1pt``, ``upwind_3pt``, ``upwind_5pt``: upwind fluxes along an axis,
  and ``plusminus``, the positive and the negative part of a velocity.

Example:
    >>> from gridax.advection.kernels import linear_4pts
    >>> linear_4pts(1.0, 2.0, 3.0, 4.0)
    2.5
"""

from gridax._src.advection.linear import (
    linear_2pts,
    linear_3pts_left,
    linear_3pts_right,
    linear_4pts,
    linear_5pts_left,
    linear_5pts_right,
    linear_6pts,
)
from gridax._src.advection.reconstruct_fn import (
    plusminus,
    upwind_1pt,
    upwind_3pt,
    upwind_5pt,
)
from gridax._src.advection.weno import (
    weno_3pts,
    weno_3pts_improved,
    weno_3pts_improved_right,
    weno_3pts_right,
    weno_5pts,
    weno_5pts_improved,
    weno_5pts_improved_right,
    weno_5pts_right,
    weno_7pts,
    weno_7pts_right,
    weno_9pts,
    weno_9pts_right,
)

__all__ = [
    "linear_2pts",
    "linear_3pts_left",
    "linear_3pts_right",
    "linear_4pts",
    "linear_5pts_left",
    "linear_5pts_right",
    "linear_6pts",
    "plusminus",
    "upwind_1pt",
    "upwind_3pt",
    "upwind_5pt",
    "weno_3pts",
    "weno_3pts_improved",
    "weno_3pts_improved_right",
    "weno_3pts_right",
    "weno_5pts",
    "weno_5pts_improved",
    "weno_5pts_improved_right",
    "weno_5pts_right",
    "weno_7pts",
    "weno_7pts_right",
    "weno_9pts",
    "weno_9pts_right",
]
