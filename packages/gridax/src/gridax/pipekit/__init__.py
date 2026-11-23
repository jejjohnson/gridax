"""gridax operators as pipekit Operators (Tier B).

Requires the ``pipekit`` extra (``pip install "gridax[pipekit]"``);
``import gridax`` never imports pipekit-jax, nor JAX-only libraries.

Every operator here is a `pipekit_jax.JaxOperator`: a pipekit ``Operator``
(JSON config, ``|`` composition, ``Sequential`` / ``Graph`` nodes) that is
also an ``eqx.Module`` (runs under ``jit`` / ``grad`` / ``vmap``). Fields
are C-grid arrays ``[*batch, Ny, Nx]``; the grid is a
`CartesianCGrid2D`, or its ``(nx_interior, ny_interior, Lx, Ly)`` config.
"""

from __future__ import annotations

try:
    import pipekit_jax as _pipekit_jax  # noqa: F401
except ImportError as err:  # pragma: no cover - exercised without the extra
    raise ImportError(
        'gridax.pipekit needs the `pipekit` extra: pip install "gridax[pipekit]"'
    ) from err

from gridax._src.pipekit.base import (
    FVXOperator,
    GridOperator,
    over_leading,
)
from gridax._src.pipekit.diagnostics import (
    STRAIN_COMPONENTS,
    AvailablePE,
    DomainIntegral,
    DomainMean,
    Enstrophy,
    KineticEnergy,
    OkuboWeiss,
    PotentialVorticity,
    QGPotentialVorticity,
    RelativeVorticity,
    Strain,
    TotalEnergy,
)
from gridax._src.pipekit.halo import graph_halo
from gridax._src.pipekit.stencils import (
    DIFFERENCES,
    GHOST_FILLERS,
    INTERPOLATIONS,
    MASK_KINDS,
    ApplyMask,
    Coriolis,
    Curl,
    Difference,
    Divergence,
    FillGhosts,
    GradPerp,
    Interp,
    Laplacian,
    LinComb,
    Pick,
)
from gridax._src.pipekit.timestep import SCHEMES, ExplicitStep

__all__ = [
    # base
    "FVXOperator",
    "GridOperator",
    "graph_halo",
    "over_leading",
    # stencils, interpolation, boundaries, masks
    "DIFFERENCES",
    "GHOST_FILLERS",
    "INTERPOLATIONS",
    "MASK_KINDS",
    "ApplyMask",
    "Coriolis",
    "Curl",
    "Difference",
    "Divergence",
    "FillGhosts",
    "GradPerp",
    "Interp",
    "Laplacian",
    # diagnostics and reductions
    "STRAIN_COMPONENTS",
    "AvailablePE",
    "DomainIntegral",
    "DomainMean",
    "Enstrophy",
    "KineticEnergy",
    "OkuboWeiss",
    "PotentialVorticity",
    "QGPotentialVorticity",
    "RelativeVorticity",
    "Strain",
    "TotalEnergy",
    # time stepping
    "SCHEMES",
    "ExplicitStep",
    # graph plumbing
    "LinComb",
    "Pick",
]
