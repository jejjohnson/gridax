"""gridax grid containers — abstract base, Cartesian, spherical."""

from gridax._src.grid.base import (
    ArakawaCGrid1D,
    ArakawaCGrid2D,
    ArakawaCGrid3D,
    CurvilinearGrid1D,
    CurvilinearGrid2D,
    CurvilinearGrid3D,
)
from gridax._src.grid.cartesian import (
    CartesianCGrid1D,
    CartesianCGrid2D,
    CartesianCGrid3D,
)
from gridax._src.grid.collocated import (
    CartesianAGrid1D,
    CartesianAGrid2D,
    CartesianAGrid3D,
    CollocatedGrid1D,
    CollocatedGrid2D,
    CollocatedGrid3D,
    SphericalAGrid2D,
    agrid_from,
)
from gridax._src.grid.coordinates import t_point_coordinates
from gridax._src.grid.spherical import SphericalCGrid2D, SphericalCGrid3D
from gridax._src.grid.stretched import (
    StretchedCGrid1D,
    StretchedCGrid2D,
    StretchedCGrid3D,
)

__all__ = [
    # Abstract topology
    "ArakawaCGrid1D",
    "ArakawaCGrid2D",
    "ArakawaCGrid3D",
    # Curvilinear (uniform metric)
    "CurvilinearGrid1D",
    "CurvilinearGrid2D",
    "CurvilinearGrid3D",
    # Cartesian concrete
    "CartesianCGrid1D",
    "CartesianCGrid2D",
    "CartesianCGrid3D",
    # Spherical concrete
    "SphericalCGrid2D",
    "SphericalCGrid3D",
    # Irregular axes
    "StretchedCGrid1D",
    "StretchedCGrid2D",
    "StretchedCGrid3D",
    # Collocated
    "CollocatedGrid1D",
    "CollocatedGrid2D",
    "CollocatedGrid3D",
    "CartesianAGrid1D",
    "CartesianAGrid2D",
    "CartesianAGrid3D",
    "SphericalAGrid2D",
    "agrid_from",
    # Point coordinates
    "t_point_coordinates",
]
