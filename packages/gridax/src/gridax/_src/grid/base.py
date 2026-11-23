"""Abstract base classes for gridax grid containers.

This module defines the layered abstraction shared by every concrete
grid type (Cartesian, spherical):

    ArakawaCGrid1D / 2D / 3D       — pure C-grid topology (Nx, Ny, Nz +
                                     ghost-cell convention).  No metric.
        ↓
    CurvilinearGrid1D / 2D / 3D    — adds a uniform metric (Lx, Ly, Lz
                                     and the accessor spacing, coords,
                                     area).  Coordinate-system agnostic;
                                     serves as the parent for all concrete
                                     grids.
        ↓
    CartesianCGrid* (cartesian.py)
    SphericalCGrid* (spherical.py)

The split lets operators that only need topology (e.g. shape and
ghost-cell handling) accept ``ArakawaCGrid*``, operators that need a
generic uniform spacing accept ``CurvilinearGrid*``, and operators
that require coordinate-system-specific metric (e.g. ``cos(lat)``)
take the concrete subclass directly.

Every class is a frozen dataclass built with NumPy, registered as a JAX
pytree when JAX can be imported (:func:`pytree_dataclass`). The integer
sizes ``Nx``, ``Ny``, ``Nz``, the ghost-ring width ``ghost`` and the axis
names ``axes`` are static pytree metadata, so a grid can be passed to plain
``jax.jit`` and still give concrete shapes. Every other field is an array
leaf: a Python float given to a constructor is stored as a 0-d float64
array, so ``jax.jit`` and ``eqx.filter_jit`` both trace it and one compiled
function serves every grid of a shape. The operators cast a metric to the
dtype of the field it scales.

The metric is read through one accessor, at any point of the staggering:
``grid.spacing(axis, at=P)``, ``grid.coords(axis, at=P)`` and
``grid.area(at=P)``, with the axis names of ``grid.axes``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np
from pipekit_array.functions import namespace_of

from gridax._src.grid.points import point_axes
from gridax._src.pytree import pytree_dataclass, static_field

# ``jaxtyping.Array`` imports JAX, so it names the type for the type checker
# only.
if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


def _as_leaf(value: Any) -> Any:
    """A Python (or NumPy) number as a 0-d float64 array; anything else as is."""
    if isinstance(value, int | float | np.integer | np.floating):
        return np.asarray(value, dtype=np.float64)
    return value


# ----------------------------------------------------------------------
# Layer 1 — pure C-grid topology (no metric)
# ----------------------------------------------------------------------


@pytree_dataclass
class ArakawaCGrid1D:
    """1-D Arakawa C-grid topology (no metric).

    Records only the integer dimension and the ghost-cell convention;
    no physical spacing.  Concrete subclasses (``CartesianCGrid1D``,
    etc.) attach a metric.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including the ghost ring).
    axes : tuple of str
        The names of the array axes, in array order (static).
    ghost : int
        Width of the ghost ring on each side (static).
    periodic : tuple of str
        The axes that wrap, for the record (static): a boundary condition says
        what fills the ring.

    Notes
    -----
    Array layout (total length ``Nx``)::

        i= 0    1    2   ...   Nx-2  Nx-1
          [g]  [ ]  [ ]  ...    [ ]  [g]
           ^    \\______ interior ______/    ^
        ghost                              ghost
        (west)                             (east)

    The 1-D C-grid uses two staggering locations (T-cell centres and
    U-east-faces); see ``CartesianCGrid1D`` for the colocation rule.
    """

    Nx: int = static_field()
    axes: tuple[str, ...] = static_field(default=("x",), kw_only=True)
    ghost: int = static_field(default=1, kw_only=True)
    periodic: tuple[str, ...] = static_field(default=(), kw_only=True)

    @property
    def shape(self) -> tuple[int, ...]:
        """The array shape of a field, ghost ring included: ``(Nx,)``."""
        return (self.Nx,)


@pytree_dataclass
class ArakawaCGrid2D:
    """2-D Arakawa C-grid topology (no metric).

    Records only the integer dimensions and the ghost-cell convention;
    no physical spacing.  Concrete subclasses (``CartesianCGrid2D``,
    ``SphericalCGrid2D``, ...) attach a metric.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including 2 ghost cells).
    Ny : int
        Total number of cells in y (including the ghost ring).
    axes : tuple of str
        The names of the array axes, in array order (static).
    ghost : int
        Width of the ghost ring on each side (static).
    periodic : tuple of str
        The axes that wrap, for the record (static): a boundary condition says
        what fills the ring.

    Notes
    -----
    Array layout (total shape ``[Ny, Nx]``)::

        j=Ny-1  +--+--+--+--+  ghost row (north)
                |  |  |  |  |
        j=Ny-2  +--+--+--+--+
                |  |  |  |  |  physical interior  [1:-1, 1:-1]
        j=1     +--+--+--+--+
                |  |  |  |  |
        j=0     +--+--+--+--+  ghost row (south)
                i=0  ...    i=Nx-1
                ghost      ghost
                (west)     (east)

    The 2-D C-grid uses four staggering locations (T, U, V, X); array
    index ``[j, i]`` encodes the **south-west** corner of the stencil
    neighbourhood.  See ``CartesianCGrid2D`` for the colocation rule.
    """

    Nx: int = static_field()
    Ny: int = static_field()
    axes: tuple[str, ...] = static_field(default=("y", "x"), kw_only=True)
    ghost: int = static_field(default=1, kw_only=True)
    periodic: tuple[str, ...] = static_field(default=(), kw_only=True)

    @property
    def shape(self) -> tuple[int, ...]:
        """The array shape of a field, ghost ring included: ``(Ny, Nx)``."""
        return (self.Ny, self.Nx)


@pytree_dataclass
class ArakawaCGrid3D:
    """3-D Arakawa C-grid topology (no metric).

    Records only the integer dimensions and the ghost-cell convention;
    no physical spacing.  Concrete subclasses (``CartesianCGrid3D``,
    ``SphericalCGrid3D``, ...) attach a metric.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including 2 ghost cells).
    Ny : int
        Total number of cells in y (including 2 ghost cells).
    Nz : int
        Total number of cells in z (including the ghost ring).
    axes : tuple of str
        The names of the array axes, in array order (static).
    ghost : int
        Width of the ghost ring on each side (static).
    periodic : tuple of str
        The axes that wrap, for the record (static): a boundary condition says
        what fills the ring.

    Notes
    -----
    Array layout (total shape ``[Nz, Ny, Nx]``).  Each z-level is a
    2-D slab identical to ``ArakawaCGrid2D``::

        k=Nz-1  +================+  ghost slab (top)
                |   2-D slab     |
        k=Nz-2  +================+
                |   2-D slab     |  physical interior  [1:-1, 1:-1, 1:-1]
        k=1     +================+
                |   2-D slab     |
        k=0     +================+  ghost slab (bottom)

    Only the horizontal axes are staggered (T, U, V, X share the same
    z-level); W-fields use the same ``[Nz, Ny, Nx]`` shape, ``w[k]`` on the
    top face of ``T[k]`` (as ``Mask3D.wet(at="W")`` and ``vertical_velocity``).
    """

    Nx: int = static_field()
    Ny: int = static_field()
    Nz: int = static_field()
    axes: tuple[str, ...] = static_field(default=("z", "y", "x"), kw_only=True)
    ghost: int = static_field(default=1, kw_only=True)
    periodic: tuple[str, ...] = static_field(default=(), kw_only=True)

    @property
    def shape(self) -> tuple[int, ...]:
        """The array shape of a field, ghost ring included: ``(Nz, Ny, Nx)``."""
        return (self.Nz, self.Ny, self.Nx)


# ----------------------------------------------------------------------
# Layer 2 — uniform curvilinear metric (Lx, spacing, ...)
# ----------------------------------------------------------------------


class _UniformMetric:
    """The metric accessor of a grid with one uniform length per axis.

    Mixed into :class:`CurvilinearGrid1D` / ``2D`` / ``3D``. Each axis of
    ``axes`` is called ``"x"``, ``"y"`` or ``"z"`` by its extent ``Lx``,
    ``Ly``, ``Lz`` and its size ``Nx``, ``Ny``, ``Nz``.
    """

    # Fields and properties of the dataclasses this mixes into.
    axes: tuple[str, ...]
    ghost: int

    if TYPE_CHECKING:

        @property
        def shape(self) -> tuple[int, ...]: ...

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):  # ty: ignore[invalid-argument-type]
            if not field.metadata.get("static", False):
                value = getattr(self, field.name)
                if value is not None:
                    object.__setattr__(self, field.name, _as_leaf(value))

    def _size(self, axis: str) -> int:
        """Number of points along ``axis``, ghost ring included."""
        if axis not in self.axes:
            raise ValueError(
                f"{type(self).__name__} has the axes {self.axes}, not {axis!r}"
            )
        return self.shape[self.axes.index(axis)]

    def _staggered(self, axis: str, at: str) -> bool:
        """Whether the points ``at`` sit half a cell along ``axis``."""
        self._size(axis)
        return axis in point_axes(at, self.axes)

    def _step(self, axis: str) -> Float[Array, ""]:
        """The uniform step along ``axis``: L / (N − 2 ghost)."""
        n_interior = self._size(axis) - 2 * self.ghost
        return getattr(self, f"L{axis}") / n_interior

    def spacing(self, axis: str, *, at: str = "T") -> Float[Array, "..."]:
        """The grid spacing along ``axis`` at the points ``at``.

        ``Δ = L / N_interior``, the same at every point of a uniform grid.

        Parameters
        ----------
        axis : str
            One of ``grid.axes``.
        at : str
            The point: ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.

        Returns
        -------
        Float[Array, ""]
            The spacing, a 0-d array.

        Examples
        --------
        >>> import gridax
        >>> grid = gridax.CartesianCGrid2D.from_interior(10, 5, 1.0, 1.0)
        >>> float(grid.spacing("x")), float(grid.spacing("y", at="V"))
        (0.1, 0.2)
        """
        self._staggered(axis, at)
        return self._step(axis)

    def coords(self, axis: str, *, at: str = "T") -> Float[Array, "N"]:
        """The coordinate of the points ``at`` along ``axis``, ghost ring included.

        ``x[i] = i Δx`` at the points centred along the axis and
        ``(i + 1/2) Δx`` at the points staggered along it (the same-index
        convention of ``docs/cgrid_discretization.md``). A grid with an origin
        (``x0``, from ``CartesianAGrid2D.staggered``) has its first interior
        point there.

        Parameters
        ----------
        axis : str
            One of ``grid.axes``.
        at : str
            The point: ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.

        Returns
        -------
        Float[Array, "N"]
            One coordinate per index along the axis.

        Examples
        --------
        >>> import gridax
        >>> grid = gridax.CartesianCGrid1D.from_interior(2, 1.0)
        >>> grid.coords("x").tolist(), grid.coords("x", at="U").tolist()
        ([0.0, 0.5, 1.0, 1.5], [0.25, 0.75, 1.25, 1.75])
        """
        shift = 0.5 if self._staggered(axis, at) else 0.0
        step = self._step(axis)
        origin = getattr(self, f"{axis}0", None)
        xp = namespace_of(step, origin)
        index = xp.arange(self._size(axis))
        if origin is None:
            # x[i] = (i + shift) Δx
            return (index + shift) * step
        # x[i] = x0 + (i - ghost + shift) Δx
        return origin + (index - self.ghost + shift) * step

    def area(self, *, at: str = "T") -> Float[Array, "..."]:
        """The horizontal cell measure around the points ``at``.

        ``Δx Δy``, broadcast to ``[Ny, Nx]`` (the cell length ``Δx``, ``[Nx]``,
        on a 1-D grid).

        Parameters
        ----------
        at : str
            The point: ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.

        Returns
        -------
        Float[Array, "..."]
            One area per horizontal cell, ghost ring included.

        Examples
        --------
        >>> import gridax
        >>> grid = gridax.CartesianCGrid2D.from_interior(4, 2, 1.0, 1.0)
        >>> grid.area().shape, float(grid.area()[0, 0])
        ((4, 6), 0.125)
        """
        return self._area(at=at)

    def _area(self, *, at: str = "T", xp: Any = None) -> Float[Array, "..."]:
        """:meth:`area` in the namespace ``xp`` (the grid's own by default)."""
        point_axes(at, self.axes)
        horizontal = [axis for axis in self.axes if axis != "z"]
        cell = self._step(horizontal[-1])
        for axis in horizontal[:-1]:
            cell = cell * self._step(axis)
        xp = namespace_of(cell) if xp is None else xp
        shape = tuple(self._size(axis) for axis in horizontal)
        return xp.full(shape, cell)


@pytree_dataclass
class CurvilinearGrid1D(_UniformMetric, ArakawaCGrid1D):
    """1-D curvilinear C-grid with a uniform metric.

    Adds the physical extent ``Lx`` to :class:`ArakawaCGrid1D`; the spacing
    follows from it (:meth:`spacing`). Concrete subclasses
    (``CartesianCGrid1D``) inherit from this layer and provide their
    coordinate-system-specific construction logic.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including the ghost ring).
    Lx : float
        Physical domain length in x, the ghost ring excluded.
    """

    Lx: Float[Array, ""]


@pytree_dataclass
class CurvilinearGrid2D(_UniformMetric, ArakawaCGrid2D):
    """2-D curvilinear C-grid with a uniform metric.

    Adds the physical extents ``Lx`` and ``Ly`` to :class:`ArakawaCGrid2D`;
    the spacings follow from them (:meth:`spacing`). Concrete subclasses
    (``CartesianCGrid2D``, ``SphericalCGrid2D``) inherit from this layer and
    provide their coordinate-system-specific construction logic.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including the ghost ring).
    Ny : int
        Total number of cells in y (including the ghost ring).
    Lx : float
        Physical domain length in x, the ghost ring excluded.
    Ly : float
        Physical domain length in y, the ghost ring excluded.
    """

    Lx: Float[Array, ""]
    Ly: Float[Array, ""]


@pytree_dataclass
class CurvilinearGrid3D(_UniformMetric, ArakawaCGrid3D):
    """3-D curvilinear C-grid with a uniform metric.

    Adds the physical extents ``Lx``, ``Ly`` and ``Lz`` to
    :class:`ArakawaCGrid3D`; the spacings follow from them
    (:meth:`spacing`). Concrete subclasses (``CartesianCGrid3D``,
    ``SphericalCGrid3D``) inherit from this layer and provide their
    coordinate-system-specific construction logic.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including the ghost ring).
    Ny : int
        Total number of cells in y (including the ghost ring).
    Nz : int
        Total number of cells in z (including the ghost ring).
    Lx : float
        Physical domain length in x, the ghost ring excluded.
    Ly : float
        Physical domain length in y, the ghost ring excluded.
    Lz : float
        Physical domain length in z, the ghost ring excluded.
    """

    Lx: Float[Array, ""]
    Ly: Float[Array, ""]
    Lz: Float[Array, ""]
