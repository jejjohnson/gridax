"""Cartesian Arakawa C-grid masks (1-D / 2-D / 3-D).

Concrete mask classes for uniform-spacing Cartesian grids.  All
construction logic uses numpy (masks are built once at setup
time and stored as NumPy arrays; JAX converts them when a kernel reads
them, and each class is a JAX pytree when JAX is importable).

Same-index colocation convention (matches the grid module's
``CartesianCGrid2D`` / ``CartesianCGrid3D`` positive-half-step rule)::

    h[j, i]                cell centre  at ( j,        i       )
    u[j, i]                east face    at ( j,        i + 1/2 )
    v[j, i]                north face   at ( j + 1/2,  i       )
    xy_corner[j, i]        NE corner    at ( j + 1/2,  i + 1/2 )  (lenient)
    xy_corner_strict[j, i] NE corner    at ( j + 1/2,  i + 1/2 )  (strict)

Staggered masks are derived from the h-mask using n-D average pooling
with **trailing-side** zero padding so the output shape equals the
input shape and ``mask.u[j, i]`` / ``mask.v[j, i]`` align with the
grid module's ``U[j, i]`` / ``V[j, i]`` positions::

    u[j, i]                = (h[j, i  ] + h[j,   i+1]) / 2 > 3/4
    v[j, i]                = (h[j, i  ] + h[j+1, i  ]) / 2 > 3/4
    xy_corner[j, i]        = sum of 4 NE-corner h-cells / 4 > 1/8  (lenient)
    xy_corner_strict[j, i] = sum of 4 NE-corner h-cells / 4 > 7/8  (strict)
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any, ClassVar, Self

from jaxtyping import Bool, Float
import numpy as np
from pipekit_array.arrays import namespace_of

from gridax._src.grid.points import POINT_AXES, Stagger, point_axes
from gridax._src.mask.base import (
    WALL_KINDS,
    MaskBase,
    StencilCapability1D,
    StencilCapability2D,
    StencilCapability3D,
)
from gridax._src.mask.utils import h_from_pooled, pool_bool, shifted
from gridax._src.pytree import pytree_dataclass

# The mask arrays are NumPy when a mask is built and JAX arrays (or tracers)
# once a transform has flattened it. ``jaxtyping.Array`` imports JAX, so it
# names the type for the type checker only.
if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


def _wet(mask: Any, axes: tuple[str, ...], at: str | Stagger, kind: str) -> Any:
    """``mask.wet``: the array of a point, and of a kind of corner."""
    staggered = point_axes(at, axes)
    if staggered == ("x", "y") and mask._u is not None:
        if kind in ("default", "strict"):
            return mask._xy_corner if kind == "default" else mask._xy_corner_strict
        if kind in WALL_KINDS:
            return _corner_kind(mask, kind)
        kinds = ("default", "strict", *WALL_KINDS)
        raise ValueError(f"kind must be one of {kinds} at X, got {kind!r}")
    if kind != "default":
        raise ValueError(f"kind={kind!r} exists at X only, not at {at!r}")
    if not staggered:
        return mask._h
    if mask._u is None:
        raise ValueError(f"at={at!r}: the mask of a collocated grid has T points only")
    return getattr(mask, {"x": "_u", "y": "_v", "z": "_w"}[staggered[0]])


def _corner_kind(mask: Any, kind: str) -> Any:
    """The corners of a wall kind, from the four faces that meet at each."""
    corner, u, v = mask._xy_corner, mask._u, mask._v
    # the faces around the corner [j, i]: u[j, i] south of it, u[j+1, i] north,
    # v[j, i] west, v[j, i+1] east; those outside the array are dry
    u_north, v_east = shifted(u, -2, 1), shifted(v, -1, 1)
    if kind == "valid":
        return corner & u & u_north & v & v_east
    on_y = ~u | ~u_north  # a vertical wall
    on_x = ~v | ~v_east  # a horizontal wall
    return corner & {"y_wall": on_y, "x_wall": on_x, "convex": on_y & on_x}[kind]


#: The stored mask of each point, and how many of the cells of its window must be
#: wet: all of them, or one for the lenient corners.
_STORED: dict[str, tuple[str, str]] = {
    "_u": ("U", "all"),
    "_v": ("V", "all"),
    "_w": ("W", "all"),
    "_xy_corner": ("X", "any"),
    "_xy_corner_strict": ("X", "all"),
}


def _window(point: str, axes: tuple[str, ...]) -> tuple[int, ...]:
    """The cells around a point: two along each axis it is staggered on."""
    return tuple(2 if axis in POINT_AXES[point] else 1 for axis in axes)


class _CartesianMask(MaskBase):
    """What `Mask1D`, `Mask2D` and `Mask3D` share: the wet points and the builders.

    The point ``[k, j, i]`` of a staggered mask takes the cells ``[k, j, i]`` and
    one step further along each axis the point is staggered on, so
    ``u[j, i]`` is wet if ``h[j, i]`` and ``h[j, i+1]`` are. Cells outside the
    array are dry.
    """

    _counts: ClassVar[Any]

    # ── wet points ───────────────────────────────────────────────────────────

    def wet(
        self, *, at: str | Stagger = "T", kind: str = "default"
    ) -> Bool[Array, "*grid"]:
        """The wet points at a point of the staggering; ``True`` = wet.

        Parameters
        ----------
        at : str or Stagger
            The point, one that the rank of the mask has, or a field's stagger
            (``CField.stagger``).
        kind : str
            At X only: ``"default"`` (at least one of the four cells is wet),
            ``"strict"`` (all four are), ``"y_wall"``, ``"x_wall"``,
            ``"convex"`` or ``"valid"``.

        Returns
        -------
        Bool[Array, "*grid"]

        Raises
        ------
        ValueError
            If the mask has no such point or kind.

        Examples
        --------
        >>> import numpy as np
        >>> from gridax import Mask2D
        >>> mask = Mask2D.from_mask(np.ones((4, 4), dtype=bool))
        >>> int(mask.wet().sum()), int(mask.wet(at="U").sum())
        (16, 12)
        """
        return _wet(self, self._axes, at, kind)

    def _rebuilt(self, h: Bool[Array, "*grid"]) -> Self:
        """The mask of the wet cells ``h``, as `from_mask` builds it.

        In the array namespace of ``h``, so on a traced mask too; `wrapped_mask`
        gives it the cells with their ghost ring wrapped.
        """
        xp = namespace_of(h)
        fields: dict[str, Any] = {
            "_h": h,
            "stencil_capability": type(self.stencil_capability)._of_wet(h),
        }
        for name, (point, wet) in _STORED.items():
            if getattr(self, name, None) is None:
                continue
            # the cells of the window of each point, as `from_mask` pools them
            cells = [h]
            for position, size in enumerate(_window(point, self._axes)):
                if size == 2:
                    cells += [shifted(c, position - len(self._axes), 1) for c in cells]
            combine = xp.logical_and if wet == "all" else xp.logical_or
            fields[name] = functools.reduce(combine, cells)
        return dataclasses.replace(self, **fields)  # ty: ignore[invalid-argument-type]

    # ── factory class-methods ─────────────────────────────────────────────────

    @classmethod
    def from_mask(
        cls,
        mask_hgrid: np.ndarray | Bool[Array, "*grid"],
        k_bottom: Array | None = None,
    ) -> Self:
        """Construct from the wet (1 / True) and dry cells.

        Everything is computed and stored with NumPy.

        Parameters
        ----------
        mask_hgrid : array-like [*grid]
            Wet / dry mask at cell centres.
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices; not in 1-D.

        Returns
        -------
        Mask1D, Mask2D or Mask3D

        Notes
        -----
        The staggered masks are :func:`pool_bool` of the cells, padded on the
        trailing side: a face is wet if both its cells are, the lenient corner
        if one of its four cells is, the strict corner if all four are. The
        stencil capability is :meth:`StencilCapability2D.from_mask`.
        """
        h = np.asarray(mask_hgrid, dtype=bool)
        cells = h.astype(np.float32)
        names = {field.name for field in dataclasses.fields(cls)}
        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "_h": h,
            "stencil_capability": cls._counts.from_mask(h),
        }
        for name, (point, wet) in _STORED.items():
            if name not in names:
                continue
            window = _window(point, cls._axes)
            size = int(np.prod(window))
            # between size - 1 and size wet cells, or between none and one
            threshold = (size - 0.5) / size if wet == "all" else 0.5 / size
            fields[name] = pool_bool(
                cells, kernel=window, threshold=threshold, direction="trailing"
            )
        if "k_bottom" in names:
            fields["k_bottom"] = None if k_bottom is None else np.asarray(k_bottom)
        elif k_bottom is not None:
            raise TypeError(f"{cls.__name__} has no k_bottom")
        return cls(**fields)

    @classmethod
    def from_grid(
        cls,
        grid: Any,
        wet_t: np.ndarray | Bool[Array, "*grid"],
        *,
        k_bottom: Array | None = None,
    ) -> Self:
        """Construct from the wet T points of a grid.

        Parameters
        ----------
        grid : grid
            The grid the mask is for, of the rank of the mask.
        wet_t : array-like [*grid]
            Wet (``True``) / dry mask at the T points, ghost ring included.
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices. See :meth:`from_mask`.

        Returns
        -------
        Mask1D, Mask2D or Mask3D

        Raises
        ------
        ValueError
            If the shape of ``wet_t`` is not the grid's.

        Examples
        --------
        >>> import numpy as np
        >>> from gridax import CartesianCGrid2D, Mask2D
        >>> grid = CartesianCGrid2D.from_interior(4, 4, 1.0, 1.0)
        >>> Mask2D.from_grid(grid, np.ones(grid.shape)).wet().shape == grid.shape
        True
        """
        if np.shape(wet_t) != grid.shape:
            raise ValueError(
                f"wet_t has shape {np.shape(wet_t)}, and the grid {grid.shape}"
            )
        mask = cls.from_mask(wet_t, k_bottom)
        if not getattr(grid, "_collocated", False):
            return mask
        # a collocated grid has T points alone
        staggered = {name for name in _STORED if hasattr(mask, name)}
        return dataclasses.replace(mask, **dict.fromkeys(staggered))  # ty: ignore[invalid-argument-type]

    @classmethod
    def from_center(
        cls,
        field: Float[Array, "*grid"],
        k_bottom: Array | None = None,
    ) -> Self:
        """Construct from a field at cell centres (T-points), ``NaN`` where dry.

        This is the natural constructor for any tracer-like quantity (SSH,
        temperature, salinity, pressure, …).

        Parameters
        ----------
        field : array-like [*grid]
            Float field at cell centres; ``NaN`` marks dry cells.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask1D, Mask2D or Mask3D
        """
        return cls.from_mask(np.isfinite(np.asarray(field)), k_bottom)

    @classmethod
    def _from_point(cls, field: Any, at: str, mode: str, k_bottom: Any) -> Self:
        """Construct from a staggered field, ``NaN`` where dry."""
        wet = np.isfinite(np.asarray(field))
        window = _window(at, cls._axes)
        return cls.from_mask(
            h_from_pooled(wet, kernel=window, mode=mode, direction="trailing"), k_bottom
        )

    @classmethod
    def from_u_face(
        cls,
        field: Float[Array, "*grid"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Self:
        """Construct from a field at u-faces, ``NaN`` where dry.

        Parameters
        ----------
        field : array-like [*grid]
            Float field at u-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            ``'permissive'`` (default) marks a cell wet if *any* of its two
            u-faces is wet, ``'conservative'`` if *all* are
            (:func:`h_from_pooled`).
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask1D, Mask2D or Mask3D
        """
        return cls._from_point(field, "U", mode, k_bottom)


class _HorizontalMask(_CartesianMask):
    """The builders of the masks that have a y axis."""

    @classmethod
    def from_v_face(
        cls,
        field: Float[Array, "*grid"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Self:
        """Construct from a field at v-faces, ``NaN`` where dry.

        Parameters
        ----------
        field : array-like [*grid]
            Float field at v-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            See :meth:`from_u_face`.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D or Mask3D
        """
        return cls._from_point(field, "V", mode, k_bottom)

    @classmethod
    def from_corner(
        cls,
        field: Float[Array, "*grid"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Self:
        """Construct from a field at xy-corners (vertices), ``NaN`` where dry.

        This is the natural constructor for vertex-stored quantities such as
        relative vorticity. Corners in 3-D are per level.

        Parameters
        ----------
        field : array-like [*grid]
            Float field at xy-corners; ``NaN`` marks dry corners.
        mode : {'permissive', 'conservative'}
            See :meth:`from_u_face`.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D or Mask3D
        """
        return cls._from_point(field, "X", mode, k_bottom)


@pytree_dataclass
class Mask1D(_CartesianMask):
    """Unified Arakawa C-grid mask for a 1-D Cartesian domain.

    Stores binary masks on the two staggerings (cell centre ``h``,
    x-face ``u``) and the directional stencil capability.  The 1-D grid
    has no v/xy-corner concepts.

    Construct via one of the factory class-methods:

    * :meth:`from_mask`       — from a binary h-grid mask array.
    * :meth:`from_center`     — from a float field at cell centres
      (NaN = dry).
    * :meth:`from_u_face`     — from a float field at u-faces, with a
      ``mode=`` choice of inversion strategy.
    * :meth:`from_dimensions` — all-ocean domain of given size.

    Parameters
    ----------
    h : Bool[Array, "Nx"]
        Wet mask at cell centres (T-points).  ``True`` = wet / ocean,
        ``False`` = dry / land.  This is the canonical mask that the
        other staggered masks are derived from.
    u : Bool[Array, "Nx"]
        Wet mask at x-faces (the boundaries between adjacent cell
        centres).  An x-face is wet iff *both* of its adjacent cell
        centres are wet — i.e. flow through the face is physically
        meaningful.
    stencil_capability : StencilCapability1D
        Directional contiguous-wet-cell counts on the h-grid.
    """

    _counts: ClassVar[Any] = StencilCapability1D
    _axes: ClassVar[tuple[str, ...]] = ("x",)

    _h: Bool[Array, "Nx"]
    _u: Bool[Array, "Nx"]
    stencil_capability: StencilCapability1D

    @classmethod
    def from_dimensions(cls, nx: int) -> Mask1D:
        """Construct an all-ocean domain of given shape.

        Parameters
        ----------
        nx : int
            Total grid dimensions (including ghost cells).

        Returns
        -------
        Mask1D
        """
        return cls.from_mask(np.ones(nx, dtype=bool))


@pytree_dataclass
class Mask2D(_HorizontalMask):
    """Unified Arakawa C-grid mask for a 2-D Cartesian domain.

    Stores binary masks on all five Arakawa C-grid staggerings (cell
    centre ``h``, x-face ``u``, y-face ``v``, xy-corner lenient
    ``xy_corner``, xy-corner strict ``xy_corner_strict``), the directional
    stencil capability, and an optional sea-floor index array.  The stencil
    tiers, the coastal ghosts, the kinds of corner and the land / coast
    classification are derived from them when asked for.

    Construct via one of the factory class-methods:

    * :meth:`from_mask`       — from a binary h-grid mask array.
    * :meth:`from_center`     — from a float field at cell centres
      (NaN = dry).
    * :meth:`from_u_face`     — from a float field at u-faces, with a
      ``mode=`` choice of inversion strategy.
    * :meth:`from_v_face`     — from a float field at v-faces, with a
      ``mode=`` choice of inversion strategy.
    * :meth:`from_corner`     — from a float field at xy-corners
      (vertices), with a ``mode=`` choice of inversion strategy.
    * :meth:`from_dimensions` — all-ocean domain of given size.

    Parameters
    ----------
    h : Bool[Array, "Ny Nx"]
        Wet mask at cell centres (T-points), where tracer-like quantities
        (temperature, salinity, SSH, pressure) live.  ``True`` = wet /
        ocean, ``False`` = dry / land.  This is the canonical mask that
        the other staggered masks are derived from.
    u : Bool[Array, "Ny Nx"]
        Wet mask at x-faces (the west/east boundaries of each cell), where
        the zonal velocity ``u`` lives.  ``u[j, i]`` is the *east* face of
        ``h[j, i]`` (positive half-step / NE convention).  Wet iff *both*
        of the zonally adjacent cell centres are wet — i.e. flow through
        the face is physically meaningful.
    v : Bool[Array, "Ny Nx"]
        Wet mask at y-faces (the south/north boundaries of each cell), where
        the meridional velocity ``v`` lives.  ``v[j, i]`` is the *north*
        face of ``h[j, i]``.  Wet iff *both* of the meridionally adjacent
        cell centres are wet.
    xy_corner : Bool[Array, "Ny Nx"]
        Lenient wet mask at xy-corners (vertices).  ``True`` wherever
        *at least one* of the four surrounding cell centres is wet —
        useful for detecting any ocean presence at a corner.
    xy_corner_strict : Bool[Array, "Ny Nx"]
        Strict wet mask at xy-corners (vertices).  ``True`` only where
        *all four* surrounding cell centres are wet — useful for
        quantities (e.g. relative vorticity) that require the full
        4-point horizontal stencil to be inside the fluid.
    stencil_capability : StencilCapability2D
        Directional contiguous-wet-cell counts on the h-grid.
    k_bottom : Array or None
        Optional 2-D array of vertical sea-floor indices (3-D domains).
    """

    _counts: ClassVar[Any] = StencilCapability2D
    _axes: ClassVar[tuple[str, ...]] = ("y", "x")

    # ── staggered masks ───────────────────────────────────────────────────────
    _h: Bool[Array, "Ny Nx"]
    _u: Bool[Array, "Ny Nx"]
    _v: Bool[Array, "Ny Nx"]
    _xy_corner: Bool[Array, "Ny Nx"]
    _xy_corner_strict: Bool[Array, "Ny Nx"]

    # ── stencil capability ────────────────────────────────────────────────────
    stencil_capability: StencilCapability2D

    # ── optional arrays ───────────────────────────────────────────────────────
    k_bottom: Array | None = None

    @classmethod
    def from_dimensions(cls, ny: int, nx: int) -> Mask2D:
        """Construct an all-ocean domain of given shape.

        Parameters
        ----------
        ny, nx : int
            Total grid dimensions (including ghost cells).

        Returns
        -------
        Mask2D
        """
        return cls.from_mask(np.ones((ny, nx), dtype=bool))

    @classmethod
    def from_bathymetry(
        cls,
        depth: np.ndarray | Float[Array, "Ny Nx"],
        min_depth: float = 0.0,
        min_area: int = 1,
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a bathymetry array on the T-points.

        ``h[j, i] = depth[j, i] > min_depth``, then wet regions with fewer
        than ``min_area`` cells (4-connected) are made dry, and the rest is
        :meth:`from_mask`. ``NaN`` depths are dry.

        This is the array-level constructor only. Reading, regridding and
        cleaning a bathymetry product (lakes, spurs, coarsening) belongs in
        ``xrtoolz.geo.bathymetry_to_mask``.

        Parameters
        ----------
        depth : array-like [Ny, Nx]
            Water depth at cell centres, positive down; zero or negative on
            land.
        min_depth : float, optional
            Cells at or shallower than this depth are dry. Default ``0.0``.
        min_area : int, optional
            Wet regions smaller than this many cells are made dry. The
            default ``1`` keeps every wet cell.
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices. See :meth:`from_mask`.

        Returns
        -------
        Mask2D

        Examples
        --------
        >>> depth = np.full((6, 6), 100.0)
        >>> depth[2:4, 2:4] = -5.0  # an island
        >>> mask = Mask2D.from_bathymetry(depth, min_depth=10.0)
        >>> int(mask.wet().sum())
        32
        """
        depth_np = np.asarray(depth, dtype=float)
        with np.errstate(invalid="ignore"):
            wet = depth_np > min_depth
        if min_area > 1:
            from scipy.ndimage import label

            labels, _ = label(wet)
            sizes = np.bincount(labels.ravel())
            small = sizes < min_area
            small[0] = False  # label 0 is the dry background
            wet = wet & ~small[labels]
        return cls.from_mask(wet, k_bottom=k_bottom)


@pytree_dataclass
class Mask3D(_HorizontalMask):
    """Unified Arakawa C-grid mask for a 3-D Cartesian domain.

    Stores binary masks on the six 3-D Arakawa C-grid staggerings, the
    directional stencil capability, and an optional sea-floor index array.
    The stencil tiers, the coastal ghosts, the kinds of corner (per z-level)
    and the land / coast classification are derived from them when asked for.

    Construct via one of the factory class-methods:

    * :meth:`from_mask`       — from a binary 3-D h-grid mask array.
    * :meth:`from_center`     — from a float field at cell centres
      (NaN = dry).
    * :meth:`from_u_face`     — from a float field at u-faces, with a
      ``mode=`` choice of inversion strategy.
    * :meth:`from_v_face`     — from a float field at v-faces, with a
      ``mode=`` choice of inversion strategy.
    * :meth:`from_w_face`     — from a float field at w-faces (vertical
      velocity), with a ``mode=`` choice of inversion strategy.
    * :meth:`from_corner`     — from a float field at xy-corners
      (vertices), with a ``mode=`` choice of inversion strategy.
    * :meth:`from_dimensions` — all-ocean domain of given size.

    Parameters
    ----------
    h : Bool[Array, "Nz Ny Nx"]
        Wet mask at cell centres (T-points), where tracer-like quantities
        (temperature, salinity, pressure) live.  ``True`` = wet / ocean,
        ``False`` = dry / land.  This is the canonical mask that the
        other staggered masks are derived from.
    u : Bool[Array, "Nz Ny Nx"]
        Wet mask at x-faces (the west/east boundaries of each cell), where
        the zonal velocity ``u`` lives.  ``u[k, j, i]`` is the *east* face
        of ``h[k, j, i]`` (positive half-step / NE convention).  Wet iff
        *both* of the zonally adjacent cell centres at the same z-level
        are wet.
    v : Bool[Array, "Nz Ny Nx"]
        Wet mask at y-faces (the south/north boundaries of each cell), where
        the meridional velocity ``v`` lives.  ``v[k, j, i]`` is the *north*
        face of ``h[k, j, i]``.  Wet iff *both* of the meridionally adjacent
        cell centres at the same z-level are wet.
    w : Bool[Array, "Nz Ny Nx"]
        Wet mask at z-faces (the top/bottom boundaries of each cell), where
        the vertical velocity ``w`` lives.  ``w[k, j, i]`` is the *top*
        face of cell ``(k, j, i)`` (positive half-step convention, matching
        ``u`` and ``v``).  Wet iff *both* of the vertically adjacent cell
        centres at the same horizontal position are wet.  Stored at the
        same shape as ``h`` (no extra ``Nz+1`` layer — the bottom surface
        of the lowest interior cell is implicit in the bottom ghost shell).
    xy_corner : Bool[Array, "Nz Ny Nx"]
        Lenient wet mask at xy-corners (vertices), per z-level.
        ``True`` wherever *at least one* of the four surrounding cell
        centres at the same z-level is wet — useful for detecting any
        ocean presence at a corner.
    xy_corner_strict : Bool[Array, "Nz Ny Nx"]
        Strict wet mask at xy-corners (vertices), per z-level.
        ``True`` only where *all four* surrounding cell centres at the
        same z-level are wet — useful for quantities (e.g. relative
        vorticity) that require the full 4-point horizontal stencil to
        be inside the fluid.
    stencil_capability : StencilCapability3D
        Directional contiguous-wet-cell counts on the h-grid (x, y, z).
    k_bottom : Array or None
        Optional 2-D ``[Ny, Nx]`` array of vertical sea-floor indices.
    """

    _counts: ClassVar[Any] = StencilCapability3D
    _axes: ClassVar[tuple[str, ...]] = ("z", "y", "x")

    # ── staggered masks ───────────────────────────────────────────────────────
    _h: Bool[Array, "Nz Ny Nx"]
    _u: Bool[Array, "Nz Ny Nx"]
    _v: Bool[Array, "Nz Ny Nx"]
    _w: Bool[Array, "Nz Ny Nx"]
    _xy_corner: Bool[Array, "Nz Ny Nx"]
    _xy_corner_strict: Bool[Array, "Nz Ny Nx"]

    # ── stencil capability ────────────────────────────────────────────────────
    stencil_capability: StencilCapability3D

    # ── optional arrays ───────────────────────────────────────────────────────
    k_bottom: Array | None = None

    @classmethod
    def from_w_face(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a field at w-faces, ``NaN`` where dry.

        This is the natural constructor for vertical-velocity fields stored at
        z-faces.

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at w-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            See :meth:`from_u_face`.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        return cls._from_point(field, "W", mode, k_bottom)

    @classmethod
    def from_dimensions(cls, nz: int, ny: int, nx: int) -> Mask3D:
        """Construct an all-ocean domain of given shape.

        Parameters
        ----------
        nz, ny, nx : int
            Total grid dimensions (including ghost cells).

        Returns
        -------
        Mask3D
        """
        return cls.from_mask(np.ones((nz, ny, nx), dtype=bool))
