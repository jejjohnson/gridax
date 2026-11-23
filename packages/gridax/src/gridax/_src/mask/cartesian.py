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
from typing import TYPE_CHECKING, Any, ClassVar

from jaxtyping import Bool, Float
import numpy as np

from gridax._src.grid.points import Stagger, point_axes
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


def _from_grid(from_mask: Any, grid: Any, wet_t: Any, **kwargs: Any) -> Any:
    """``from_mask`` on the shape of a grid, with T alone if it is collocated."""
    if np.shape(wet_t) != grid.shape:
        raise ValueError(
            f"wet_t has shape {np.shape(wet_t)}, and the grid {grid.shape}"
        )
    mask = from_mask(wet_t, **kwargs)
    if not getattr(grid, "_collocated", False):
        return mask
    staggered = [
        field.name
        for field in dataclasses.fields(mask)
        if field.name.startswith("_") and field.name != "_h"
    ]
    return dataclasses.replace(mask, **dict.fromkeys(staggered))


# ──────────────────────────────────────────────────────────────────────────────
# Mask1D
# ──────────────────────────────────────────────────────────────────────────────


@pytree_dataclass
class Mask1D(MaskBase):
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

    _axes: ClassVar[tuple[str, ...]] = ("x",)

    _h: Bool[Array, "Nx"]
    _u: Bool[Array, "Nx"]
    stencil_capability: StencilCapability1D

    # ── wet points ───────────────────────────────────────────────────────────

    def wet(
        self, *, at: str | Stagger = "T", kind: str = "default"
    ) -> Bool[Array, "Nx"]:
        """The wet points at a point of the staggering; ``True`` = wet.

        Parameters
        ----------
        at : str or Stagger
            The point, ``"T"`` or ``"U"``, or a field's stagger (``CField.stagger``).
        kind : str
            ``"default"``, the only one in 1-D.

        Returns
        -------
        Bool[Array, "Nx"]

        Raises
        ------
        ValueError
            If the mask has no such point or kind.

        Examples
        --------
        >>> import numpy as np
        >>> from gridax import Mask1D
        >>> mask = Mask1D.from_mask(np.ones((4,), dtype=bool))
        >>> int(mask.wet().sum()), int(mask.wet(at="U").sum())
        (4, 3)
        """
        return _wet(self, self._axes, at, kind)

    # ── factory class-methods ─────────────────────────────────────────────────

    @classmethod
    def from_mask(
        cls,
        mask_hgrid: np.ndarray | Bool[Array, "Nx"],
    ) -> Mask1D:
        """Construct from a binary h-grid (cell-centre) mask.

        Parameters
        ----------
        mask_hgrid : array-like [Nx]
            Binary wet (1 / True) / dry (0 / False) mask at cell centres.

        Returns
        -------
        Mask1D

        Notes
        -----
        The staggered u-face mask is derived from ``mask_hgrid`` via
        :func:`pool_bool` with the following kernel/threshold pair
        (trailing-pad direction → ``u[i] = h[i] AND h[i+1]``):

        ====== ========= =========  ====================================
        Mask   Kernel    Threshold  Semantics
        ====== ========= =========  ====================================
        ``u``  ``(2,)``  ``3/4``    both adjacent h-cells wet (strict)
        ====== ========= =========  ====================================

        The stencil capability is built via
        :meth:`StencilCapability1D.from_mask`.
        """
        h_np = np.asarray(mask_hgrid, dtype=bool)
        hf = h_np.astype(np.float32)

        # Staggered u-face (east face of T[i]): kernel (2,) trailing
        # → u[i] = (h[i] + h[i+1]) / 2 > 3/4 (both adjacent h-cells wet)
        u_np = pool_bool(hf, kernel=(2,), threshold=3.0 / 4.0, direction="trailing")

        # ── stencil capability ────────────────────────────────────────────
        sc = StencilCapability1D.from_mask(h_np)

        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "_h": h_np,
            "_u": u_np,
            "stencil_capability": sc,
        }
        return cls(**fields)

    @classmethod
    def from_grid(
        cls,
        grid: Any,
        wet_t: np.ndarray | Bool[Array, "Nx"],
    ) -> Mask1D:
        """Construct from the wet T points of a grid.

        Parameters
        ----------
        grid : CartesianCGrid1D
            The grid the mask is for.
        wet_t : array-like [Nx]
            Wet (``True``) / dry mask at the T points, ghost ring included.

        Returns
        -------
        Mask1D

        Raises
        ------
        ValueError
            If the shape of ``wet_t`` is not the grid's.

        Examples
        --------
        >>> import numpy as np
        >>> from gridax import CartesianCGrid1D, Mask1D
        >>> grid = CartesianCGrid1D.from_interior(4, 1.0)
        >>> Mask1D.from_grid(grid, np.ones(grid.shape)).wet().shape == grid.shape
        True
        """
        return _from_grid(cls.from_mask, grid, wet_t)

    @classmethod
    def from_center(
        cls,
        field: Float[Array, "Nx"],
    ) -> Mask1D:
        """Construct from a field at cell centres (T-points).

        ``NaN`` values are treated as dry; finite values as wet.  This
        is the natural constructor for any tracer-like quantity (SSH,
        temperature, salinity, pressure, …) that lives at cell centres.

        Parameters
        ----------
        field : array-like [Nx]
            Float field at cell centres; ``NaN`` marks dry cells.

        Returns
        -------
        Mask1D
        """
        h_mask = np.isfinite(np.asarray(field))
        return cls.from_mask(h_mask)

    @classmethod
    def from_u_face(
        cls,
        field: Float[Array, "Nx"],
        mode: str = "permissive",
    ) -> Mask1D:
        """Construct from a field at u-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry u-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel ``(2,)``.

        Parameters
        ----------
        field : array-like [Nx]
            Float field at u-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  ``'permissive'`` (default) marks an
            h-cell wet if *any* of its two adjacent u-faces is wet
            (logical OR).  ``'conservative'`` requires *all* adjacent
            u-faces to be wet (logical AND).

        Returns
        -------
        Mask1D
        """
        u_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(u_mask, kernel=(2,), mode=mode, direction="trailing")
        return cls.from_mask(h_mask)

    @classmethod
    def from_dimensions(
        cls,
        nx: int,
    ) -> Mask1D:
        """Construct an all-ocean 1-D domain of given length.

        Parameters
        ----------
        nx : int
            Total grid length (including ghost cells).

        Returns
        -------
        Mask1D
        """
        return cls.from_mask(np.ones(nx, dtype=bool))


# ──────────────────────────────────────────────────────────────────────────────
# Mask2D
# ──────────────────────────────────────────────────────────────────────────────


@pytree_dataclass
class Mask2D(MaskBase):
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

    # ── wet points ───────────────────────────────────────────────────────────

    def wet(
        self, *, at: str | Stagger = "T", kind: str = "default"
    ) -> Bool[Array, "Ny Nx"]:
        """The wet points at a point of the staggering; ``True`` = wet.

        Parameters
        ----------
        at : str or Stagger
            The point, ``"T"``, ``"U"``, ``"V"`` or ``"X"``, or a field's stagger (``CField.stagger``).
        kind : str
            At X only: ``"default"`` (at least one of the four cells is wet),
            ``"strict"`` (all four are), ``"y_wall"``, ``"x_wall"``,
            ``"convex"`` or ``"valid"``.

        Returns
        -------
        Bool[Array, "Ny Nx"]

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

    # ── factory class-methods ─────────────────────────────────────────────────

    @classmethod
    def from_mask(
        cls,
        mask_hgrid: np.ndarray | Bool[Array, "Ny Nx"],
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a binary h-grid (cell-centre) mask.

        All intermediate computations use numpy, and so are the stored
        arrays.

        Parameters
        ----------
        mask_hgrid : array-like [Ny, Nx]
            Binary wet (1 / True) / dry (0 / False) mask at cell centres.
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices for 3-D domains.

        Returns
        -------
        Mask2D

        Notes
        -----
        The four staggered masks are derived from ``mask_hgrid`` via
        :func:`pool_bool` with the following kernel/threshold pairs
        (trailing-pad direction → positive half-step convention):

        ==================== ========== =========  ====================================
        Mask                 Kernel     Threshold  Semantics
        ==================== ========== =========  ====================================
        ``u`` (east face)    ``(1, 2)`` ``3/4``    ``h[j, i] AND h[j, i+1]``
        ``v`` (north face)   ``(2, 1)`` ``3/4``    ``h[j, i] AND h[j+1, i]``
        ``xy_corner``        ``(2, 2)`` ``1/8``    ≥ 1 of 4 NE-corner h-cells wet
        ``xy_corner_strict`` ``(2, 2)`` ``7/8``    all 4 NE-corner h-cells wet
        ==================== ========== =========  ====================================

        The stencil capability is built via
        :meth:`StencilCapability2D.from_mask`.
        """
        h_np = np.asarray(mask_hgrid, dtype=bool)
        hf = h_np.astype(np.float32)

        # ── staggered masks ───────────────────────────────────────────────
        # All staggered masks use the trailing-pad (positive half-step)
        # convention, matching the grid module's same-index rule:
        # U[j, i] at east face (i+1/2), V[j, i] at north face (j+1/2),
        # X[j, i] at NE corner (i+1/2, j+1/2).

        # u[j, i] = (h[j, i] + h[j, i+1]) / 2 > 3/4  (east face = U-point)
        u_np = pool_bool(hf, kernel=(1, 2), threshold=3.0 / 4.0, direction="trailing")
        # v[j, i] = (h[j, i] + h[j+1, i]) / 2 > 3/4  (north face = V-point)
        v_np = pool_bool(hf, kernel=(2, 1), threshold=3.0 / 4.0, direction="trailing")
        # xy_corner[j, i]: at least 1 of 4 NE-corner h-cells wet  (lenient)
        xy_corner_np = pool_bool(
            hf, kernel=(2, 2), threshold=1.0 / 8.0, direction="trailing"
        )
        # xy_corner_strict[j, i]: all 4 NE-corner h-cells wet     (strict)
        xy_corner_strict_np = pool_bool(
            hf, kernel=(2, 2), threshold=7.0 / 8.0, direction="trailing"
        )

        # ── stencil capability ────────────────────────────────────────────
        sc = StencilCapability2D.from_mask(h_np)

        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "_h": h_np,
            "_u": u_np,
            "_v": v_np,
            "_xy_corner": xy_corner_np,
            "_xy_corner_strict": xy_corner_strict_np,
            "stencil_capability": sc,
            "k_bottom": np.asarray(k_bottom) if k_bottom is not None else None,
        }
        return cls(**fields)

    @classmethod
    def from_grid(
        cls,
        grid: Any,
        wet_t: np.ndarray | Bool[Array, "Ny Nx"],
        *,
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from the wet T points of a grid.

        Parameters
        ----------
        grid : CartesianCGrid2D or SphericalCGrid2D
            The grid the mask is for.
        wet_t : array-like [Ny Nx]
            Wet (``True``) / dry mask at the T points, ghost ring included.
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices. See :meth:`from_mask`.

        Returns
        -------
        Mask2D

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
        return _from_grid(cls.from_mask, grid, wet_t, k_bottom=k_bottom)

    @classmethod
    def from_center(
        cls,
        field: Float[Array, "Ny Nx"],
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a field at cell centres (T-points).

        ``NaN`` values are treated as dry; finite values as wet.  This
        is the natural constructor for any tracer-like quantity (SSH,
        temperature, salinity, pressure, …) that lives at cell centres.

        Parameters
        ----------
        field : array-like [Ny, Nx]
            Float field at cell centres; ``NaN`` marks dry cells.
        k_bottom : array-like, optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D
        """
        h_mask = np.isfinite(np.asarray(field))
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_u_face(
        cls,
        field: Float[Array, "Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a field at u-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry u-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel ``(1, 2)``
        (trailing-pad direction, matching the grid's positive-half-step
        convention) and the requested ``mode``; the remaining staggered
        masks are produced by :meth:`from_mask`.

        Parameters
        ----------
        field : array-like [Ny, Nx]
            Float field at u-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like, optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D
        """
        u_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(u_mask, kernel=(1, 2), mode=mode, direction="trailing")
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_v_face(
        cls,
        field: Float[Array, "Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a field at v-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry v-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel ``(2, 1)``
        (trailing-pad direction) and the requested ``mode``.

        Parameters
        ----------
        field : array-like [Ny, Nx]
            Float field at v-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like, optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D
        """
        v_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(v_mask, kernel=(2, 1), mode=mode, direction="trailing")
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_corner(
        cls,
        field: Float[Array, "Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a field at xy-corners (vertices).

        ``NaN`` values are treated as dry corners.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel ``(2, 2)``
        (trailing-pad direction) and the requested ``mode``.  This is
        the natural constructor for vertex-stored quantities such as
        relative vorticity.

        Parameters
        ----------
        field : array-like [Ny, Nx]
            Float field at xy-corners; ``NaN`` marks dry corners.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like, optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D
        """
        c_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(c_mask, kernel=(2, 2), mode=mode, direction="trailing")
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_dimensions(
        cls,
        ny: int,
        nx: int,
    ) -> Mask2D:
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


# ──────────────────────────────────────────────────────────────────────────────
# Mask3D
# ──────────────────────────────────────────────────────────────────────────────


@pytree_dataclass
class Mask3D(MaskBase):
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

    # ── wet points ───────────────────────────────────────────────────────────

    def wet(
        self, *, at: str | Stagger = "T", kind: str = "default"
    ) -> Bool[Array, "Nz Ny Nx"]:
        """The wet points at a point of the staggering; ``True`` = wet.

        Parameters
        ----------
        at : str or Stagger
            The point, ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``, or a field's stagger (``CField.stagger``).
        kind : str
            At X only: ``"default"`` (at least one of the four cells is wet),
            ``"strict"`` (all four are), ``"y_wall"``, ``"x_wall"``,
            ``"convex"`` or ``"valid"``.

        Returns
        -------
        Bool[Array, "Nz Ny Nx"]

        Raises
        ------
        ValueError
            If the mask has no such point or kind.

        Examples
        --------
        >>> import numpy as np
        >>> from gridax import Mask3D
        >>> mask = Mask3D.from_mask(np.ones((3, 4, 4), dtype=bool))
        >>> int(mask.wet().sum()), int(mask.wet(at="U").sum())
        (48, 36)
        """
        return _wet(self, self._axes, at, kind)

    # ── factory class-methods ─────────────────────────────────────────────────

    @classmethod
    def from_mask(
        cls,
        mask_hgrid: np.ndarray | Bool[Array, "Nz Ny Nx"],
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a binary 3-D h-grid (cell-centre) mask.

        Parameters
        ----------
        mask_hgrid : array-like [Nz, Ny, Nx]
            Binary wet (1 / True) / dry (0 / False) mask at cell centres.
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices (one per horizontal column).

        Returns
        -------
        Mask3D

        Notes
        -----
        The five staggered masks are derived from ``mask_hgrid`` via
        :func:`pool_bool` with the following kernel/threshold pairs
        (trailing-pad direction → positive half-step convention):

        ==================== ============= =========  =========================================
        Mask                 Kernel        Threshold  Semantics
        ==================== ============= =========  =========================================
        ``u`` (east face)    ``(1, 1, 2)`` ``3/4``    ``h[k, j, i] AND h[k, j, i+1]``
        ``v`` (north face)   ``(1, 2, 1)`` ``3/4``    ``h[k, j, i] AND h[k, j+1, i]``
        ``w`` (top z-face)   ``(2, 1, 1)`` ``3/4``    ``h[k, j, i] AND h[k+1, j, i]``
        ``xy_corner``        ``(1, 2, 2)`` ``1/8``    ≥ 1 of 4 NE-corner h-cells wet (per k)
        ``xy_corner_strict`` ``(1, 2, 2)`` ``7/8``    all 4 NE-corner h-cells wet (per k)
        ==================== ============= =========  =========================================

        The stencil capability is built via
        :meth:`StencilCapability3D.from_mask`.
        """
        h_np = np.asarray(mask_hgrid, dtype=bool)
        hf = h_np.astype(np.float32)

        # ── staggered masks ───────────────────────────────────────────────
        # All staggered masks use the trailing-pad (positive half-step)
        # convention, matching the grid module's same-index rule.

        # u[k, j, i] = (h[k, j, i] + h[k, j, i+1]) / 2 > 3/4  (east face = U-point)
        u_np = pool_bool(
            hf, kernel=(1, 1, 2), threshold=3.0 / 4.0, direction="trailing"
        )
        # v[k, j, i] = (h[k, j, i] + h[k, j+1, i]) / 2 > 3/4  (north face = V-point)
        v_np = pool_bool(
            hf, kernel=(1, 2, 1), threshold=3.0 / 4.0, direction="trailing"
        )
        # w[k, j, i] = (h[k, j, i] + h[k+1, j, i]) / 2 > 3/4  (top z-face)
        w_np = pool_bool(
            hf, kernel=(2, 1, 1), threshold=3.0 / 4.0, direction="trailing"
        )
        # xy_corner[k, j, i]: at least 1 of 4 NE-corner h-cells wet (lenient)
        xy_corner_np = pool_bool(
            hf, kernel=(1, 2, 2), threshold=1.0 / 8.0, direction="trailing"
        )
        # xy_corner_strict: all 4 NE-corner h-cells wet (strict)
        xy_corner_strict_np = pool_bool(
            hf, kernel=(1, 2, 2), threshold=7.0 / 8.0, direction="trailing"
        )

        # ── stencil capability (3-D, includes z) ──────────────────────────
        sc = StencilCapability3D.from_mask(h_np)

        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "_h": h_np,
            "_u": u_np,
            "_v": v_np,
            "_w": w_np,
            "_xy_corner": xy_corner_np,
            "_xy_corner_strict": xy_corner_strict_np,
            "stencil_capability": sc,
            "k_bottom": np.asarray(k_bottom) if k_bottom is not None else None,
        }
        return cls(**fields)

    @classmethod
    def from_grid(
        cls,
        grid: Any,
        wet_t: np.ndarray | Bool[Array, "Nz Ny Nx"],
        *,
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from the wet T points of a grid.

        Parameters
        ----------
        grid : CartesianCGrid3D or SphericalCGrid3D
            The grid the mask is for.
        wet_t : array-like [Nz Ny Nx]
            Wet (``True``) / dry mask at the T points, ghost ring included.
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices. See :meth:`from_mask`.

        Returns
        -------
        Mask3D

        Raises
        ------
        ValueError
            If the shape of ``wet_t`` is not the grid's.

        Examples
        --------
        >>> import numpy as np
        >>> from gridax import CartesianCGrid3D, Mask3D
        >>> grid = CartesianCGrid3D.from_interior(4, 4, 2, 1.0, 1.0, 1.0)
        >>> Mask3D.from_grid(grid, np.ones(grid.shape)).wet().shape == grid.shape
        True
        """
        return _from_grid(cls.from_mask, grid, wet_t, k_bottom=k_bottom)

    @classmethod
    def from_center(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a 3-D field at cell centres (T-points).

        ``NaN`` values are treated as dry; finite values as wet.

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at cell centres; ``NaN`` marks dry cells.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        h_mask = np.isfinite(np.asarray(field))
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_u_face(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a 3-D field at u-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry u-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel
        ``(1, 1, 2)`` (trailing-pad direction) and the requested
        ``mode``.

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at u-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        u_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(
            u_mask, kernel=(1, 1, 2), mode=mode, direction="trailing"
        )
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_v_face(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a 3-D field at v-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry v-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel
        ``(1, 2, 1)`` (trailing-pad direction) and the requested
        ``mode``.

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at v-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        v_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(
            v_mask, kernel=(1, 2, 1), mode=mode, direction="trailing"
        )
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_w_face(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a 3-D field at w-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry w-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel
        ``(2, 1, 1)`` (trailing-pad direction) and the requested
        ``mode``.  This is the natural constructor for vertical-velocity
        fields stored at z-faces.

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at w-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        w_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(
            w_mask, kernel=(2, 1, 1), mode=mode, direction="trailing"
        )
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_corner(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a 3-D field at xy-corners (vertices).

        ``NaN`` values are treated as dry corners.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel
        ``(1, 2, 2)`` (trailing-pad direction) and the requested
        ``mode``.  Corners in 3-D are per-z-level (no z-staggering on
        corners).

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at xy-corners; ``NaN`` marks dry corners.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        c_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(
            c_mask, kernel=(1, 2, 2), mode=mode, direction="trailing"
        )
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_dimensions(
        cls,
        nz: int,
        ny: int,
        nx: int,
    ) -> Mask3D:
        """Construct an all-ocean 3-D domain of given shape.

        Parameters
        ----------
        nz, ny, nx : int
            Total grid dimensions (including ghost cells).

        Returns
        -------
        Mask3D
        """
        return cls.from_mask(np.ones((nz, ny, nx), dtype=bool))
