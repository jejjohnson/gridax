"""Shared mask infrastructure: StencilCapability1D / 2D / 3D, and `MaskBase`.

These small data classes record, at every grid cell, the number of
contiguous wet neighbours reachable in each axis-aligned direction
without crossing a dry cell or the domain edge.  They are stored on
the dimensional ``Mask*`` classes and consumed by adaptive WENO /
TVD reconstruction code in the advection module.

Construction is numpy-only (the recurrence does not vectorise without
extra machinery, and masks are built once at setup time, not inside
JIT-traced step functions).  Stored arrays are NumPy int8: a count
saturates at 127, far above the widest stencil half-width (5), so every
consumer's comparison ``count >= half_width`` is unchanged.

`MaskBase` holds what the three masks derive from their stored arrays: the
stencil tiers, the coastal ghosts and the land / coast classification. None of
it is stored, so it adds nothing to the pytree, and all of it works on a
traced mask.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from jaxtyping import Bool, Int
import numpy as np

from gridax._src.grid.points import Stagger, point_axes
from gridax._src.mask.utils import count_contiguous, dilate_mask, shifted
from gridax._src.pytree import array_namespace, pytree_dataclass, static_field

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

#: Counts saturate here; it bounds the int8 storage, not any stencil.
_COUNT_CAP = np.iinfo(np.int8).max


def _counts(h: np.ndarray, axis: int, forward: bool) -> np.ndarray:
    """Contiguous wet counts along ``axis``, saturated at 127, as int8."""
    counts = count_contiguous(h, axis=axis, forward=forward)
    return np.minimum(counts, _COUNT_CAP).astype(np.int8)


@pytree_dataclass
class StencilCapability1D:
    """Directional count of contiguous wet neighbours for each 1-D grid cell.

    At each cell ``i``, stores the number of consecutive wet cells (including
    the cell itself) reachable before hitting a dry cell or the domain edge.

    Parameters
    ----------
    x_pos : Int[Array, "Nx"]
        Count in the +x direction.
    x_neg : Int[Array, "Nx"]
        Count in the −x direction.
    """

    x_pos: Int[Array, "Nx"]
    x_neg: Int[Array, "Nx"]

    @classmethod
    def from_mask(cls, h: np.ndarray | Bool[Array, "Nx"]) -> StencilCapability1D:
        """Build stencil capability from a 1-D wet/dry mask.

        Parameters
        ----------
        h : array-like [Nx] bool

        Returns
        -------
        StencilCapability1D
        """
        h_np = np.asarray(h, dtype=bool)
        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "x_pos": _counts(h_np, axis=0, forward=True),
            "x_neg": _counts(h_np, axis=0, forward=False),
        }
        return cls(**fields)


@pytree_dataclass
class StencilCapability2D:
    """Directional count of contiguous wet neighbours for each 2-D grid cell.

    At each cell ``(j, i)``, stores the number of consecutive wet cells
    (including the cell itself) reachable before hitting a dry cell or
    the domain edge.

    Parameters
    ----------
    x_pos : Int[Array, "Ny Nx"]
        Count in the +x direction.
    x_neg : Int[Array, "Ny Nx"]
        Count in the −x direction.
    y_pos : Int[Array, "Ny Nx"]
        Count in the +y direction.
    y_neg : Int[Array, "Ny Nx"]
        Count in the −y direction.
    """

    x_pos: Int[Array, "Ny Nx"]
    x_neg: Int[Array, "Ny Nx"]
    y_pos: Int[Array, "Ny Nx"]
    y_neg: Int[Array, "Ny Nx"]

    @classmethod
    def from_mask(cls, h: np.ndarray | Bool[Array, "Ny Nx"]) -> StencilCapability2D:
        """Build stencil capability from a 2-D wet/dry mask.

        Construction uses numpy; stored arrays are NumPy int8.

        Parameters
        ----------
        h : array-like [Ny, Nx] bool
            Wet (True) / dry (False) mask.

        Returns
        -------
        StencilCapability2D
        """
        h_np = np.asarray(h, dtype=bool)
        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "x_pos": _counts(h_np, axis=1, forward=True),
            "x_neg": _counts(h_np, axis=1, forward=False),
            "y_pos": _counts(h_np, axis=0, forward=True),
            "y_neg": _counts(h_np, axis=0, forward=False),
        }
        return cls(**fields)


@pytree_dataclass
class StencilCapability3D:
    """Directional count of contiguous wet neighbours for each 3-D grid cell.

    At each cell ``(k, j, i)``, stores the number of consecutive wet cells
    (including the cell itself) reachable before hitting a dry cell or
    the domain edge.

    Parameters
    ----------
    x_pos : Int[Array, "Nz Ny Nx"]
        Count in the +x direction.
    x_neg : Int[Array, "Nz Ny Nx"]
        Count in the −x direction.
    y_pos : Int[Array, "Nz Ny Nx"]
        Count in the +y direction.
    y_neg : Int[Array, "Nz Ny Nx"]
        Count in the −y direction.
    z_pos : Int[Array, "Nz Ny Nx"]
        Count in the +z direction.
    z_neg : Int[Array, "Nz Ny Nx"]
        Count in the −z direction.
    """

    x_pos: Int[Array, "Nz Ny Nx"]
    x_neg: Int[Array, "Nz Ny Nx"]
    y_pos: Int[Array, "Nz Ny Nx"]
    y_neg: Int[Array, "Nz Ny Nx"]
    z_pos: Int[Array, "Nz Ny Nx"]
    z_neg: Int[Array, "Nz Ny Nx"]

    @classmethod
    def from_mask(cls, h: np.ndarray | Bool[Array, "Nz Ny Nx"]) -> StencilCapability3D:
        """Build stencil capability from a 3-D wet/dry mask.

        Parameters
        ----------
        h : array-like [Nz, Ny, Nx] bool

        Returns
        -------
        StencilCapability3D
        """
        h_np = np.asarray(h, dtype=bool)
        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "x_pos": _counts(h_np, axis=2, forward=True),
            "x_neg": _counts(h_np, axis=2, forward=False),
            "y_pos": _counts(h_np, axis=1, forward=True),
            "y_neg": _counts(h_np, axis=1, forward=False),
            "z_pos": _counts(h_np, axis=0, forward=True),
            "z_neg": _counts(h_np, axis=0, forward=False),
        }
        return cls(**fields)


#: The tier sizes of every scheme up to order 9.
TIER_SIZES: tuple[int, ...] = (2, 4, 6, 8, 10)

#: The kinds of corner that are derived from the faces around it.
WALL_KINDS: tuple[str, ...] = ("y_wall", "x_wall", "convex", "valid")


@pytree_dataclass
class CoastGhosts:
    """The dry points one side of a coast, and where the water is from them.

    Parameters
    ----------
    points : Bool[Array, "*grid"]
        ``True`` on the dry points.
    offset : int
        Cells along the axis from such a point to the nearest wet one:
        negative if the water is on the low side.
    """

    points: Bool[Array, "*grid"]
    offset: int = static_field()


class MaskBase:
    """What `Mask1D`, `Mask2D` and `Mask3D` derive from the arrays they store."""

    #: The axes of the arrays.
    _axes: ClassVar[tuple[str, ...]]
    _h: Any
    stencil_capability: Any

    def wet(self, *, at: str | Stagger = "T", kind: str = "default") -> Any:
        raise NotImplementedError

    def _position(self, axis: str) -> int:
        if axis not in self._axes:
            raise ValueError(f"axis must be one of {self._axes}, got {axis!r}")
        return self._axes.index(axis) - len(self._axes)

    # ── stencil tiers ─────────────────────────────────────────────────────────

    def tiers(
        self,
        axis: str,
        *,
        at: str | Stagger = "T",
        kind: str = "default",
        sizes: Sequence[int] = TIER_SIZES,
    ) -> dict[int, Bool[Array, "*grid"]]:
        """The widest stencil each point has room for on wet points.

        A symmetric stencil of size ``s`` fits at a point if the ``s/2``
        points on either side of it, itself included, are wet and inside the
        array. Each point is in the tier of the largest size that fits, and in
        none if it is dry.

        At T the tiers are read off the stored counts
        (``stencil_capability``), so any sizes cost a comparison each. At the
        other points they are counted from :meth:`wet`.

        Parameters
        ----------
        axis : str
            The axis of the stencil: ``"x"``, ``"y"`` or ``"z"``.
        at : str or Stagger
            The point of the stencil.
        kind : str
            The kind of wet point, as in :meth:`wet`.
        sizes : sequence of int
            The sizes, even. The default covers every scheme up to order 9:
            ``Scheme.tiers()`` has sizes ``2 halo + 2``.

        Returns
        -------
        dict of int to Bool[Array, "*grid"]
            The points of each size; they exclude each other.

        Raises
        ------
        ValueError
            If the axis, the point or a size is not valid.

        Examples
        --------
        >>> import numpy as np
        >>> from gridax import Mask1D
        >>> tiers = Mask1D.from_mask([0, 1, 1, 1, 1, 1, 0]).tiers("x", sizes=(2, 4))
        >>> tiers[2].astype(int).tolist(), tiers[4].astype(int).tolist()
        ([0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 0, 0])
        """
        ordered = sorted(set(sizes))
        if not ordered or any(s < 2 or s % 2 for s in ordered):
            raise ValueError(f"sizes must be even and at least 2, got {tuple(sizes)}")
        reach = self._reach(axis, at, kind, ordered[-1] // 2)
        out = {}
        for s, wider in zip(ordered, [*ordered[1:], None], strict=True):
            fits = reach >= s // 2
            out[s] = fits if wider is None else fits & (reach < wider // 2)
        return {s: out[s] for s in sizes}

    def _reach(
        self, axis: str, at: str | Stagger, kind: str, upto: int
    ) -> Int[Array, "*grid"]:
        """How many wet points there are on either side of each, itself included.

        Counted up to at least ``upto``, as int8.
        """
        position = self._position(axis)
        if not point_axes(at, self._axes) and kind == "default":
            counts = self.stencil_capability
            pos, neg = getattr(counts, f"{axis}_pos"), getattr(counts, f"{axis}_neg")
            return array_namespace(pos, neg).minimum(pos, neg)
        wet = self.wet(at=at, kind=kind)
        xp = array_namespace(wet)
        fits = wet
        reach = xp.asarray(fits, dtype=xp.int8)
        for k in range(1, upto):
            fits = fits & shifted(wet, position, k) & shifted(wet, position, -k)
            reach = reach + xp.asarray(fits, dtype=xp.int8)
        return reach

    # ── coastal ghosts ────────────────────────────────────────────────────────

    def coast_ghosts(
        self,
        axis: str,
        side: str,
        *,
        at: str | Stagger = "T",
        kind: str = "default",
        layer: int = 1,
    ) -> CoastGhosts:
        """The dry points behind a coast, as the ghost points of a domain edge.

        A coast on the high side of the water (``side="high"``) has the wet
        point ``i - 1`` and the dry one ``i``. Layer ``k`` holds the dry points
        with ``k - 1`` dry points between them and the water, as layer ``k`` of
        the ghost ring does, so a boundary condition can write them from the
        same wet points: ``φ_m`` is ``offset`` and ``m - 1`` more cells away.

        Parameters
        ----------
        axis : str
            The axis normal to the coast: ``"x"``, ``"y"`` or ``"z"``.
        side : {"low", "high"}
            The side of the water the coast is on.
        at : str or Stagger
            The point of the field.
        kind : str
            The kind of wet point, as in :meth:`wet`.
        layer : int
            The layer, from 1.

        Returns
        -------
        CoastGhosts
            The dry points, and the offset to the nearest wet one.

        Raises
        ------
        ValueError
            If the axis, the side, the point or the layer is not valid.

        Examples
        --------
        >>> from gridax import Mask1D
        >>> ghosts = Mask1D.from_mask([0, 1, 1, 0, 0, 0]).coast_ghosts("x", "high")
        >>> ghosts.points.astype(int).tolist(), ghosts.offset
        ([0, 0, 0, 1, 0, 0], -1)
        """
        position = self._position(axis)
        if side not in ("low", "high"):
            raise ValueError(f"side must be 'low' or 'high', got {side!r}")
        if layer < 1:
            raise ValueError(f"layer must be at least 1, got {layer}")
        step = 1 if side == "low" else -1
        wet = self.wet(at=at, kind=kind)
        points = shifted(wet, position, step * layer)
        for k in range(layer):
            points = points & ~shifted(wet, position, step * k)
        return CoastGhosts(points=points, offset=step * layer)

    # ── land / coast classification ───────────────────────────────────────────

    @property
    def classification(self) -> Int[Array, "*grid"]:
        """0 on land, 1 on the coast, 2 near the coast, 3 in the open ocean.

        A coast cell has land on one of its faces, and a near-coast cell a
        coast cell.
        """
        xp = array_namespace(self._h)
        near_land = dilate_mask(~self._h)
        levels = (self._h, self._h & ~near_land, self._h & ~dilate_mask(near_land))
        return sum(xp.asarray(level, dtype=xp.int8) for level in levels)

    @property
    def ind_land(self) -> Bool[Array, "*grid"]:
        """The land cells (``classification == 0``)."""
        return self.classification == 0

    @property
    def ind_coast(self) -> Bool[Array, "*grid"]:
        """The coast cells (``classification == 1``)."""
        return self.classification == 1

    @property
    def ind_near_coast(self) -> Bool[Array, "*grid"]:
        """The near-coast cells (``classification == 2``)."""
        return self.classification == 2

    @property
    def ind_ocean(self) -> Bool[Array, "*grid"]:
        """The open-ocean cells (``classification == 3``)."""
        return self.classification == 3

    @property
    def ind_boundary(self) -> Bool[Array, "*grid"]:
        """The outermost ring of the array."""
        out = np.ones(self._h.shape, dtype=bool)
        out[tuple(slice(1, -1) for _ in self._h.shape)] = False
        return out
