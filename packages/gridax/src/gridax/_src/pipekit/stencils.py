"""Stencil, interpolation, boundary and mask operators (2-D Cartesian).

Each operator is one Tier-A call. Fields are ``[*batch, Ny, Nx]`` with the
ghost ring included; outputs follow Tier A exactly (interior written, ghost
ring zero, unless the operator fills it).

Staggering is carried by configuration, not checked on the carrier::

    Difference(grid=G, axis="x", src="T", dst="U")      δₓ : T → U
    Interp(grid=G, src="U", dst="T")                     U → T (two-point mean)

The valid ``(axis, src, dst)`` and ``(src, dst)`` sets are read off the
Tier-A method names (``Difference2D.diff_x_T_to_U``,
``Interpolation2D.U_to_T``) at import, so a Tier-A method added or renamed
changes Tier B with it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
from geophysix.rotation import beta_plane
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gridax import Mask2D
from gridax._src.boundary.boundary import (
    enforce_periodic,
    free_slip_boundaries,
    no_flux_boundaries,
    no_slip_boundaries,
    wall_boundaries,
    zero_boundaries,
    zero_gradient_boundaries,
)
from gridax._src.operators.calculus import curl, divergence, grad_perp, laplacian
from gridax._src.operators.coriolis import coriolis
from gridax._src.operators.moves import difference, interpolate
from gridax._src.operators.triples import PAIRS, TRIPLES, choose
from gridax._src.pipekit.base import FVXOperator, GridOperator

Field = Float[Array, "*batch Ny Nx"]

STAGGERS = ("T", "U", "V", "X")


#: ``(axis, src, dst)`` of the 2-D differences, e.g. ``("x", "T", "U")``.
DIFFERENCES = tuple(sorted(TRIPLES["y", "x"]))
#: ``(src, dst)`` of the 2-D interpolations, e.g. ``("U", "T")``.
INTERPOLATIONS = tuple(sorted(PAIRS["y", "x"]))


class Difference(GridOperator):
    """One-sided difference between staggered points, ``δ / Δ``.

    ``axis="x", src="T", dst="U"``: δₓh[j, i+½] = (h[j, i+1] − h[j, i]) / Δx.

    Attributes
    ----------
    grid : CartesianCGrid2D
        The grid, or ``(nx_interior, ny_interior, Lx, Ly)``.
    axis : {"x", "y"}
        Difference direction.
    src, dst : {"T", "U", "V", "X"}
        Source and destination stagger; see `DIFFERENCES` for the valid
        triples.
    """

    axis: str = eqx.field(static=True, kw_only=True)
    src: str = eqx.field(static=True, kw_only=True)
    dst: str = eqx.field(static=True, kw_only=True)

    def __check_init__(self) -> None:
        choose("difference", (self.axis, self.src, self.dst), TRIPLES["y", "x"])

    def _apply(self, x: Field) -> Field:
        return difference(x, self.grid, axis=self.axis, src=self.src, dst=self.dst)


class Divergence(GridOperator):
    """∇·(u, v) at T-points: δₓu / Δx + δᵧv / Δy.  ``(U, V) → T``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return divergence(u, v, self.grid)


class Curl(GridOperator):
    """Relative vorticity ζ = δₓv / Δx − δᵧu / Δy at X-points.  ``(U, V) → X``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return curl(u, v, self.grid)


class Laplacian(GridOperator):
    """Five-point Laplacian ∇²h at T-points.  ``T → T``."""

    def _apply(self, h: Field) -> Field:
        return laplacian(h, self.grid)


class GradPerp(GridOperator):
    """Rotated gradient (u, v) = (−∂ᵧψ, ∂ₓψ) of a streamfunction.  ``T → (U, V)``."""

    def _apply(self, psi: Field) -> tuple[Field, Field]:
        return grad_perp(psi, self.grid, at="T")


class Interp(GridOperator):
    """Average between staggered points (two- or four-point mean).

    Attributes
    ----------
    grid : CartesianCGrid2D
        The grid, or ``(nx_interior, ny_interior, Lx, Ly)``.
    src, dst : {"T", "U", "V", "X"}
        Source and destination stagger; see `INTERPOLATIONS`.
    """

    src: str = eqx.field(static=True, kw_only=True)
    dst: str = eqx.field(static=True, kw_only=True)

    def __check_init__(self) -> None:
        choose("interpolation", (self.src, self.dst), PAIRS["y", "x"])

    def _apply(self, x: Field) -> Field:
        return interpolate(x, self.grid, src=self.src, dst=self.dst)


class Coriolis(GridOperator):
    """Coriolis tendencies on a β-plane, ``(U, V) → (U, V)``.

    f = f₀ + β (y − y₀) at T-points, ghost ring included (y = j Δy), then
    ``gridax.coriolis``: (+f v̄ at U, −f ū at V), energy conserving.
    ``f0`` and ``beta`` are learnable leaves.

    Attributes
    ----------
    grid : CartesianCGrid2D
        The grid, or ``(nx_interior, ny_interior, Lx, Ly)``.
    f0 : Array
        Reference Coriolis parameter.
    beta : Array
        Meridional gradient ∂f/∂y.
    y0 : float or None
        Reference latitude coordinate; ``None`` uses the mean of the T-point
        y coordinates.
    """

    f0: jax.Array = eqx.field(converter=jnp.asarray, kw_only=True)
    beta: jax.Array = eqx.field(converter=jnp.asarray, default=0.0, kw_only=True)
    y0: float | None = eqx.field(static=True, default=None, kw_only=True)

    def coriolis_parameter(self, dtype: Any = None) -> Float[Array, "Ny Nx"]:
        """f at the T-points of the grid, ghost ring included."""
        g = self.grid
        y = jnp.asarray(g.coords("y"), dtype=dtype)  # T[j] sits at y = j Δy
        Y = jnp.broadcast_to(y[:, None], (g.Ny, g.Nx))
        # f = f0 + beta (y - y0), with y0 the mean T-point y when unset
        y0 = jnp.mean(Y) if self.y0 is None else self.y0
        return beta_plane(Y, f0=self.f0, beta=self.beta, y0=y0)

    def _apply(self, u: Field, v: Field) -> tuple[Field, Field]:
        f = self.coriolis_parameter(jnp.result_type(u, v))
        return coriolis(u, v, self.grid, f=f)


#: ``kind -> ghost-ring filler`` for `FillGhosts`.
GHOST_FILLERS: dict[str, Callable[..., Field]] = {
    "periodic": enforce_periodic,
    "zero": zero_boundaries,
    "zero_gradient": zero_gradient_boundaries,
    "no_flux": no_flux_boundaries,
    "no_slip": no_slip_boundaries,
    "free_slip": free_slip_boundaries,
    "wall": wall_boundaries,
}


class FillGhosts(FVXOperator):
    """Fill the ghost ring with a boundary condition.  ``x → x``.

    Attributes
    ----------
    kind : str
        One of `GHOST_FILLERS`: ``periodic``, ``zero``, ``zero_gradient``,
        ``no_flux``, ``no_slip``, ``free_slip`` or ``wall``.
    staggering : {"h", "u", "v", "q"}
        Only for ``kind="wall"``: the C-grid variable (see
        `wall_boundaries`).
    """

    kind: str = eqx.field(static=True)
    staggering: str = eqx.field(static=True, default="h")

    def __check_init__(self) -> None:
        if self.kind not in GHOST_FILLERS:
            raise ValueError(
                f"kind {self.kind!r} is not available; valid: {sorted(GHOST_FILLERS)}"
            )

    def _apply(self, x: Field) -> Field:
        if self.kind == "wall":
            return wall_boundaries(x, staggering=self.staggering)
        return GHOST_FILLERS[self.kind](x)


#: ``stagger -> kind`` of the wet points `ApplyMask` keeps (contract 3).
MASK_KINDS = {"T": "default", "U": "default", "V": "default", "X": "strict"}


class ApplyMask(FVXOperator):
    """Zero a field on the dry points of its stagger (``jnp.where``, NaN-safe).

    The mask is an array-holding module, so the config has no JSON form:
    the operator is ``forbid_in_yaml`` and persists through
    ``serialize_weights`` / ``with_weights``.

    Attributes
    ----------
    mask : Mask2D
        Land / ocean mask.
    stagger : {"T", "U", "V", "X"}
        Which wet mask to apply: T → ``mask.wet()``, U → ``mask.wet(at="U")``, V →
        ``mask.wet(at="V")``, X → ``mask.wet(at="X", kind="strict")``.
    """

    forbid_in_yaml: ClassVar[bool] = True

    mask: Mask2D
    stagger: str = eqx.field(static=True, default="T")

    def __check_init__(self) -> None:
        if self.stagger not in MASK_KINDS:
            raise ValueError(
                f"stagger {self.stagger!r} is not available; valid: {list(STAGGERS)}"
            )

    def _apply(self, x: Field) -> Field:
        wet = self.mask.wet(at=self.stagger, kind=MASK_KINDS[self.stagger])
        return jnp.where(wet, x, jnp.zeros((), x.dtype))


class Pick(FVXOperator):
    """Select one output of a tuple-valued node: ``(a, b, …) → [index]``.

    Attributes
    ----------
    index : int
        Position to select.
    """

    index: int = eqx.field(static=True)

    def _apply(self, xs: tuple[Any, ...]) -> Any:
        return xs[self.index]


class LinComb(FVXOperator):
    """Linear combination y = ∑ᵢ cᵢ xᵢ (coefficients are learnable leaves).

    Attributes
    ----------
    coeffs : Array
        ``[n]`` coefficients, one per input.
    """

    coeffs: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, *xs: Field) -> Field:
        if len(xs) != self.coeffs.shape[0]:
            raise ValueError(
                f"LinComb has {self.coeffs.shape[0]} coefficients, got {len(xs)} inputs"
            )
        return sum((c * x for c, x in zip(self.coeffs, xs, strict=True)), start=0.0)
