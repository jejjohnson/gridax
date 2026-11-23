"""Diagnostic and reduction operators (2-D Cartesian).

Thin wrappers over the canonical diagnostics: each body is a call of one. Fields are ``[*batch, Ny, Nx]``;
the reductions keep the leading axes::

    x : [T, Ny, Nx]  ──DomainIntegral──►  [T]

so a conservation check over a trajectory is a three-node graph::

    E = TotalEnergy(grid=G)(
        KineticEnergy(grid=G)(u, v), AvailablePE(grid=G, g_prime=g)(h)
    )
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gridax._src.boundary.regions import embed, interior
from gridax._src.operators.calculus import curl
from gridax._src.operators.diagnostics import (
    available_potential_energy,
    enstrophy,
    kinetic_energy,
    okubo_weiss_parameter,
    qg_potential_vorticity,
    shear_strain,
    strain_rate_squared,
    sw_potential_vorticity,
    tensor_strain,
)
from gridax._src.operators.reductions import area_mean, area_sum
from gridax._src.pipekit.base import GridOperator, over_leading

Field = Float[Array, "*batch Ny Nx"]
Scalar = Float[Array, "*batch"]


class RelativeVorticity(GridOperator):
    """ζ = δₓv / Δx − δᵧu / Δy at X-points.  ``(U, V) → X``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return curl(u, v, self.grid)


class PotentialVorticity(GridOperator):
    """Shallow-water PV q = (ζ + f) / h at X-points.  ``(U, V, T, T) → X``."""

    def _apply(self, u: Field, v: Field, h: Field, f: Field) -> Field:
        return sw_potential_vorticity(u, v, h, f, self.grid)


class Enstrophy(GridOperator):
    """½ ζ² at X-points.  ``(U, V) → X``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return enstrophy(curl(u, v, self.grid))


class QGPotentialVorticity(GridOperator):
    """Single-layer QG PV q = (∇²ψ + β (y − y₀)) / f₀ at T-points.  ``T → T``.

    y = j Δy at the T-points (ghost ring included).

    Attributes
    ----------
    grid : CartesianCGrid2D
        The grid, or ``(nx_interior, ny_interior, Lx, Ly)``.
    f0 : Array
        Reference Coriolis parameter (learnable).
    beta : Array
        Meridional gradient ∂f/∂y (learnable).
    y0 : float or None
        Reference y; ``None`` is the domain centre, ``(Ny − 1) Δy / 2``.
    """

    f0: jax.Array = eqx.field(converter=jnp.asarray, kw_only=True)
    beta: jax.Array = eqx.field(converter=jnp.asarray, default=0.0, kw_only=True)
    y0: float | None = eqx.field(static=True, default=None, kw_only=True)

    def _apply(self, psi: Field) -> Field:
        g = self.grid
        y = jnp.asarray(g.coords("y"), dtype=psi.dtype)  # T[j] sits at y = j Δy
        Y = jnp.broadcast_to(y[:, None], (g.Ny, g.Nx))
        y0 = (y[0] + y[-1]) / 2 if self.y0 is None else self.y0
        return qg_potential_vorticity(psi, g, f0=self.f0, beta=self.beta, y=Y, y0=y0)


class KineticEnergy(GridOperator):
    """½ (ū² + v̄²) at T-points (velocities averaged to T).  ``(U, V) → T``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return kinetic_energy(u, v, self.grid)


class AvailablePE(GridOperator):
    """½ g′ (h − H)² at T-points.  ``T → T``.

    Attributes
    ----------
    grid : CartesianCGrid2D
        The grid, or ``(nx_interior, ny_interior, Lx, Ly)``.
    g_prime : Array
        Reduced gravity (learnable).
    H : Array
        Reference thickness (learnable), broadcast over the grid.
    """

    g_prime: jax.Array = eqx.field(converter=jnp.asarray, kw_only=True)
    H: jax.Array = eqx.field(converter=jnp.asarray, default=0.0, kw_only=True)

    def _apply(self, h: Field) -> Field:
        H = jnp.broadcast_to(self.H, h.shape)
        h, H = interior(h, self.grid), interior(H, self.grid)
        return embed(available_potential_energy(h, H, self.g_prime), self.grid)


#: ``component -> canonical function`` of `Strain`.
STRAIN_COMPONENTS = {
    "shear": shear_strain,
    "tensor": tensor_strain,
    "magnitude_squared": strain_rate_squared,
}


class Strain(GridOperator):
    """A strain component: shear σₛ (X), tensor σₙ (T) or σₛ² + σₙ² (T).

    Attributes
    ----------
    grid : CartesianCGrid2D
        The grid, or ``(nx_interior, ny_interior, Lx, Ly)``.
    component : {"shear", "tensor", "magnitude_squared"}
        Which component.
    """

    component: str = eqx.field(static=True, kw_only=True)

    def __check_init__(self) -> None:
        if self.component not in STRAIN_COMPONENTS:
            raise ValueError(
                f"component {self.component!r} is not available; "
                f"valid: {sorted(STRAIN_COMPONENTS)}"
            )

    def _apply(self, u: Field, v: Field) -> Field:
        return STRAIN_COMPONENTS[self.component](u, v, self.grid)


class OkuboWeiss(GridOperator):
    """W = σₙ² + σₛ² − ζ² at T-points (W < 0: vortex cores).  ``(U, V) → T``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return okubo_weiss_parameter(u, v, self.grid)


class _Reduction(GridOperator):
    """A number per leading index, from the whole domain."""

    @property
    def halo(self) -> int:
        raise AttributeError(
            f"{type(self).__name__} reads the whole domain and has no halo"
        )


class TotalEnergy(_Reduction):
    """E = ∑_interior (KE + APE) ΔA (`area_sum`), one value per leading index.

    ``(T, T) → [*batch]``.
    """

    def _apply(self, ke: Field, ape: Field) -> Scalar:
        return over_leading(lambda f: area_sum(f, self.grid))(ke + ape)


class DomainIntegral(_Reduction):
    """∑_interior x ΔA over the T-cells (`area_sum`).  ``T → [*batch]``."""

    def _apply(self, x: Field) -> Scalar:
        return over_leading(lambda f: area_sum(f, self.grid))(x)


class DomainMean(_Reduction):
    """Area-weighted interior mean ⟨x⟩ (`area_mean`).  ``T → [*batch]``."""

    def _apply(self, x: Field) -> Scalar:
        return over_leading(lambda f: area_mean(f, self.grid))(x)
