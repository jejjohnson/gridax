# Vorticity and the Jacobian

The relative vorticity is [`curl`](divergence.md), and the potential vorticity is [`sw_potential_vorticity`](diagnostics.md).

## PV flux

::: gridax.pv_flux

::: gridax.vorticity_flux_upwind

## Jacobian

`arakawa_jacobian` returns the full `[..., Ny, Nx]` T-point array with a zero ghost ring.  Under a mask the fields are treated as zero on dry T-cells (so land `NaN`s cannot reach the stencil) and dry output cells are exactly zero.

::: gridax.arakawa_jacobian
