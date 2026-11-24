# Diagnostic Quantities

Pointwise and domain-integrated diagnostic quantities on Arakawa C-grids.

The diagnostics that need a grid are canonical functions: `grid` positional, `mask=` a keyword, and the dry points of the output zero. The pointwise ones take fields that already share a point, no grid and no mask.

A domain integral is `area_sum(field, grid, mask=mask)`, e.g. `area_sum(ke + ape, grid)` for the total energy.

See [Masks: how the functions use a mask](../masks/#operator-api-how-operators-consume-masks)
for the points each output is masked at.

| Diagnostic | Function | Output |
|---|---|---|
| Kinetic energy | `kinetic_energy` | T |
| Bernoulli potential | `bernoulli_potential` | T |
| Available potential energy | `available_potential_energy` | pointwise |
| Relative vorticity | `curl` | X |
| Shallow-water PV | `sw_potential_vorticity`, pointwise `potential_vorticity` | X |
| Enstrophy | `enstrophy` of `curl` | pointwise |
| Potential enstrophy | `potential_enstrophy` | pointwise |
| Shear strain | `shear_strain` | X |
| Tensor strain | `tensor_strain` | T |
| Strain magnitude² | `strain_rate_squared`, pointwise `strain_magnitude_squared` | T |
| Okubo–Weiss | `okubo_weiss_parameter`, pointwise `okubo_weiss` | T |
| QG PV, one layer or several (`A=`) | `qg_potential_vorticity` | T |
| Stretching term | `stretching_term` | pointwise |
| Geostrophic velocity, on a spherical C-grid | `geostrophic_velocity` | U, V |
| Vertical velocity | `vertical_velocity` | W |

`sw_potential_vorticity` is `NaN` at a wet corner of zero thickness, so a degenerate layer is flagged rather than hidden; dry corners are exactly `0` under a mask.

On a [collocated grid](grids.md) (`CartesianAGrid2D`, `SphericalAGrid2D`) the kinetic energy, the two strains, `strain_rate_squared`, `okubo_weiss_parameter`, `sw_potential_vorticity` and `geostrophic_velocity` take and give T points and average nothing. Dry cells are `NaN` there, and a wet cell next to them takes the widest difference that fits. On the sphere the strains carry the metric terms $u \tan\varphi / R$ and $v \tan\varphi / R$, so solid-body rotation has no strain to second order, and a row at a pole is `NaN`. `geostrophic_velocity` is `NaN` where $f = 0$; leaving out a band around the equator is the caller's mask.

## Functions of the grid

### Kinetic Energy

::: gridax.kinetic_energy

### Bernoulli Potential

::: gridax.bernoulli_potential

### Potential Vorticity

::: gridax.sw_potential_vorticity

::: gridax.qg_potential_vorticity

### Strain and the Okubo-Weiss Parameter

::: gridax.shear_strain

::: gridax.tensor_strain

::: gridax.strain_rate_squared

::: gridax.okubo_weiss_parameter

### Geostrophic Velocity

::: gridax.geostrophic_velocity

### Vertical Velocity

::: gridax.vertical_velocity

## Pointwise functions

::: gridax.potential_vorticity

::: gridax.stretching_term

::: gridax.strain_magnitude_squared

::: gridax.okubo_weiss

::: gridax.enstrophy

::: gridax.potential_enstrophy

::: gridax.available_potential_energy

## Sea level and streamfunction

::: gridax.ssh_to_streamfn

::: gridax.streamfn_to_ssh
