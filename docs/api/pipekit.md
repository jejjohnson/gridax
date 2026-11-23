# pipekit Operators

`gridax.pipekit` wraps the 2-D Cartesian operators as
[pipekit](https://github.com/jejjohnson/pipekit) operators: JSON
configuration, `Sequential` / `Graph` composition, and Equinox pytree leaves
so `jit`, `grad` and `vmap` see through them. Each operator is one Tier-A
call on `[*batch, Ny, Nx]` fields. Requires the `pipekit` extra:
`pip install "gridax[pipekit]"`. Walkthrough:
[Composing finitevolX with pipekit](../pipekit-composition/).

| Tier-B operator | Tier-A counterpart | in → out |
|---|---|---|
| `Difference(axis, src, dst, scheme)` | `difference` | 1 → 1 |
| `Divergence` | `divergence` | (U, V) → T |
| `Curl` | `curl` | (U, V) → X |
| `Laplacian` | `laplacian` | T → T |
| `GradPerp` | `grad_perp` | T → (U, V) |
| `Interp(src, dst, scheme)` | `interpolate` | 1 → 1 |
| `Coriolis(f0, beta, y0)` | `geophysix.rotation.beta_plane` + `coriolis` | (U, V) → (U, V) |
| `FillGhosts(kind, stagger)` | `fill_ghosts` with `BC.periodic()`, `BC.no_flux()`, … | 1 → 1 |
| `ApplyMask(mask, stagger)` | `jnp.where(mask.{h,u,v,xy_corner_strict}, x, 0)` | 1 → 1 |
| `RelativeVorticity` | `curl` | (U, V) → X |
| `PotentialVorticity` | `sw_potential_vorticity` | (U, V, T, T) → X |
| `Enstrophy` | `enstrophy` of `curl` | (U, V) → X |
| `QGPotentialVorticity(f0, beta, y0)` | `qg_potential_vorticity` | T → T |
| `KineticEnergy` | `kinetic_energy` | (U, V) → T |
| `AvailablePE(g_prime, H)` | `available_potential_energy` | T → T |
| `Strain(component)` | `shear_strain` / `tensor_strain` / `strain_rate_squared` | (U, V) → X / T |
| `OkuboWeiss` | `okubo_weiss_parameter` | (U, V) → T |
| `TotalEnergy` | `area_sum` of KE + APE | (T, T) → `[*batch]` |
| `DomainIntegral` / `DomainMean` | `area_sum` / `area_mean` | T → `[*batch]` |
| `ExplicitStep(rhs, dt, scheme)` | `integrix.march.euler_step` / `heun_step` / `rk3_ssp_step` / `rk4_step` | state → state |
| `Pick(index)`, `LinComb(coeffs)` | — (graph plumbing) | |

## Halos

Every operator says how many cells around a point its result reads: `op.halo` is 1 for a one-cell stencil, 2 for `scheme=Centred(4)`, 0 for an operation point by point. `graph_halo(graph)` sums them along each path of a `Graph` or `Sequential` and returns the largest, which is the overlap a patch needs for its interior to have the values of the whole domain.

```python
from gridax.pipekit import Difference, Interp, graph_halo
from gridax.schemes import Centred

G = (64, 64, 1.0, 1.0)
to_u = Interp(grid=G, src="T", dst="U", scheme=Centred(4))
back = Difference(grid=G, axis="x", src="U", dst="T", scheme=Centred(4))
graph_halo(to_u | back)  # 4
```

A reduction (`TotalEnergy`, `DomainMean`) reads the whole domain and has no halo, so `graph_halo` raises on it.

## Base classes and helpers

::: gridax.pipekit.FVXOperator

::: gridax.pipekit.GridOperator

::: gridax.pipekit.over_leading

::: gridax.pipekit.graph_halo

::: gridax.pipekit.testing.check_operator

## Stencils, interpolation, boundaries and masks

::: gridax.pipekit.Difference

::: gridax.pipekit.Divergence

::: gridax.pipekit.Curl

::: gridax.pipekit.Laplacian

::: gridax.pipekit.GradPerp

::: gridax.pipekit.Interp

::: gridax.pipekit.Coriolis

::: gridax.pipekit.FillGhosts

::: gridax.pipekit.ApplyMask

## Diagnostics and reductions

::: gridax.pipekit.RelativeVorticity

::: gridax.pipekit.PotentialVorticity

::: gridax.pipekit.Enstrophy

::: gridax.pipekit.QGPotentialVorticity

::: gridax.pipekit.KineticEnergy

::: gridax.pipekit.AvailablePE

::: gridax.pipekit.Strain

::: gridax.pipekit.OkuboWeiss

::: gridax.pipekit.TotalEnergy

::: gridax.pipekit.DomainIntegral

::: gridax.pipekit.DomainMean

## Time stepping and graph plumbing

::: gridax.pipekit.ExplicitStep

::: gridax.pipekit.Pick

::: gridax.pipekit.LinComb
