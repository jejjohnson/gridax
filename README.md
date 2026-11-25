<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="gridax" src="docs/assets/logo-light.svg" width="340">
</picture>

**Arakawa C-grids, land and ocean masks and the finite-volume operators on them, on the array API: NumPy, SciPy and geophysix, with JAX when it is installed.**

[![Tests](https://github.com/jejjohnson/gridax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/gridax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/gridax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/gridax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/gridax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/gridax/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/gridax/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/gridax/)
[![codecov](https://codecov.io/gh/jejjohnson/gridax/branch/main/graph/badge.svg?token=YGPQQEAK91)](https://codecov.io/gh/jejjohnson/gridax)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

📖 [Documentation](https://jejjohnson.github.io/gridax/) ·
🚀 [Quickstart](https://jejjohnson.github.io/gridax/quickstart/) ·
🧱 [Derivative stack](https://jejjohnson.github.io/gridax/stencils/#derivative-stack) ·
📚 [API reference](https://jejjohnson.github.io/gridax/reference/) ·
🧭 [Design](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/gridax.md)

</div>

---

An ocean or atmosphere model on a structured grid spends much of its code on bookkeeping: which point of a cell each variable sits on, which ghost cells a stencil reads, and which faces touch land. gridax holds that bookkeeping as data, with the operators that respect it. Its central object is the Arakawa C-grid. Tracers sit at the cell centres (T points), velocities on the east and north faces (U and V points) and vorticity at the corners (X points). One ghost ring holds the boundary conditions, and a land and ocean mask says which of those points are wet.

```math
\frac{\mathrm{d}\bar{q}_{j,i}}{\mathrm{d}t} = -\frac{F^x_{j,i+1/2} - F^x_{j,i-1/2}}{\Delta x} - \frac{F^y_{j+1/2,i} - F^y_{j-1/2,i}}{\Delta y}
```

Here $`\bar{q}_{j,i}`$ is the average of a tracer over cell $`(j, i)`$, held at the T points in an array of shape $`(N_y, N_x)`$ that includes the ghost ring. $`F^x`$ and $`F^y`$ are the fluxes through the east and north faces, held at the U and V points in arrays of the same shape, and $`\Delta x`$ and $`\Delta y`$ are the grid spacings. The flux out of one cell is the flux into its neighbour, so the total of $`\bar{q}`$ changes only through the walls. Most of gridax computes a piece of this equation: the face values that build the fluxes, the differences between them, the masks that close the dry faces and the ghost cells the walls need.

```python
# Shapes: N_y × N_x, the interior and one ghost ring
# (grid, mask, q, u and v as in the quickstart below)

# today (gridax)
# dq̄/dt, with WENO-5 face values
tendency = gridax.advect(q, u, v, grid, scheme=WENO(5), mask=mask)  # (N_y, N_x) → (N_y, N_x)
# today (gridax)
# ∑ A dq̄/dt over the wet cells, zero to rounding
budget = gridax.area_sum(tendency, grid, mask=mask)  # (N_y, N_x) → ()
```

None of it needs JAX. We write gridax against the [array API](https://data-apis.org/array-api/), so one function runs on [NumPy](https://numpy.org) arrays, on [JAX](https://github.com/jax-ml/jax) arrays and tracers, or on any other backend that follows the standard. Grids and masks are built with NumPy, and the masks take their dilations from [SciPy](https://scipy.org)'s `ndimage`. Physical constants are [geophysix](https://github.com/jejjohnson/geophysix) records and appear only as parameter defaults, so every operator stays unit-agnostic. When JAX is importable, every grid, mask and operator registers as a JAX pytree, so they pass through `jax.jit`, `jax.vmap` and `jax.grad`. [jaxtyping](https://github.com/patrick-kidger/jaxtyping) annotates the shapes.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/gridax_overview/gridax_overview.png" alt="gridax in one picture: one C-grid at the top, with its finite-volume equation, the same-index staggering of the T, U, V and X points and the mask; six operations below it, difference, advect, diffuse, rotate, fill the ghost ring and diagnose, each with its equation and its main names; gridax.stencils and gridax.analysis underneath, and NumPy, SciPy, geophysix and JAX at the bottom" width="100%"></p>

## 🌟 Highlights

- 🧱 **Grids and masks as data**: `CartesianCGrid1D/2D/3D` and `SphericalCGrid2D/3D` add the ghost ring, and `Mask2D.from_mask` (or `Mask2D.from_bathymetry`) derives the wet T, U, V and X points, the coast classification and the stencil capability from one T-point mask, with NumPy and SciPy alone.
- 🔢 **Exact stencil coefficients**: `gridax.stencils.difference_coeffs` solves the Taylor conditions in rational arithmetic, for point values or cell averages, collocated or staggered, and the WENO sub-stencils, optimal weights and smoothness matrices come from the same solver.
- 🌊 **Advection that is safe at the coast**: `gridax.advect` builds the fluxes from upwind face values of first to third order, four TVD limiters, or WENO-3, 5, 7 and 9 (and WENO-Z 5), and next to land each face falls back to the widest stencil that fits; `gridax.reconstruct` is the flux along one axis.
- 🧮 **One function per operation**: `difference`, `interpolate`, `divergence`, `curl`, `laplacian`, `coriolis`, `diffuse`, `biharmonic`, `advect_momentum`, `arakawa_jacobian`, …, for 1-D, 2-D and 3-D, Cartesian, spherical and collocated grids: the type of the grid picks the stencil. `GridOps(grid, mask)` has them all as methods.
- 🧭 **Boundary conditions per face**: nine conditions, from `Dirichlet` and `Neumann` to `Sponge` and `Outflow`, named per face by `BC` and written into the ghost layers of a field by `fill_ghosts`, so that each condition holds on the face.
- 📊 **Diagnostics and integrals**: kinetic energy, enstrophy, potential vorticity, strain and the Okubo–Weiss parameter as functions, and area and volume sums and means that skip the dry cells (`area_sum`, `masked_mean`, …), on Cartesian and spherical grids.
- 🔁 **One code path for NumPy and JAX**: a call returns NumPy arrays for NumPy inputs and traces under `jax.jit` for JAX inputs, and the masked-operator goldens are checked on both (`tests/test_numpy_parity.py`).

## 📐 One derivative stack, four layers

Every derivative and every face value in gridax comes from one stack of four layers, collocated and staggered alike. Each layer is built from the one below it, and a model reaches in at whichever layer it needs.

| Layer | Holds | Example | Names |
|---|---|---|---|
| 0, coefficients | offsets and exact weights, no arrays | $`(1, -27, 27, -1)/24`$, a fourth-order staggered first derivative | `gridax.stencils`: `difference_offsets`, `difference_coeffs`, `nonuniform_coeffs`, `weno_stencils`, `optimal_weights`, `smoothness_matrix` |
| 1, raw stencils | index arithmetic, with no metric and no mask | $`h_{j,i+1} - h_{j,i}`$ | `gridax.stencils`: the 32 named stencils (`diff_x_fwd_2d`, `avg_xfwd_yfwd_2d`, …), `staggered_difference`, `staggered_average`, `interior` |
| 2, canonical functions | one per operation, scaled by the metric | $`(h_{j,i+1} - h_{j,i}) / \Delta x`$ | `difference`, `interpolate`, `divergence`, `curl`, `laplacian`, `advect`, `reconstruct`, `diffuse`, `coriolis`, `arakawa_jacobian`, `kinetic_energy`; on bare arrays `gridax.analysis` |
| 3, the binding | a grid and a mask, held as fields | $`\mathrm{d}\bar{q}_{j,i}/\mathrm{d}t`$ above, zero on the dry points | `GridOps` |

Layer 0 is one solver. A stencil with offsets $`s_1, \dots, s_K`$, in units of the grid step, approximates the $`n`$-th derivative with coefficients $`c_1, \dots, c_K`$ that solve

```math
\sum_{k=1}^{K} c_k \mu_m(s_k) = n! \delta_{mn}
```

for $`m = 0, 1, \dots, K - 1`$, where $`\delta_{mn}`$ is 1 when $`m = n`$ and 0 otherwise, and the moment $`\mu_m(s)`$ is $`s^m`$ for point values and the average of $`x^m`$ over the unit cell centred on $`s`$ for cell averages. Integer offsets give collocated stencils and half-integer offsets give staggered ones. The two kinds of sample agree on two points and differ from three on, which is why the [derivative stack page](https://jejjohnson.github.io/gridax/stencils/#derivative-stack) tells you which to use for velocities and which for a finite-volume tracer.

## 📥 Install

gridax needs Python 3.12 or 3.13. The base install brings in NumPy, SciPy, jaxtyping, geophysix and pipekit-array (with array-api-compat) and never imports JAX itself: install JAX beside it, and the same calls take JAX arrays. Two extras add the parts that are built on JAX, and `import gridax` imports neither of them:

```bash
pip install gridax              # grids, masks, stencils, operators, analysis, advection
pip install "gridax[jax]"       # + jax, equinox, lineax, gaussx, spectraldiffx: import gridax.solvers
pip install "gridax[pipekit]"   # + pipekit-ops, pipekit-jax, integrix: import gridax.pipekit
```

gridax is not on PyPI yet ([#426](https://github.com/jejjohnson/gridax/issues/426)), so for now install it from this repository with [uv](https://docs.astral.sh/uv/). geophysix is not on PyPI either, and uv reads `[tool.uv.sources]` only from the project it installs, not from that project's dependencies, so a project that uses gridax names both sources in its own `pyproject.toml` (for an extra, add the sources that `packages/gridax/pyproject.toml` lists for spectraldiffx, gaussx, integrix and pipekit, and the gaussx override in the root `pyproject.toml`):

```toml
[project]
dependencies = ["gridax"]

[tool.uv.sources]
gridax = { git = "https://github.com/jejjohnson/gridax", tag = "gridax-v0.2.0", subdirectory = "packages/gridax" }
geophysix = { git = "https://github.com/jejjohnson/geophysix", tag = "v0.0.3" }
```

## 🌊 Quickstart: a tracer round an island

We carry a blob of tracer round a closed basin with an island in it and check that the tracer total stays fixed. The grid is the unit square with 32 × 32 cells, and the flow is one gyre with streamfunction $`\psi = \sin(\pi x) \sin(\pi y) / \pi`$, so $`u = -\partial \psi / \partial y`$ on the U points and $`v = \partial \psi / \partial x`$ on the V points. The mask closes the basin with a ring of dry cells and cuts out an island of radius 0.15. The tracer $`\bar{q}`$ starts as a Gaussian blob west of the island, $`A_{j,i}`$ are the cell areas, and we step the equation above with Heun's method and $`\Delta t = 0.2 \Delta x`$.

```python
import gridax
from gridax.schemes import WENO
import numpy as np

# Shapes: N_y = N_x = 34, a 32 × 32 interior and one ghost ring

# today (gridax)
grid = gridax.CartesianCGrid2D.from_interior(32, 32, Lx=1.0, Ly=1.0)
# (x_i, y_j), the T-point coordinates
x, y = np.meshgrid(
    (np.arange(grid.Nx) - 0.5) * grid.spacing("x"), (np.arange(grid.Ny) - 0.5) * grid.spacing("y")
)  # → (N_y, N_x), (N_y, N_x)
# a closed basin with a round island in the middle
wet = (x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.15**2  # → (N_y, N_x)
wet[:2, :] = wet[-2:, :] = wet[:, :2] = wet[:, -2:] = False
# today (gridax)
mask = gridax.Mask2D.from_mask(wet)
# ψ = sin(πx) sin(πy) / π, one gyre
psi = np.sin(np.pi * x) * np.sin(np.pi * y) / np.pi  # → (N_y, N_x)
# today (gridax)
ops = gridax.GridOps(grid, mask)
# today (gridax)
# u = −∂ψ/∂y at U points, v = ∂ψ/∂x at V points
u, v = ops.grad_perp(psi, at="T")  # → (N_y, N_x), (N_y, N_x)
# a Gaussian blob of tracer west of the island
q = np.exp(-((x - 0.25) ** 2 + (y - 0.5) ** 2) / 0.01) * wet  # → (N_y, N_x)

# today (gridax)
# ∑ A_ji q̄_ji over the wet cells
total = gridax.area_sum(q, grid, mask=mask)  # → ()
dt = 0.2 * grid.spacing("x")
for _ in range(200):
    # Heun's method on dq̄/dt, with WENO-5 face values
    q1 = q + dt * ops.advect(q, u, v, scheme=WENO(5))  # (N_y, N_x) → (N_y, N_x)
    q = 0.5 * (q + q1 + dt * ops.advect(q1, u, v, scheme=WENO(5)))
# today (gridax)
change = gridax.area_sum(q, grid, mask=mask) / total - 1.0  # → ()
print(f"{float(change):.1e} {float(q.max()):.3f} {float(q.min()):.1e}")
```

This prints `0.0e+00 0.807 -4.8e-14`. After 200 steps the tracer total equals the initial one to the last bit, because every flux leaves one cell and enters the next and the dry faces carry none. The peak has fallen from 0.952 to 0.807 as WENO-5 smooths the blob on this coarse grid, and the smallest value, $`-4.8 \times 10^{-14}`$, is an undershoot at the level of rounding.

Every array in the block is a NumPy array, and nothing in it imports JAX. Pass `jax.numpy` arrays instead and the same lines trace under `jax.jit` and differentiate under `jax.grad`. Swap `WENO(5)` for `Upwind(1)`, `TVD("van_leer")` or `WENO(9)` of `gridax.schemes`, or the mask for one from `Mask2D.from_bathymetry`, and the rest of the block stays the same. The [docs quickstart](https://jejjohnson.github.io/gridax/quickstart/) explains each step, says why the basin needs its ring of dry cells, and runs the tendency under `jit` and `grad`.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/gridax_roles/gridax_roles-dark.svg">
    <img alt="gridax's main public names on the GeoStack role ladder: the operators bound to a grid and a mask as models; the advection schemes and the boundary atoms and sets as strategies; raw stencils, collocated differences, face values, limiters, C-grid terms, diagnostics, integrals, ghost-ring fillers and smooth surrogates as primitives; the grids, masks and arrays as structures; geophysix's constants and the setup functions beside the ladder" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/gridax_roles/gridax_roles-light.svg" width="100%">
  </picture>
</p>

Each area links to its API page, and the [capability index](https://jejjohnson.github.io/gridax/capabilities/) lists every public name in one place. Names without a module are in the top-level `gridax` namespace.

| Area | Symbols | Notes |
|---|---|---|
| [Grids](https://jejjohnson.github.io/gridax/reference/grids/) | `ArakawaCGrid1D/2D/3D`, `CurvilinearGrid1D/2D/3D`, `CartesianCGrid1D/2D/3D`, `SphericalCGrid2D/3D`, `t_point_coordinates` | topology, then a uniform metric, then the coordinates |
| [Masks](https://jejjohnson.github.io/gridax/reference/masks/) | `Mask1D/2D/3D`, `StencilCapability1D/2D/3D`; `dilate_mask`, `make_sponge`, `pool_bool`, `h_from_pooled`, `count_contiguous` | built once, eagerly, from a T-point mask |
| [Coefficients and raw stencils](https://jejjohnson.github.io/gridax/reference/stencils/) | `gridax.stencils`: `difference_offsets`, `difference_coeffs`, `nonuniform_coeffs`, `weno_stencils`, `optimal_weights`, `smoothness_matrix`; `diff_x_fwd_2d`, `avg_y_bwd_3d`, …; `staggered_difference`, `staggered_average`, `interior` | Layers 0 and 1 |
| [Collocated kernels](https://jejjohnson.github.io/gridax/reference/analysis/) | `gridax.analysis`: `difference`, `remap_axis`, `remap_axis_columns`, `Extrapolate` | along one axis of a bare array; the calculus on a `CartesianAGrid` or `SphericalAGrid2D` is the functions below |
| [Differences](https://jejjohnson.github.io/gridax/reference/difference/) and [interpolation](https://jejjohnson.github.io/gridax/reference/interpolation/) | `difference`, `interpolate`; `gradient`, `divergence`, `curl`, `laplacian`, `grad_perp` | moves between the T, U, V and X points, on any grid |
| [Vorticity](https://jejjohnson.github.io/gridax/reference/vorticity/) and [Coriolis](https://jejjohnson.github.io/gridax/reference/coriolis/) | `curl`, `pv_flux`, `vorticity_flux_upwind`; `coriolis`; `arakawa_jacobian` | the Jacobian conserves energy and enstrophy |
| [Advection](https://jejjohnson.github.io/gridax/reference/advection/) and [reconstruction](https://jejjohnson.github.io/gridax/reference/reconstruction/) | `advect`, `advect_vertical`, `reconstruct`; [`gridax.schemes`](https://jejjohnson.github.io/gridax/schemes/): `Upwind`, `Centred`, `Biased`, `WENO`, `TVD`; `gridax.advection.kernels`; `minmod`, `van_leer`, `superbee`, `mc`; `upwind_flux`, `rusanov_flux` | flux form, closed walls by default |
| [Diffusion](https://jejjohnson.github.io/gridax/reference/diffusion/) and [momentum](https://jejjohnson.github.io/gridax/reference/momentum/) | `diffuse`, `diffusive_fluxes`, `diffuse_vertical`, `biharmonic`; `advect_momentum`, `vortex_force` | harmonic and biharmonic; vector-invariant momentum |
| [Boundary conditions](https://jejjohnson.github.io/gridax/reference/boundary_conditions/) | `Dirichlet`, `Neumann`, `Robin`, `Periodic`, `Slip`, `Sponge`, `Reflective`, `Extrapolation`, `Outflow`; `BC` and its presets (`BC.no_flux`, `BC.periodic`, `BC.wall_normal`, …), `fill_ghosts` | one condition per face, any grid dimension, any number of ghost layers |
| [Diagnostics](https://jejjohnson.github.io/gridax/reference/diagnostics/) | `kinetic_energy`, `bernoulli_potential`, `shear_strain`, `tensor_strain`, `okubo_weiss_parameter`, `sw_potential_vorticity`, `qg_potential_vorticity`, `geostrophic_velocity`, `vertical_velocity`, `enstrophy`, …; `area_sum`, `area_mean`, `volume_sum`, `volume_mean`, `masked_mean`, `masked_moments`, … | the sums and means skip dry cells |
| Smooth surrogates | `smooth_abs`, `smooth_max`, `smooth_clamp` | for the non-smooth steps on a path you differentiate |

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [Masks](https://jejjohnson.github.io/gridax/demo-masks/) | building masks for several basins, the staggered masks they derive, stencil capability and the coast classification |
| 2 | [1-D advection schemes](https://jejjohnson.github.io/gridax/advection-1d-schemes/) | six upwind, TVD and WENO schemes on one profile, side by side |
| 3 | [2-D rotating advection](https://jejjohnson.github.io/gridax/advection-2d-rotation/) | a cosine bell carried once round a solid-body rotation |
| 4 | [2-D masked advection](https://jejjohnson.github.io/gridax/advection-2d-masked/) | advection past land, with the stencils narrowing at the coast |

## 📖 Documentation

- **[Quickstart](https://jejjohnson.github.io/gridax/quickstart/)**: the install, this README's example with each step explained, and the same tendency under `jit` and `grad`.
- **The C-grid**: [discretisation, staggering and ghost cells](https://jejjohnson.github.io/gridax/cgrid-discretization/) and [masks](https://jejjohnson.github.io/gridax/masks/).
- **[The derivative stack](https://jejjohnson.github.io/gridax/stencils/)**: the four layers, the raw stencils and the two wall rules, with [writing fast stencils](https://jejjohnson.github.io/gridax/performance/) beside it.
- **Operators**: [differences](https://jejjohnson.github.io/gridax/difference/), [interpolation](https://jejjohnson.github.io/gridax/interpolation/), [advection](https://jejjohnson.github.io/gridax/advection/), [boundary conditions](https://jejjohnson.github.io/gridax/boundary-conditions/) and the [other spatial operators](https://jejjohnson.github.io/gridax/spatial-operators/).
- **[API reference](https://jejjohnson.github.io/gridax/reference/)**: every module, with the [capability index](https://jejjohnson.github.io/gridax/capabilities/) listing each public name.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/gridax_ecosystem/gridax_ecosystem-dark.svg">
    <img alt="The packages that import gridax: diagnostix in GeoNumerics, which uses the collocated grids, difference and gradient; xrtoolz-ops in GeoStack, which uses the collocated grids, divergence and the strains; and in GeoModels sesmax-core, which uses CField, GridOps, the masks and the solvers, sesmax-ocean, which uses GridOps, CField, fill_ghosts and area_sum, and plumax, which uses BC, advect, diffuse and the vertical terms; gridax builds on NumPy, SciPy, jaxtyping, geophysix and pipekit-array, its jax extra adds JAX, equinox, lineax, gaussx and spectraldiffx, and its pipekit extra pipekit-jax and integrix" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/gridax_ecosystem/gridax_ecosystem-light.svg" width="100%">
  </picture>
</p>

gridax owns the C-grid, its masks and the operators on it, and composes the rest in:

| Concern | Owner |
|---|---|
| Arrays and the array API | [`numpy`](https://numpy.org), or any backend that follows the [array API](https://data-apis.org/array-api/) |
| Dilations and other mask morphology, at construction | [`scipy`](https://scipy.org) (`scipy.ndimage`) |
| The array namespace of a call | [`pipekit-array`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array) (`pipekit_array.arrays`) |
| Physical constants, used as parameter defaults | [`geophysix`](https://github.com/jejjohnson/geophysix) |
| Shape annotations | [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) |
| Pytrees, `jit`, `vmap` and `grad`, when it is installed | [`jax`](https://github.com/jax-ml/jax) |
| Spectral solves and structured linear algebra in `gridax.solvers` (the `jax` extra) | [`spectraldiffx`](https://github.com/jejjohnson/spectraldiffx), [`gaussx`](https://github.com/jejjohnson/gaussx), [`lineax`](https://github.com/patrick-kidger/lineax), [`equinox`](https://github.com/patrick-kidger/equinox) |
| Operators as pipekit `Operator`s in `gridax.pipekit` (the `pipekit` extra) | [`pipekit`](https://github.com/jejjohnson/pipekit), [`integrix`](https://github.com/jejjohnson/integrix) for `ExplicitStep` |

Dependencies point one way: the base install imports geophysix, pipekit-array and pipekit-ops and no other [GeoStack](https://github.com/jejjohnson/geostack) library (`gridax.solvers` adds spectraldiffx and gaussx, and `gridax.pipekit` adds pipekit-jax and integrix), and the libraries built on it import gridax, never the reverse. diagnostix, xrtoolz-ops, sesmax and plumax import gridax; somax still imports `finitevolx`, the name before the rename, from a pinned commit. Time stepping belongs to integrix (`integrix.march`), which takes a gridax tendency as its right-hand side.

## 🌳 Repository layout

```text
packages/gridax/
  src/gridax/
    __init__.py    ← grids, masks, operators, boundary conditions, diagnostics
    stencils.py    ← Layer 0 coefficients and Layer 1 raw stencils
    analysis.py    ← collocated differences and axis remapping
    advection.py   ← reconstruction along one axis, and its schemes
    solvers/       ← spectral, capacitance, masked-operator and multigrid solvers (the jax extra)
    pipekit/       ← the operators as pipekit Operators (the pipekit extra)
    _src/          ← the implementations behind the public modules
  tests/           ← the suite that runs with NumPy alone
docs/              ← MyST guides, notebooks and the API reference
tests/             ← the suite that needs JAX: operators under jit, the solvers, the pipekit layer
scripts/           ← example models, the capability index and the docs build
```

## ⚙️ Development

```bash
git clone https://github.com/jejjohnson/gridax.git
cd gridax
make install     # uv sync --all-extras and the pre-commit hooks
make test        # the fast tier, on two workers
make uv-lint     # ruff check, ruff format --check and ty, as CI runs them
make docs        # the MyST site and the API reference, with every link checked
```

[`AGENTS.md`](AGENTS.md) holds the rules for contributors and coding agents, from the module map and the C-grid contracts to the tests that enforce them, and [`docs/capabilities.md`](docs/capabilities.md) lists every public name before you write a new one. The icon and the logos come from [`docs/assets/render.py`](docs/assets/render.py), and the schematics are drawn in [geostack](https://github.com/jejjohnson/geostack/tree/main/docs/images/libraries) with the stack's other library figures.

## 🎓 References

- A. Arakawa and V. R. Lamb, 1977. Computational design of the basic dynamical processes of the UCLA general circulation model. *Methods in Computational Physics*, 17, 173–265.
- R. Sadourny, 1975. The dynamics of finite-difference models of the shallow-water equations. *Journal of the Atmospheric Sciences*, 32, 680–689.
- R. J. LeVeque, 2002. *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
- G.-S. Jiang and C.-W. Shu, 1996. Efficient implementation of weighted ENO schemes. *Journal of Computational Physics*, 126, 202–228.
- R. Borges, M. Carmona, B. Costa and W. S. Don, 2008. An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws. *Journal of Computational Physics*, 227.
- G. Roullet and T. Gaillard, 2021. [A fast monotone discretization of the rotating shallow water equations](https://doi.org/10.1029/2021MS002663). *Journal of Advances in Modeling Earth Systems*.
- L. Thiry et al., 2023. [MQGeometry-1.0: a multi-layer quasi-geostrophic solver on non-rectangular geometries](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-1715/). *EGUsphere preprint*.

## 📄 License

MIT, see [LICENSE](LICENSE).
