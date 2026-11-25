---
name: finite-volume-models-with-finitevolx
description: Write finite-volume ocean and atmosphere models in JAX on finitevolX — Arakawa C-grids with a ghost ring (Cartesian and spherical, 1-D / 2-D / 3-D, multilayer), raw stencils and difference / interpolation / divergence / vorticity operators, WENO / TVD / upwind advection, diffusion, Coriolis and momentum advection, land-sea masks with coastal stencil fallback, boundary conditions, energy / PV / strain diagnostics, wind and drag forcing, spectral / capacitance / CG / multigrid Poisson and Helmholtz solvers, and SSP-RK, multistep, IMEX and diffrax time steppers. Use whenever a task discretises a PDE on a staggered grid, writes a stencil or a ghost-cell fill, advects a tracer, inverts vorticity or PV, or time-steps a shallow-water / QG / primitive-equation model, in a project that uses (or could use) finitevolX.
---

# Finite-volume models on finitevolX

finitevolX gives every piece of a C-grid model as a JAX pytree or a pure
function: grids that carry the ghost ring, stencils that know
which face they map to, masks that switch stencils off at the coast,
boundary conditions that fill ghosts at the right half-cell, solvers and
time steppers. Everything runs under `jit`, `vmap` and `grad`. Before
writing a shifted slice, a ghost-cell fill, a WENO reconstruction, a
Poisson solver or an RK loop, look it up:

1. **The capability index** lists every public name with a one-line
   summary, plus the spectraldiffx and gaussx APIs underneath:
   <https://jejjohnson.github.io/gridax/capabilities/>. Or list the
   installed version:

   ```python
   import importlib
   import inspect

   for module_name in ("gridax", "gridax.solvers"):
       module = importlib.import_module(module_name)
       for name in module.__all__:
           doc = (inspect.getdoc(getattr(module, name)) or "").split("\n")[0]
           print(f"{module_name}.{name}: {doc}")
   ```

2. Compose what exists. finitevolX is now the `gridax` package: `import gridax as fvx` for the grids, masks, stencils and operators (NumPy or JAX), and `import gridax.solvers as gs` for the solvers, which need the `jax` extra. It is not on PyPI: `uv add "gridax[jax] @ git+https://github.com/jejjohnson/gridax.git#subdirectory=packages/gridax"`, with the git sources of spectraldiffx, gaussx and geophysix that `packages/gridax/pyproject.toml` lists.

## What lives where

| You need… | Use |
|---|---|
| A grid | `fvx.CartesianCGrid1D/2D/3D.from_interior(...)`, `fvx.SphericalCGrid2D/3D.from_interior(...)`; arrays are `[Ny, Nx]` including one ghost ring, interior `[1:-1, 1:-1]` |
| Raw shifted differences / averages | `fvx.diff_x_fwd_2d`, `fvx.diff_y_bwd_2d`, `fvx.avg_xfwd_yfwd_2d`, … (+ `_1d` / `_3d`); pad back with `interior(values, like)` of `gridax.stencils` |
| Gradients, divergence, curl, Laplacian | `fvx.difference(h, grid, axis="x", src="T", dst="U")`, `fvx.gradient`, `fvx.divergence`, `fvx.curl`, `fvx.laplacian`, `fvx.grad_perp`, on any grid (1-D to 3-D, Cartesian or spherical); `fvx.GridOps(grid, mask)` has them as methods |
| Moving fields between T / U / V / X points | `fvx.interpolate(f, grid, src="T", dst="U")` |
| Tracer advection | `fvx.advect(q, u, v, grid, scheme=WENO(5), mask=mask)`, with a scheme of `gridax.schemes`: `Upwind(k)`, `Centred(k)`, `WENO(k)`, `TVD(limiter)` |
| Vertical advection and diffusion of a 3-D tracer | `fvx.advect_vertical(q, w, grid, scheme=...)` (`w` at W), `fvx.diffuse_vertical(q, grid, kappa=...)` |
| Diffusion, momentum advection, Coriolis | `fvx.diffuse`, `fvx.biharmonic`, `fvx.advect_momentum`, `fvx.coriolis` (the grid positional, `mask=`) |
| Land and coasts | `fvx.Mask2D.from_mask(wet)` → `mask.wet(at="T" / "U" / "V" / "X")`; pass `mask=` to an operator |
| Ghost cells | `fvx.fill_ghosts(u, grid, fvx.BC(x=fvx.Dirichlet(0.0), y=fvx.Neumann()), stagger="U")`, with the presets `fvx.BC.periodic()`, `fvx.BC.no_flux()`, `fvx.BC.zero()`, `fvx.BC.wall_normal()`, `fvx.BC.wall_tangential(slip="no")` (joined with `\|`) |
| Diagnostics | `fvx.kinetic_energy`, `fvx.bernoulli_potential`, `fvx.shear_strain`, `fvx.okubo_weiss_parameter`, `fvx.sw_potential_vorticity`, `fvx.qg_potential_vorticity`, `fvx.arakawa_jacobian`, … |
| Integrals and means | `fvx.area_sum`, `fvx.area_mean`, `fvx.volume_mean`, `fvx.masked_mean` |
| Poisson / Helmholtz | `gs.solve_helmholtz_dst` / `_dct` / `_fft` on a rectangle (slice the unknowns, embed the result in zeros); the one-call wrappers `streamfunction_from_vorticity`, `pressure_from_divergence` and `pv_inversion` moved to `sesmax_core.gfd`; `gaussx.linear_solve_with_info` with `gaussx.CGSolver` (CG is gaussx's, with `gaussx.NystromPreconditioner` for a low-rank preconditioner and `gs.make_spectral_preconditioner` for a spectral one), `gs.build_capacitance_solver`, `gs.build_multigrid_solver(...).as_preconditioner()` or `gs.multigrid_levels` for gaussx's multigrid |
| The Laplacian as a linear operator (GP / DA priors, log-determinants, sampling) | `gs.helmholtz_operator(shape, dx, dy, lambda_, bc)` (a `gaussx.DiagonalizedOperator`: `gaussx.solve` / `logdet` / `sqrt` / `inv`), `gs.masked_helmholtz_operator(mask, dx, dy, lambda_)` (a `gaussx.MaskedOperator`, capacitance solve); an SPDE / Matérn precision `τ²(κ² − ∇²)^α` from its eigenvalues; `gs.build_multigrid_solver(...).as_preconditioner()` for `gaussx.CGSolver` |
| Forcing | moved to `sesmax_core.forcing` (#422): `WindStress2D`, `LinearDrag2D`, `QuadraticDrag2D`, `RayleighDamping2D`, combined with `ForcingSum` / `ForcingProduct` / `TimeVaryingForcing` |
| Time stepping | `integrix.march.rk3_ssp_step(state, rhs_fn, dt)`, `integrix.march.rk4_step`, `integrix.march.ab2_step`, `integrix.march.leapfrog_raf_step`, `integrix.march.imex_ssp2_step`; diffrax solvers `integrix.march.RK3SSP`, `integrix.march.SSP_RK104`, `integrix.march.IMEX_SSP2` with `sesmax_core.solve_pde` (the former `solve_ocean_pde`, moved from finitevolX) |
| Several layers | `sesmax_core.vertical.multilayer(op.method)` over `[nl, Ny, Nx]`; vertical modes `sesmax_core.vertical.decompose_vertical_modes`, `layer_to_mode` (moved from finitevolX, #422) |

Spectral transforms and solvers come from spectraldiffx (finitevolX
re-exports them); structured linear algebra (tridiagonal solves, Nyström
preconditioning) comes from gaussx, and `gs.helmholtz_operator` /
`gs.masked_helmholtz_operator` hand the grid's own Laplacian back to gaussx
as a structured operator (never a dense matrix).

## The rules your code must keep

- **Same-index staggering.** `T[j, i]` is the cell centre, `U[j, i]` its
  east face, `V[j, i]` its north face, `X[j, i]` its NE corner. Name the
  move (`difference(h, grid, axis="x", src="T", dst="U")`,
  `interpolate(u, grid, src="U", dst="T")`) instead of writing the slice.
- **The caller owns the ghost ring.** Operators write only the interior
  and leave the ghost ring zero; fill ghosts (`fill_ghosts` with a `BC`) on every field before the next operator reads
  it, including inside the RHS of a time stepper. Backward operators (U→T,
  V→T) read the west / south ghost that nothing else fills.
- **Masks are built once and passed to the functions.** `Mask2D.from_mask`
  runs NumPy / SciPy eagerly; give it to the function as `mask=` (or to
  `GridOps(grid, mask)`) so the
  output is zero on land and advection falls back to narrower stencils at
  the coast. Never store `NaN` on land unless the operator documents that
  it sanitises its inputs.
- **Closed walls by default.** `advect` and `diffuse` take
  `wall="closed"` (no flux through the outer walls); use `wall="open"`
  with filled ghosts for periodic or open boundaries (the flux through the
  wall then comes from the ghost ring and mass is conserved exactly).
- **Units are yours.** Physical constants are parameters with SI defaults
  (`gravity` / `g` default to 9.80665, `omega` to Earth's rotation rate,
  `radius` to Earth's radius), never baked into the numerics, so a
  nondimensional model passes its own values and the operators work
  unchanged.
- **JAX rules.** Pure functions, `.at[...].set` instead of mutation, no
  Python control flow on traced values, float32 in gives float32 out; enable
  x64 for solver-accuracy work.

## Worked example

```python
import jax
import jax.numpy as jnp
import numpy as np

import gridax as fvx
from gridax.schemes import WENO
import gridax.solvers as gs
from integrix.march import rk3_ssp_step

jax.config.update("jax_enable_x64", True)

# 1. A doubly periodic 64 × 64 C-grid on [0, 1]²: arrays are [Ny, Nx] = [66, 66]
grid = fvx.CartesianCGrid2D.from_interior(64, 64, 1.0, 1.0)
xc = (jnp.arange(grid.Nx) - 0.5) * grid.spacing("x")  # T-point x, ghost columns included
yc = (jnp.arange(grid.Ny) - 0.5) * grid.spacing("y")
X, Y = jnp.meshgrid(xc, yc)  # [Ny, Nx] T-point coordinates

# 2. Streamfunction at T-points → non-divergent velocities on the faces
psi = jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)  # [Ny, Nx]
ops = fvx.GridOps(grid)
u, v = ops.grad_perp(psi, at="T")  # u at U-points (east faces), v at V-points (north faces)
around = fvx.BC.periodic()  # the caller owns the ghosts
u = fvx.fill_ghosts(u, grid, around, stagger="U")
v = fvx.fill_ghosts(v, grid, around, stagger="V")
div = ops.divergence(u, v)  # [Ny, Nx] at T-points

# 3. Invert the discrete Laplacian back to ψ with the periodic (FFT) solver
lap = ops.laplacian(psi)  # ∇²ψ at T-points, zero ghost ring
psi_rec = jnp.zeros_like(lap).at[1:-1, 1:-1].set(
    gs.solve_helmholtz_fft(lap[1:-1, 1:-1], grid.spacing("x"), grid.spacing("y"))
)  # slice the T interior, solve (zero-mean), embed in zeros
inv_err = jnp.max(jnp.abs(psi_rec[1:-1, 1:-1] - psi[1:-1, 1:-1]))

# 4. Advect a tracer with WENO5 fluxes and SSP-RK3, periodic and conservative
def rhs(q):
    q = fvx.fill_ghosts(q, grid, around)  # fill the ghost ring before every stencil
    return ops.advect(q, u, v, scheme=WENO(5), wall="open")  # -∇·(q u) at T-points


q0 = jnp.exp(-((X - 0.3) ** 2 + (Y - 0.5) ** 2) / 0.005)  # [Ny, Nx]
dt = 0.4 * grid.spacing("x") / jnp.max(jnp.abs(u))  # CFL number 0.4
step = jax.jit(lambda q: rk3_ssp_step(q, rhs, dt))
q = q0
for _ in range(100):
    q = step(q)
mass_drift = (fvx.area_sum(q, grid) - fvx.area_sum(q0, grid)) / fvx.area_sum(q0, grid)

# 5. A closed basin with an island: masked operators return exactly zero on land
wet = np.ones((grid.Ny, grid.Nx), dtype=bool)
wet[0, :] = wet[-1, :] = wet[:, 0] = wet[:, -1] = False
wet[28:38, 28:38] = False
mask = fvx.Mask2D.from_mask(wet)
dq = fvx.advect(q0, u, v, grid, scheme=WENO(5), mask=mask)
land_max = jnp.max(jnp.abs(jnp.where(mask.wet(), 0.0, dq)))
```

With x64 on, the velocities from `grad_perp` have a divergence of order
1e-13 (round-off against velocities of order 6), the FFT solver inverts
the five-point Laplacian to about 1e-15, the periodic WENO5 / SSP-RK3 run
conserves the tracer's mass to about 1e-14 after 100 steps, and the masked
tendency is exactly zero on every land cell.

## Don't write it — use finitevolX

| Don't write… | Use |
|---|---|
| `(h[1:-1, 2:] - h[1:-1, 1:-1]) / dx` into a zeroed array | `fvx.difference(h, grid, axis="x", src="T", dst="U")` or `interior(fvx.diff_x_fwd_2d(h) / dx, h)` with `from gridax.stencils import interior` |
| `0.5 * (u[:, 1:] + u[:, :-1])` to move u to cell centres | `fvx.interpolate(u, grid, src="U", dst="T")` |
| An upwind / WENO face reconstruction and flux divergence | `fvx.advect(q, u, v, grid, scheme=WENO(5))` |
| `q[:, 0] = q[:, -2]` (periodic ghosts) and friends | `fvx.fill_ghosts(q, grid, fvx.BC.periodic())` |
| A wet-face mask `wet[:, :-1] & wet[:, 1:]` | `fvx.Mask2D.from_mask(wet).wet(at="U")` |
| `scipy.fft.dstn` Poisson solver | `gs.solve_helmholtz_dst` (or `sesmax_core.gfd.streamfunction_from_vorticity` for the full-array one-call form) |
| A conjugate-gradient loop on a masked basin | `gaussx.linear_solve_with_info(gaussx.as_linear_operator(matvec, shape=(n, n), negative_definite=True), rhs.ravel(), solver=gaussx.CGSolver(), preconditioner=gs.build_multigrid_solver(...).as_preconditioner())` |
| A dense `(N, N)` Laplacian matrix for a prior covariance, its Cholesky or log-determinant | `gs.helmholtz_operator` / `gs.masked_helmholtz_operator` with `gaussx.solve`, `gaussx.logdet`, `gaussx.sqrt` |
| A Thomas algorithm for vertical diffusion | `gs.solve_tridiagonal` |
| A hand-written RK3 / AB2 / leapfrog loop | `integrix.march.rk3_ssp_step`, `integrix.march.ab2_step`, `integrix.march.leapfrog_raf_step`, or `integrix.march.RK3SSP` with `sesmax_core.solve_pde.solve_ocean_pde` |
| `jnp.sum(q * dx * dy)` | `fvx.area_sum(q, grid)` (excludes the ghost ring; `mask=` drops land) |
| Kinetic energy / PV / Okubo–Weiss by hand | `fvx.kinetic_energy`, `fvx.sw_potential_vorticity`, `fvx.okubo_weiss_parameter` |
| A `vmap` over layers | `sesmax_core.vertical.multilayer(ops.laplacian)(psi_layers)` |

## Self-check before you finish

- Every field read by a stencil has its ghost ring filled for the boundary
  condition you mean (periodic, wall, open), at every RHS evaluation.
- Each move has the stagger you think: T fields into `src="T"`,
  velocities on U / V faces, vorticity on X corners.
- The total of a conserved tracer (`fvx.area_sum`) is constant over a run
  with periodic or closed boundaries.
- `dt` respects the CFL limit (`dt ≲ dx / max|u|`; on the sphere use
  `grid.min_cell_width`).
- A float32 input stays float32; `jax.jit` and `jax.grad` run through your
  RHS.

If finitevolX lacks what you need, keep your addition small and shaped like
finitevolX (a pytree holding `grid` and `mask`, built on the raw
stencils and writing only the interior) and consider proposing it upstream
at <https://github.com/jejjohnson/gridax/issues>.

<!-- skill-tree:begin · generated by geostack scripts/skill_tree.py; do not edit by hand -->
## Builds on

Each card below says what a library under this skill guarantees.
A card is enough to call it.
Load a skill only when its card's last line applies.

### gaussx · extra · 2 of 7 units

#### Structures and primitives · ⓪ ① · skill `gaussx:structured-gaussian-linalg`

- **Provides.** Covariance and precision operators that keep their structure, and the verbs that use it.
- **Nouns.** `gaussx.Kronecker`, `gaussx.SumOfKroneckers`, `gaussx.BlockDiag`, `gaussx.BlockTriDiag`, `gaussx.Toeplitz`, `gaussx.SparseOperator`, `gaussx.low_rank_plus_diag`; any `lineax.AbstractLinearOperator`, tagged.
- **Verbs.** `gaussx.solve`, `gaussx.logdet`, `gaussx.cholesky`, `gaussx.sqrt`, `gaussx.inv`, `gaussx.diag`, `gaussx.trace`; `gaussx.solve_matrix`, `gaussx.woodbury_solve`, `gaussx.safe_cholesky`.
- **Adverbs.** `solver=` on `gaussx.solve`; left out, the verb dispatches on the structure.
- **Keep.** A dense matrix enters as `lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)`; factors are operators, not arrays.
- **Never.** Call `as_matrix()` and then a dense `jnp.linalg` solve, or build `jnp.kron(A, B)`.
- **One rung down.** Open `gaussx:structured-gaussian-linalg` to build a new structure or to see which verb has a fast path for which noun.

#### Strategies · ②, setup · skill `gaussx:solvers-and-preconditioners`

- **Provides.** The ways a solve or a log-determinant is computed, as objects a caller passes in.
- **Nouns.** `gaussx.AutoSolver`, `gaussx.DenseSolver`, `gaussx.CGSolver`, `gaussx.PreconditionedCGSolver`, `gaussx.BBMMSolver`, `gaussx.MINRESSolver`, `gaussx.LSMRSolver`, `gaussx.SparseCholeskySolver`; `gaussx.SLQLogdet`, `gaussx.DenseLogdet`; `gaussx.NystromPreconditioner`, `gaussx.PartialCholeskyPreconditioner`.
- **Verbs.** `strategy.solve(operator, vector)` and `strategy.logdet(operator, key=key)`; `gaussx.linear_solve(operator, vector, solver=...)` for an operator or a `(matvec, shape)` pair.
- **Adverbs.** `preconditioner=` on the CG family; `key=` for a stochastic log-determinant, or `gaussx.KeyedSolver`; `gaussx.ComposedSolver` to take the solve from one strategy and the log-determinant from another.
- **Keep.** A function above gaussx takes `solver: gaussx.AbstractSolverStrategy | None = None` and hands it down unchanged; `None` means structural dispatch.
- **Never.** Write a CG, Lanczos or Hutchinson loop, or fix a solver inside a function that could take one.
- **One rung down.** Open `gaussx:solvers-and-preconditioners` to choose between strategies, tune a preconditioner or build a randomized factorisation.

### integrix · extra · 0 of 2 units

It uses fewer than 3 names of each unit, so no card. The front door is `integrix:quadrature-and-expectations-with-integrix`.

### pipekit · base dependency · no skill yet

- **Provides.** Carrier-agnostic operator graphs: the `Operator` base class, composition into pipelines and graphs, named-array operators, JAX and Equinox operators, and time-stepping and data-assimilation cycles, split over seven packages that each have their own import name.
- **Start from.** `pipekit_ops.Operator`, `pipekit_ops.Sequential`, `pipekit_ops.Graph`, `pipekit_ops.Input`, `pipekit_ops.Lambda`, `pipekit_ops.StatefulOperator`, `pipekit_array.DimArray`, `pipekit_jax.JaxOperator`.
- **Keep.** Subclass `Operator` and implement `_apply`, and store each `__init__` argument as an attribute of the same name so `get_config()` and the serialised state round-trip; import names from `pipekit_ops`, `pipekit_array`, `pipekit_cycle` or `pipekit_jax`, because `import pipekit` exports nothing.
- **Never.** Override `__call__` on an operator; implement `_apply`, because `__call__` decides between running eagerly and recording a `Node` in a `Graph`.
- **Open the repository when.** You need cycles, stores, caching or parallel maps, or more than these names: it has no skill yet, so read its `AGENTS.md` and its section of GeoStack's capability index.

### geophysix · base dependency · no skill yet

- **Provides.** Physical facts about the Earth as closed forms, on NumPy, JAX and other array libraries: constants and trace gases, the ellipsoid and the sphere, rotation, the gas law, sea-water density, column units and the wind.
- **Start from.** `geophysix.constants.STANDARD`, `geophysix.earth.WGS84`, `geophysix.earth.SPHERE`, `geophysix.earth.cell_area`, `geophysix.rotation.coriolis`, `geophysix.thermo.number_density`, `geophysix.columns.convert`, `geophysix.seawater.linear_density`, `geophysix.wind.wind_speed`.
- **Keep.** SI units, degrees for angles, and constants as records: pass another `earth=` or `constants=` record for a sphere or an old value; no function reads a global.
- **Never.** Write an Earth radius, g₀, Ω, k_B or a molar mass as a literal, or import a constant from a library above this one; take it from the record.
- **Open the repository when.** You need more than these names, or a formula it does not have: it has no skill yet, so read its `AGENTS.md` and its section of GeoStack's capability index.

### spectraldiffx · extra · skill `spectraldiffx:spectral-methods-with-spectraldiffx`

- **Provides.** Pseudospectral tools in JAX: Fourier, Chebyshev and spherical-harmonic grids that own their wavenumbers, spectral derivatives with dealiasing, DCT and DST transforms, and Poisson and Helmholtz solvers for periodic, Dirichlet and Neumann boundaries.
- **Start from.** `spectraldiffx.FourierGrid2D`, `spectraldiffx.SpectralDerivative2D`, `spectraldiffx.SpectralHelmholtzSolver2D`, `spectraldiffx.solve_helmholtz_2d`, `spectraldiffx.solve_poisson_2d`, `spectraldiffx.build_capacitance_solver`, `spectraldiffx.dct`, `spectraldiffx.dst`.
- **Keep.** Take wavenumbers from the grid (`grid.k`, `grid.K2`), put x on the last array axis, and pass products of fields through `jacobian`, `advection_scalar` or `apply_dealias`, not linear operators.
- **Never.** Solve a domain with walls using an FFT; pass the boundary condition per axis to `solve_helmholtz_2d` and let it pick FFT, DST or DCT.
- **Open the skill when.** You need Chebyshev or spherical grids, mixed per-axis boundary conditions or non-zero boundary values, a land-masked basin solve, spectral filters, or the exact meaning of the `dirichlet` and `neumann` options.

Used by (pointers, not skills to load): geotoolz, plumax, sesmax, tracerax.

<!-- skill-tree:end -->
