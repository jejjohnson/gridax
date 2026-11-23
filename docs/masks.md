# Arakawa C-Grid Masks

finitevolX provides three mask classes — `Mask1D`, `Mask2D`, `Mask3D` —
that describe land/ocean geometry on an Arakawa C-grid.  Every function
that needs a grid takes the mask of the matching dimension as the keyword
`mask=`, and `GridOps(grid, mask)` holds both.  See
[how the functions use a mask](#operator-api-how-operators-consume-masks)
below for the point each output is masked at.

The rest of this page documents ``Mask2D`` — the 2-D class — but the
same factories (`from_mask`, `from_dimensions`, `from_center`,
`from_u_face` …) and the same staggered-derivation pipeline are
available on ``Mask1D`` (no v-face, no corner) and ``Mask3D`` (adds a
``w`` vertical face and native 3-D classification / stencil capability).

## The Arakawa C-grid layout

`Mask2D` builds all staggered masks from a single cell-centre
wet/dry field, following the Arakawa & Lamb (1977) grid layout:

```
y
^
:           :
w-----v-----w..
|           |
|           |
u     h     u
|           |
|           |
w-----v-----w..   > x
```

| Point                | Location                  | Variable                         |
|----------------------|---------------------------|----------------------------------|
| **h**                | cell centre               | tracers, height, pressure        |
| **u**                | east face                 | zonal velocity                   |
| **v**                | north face                | meridional velocity              |
| **xy_corner**        | NE corner (lenient)       | vorticity                        |
| **xy_corner_strict** | NE corner (strict)        | streamfunction                   |

All staggered points use the *same-index, positive half-step* convention:
``u[j, i]`` is the east face of ``h[j, i]``, ``v[j, i]`` is the north face,
and ``xy_corner[j, i]`` is the NE corner — matching ``CartesianCGrid2D``.
See [the Arakawa C-grid discretization notes](cgrid_discretization.md) for
the underlying convention and a comparison with negative half-step
alternatives.

## Creating masks

All you need is a binary h-grid mask (True = ocean, False = land).
The factory method derives everything else:

```python
import numpy as np
from gridax import Mask2D

# Rectangular basin with land boundaries
n = 10
h_mask = np.ones((n, n), dtype=bool)
h_mask[0, :] = h_mask[-1, :] = False
h_mask[:, 0] = h_mask[:, -1] = False

# With an island
h_mask[4:7, 4:7] = False

masks = Mask2D.from_mask(h_mask)
```

For an all-ocean domain, use the shortcut:

```python
masks = Mask2D.from_dimensions(ny=12, nx=12)
```

Or construct from any float field at a known grid position, where
`NaN` marks dry cells:

```python
# From a field at cell centres (T-points): SSH, temperature, pressure, …
masks = Mask2D.from_center(ssh_field)

# From a field at u-faces, v-faces, or xy-corners.  Pass `mode=` to
# choose the inversion strategy when the inverse mapping back to the
# h-grid is non-unique:
#   mode='permissive'  → h wet iff *any* surrounding face/corner wet
#   mode='conservative' → h wet iff *all* surrounding face/corner wet
masks = Mask2D.from_u_face(u_field, mode="permissive")
masks = Mask2D.from_v_face(v_field, mode="permissive")
masks = Mask2D.from_corner(vorticity_field, mode="conservative")
```

[`Mask2D.from_grid(grid, wet_t)`](xref:api#gridax.Mask2D) is `from_mask` with the shape checked against the grid.

## Staggered variable locations

The wet points are read by point of the staggering, with the same names the grid and the fields use, or a field's stagger:

```python
masks.wet()  # T: the cell centres
masks.wet(at="U"), masks.wet(at="V")  # a face is wet where both its cells are
masks.wet(at="X")  # a corner with at least one wet cell
masks.wet(at="X", kind="strict")  # a corner whose four cells are wet
masks.wet(at="X", kind="y_wall")  # also "x_wall", "convex" and "valid"
```

Each variable type sits at a different position within a grid cell.
The figures below show the actual staggered positions for several
domain topologies:

### Rectangular basin

![Rectangular basin](images/demo_masks/staggered_basin.png)

### Basin with island

![Basin with island](images/demo_masks/staggered_island.png)

### Zonal channel

![Zonal channel](images/demo_masks/staggered_channel.png)

### Irregular coastline

![Irregular coastline](images/demo_masks/staggered_irregular.png)

## Land / coast classification

`mask.classification` has four levels (0 = land, 1 = coast,
2 = near-coast, 3 = open ocean), derived from the wet cells when asked for:

### Rectangular basin

![Classification: basin](images/demo_masks/classification_basin.png)

### Basin with island

![Classification: island](images/demo_masks/classification_island.png)

### Zonal channel

![Classification: channel](images/demo_masks/classification_channel.png)

### Irregular coastline

![Classification: irregular](images/demo_masks/classification_irregular.png)

## Vorticity boundary classification

At xy-corner points, cells are classified based on their relationship
to adjacent velocity faces:

- **xy_corner_valid** — interior: all 4 adjacent velocity faces are wet
- **xy_corner_y_wall** — on a vertical (y-direction) boundary
- **xy_corner_x_wall** — on a horizontal (x-direction) boundary
- **xy_corner_convex** — at convex corners (both boundary types)

## Stencil capability and adaptive WENO

Each cell stores how many contiguous wet neighbours it has in each
direction via `StencilCapability2D`. This drives adaptive stencil
selection for WENO reconstruction near coastlines:

```python
# Mutually-exclusive masks: largest usable stencil at each point
adaptive = masks.tiers("x")  # at="U", sizes=(2, 4, 6), …
# adaptive[2]  → 1st-order upwind only
# adaptive[4]  → WENO3
# adaptive[6]  → WENO5
# adaptive[8]  → WENO7
# adaptive[10] → WENO9
```

## Coastal ghosts

The dry points next to the water are to a coast what the ghost ring is to a domain edge. `coast_ghosts` gives them one side at a time, with the offset of the wet point they mirror, at any point of the staggering and for as many layers as a stencil reads:

```python
east = masks.coast_ghosts("x", "high")  # coasts on the east side of the water
east.points  # the dry T points whose west neighbour is wet
east.offset  # -1: that neighbour, the φ₁ of the mirror rules
masks.coast_ghosts("y", "low", at="V", layer=2)  # the second layer of V points
```

## Jupyter notebook

A complete demo is in the [masks notebook](notebooks/demo_masks.ipynb).

---

(operator-api-how-operators-consume-masks)=
## How the functions use a mask

```python
import gridax as fvx

grid = fvx.CartesianCGrid2D.from_interior(64, 64, 1e6, 1e6)
mask = fvx.Mask2D.from_mask(h_mask)

# as a keyword of a function
dq = fvx.advect(q, u, v, grid, mask=mask)

# held once, with the grid
ops = fvx.GridOps(grid, mask)
dq = ops.advect(q, u, v)
dh_dx = ops.difference(h, axis="x", src="T", dst="U")
ke = ops.kinetic_energy(u, v)  # dry T cells exactly 0

# without a mask
dq = fvx.advect(q, u, v, grid)
```

### Where the mask enters

The coefficients and the raw stencils know no mask. A canonical function (Layer 2) takes it as the keyword `mask=`, and `GridOps` (Layer 3) holds it and passes it on ([ADR 1](design/adr/0001-canonical-functions.md)).

| Layer | Example | Mask-aware? |
|---|---|---|
| Layer 0 — coefficients | `difference_offsets`, `difference_coeffs`, `weno_stencils`, … in `gridax.stencils` (numbers, no arrays) | **No** |
| Layer 1 — raw stencils | `diff_x_fwd_2d`, `avg_y_bwd_2d`, `staggered_difference`, … in `gridax.stencils` | **No** |
| Layer 2 — canonical functions | `difference`, `interpolate`, `divergence`, `curl`, `advect`, `reconstruct`, `diffuse`, `biharmonic`, `coriolis`, `advect_momentum`, `arakawa_jacobian`, `kinetic_energy`, `shear_strain`, `area_sum`, … | **Yes**, `mask=` |
| Pointwise and array helpers | `enstrophy`, `okubo_weiss`, `potential_vorticity`, the collocated `gridax.analysis`, … | **No** |
| Layer 3 — the binding | `GridOps(grid, mask)` | **Yes**, it holds it |

For a helper that takes no mask, apply it at the call site: `gridax.apply_mask(f, mask, at=…)`, or `jnp.where(mask.wet(at="X", kind="strict"), enstrophy(omega), 0.0)`. The diagnostics that need an average from X to T have a canonical function that does it: `strain_rate_squared(u, v, grid, mask=mask)` and `okubo_weiss_parameter(u, v, grid, mask=mask)`, beside the pointwise `strain_magnitude_squared(sn, ss)` and `okubo_weiss(sn, ss, omega)`.

### What a dry point may hold

Anything, `NaN` too: no operation with a mask reads a dry value off the ghost ring, so land values reach neither the result nor its gradient (`tests/test_dry_inputs.py` checks every operation). Each kind of field has its rule for the stencils that reach across a coast:

| Field | Beyond the coast | Because |
|---|---|---|
| velocities and fluxes, at U, V or W | zero | nothing flows through a coast |
| a tracer at T, in `advect`, `diffuse`, `biharmonic`, `laplacian` | no flux through the face | the coast is a closed wall |
| ψ at T, in `grad_perp` | a corner is the mean of its wet cells | one value per corner, so the velocity stays non-divergent |
| h, in `geostrophic_velocity` | the slope is the mean of the wet differences around the face | a uniform slope is kept |
| a field at X | a corner is read if one of its cells is wet | corners on a coast hold boundary values |

A velocity point next to land in `laplacian(u, grid, at="U", mask=mask)` reads zero on the dry points, which is right for the wall the component is normal to; a tangential wall needs `fill_ghosts(…, mask=mask)` first, on a scheme with `Adaptive(mask="none")`.

`Adaptive(scheme, mask="none")` on `difference` and `interpolate` is the one way to read land: it masks the output only. The ghost ring is BC-owned: it is passed through unchanged, whatever the mask says there, and must be finite.

### Dimension → mask type

| Grid | Mask |
|---|---|
| 1-D | `Mask1D` |
| 2-D, Cartesian or spherical | `Mask2D` |
| 3-D, Cartesian or spherical | `Mask3D` |

The mask classes are coordinate-agnostic: a spherical grid takes the same `Mask2D` or `Mask3D` as a Cartesian one.

### Output point → wet points

The convention is **one output → one set of wet points**, chosen by the point the output lives at:

| Output at | Wet points | Functions |
|---|---|---|
| T (cell centre) | `mask.wet()` | `divergence`, `laplacian`, differences and interpolations to T, `advect`, `diffuse`, `biharmonic`, `kinetic_energy`, `bernoulli_potential`, `tensor_strain`, `strain_rate_squared`, `okubo_weiss_parameter`, `qg_potential_vorticity`, `arakawa_jacobian` |
| U (east face) | `mask.wet(at="U")` | differences and interpolations to U, and the first component of `grad_perp`, `gradient`, `coriolis`, `advect_momentum`, `pv_flux` and `geostrophic_velocity` |
| V (north face) | `mask.wet(at="V")` | differences and interpolations to V, and the second component of the same functions |
| X (NE corner) | `mask.wet(at="X", kind="strict")` | `curl`, differences and interpolations to X, `sw_potential_vorticity`, `shear_strain` |
| W (vertical face) | `mask.wet(at="W")` | `vertical_velocity` |

The X points use the **strict** 4-of-4 corner mask (`mask.wet(at="X", kind="strict")`) because a corner value is trusted only when all four surrounding T cells are wet.  No function reads the lenient `mask.wet(at="X")`; it is there for your own.

### Four ways to apply a mask

The result is always "dry points of the output are exactly 0". How a function gets there depends on what its stencil reads:

**Mask the output.** The moves, the calculus on one component, `coriolis` and the diagnostics zero the dry values of their inputs, compute as if every point were wet, and then mask the output with `where(wet, out, 0)`.

**Mask the intermediate fields.** `advect_momentum`, `sw_potential_vorticity`, `strain_rate_squared` and `okubo_weiss_parameter` mask the staggered fields they are built from before combining them, so a coastal T cell averages in zeros from its dry corners.

* `sw_potential_vorticity` divides by the thickness at the corners, which is exactly `0` at a dry one: it divides by `1` there and masks with `where`, so the value *and* its reverse-mode gradient stay finite. A wet corner of zero thickness is still `NaN`. `geostrophic_velocity` faces the same division (a dry face can have `f = 0`) and does the same.

**Mask the fluxes.** `advect` and `diffuse` zero the flux through a face with a dry side before the divergence. Masking only the tendency is **not** enough: a wet T cell next to land would be contaminated by the fluxes of its dry faces.

```
flux_x *= mask.wet(at="U")  # U faces
flux_y *= mask.wet(at="V")  # V faces
tendency *= mask.wet()      # final output
```

**Apply the masked harmonic operator twice.** The tendency of `biharmonic` is `−κ · Dm(Dm(h))`, where `Dm` is the masked no-flux `diffuse`.  Because the second pass also zeroes the wet→dry fluxes, the dry zeros of the intermediate `Dm(h)` are never read: the coast is a no-flux wall for both `h` and `∇²h`, exactly like the closed domain walls.  The result is conservative (`Σ_wet ∂h/∂t = 0`) and symmetric on the wet cells, and no land value reaches a wet cell.

On a 3-D grid the functions act on the last two axes of the interior levels, with the 3-D masks, and pad a zero ghost slab.

### Stencil tiers in advection

`advect` and `reconstruct` read `mask.tiers(axis)` at each call. The stored counts make a tier one comparison, so nothing is cached beside the mask, and a mask takes any scheme.

### Changing the mask

A mask is an argument: pass another one, or build another `GridOps`. Both are pytrees, so a mask can change under `jit` without a retrace when its shape does not:

```python
ops_summer = fvx.GridOps(grid, summer_mask)
ops_winter = fvx.GridOps(grid, winter_mask)
dq = ops_summer.advect(q, u, v, mask=winter_mask)  # for one call
```
