# Arakawa C-Grid Masks

finitevolX provides three mask classes — `Mask1D`, `Mask2D`, `Mask3D` —
that describe land/ocean geometry on an Arakawa C-grid.  Every class
operator in the package takes an optional ``mask`` class attribute of
the matching dimension, so masks travel with operators rather than
getting threaded through every call.  See the
[Operator API](#operator-api-how-operators-consume-masks) section below
for the full pattern and the stagger → mask field mapping.

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
## Operator API: how operators consume masks

Every class operator in `gridax` takes an optional `mask` class
attribute of the matching dimension.  The mask travels with the
operator; once constructed, method calls don't need to pass the mask
through per call.

```python
import gridax as fvx

grid = fvx.CartesianCGrid2D.from_interior(64, 64, 1e6, 1e6)
mask = fvx.Mask2D.from_mask(h_mask)

# Mask as a class attribute — set once, consumed on every call.
diff = fvx.Difference2D(grid=grid, mask=mask)
adv  = fvx.Advection2D(grid=grid, mask=mask)
diff_op = fvx.Diffusion2D(grid=grid, mask=mask)

# Unmasked operators are the default — just omit `mask=`.
diff_noop = fvx.Difference2D(grid=grid)   # mask=None
```

### Where the mask enters

The coefficients and the raw stencils know no mask. A canonical function (Layer 2) takes it as the keyword `mask=`, and a binding (Layer 3) holds it and passes it on ([ADR 1](design/adr/0001-canonical-functions.md)). The code is moving to that rule family by family; the table says where each family is today:

| Layer | Example | Mask-aware? |
|---|---|---|
| Layer 0 — coefficients | `difference_offsets`, `difference_coeffs`, `weno_stencils`, … in `gridax.stencils` (numbers, no arrays) | **No** |
| Layer 1 — raw stencils | `diff_x_fwd_2d`, `avg_y_bwd_2d`, `staggered_difference`, … in `gridax.stencils` | **No** |
| Layer 2 — canonical functions | `difference`, `interpolate`, `divergence`, `curl`, `advect`, `reconstruct`, `diffuse`, `biharmonic`, `coriolis`, `arakawa_jacobian`, `kinetic_energy`, `shear_strain`, `tensor_strain`, `area_sum`, … | **Yes**, `mask=` |
| Pointwise and array helpers | `enstrophy`, `okubo_weiss`, `potential_vorticity`, the collocated `gridax.analysis` calculus, … | **No** (`gridax.analysis.adaptive_*` take one) |
| Layer 3 — bindings | `Difference2D`, `Interpolation2D`, `Divergence2D`, `Vorticity2D`, `Coriolis2D`, `Advection2D`, `Diffusion2D`, `MomentumAdvection2D`, `Energetics2D`, `Strain2D`, `QGPotentialVorticity2D`, `ArakawaJacobian2D`, and their `1D` / `3D` / `Spherical*` siblings | **Yes**, via `self.mask` |

The pointwise and stencil diagnostics below have Layer-3 class forms, so
in a masked domain you don't need to mask them by hand:

| Helper | Layer-3 class form |
|---|---|
| `kinetic_energy`, `bernoulli_potential`, `available_potential_energy` | `Energetics2D(grid, mask)` methods of the same name |
| `shear_strain`, `tensor_strain` | `Strain2D.shear`, `Strain2D.tensor` |
| `strain_magnitude_squared`, `okubo_weiss` | `Strain2D.magnitude_squared(u, v)`, `Strain2D.okubo_weiss(u, v)` |
| `sw_potential_vorticity` | `Vorticity2D.potential_vorticity` (the relative vorticity is `Vorticity2D.relative_vorticity` or `Difference2D.curl`) |
| `enstrophy`, `potential_enstrophy` | `Vorticity2D.enstrophy(u, v)`, `Vorticity2D.potential_enstrophy(u, v, h, f)` |
| `qg_potential_vorticity`, `stretching_term`, `potential_vorticity_multilayer` | `QGPotentialVorticity2D.__call__`, `.stretching`, `.multilayer` |
| `arakawa_jacobian` | `ArakawaJacobian2D(grid, mask)` (full-shape output, zero ghost ring) |
| `_geostrophic_velocity_sphere` (private) | `SphericalDifference2D.geostrophic_velocity(h, f, gravity)` |

```python
energetics = fvx.Energetics2D(grid=grid, mask=mask)
ke = energetics.kinetic_energy(u, v)       # dry T-cells exactly 0

strain = fvx.Strain2D(grid=grid, mask=mask)
ow = strain.okubo_weiss(u, v)              # T-points, dry T-cells exactly 0
```

`Energetics2D`, `Strain2D`, `QGPotentialVorticity2D` and
`ArakawaJacobian2D` zero every field input on the dry *interior* cells of
its stagger before any stencil reads it, and mask outputs with
`jnp.where`, so interior land values stored as `NaN` (e.g. from
`Mask2D.from_center`) neither leak into wet cells nor survive at dry ones.
The ghost ring is BC-owned: it is passed through unchanged and must be
finite.

`SphericalDifference2D.geostrophic_velocity` only guards the division (it
sets the face-averaged `f` to `1` on dry faces); its 4-point stencils read `h` on neighbouring
land cells for wet coastal faces, so **land `h` must be finite** there.
The same holds for the other existing class operators, which mask outputs
but do not sanitize inputs.

Diagnostics **without** a class form:

- The domain integrals are `area_sum(field, grid, mask=mask)`.
- `vertical_velocity` takes a per-call `mask=Mask3D`: it zeroes the
  divergence on dry T-points (`mask.wet()`) and its output on dry W-faces
  (`mask.wet(at="W")`); the output has the same `[Nz, Ny, Nx]` shape as `mask.wet(at="W")`.

The canonical functions take the mask: `kinetic_energy(u, v, grid, mask=mask)`.
For a helper that does not, apply it at the call site:
`jnp.where(mask.wet(at="X", kind="strict"), enstrophy(omega), 0.0)`.  For
pointwise helpers whose inputs live on different staggers (e.g.
`okubo_weiss(sn, ss, omega)` with `sn` at T-points and `ss`, `omega` at
X-points), the class form also does the X → T averaging for you.

### Dimension → mask type mapping

Operators take the mask class matching their dimension:

| Operator class | Mask type | Example |
|---|---|---|
| `*1D` (Difference1D, Interpolation1D, Advection1D) | `Mask1D \| None` | `Difference1D(grid=g, mask=Mask1D.from_mask(h))` |
| `*2D` (every 2-D operator) | `Mask2D \| None` | `Advection2D(grid=g, mask=Mask2D.from_mask(h))` |
| `*3D` (every 3-D operator) | `Mask3D \| None` | `Diffusion3D(grid=g, mask=Mask3D.from_mask(h))` |

Spherical 2-D / 3-D operators take the **Cartesian** `Mask2D` /
`Mask3D` — the mask geometry is coordinate-agnostic at the
post-compute-multiply layer.  A dedicated `SphericalMask2D` /
`SphericalMask3D` is a TODO for future work.

### Stagger → mask field mapping

The core convention is **one operator output → one mask field**, chosen
by the operator's output stagger.  Every operator follows the same
mapping:

| Output stagger | Mask field | Operators / methods |
|---|---|---|
| T (cell centre) | `mask.wet()` | divergence, laplacian, U→T / V→T / X→T differences and interpolations, `Advection*`, `Diffusion*`, `BiharmonicDiffusion*`, `Energetics2D.*`, `Strain2D.tensor` / `.magnitude_squared` / `.okubo_weiss`, `QGPotentialVorticity2D.*`, `ArakawaJacobian2D` |
| U (east face) | `mask.wet(at="U")` | T→U / X→U / V→U differences and interpolations, `grad_perp` u-component, `Coriolis*` du_cor, `MomentumAdvection*` du_adv, `SphericalDifference2D.geostrophic_velocity` u_g |
| V (north face) | `mask.wet(at="V")` | T→V / X→V / U→V, `grad_perp` v-component, `Coriolis*` dv_cor, `MomentumAdvection*` dv_adv, `SphericalDifference2D.geostrophic_velocity` v_g |
| X (NE corner) | `mask.wet(at="X", kind="strict")` | curl, T→X / U→X / V→X, relative & potential vorticity, `Vorticity2D.enstrophy` / `.potential_enstrophy`, `Strain2D.shear`, `pv_flux_*` U/V-stagger outputs where applicable |

The X-output mapping uses the **strict** 4-of-4 corner mask
(`mask.wet(at="X", kind="strict")`) because a corner-output value is trusted only when
all four surrounding T-cells are wet.  The lenient
`mask.wet(at="X")` mask is not currently consumed by any operator — it's
available for users building their own.

### Four patterns for mask application

Under the hood, operators apply masks in one of four patterns
depending on what's safe.  You don't need to know which is which for
normal usage — the result is always "dry cells at the output stagger
are exactly 0" — but the distinctions matter when customising or
inheriting:

**Pattern 1 — Post-compute multiply.** Used by the majority of
operators: `Difference*`, `Interpolation*`, `Vorticity3D`,
`Spherical*`, `Energetics2D`, `QGPotentialVorticity2D`,
`ArakawaJacobian2D`.  Compute the output as if all-ocean, then multiply by the
stagger-matched mask field at the end.  Cheap and simple.  The #206
diagnostic classes (`Energetics2D`, `Strain2D`, `QGPotentialVorticity2D`,
`ArakawaJacobian2D`) use the `NaN`-safe variant: zero the inputs on dry
cells first, then select with `jnp.where(mask, out, 0)` instead of
multiplying.

**Pattern 2 — Pass-down into sub-operators.** Used by composed
operators with no tricky division — `Divergence2D`, `Vorticity2D`,
`MomentumAdvection2D`, `Strain2D` (its T-point combinations mask the
X-point shear and vorticity before averaging them X → T, so a coastal
T-cell averages in zeros from its dry corners).  The outer class wires its mask into every
internal sub-operator at `__init__` time, so intermediate staggered
fields are already zero at dry cells.  The outer method trusts the
pass-down and typically doesn't need an extra multiply.

* **Special case for `Vorticity2D.potential_vorticity`**: the
  `(zeta + f) / h` division would produce NaN at every dry X-corner
  under pass-down (because `h_on_q == 0`).  The method preserves the
  existing "NaN at wet zero-thickness" sentinel but replaces
  mask-induced NaNs with exact 0 via a final
  `jnp.where(mask.wet(at="X", kind="strict"), out, 0)`.  Same fix applies to
  `SphericalVorticity2D.potential_vorticity` and
  `Vorticity2D.potential_enstrophy` (built on it).
  `SphericalDifference2D.geostrophic_velocity` faces the same division
  (a dry face can have `f_on_face = 0`): it sets the face-averaged `f`
  to `1` on every dry face right before dividing, so the value *and* its
  reverse-mode gradient stay finite, then masks with `jnp.where`.
  `Vorticity2D.potential_vorticity` likewise divides by `1` at dry
  corners, where pass-down masking makes `h_on_q` exactly `0`.

**Pattern 3 — Intermediate flux masking.** Used by `Diffusion2D` and
`Diffusion3D` only.  Post-compute multiply is **not** sufficient for
diffusion because the divergence at wet T-cells adjacent to land would
be contaminated by polluted dry-side face fluxes.  Instead the class
wrappers apply the mask in three steps:

```
flux_x *= mask.wet(at="U")        # U-face stage
flux_y *= mask.wet(at="V")        # V-face stage
tendency *= mask.wet()      # final output
```

The canonical function `diffuse` does it, on the levels of a 3-D grid at once;
the classes call it.

**Pattern 4 — The masked harmonic operator applied twice.**
Used by `BiharmonicDiffusion2D` / `3D` and `SphericalBiharmonicDiffusion2D`
/ `3D`.  The tendency is `−κ · Dm(Dm(h))`, where `Dm` is the masked
no-flux harmonic operator of Pattern 3 (`biharmonic` applies `diffuse` twice with the same mask).  Dry inputs are
zeroed with `jnp.where` first, so land may hold `NaN` or a fill value.
Because the second pass also zeroes the wet→dry fluxes, the dry zeros of
the intermediate `Dm(h)` are never read: the coast is a no-flux wall for
both `h` and `∇²h`, exactly like the closed domain walls.  The result is
conservative (`Σ_wet ∂h/∂t = 0`) and symmetric on the wet cells, and no
land value reaches a wet cell.  (Before #292 the inner Laplacian was
unmasked and post-multiplied by `mask.wet()`, which read land values and was
not conservative.)

### Pattern A vs Pattern B for 3-D operators

The 3-D wrappers that vmap an inner 2-D operator over z-levels
(`Vorticity3D`, `Spherical*3D`, `MomentumAdvection3D`)
use **Pattern A (post-compute)**: the inner 2-D op is always
constructed `mask=None` and the 3-D wrapper owns the mask, applying `* self.mask.<stagger>` after the vmap and
`zero_z_ghosts`.  This is cleaner than trying to thread per-z mask
slices into the vmap'd 2-D op.

The canonical functions need no such pattern: `advect`, `diffuse`, `coriolis`
and the others act on the last two axes of the interior levels, with the 3-D
masks, and pad a zero ghost slab.

### Stencil tiers in advection

`advect` and `reconstruct` read `mask.tiers(axis)` at each call. The stored counts make a tier one comparison, so nothing is cached on an operator: `Advection2D` with a mask is `advect` with that mask, and takes any scheme.

### Changing the mask on an operator

`equinox` Modules are immutable — `op.mask = new_mask` will raise.  To
"change" a mask on an operator, use `eqx.tree_at` to produce a new
operator instance with the field updated:

```python
import equinox as eqx

# Start with one mask ...
adv_summer = fvx.Advection2D(grid=grid, mask=summer_mask)

# ... and produce a new instance with a different mask.  The grid and
# reconstruction module are shared (no copy); only `mask` (and, for
# Advection, the pre-built hierarchies — eqx.tree_at reconstructs the
# __init__ path correctly for all affected fields) is swapped.
adv_winter = eqx.tree_at(lambda o: o.mask, adv_summer, winter_mask)
```

For composed operators (Divergence2D, Vorticity2D, MomentumAdvection2D,
etc.) the outer `self.mask` is independent of the mask stored on the
internal sub-operators, so you may also need to swap those — the
simpler alternative is usually to just construct a fresh operator
from scratch:

```python
# Cleaner for composed ops
vort_winter = fvx.Vorticity2D(grid=grid, mask=winter_mask)
```

### Migration from the old per-call kwarg API

Some operators used to take `mask=` as a per-call keyword argument.
In the current API, the mask is always a class field.  Translation:

```python
# Old (removed)
adv = fvx.Advection2D(grid=grid)
adv(h, u, v, scheme=WENO(5), mask=mask)

cor = fvx.Coriolis2D(grid=grid)
cor(u, v, f, mask=mask)

# New
adv = fvx.Advection2D(grid=grid, mask=mask)
adv(h, u, v, scheme=WENO(5))

cor = fvx.Coriolis2D(grid=grid, mask=mask)
cor(u, v, f)
```

`Diffusion2D`, `Diffusion3D`, `Difference2D.grad_perp`,
`kinetic_energy`, `enstrophy`, and `potential_enstrophy` used to take
raw-array `mask_h` / `mask_u` / `mask_v` keyword arguments.  Those are
**all removed**:

* `Diffusion2D` / `Diffusion3D` take a `Mask2D` / `Mask3D` class field
  instead.
* `Difference2D.grad_perp` uses the `self.mask` class field (set on
  `Difference2D` at construction).
* `kinetic_energy` takes the grid and the mask. `enstrophy` and
  `potential_enstrophy` are pointwise and mask-free; apply masks at the call
  site:

  ```python
  ke = fvx.kinetic_energy(u, v, grid, mask=mask)
  z  = fvx.enstrophy(omega)     * mask.wet(at="X", kind="strict")
  pe = fvx.potential_enstrophy(q, h) * mask.wet(at="X", kind="strict")
  ```
