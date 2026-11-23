# Schemes

A scheme says which values an operation reads around its output, and with what weights. `gridax.schemes` has five, one object each, and the same object goes to every operation that takes `scheme=`.

## The five schemes

| Scheme | Arguments (default) | Stencil | Taken by |
|---|---|---|---|
| `Centred(order, data)` | any even `order` (2) | symmetric | `difference`, `interpolate`, `reconstruct` |
| `Biased(order, side, data)` | any `order` (2), `side` required | leans to `side` | `difference`, `interpolate`, `reconstruct` |
| `Upwind(order)` | any odd `order` (1) | leans upwind | `reconstruct` |
| `WENO(order, variant)` | any odd `order` ≥ 3 (5); `variant` `"js"`, or `"z"` up to order 9 | nonlinear, upwind | `reconstruct` |
| `TVD(limiter)` | `limiter` (`"minmod"`) | nonlinear, upwind | `reconstruct` |

**The order** is the formal order of accuracy everywhere. A centred stencil has an even order by symmetry. An upwind stencil of order $2k - 1$ is the centred stencil of order $2k - 2$ with one more upwind cell; its leading error is an even derivative, so it is dissipative. WENO of order $2r - 1$ blends $r$ sub-stencils of $r$ cells; its linear weights, its sub-stencils and their smoothness come from Layer 0 for any $r$, and the orders up to 9 have hand-written kernels. Jiang–Shu weights lose some of the order at a crest, where the first derivative vanishes: on a sine, `WENO(7)` converges at order 6 in the mean, `WENO(11)` at 10.2 to 10.4, and WENO-Z keeps 7 at order 7. `Upwind` has no such loss. `Upwind(2)` is fully one-sided and dispersive, and is kept for compatibility.

**The data.** `data="point"` says the values are point values, as C-grid velocities are, and `data="cell"` that they are cell averages, as a finite-volume tracer carries. The two give different weights from the third value on. `None` leaves it to the operation: `reconstruct` reads cell averages, `difference` and `interpolate` point values. `Upwind`, `WENO` and `TVD` always read cell averages.

**The side.** On a staggered grid a `Biased` stencil has one more point on `side` than on the other if their number is odd and two more if it is even, so `Biased(3, "left")` is `Upwind(3)` for flow towards `+x`, whatever the velocity. On a collocated grid it is fully one-sided: `"right"` is the forward difference and `"left"` the backward one.

## Face values

Write $q_i$ for the value in cell $i$ and $\hat q$ for the face value at $i + 1/2$. For flow towards $+x$:

| Scheme | $\hat q$ |
|---|---|
| `Centred(2)` | $(q_i + q_{i+1}) / 2$ |
| `Centred(4, data="point")` | $(-q_{i-1} + 9 q_i + 9 q_{i+1} - q_{i+2}) / 16$ |
| `Centred(4, data="cell")` | $(-q_{i-1} + 7 q_i + 7 q_{i+1} - q_{i+2}) / 12$ |
| `Upwind(1)` | $q_i$ |
| `Upwind(2)` | $(3 q_i - q_{i-1}) / 2$ |
| `Upwind(3)` | $(-q_{i-1} + 5 q_i + 2 q_{i+1}) / 6$ |
| `TVD(limiter)` | $q_i + \frac12 \psi(r) (q_{i+1} - q_i)$ |
| `WENO(5)` | $\sum_k \omega_k \hat q_k$ |

In the TVD row $\psi$ is the limiter and $r = (q_i - q_{i-1}) / (q_{i+1} - q_i)$. In the WENO row $\hat q_k$ are the face values of the three third-order sub-stencils and $\omega_k$ their nonlinear weights.

## What a scheme answers

A scheme is a ladder of `Tier` s, widest first, and only `tiers()` and `order` are written per scheme. The rest is derived from the tiers and from [Layer 0](stencils.md), which solves the Taylor conditions exactly, so the weights are fractions.

```python
import gridax
from gridax.schemes import WENO, Centred, Upwind

grid = gridax.CartesianCGrid2D.from_interior(60, 40, 120.0e3, 40.0e3)  # one ghost ring

Centred(4, data="point")  # Centred(order=4, data='point') # (-1, 9, 9, -1)/16

# cells read around the face i + 1/2, and the ghost cells they need
Centred(4).footprint(staggered=True)  # (-1, 2)
Centred(4).halo(staggered=True)  # 2
WENO(5).footprint(staggered=True)  # (-2, 3), both flow directions

# ∑ₖ cₖ sₖᵐ = n! δₘₙ, s = (-3/2, -1/2, 1/2, 3/2), n = 1
Centred(4).weights(derivative=1, staggered=True)  # (1, -27, 27, -1)/24, as Fractions

# does the widest stencil fit next to the edges of the grid?
Centred(4).degrades_on(grid)  # True: it needs two ghost cells

Upwind(3).linear, Upwind(3).upwinded  # (True, True)
```

| Scheme | Footprint (lo, hi) | Halo |
|---|---|---|
| `Centred(2)` | (0, 1) | 1 |
| `Centred(4)` | (-1, 2) | 2 |
| `WENO(5)` | (-2, 3) | 3 |

An upwinded scheme reads a stencil or its mirror, and its footprint holds both. Each `Tier` also has a `halo`, the cells it reads on either side of the upwind cell, and a `size`, the key of the adaptive mask that allows it: `size = 2 * halo + 2`.

## Differences and interpolation

`gridax.difference` and `gridax.interpolate` take a scheme that is linear and not upwinded, and raise on any other. The default, `Centred(2)`, is the named two-point stencil.

```python
import gridax
from gridax.schemes import Biased, Centred

# Shapes: Ny, Nx include one ghost ring; *batch any leading dims

# δₓh[j, i+½] = (h[j, i−1] − 27h[j, i] + 27h[j, i+1] − h[j, i+2]) / (24Δx)
# [*batch, Ny, Nx] T → [*batch, Ny, Nx] U
dhdx4 = gridax.difference(h, cgrid, axis="x", src="T", dst="U", scheme=Centred(4))

# along two axes: the product of the weights
# [*batch, Ny, Nx] T → [*batch, Ny, Nx] X
hx = gridax.interpolate(h, cgrid, src="T", dst="X", scheme=Centred(4))

# ∂ₓf[i] = (f[i+1] − f[i]) / Δx, on a collocated grid
# [*batch, Ny, Nx] → [*batch, Ny, Nx]
dfdx = gridax.difference(f, agrid, axis="x", scheme=Biased(1, "right"))
```

## Next to a wall

Where the widest stencil does not fit, a face value takes the next narrower tier, down to the upwind cell (`Centred`: down to the two cells of the face). With a mask, a dry cell ends the stencil as the edge of the array does, so a face value never reads a dry cell. A difference or an average next to the edge of the array shifts its stencil inwards instead (see [the two wall rules](stencils.md)), and with a mask it takes the next narrower stencil wherever it would read a dry point, down to the two points of `Centred(2)`. A collocated difference is one-sided there.

## Saying what happens there: `Adaptive`

Those are the defaults of each operation. `Adaptive(scheme, edges=…, mask=…)` wraps a scheme and says it instead, for every operation that takes a `scheme=`:

| Policy | `edges=`, at the array bounds | `mask=`, next to dry points |
|---|---|---|
| `"narrow"` | the next narrower stencil that fits | the widest narrower stencil on wet points |
| `"bias"` | as many points, shifted inwards | a stencil of the same order, shifted onto wet points |
| `"extend"` | the full stencil, outside filled from `bc=` | |
| `"periodic"` | the full stencil, wrapped around | |
| `"fill"` | | the full stencil, dry points filled from `bc=` |
| `"none"` | zero (NaN if collocated) where it does not fit | the full stencil, dry points read as they are |

| Operation | Default `edges` | Default `mask` |
|---|---|---|
| `reconstruct`, `advect` | `"narrow"` | `"narrow"` |
| `difference`, `interpolate` on a C-grid | `"bias"` | `"narrow"` |
| `difference` on an A-grid | `"bias"` | shift, then narrow |

With `bc=` the default at the edges is `"extend"`.

```python
import gridax
import numpy as np
from gridax.schemes import WENO, Adaptive, Centred

grid = gridax.CartesianCGrid2D.from_interior(32, 32, 1.0, 1.0)
wet = np.ones(grid.shape, dtype=bool)
wet[10:16, 12:20] = False
mask = gridax.Mask2D.from_grid(grid, wet)
h, u, v = np.random.default_rng(0).normal(size=(3, *grid.shape))

# fourth order up to the coast, with a stencil that leans away from it
leaning = Adaptive(Centred(4), mask="bias")
dhdx = gridax.difference(h, grid, axis="x", src="T", dst="U", scheme=leaning, mask=mask)

# WENO-5 everywhere: across the seam, and through no-flux ghosts behind the coast
bc = gridax.BC(
    x=gridax.Periodic(),
    y=gridax.Periodic(),
    coast=gridax.BC(x=gridax.Neumann(), y=gridax.Neumann()),
)
full = Adaptive(WENO(5), mask="fill")
dhdt = gridax.advect(h, u, v, grid, scheme=full, mask=mask, bc=bc)

Adaptive(Centred(4), mask="bias").degrades_on(grid, mask)  # True: at the edges
```

A flux has one value per face whatever the policy, so `advect` conserves to rounding under all of them. An upwinded scheme has no `"bias"`, which would make it read downwind cells. The policy at the edges is chosen from the shape of the array and costs nothing; next to a coast every candidate is computed and one is selected per point.
