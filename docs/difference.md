# Finite-Difference Operators

This page covers the theory behind the finite-difference operators in finitevolX,
explains the staggered-grid indexing conventions, and provides practical guidance
for choosing and composing operators.

---

## The Problem

Ocean and atmosphere models compute spatial derivatives of scalar and vector fields
at each time step.  On an Arakawa C-grid the variables live at different staggered
locations, so the derivative of a quantity at one location must be evaluated at a
*different* location — and the rule for where the output lives is determined by
the direction and sign of the shift.

The core question is:

> Given a field at location **A**, how do we compute its derivative and what
> location does the result live at?

finitevolX answers this through a consistent **forward / backward** convention
that is the backbone of all higher-level operators (divergence, vorticity,
Laplacian, diffusion, advection, etc.).

A move is named by its axis and its points, and the type of the grid gives the rank and the geometry:

```python
import numpy as np
import gridax

grid = gridax.CartesianCGrid2D.from_interior(16, 16, 1.0, 1.0)
h = np.ones(grid.shape)

# dh/dx at the east faces, dh[j, i+1/2] = (h[j, i+1] - h[j, i]) / dx
dh_dx = gridax.difference(h, grid, axis="x", src="T", dst="U")
# h at the corners, the mean of the four cells
h_x = gridax.interpolate(h, grid, src="T", dst="X")
```

`GridOps(grid, mask)` has both as methods, with the grid and the mask filled in.

---

## Staggered Grid Locations

All arrays share the same shape `[Ny, Nx]` with a one-cell ghost ring on each side.
The physical interior is `[1:-1, 1:-1]`.  Four location types are defined:

| Symbol | Location | Array indices | Description |
|--------|----------|---------------|-------------|
| **T** | Cell centre | `T[j, i]` at `(j, i)` | Scalars: thickness, pressure, temperature |
| **U** | East face | `U[j, i]` at `(j, i+½)` | x-velocity, x-flux |
| **V** | North face | `V[j, i]` at `(j+½, i)` | y-velocity, y-flux |
| **X** | NE corner | `X[j, i]` at `(j+½, i+½)` | Vorticity, PV, corner quantities |

:::{note} Same-index convention
`U[j, i]` stores the value at the **east** face of cell `(j, i)`, i.e. at
position `(j, i+½)`.  Similarly `V[j, i]` is at the **north** face `(j+½, i)`.
This means the forward difference `T → U` uses `h[i+1] − h[i]`, whereas
the backward difference `U → T` uses `u[i] − u[i−1]`.
:::

---

## Forward vs. Backward Differences

### Forward differences

A **forward** difference moves the output one half-step in the **positive**
direction relative to the input:

$$
T \xrightarrow{\partial/\partial x} U, \qquad
T \xrightarrow{\partial/\partial y} V, \qquad
U \xrightarrow{\partial/\partial y} X, \qquad
V \xrightarrow{\partial/\partial x} X
$$

| Move | Stencil | Use case |
|----------|---------|----------|
| ∂/∂x, T to U | $(h_{i+1} - h_i)/\Delta x$ | Pressure gradient force, diffusion flux |
| ∂/∂y, T to V | $(h_{j+1} - h_j)/\Delta y$ | Pressure gradient force, diffusion flux |
| ∂/∂y, U to X | $(u_{j+1} - u_j)/\Delta y$ | Part of vorticity $\partial u / \partial y$ |
| ∂/∂x, V to X | $(v_{i+1} - v_i)/\Delta x$ | Part of vorticity $\partial v / \partial x$ |

### Backward differences

A **backward** difference moves the output one half-step in the **negative**
direction:

$$
U \xrightarrow{\partial/\partial x} T, \qquad
V \xrightarrow{\partial/\partial y} T, \qquad
X \xrightarrow{\partial/\partial y} U, \qquad
X \xrightarrow{\partial/\partial x} V
$$

| Move | Stencil | Use case |
|----------|---------|----------|
| ∂/∂x, U to T | $(u_i - u_{i-1})/\Delta x$ | Divergence, tracer tendency from flux |
| ∂/∂y, V to T | $(v_j - v_{j-1})/\Delta y$ | Divergence, tracer tendency from flux |
| ∂/∂y, X to U | $(q_j - q_{j-1})/\Delta y$ | Coriolis & vorticity-flux term |
| ∂/∂x, X to V | $(q_i - q_{i-1})/\Delta x$ | Coriolis & vorticity-flux term |

:::{tip} Forward → Backward pairs
Every forward operator has a corresponding backward operator that reverses
the shift.  Composing ∂/∂x from T to U with ∂/∂x from U to T
gives a second-order centred difference at T-points (i.e. the Laplacian
in x), which is the basis for the diffusion operator.
:::

---

## Compound Operators

### Divergence

The **discrete divergence** of a 2-D velocity field $(u, v)$ at T-points:

$$
\delta_{j,i} = \frac{u_{j,i+\frac{1}{2}} - u_{j,i-\frac{1}{2}}}{\Delta x}
             + \frac{v_{j+\frac{1}{2},i} - v_{j-\frac{1}{2},i}}{\Delta y}
$$

This is the sum of two backward differences, ∂/∂x from U to T and ∂/∂y from V to T.
It is `gridax.divergence(u, v, grid)`; without a grid object, write `interior(diff_x_bwd_2d(u) / dx + diff_y_bwd_2d(v) / dy, u)`.

### Curl (Relative Vorticity)

The **discrete curl** of $(u, v)$ at X-points (corners):

$$
\zeta_{j+\frac{1}{2}, i+\frac{1}{2}}
  = \frac{v_{j+\frac{1}{2}, i+1} - v_{j+\frac{1}{2}, i}}{\Delta x}
  - \frac{u_{j+1, i+\frac{1}{2}} - u_{j, i+\frac{1}{2}}}{\Delta y}
$$

This is ∂/∂x from V to X minus ∂/∂y from U to X, available as `gridax.curl(u, v, grid)`.

### Laplacian

The **5-point discrete Laplacian** at T-points:

$$
\nabla^2 h_{j,i}
  = \frac{h_{j,i+1} - 2h_{j,i} + h_{j,i-1}}{\Delta x^2}
  + \frac{h_{j+1,i} - 2h_{j,i} + h_{j-1,i}}{\Delta y^2}
$$

Available as `gridax.laplacian(h, grid)`.  Used directly in the diffusion and
multigrid solver operators.

### Perpendicular Gradient (Geostrophic Velocity)

The operator `grad_perp` maps a T-point streamfunction $\psi$ to C-grid face
velocities via the geostrophic relation $(u, v) = (-\partial\psi/\partial y,\,
+\partial\psi/\partial x)$, using a compact 4-point stencil that reads T-point
ghost cells directly:

$$
u_{j, i+\frac{1}{2}}
  = -\frac{\psi_{j+1,i} + \psi_{j+1,i+1} - \psi_{j-1,i} - \psi_{j-1,i+1}}{4\,\Delta y}
$$

$$
v_{j+\frac{1}{2}, i}
  = +\frac{\psi_{j,i+1} + \psi_{j+1,i+1} - \psi_{j,i-1} - \psi_{j+1,i-1}}{4\,\Delta x}
$$

The resulting velocity field is **discretely non-divergent**: $\delta(u, v) = 0$
at all interior T-points.  This is the standard geostrophic velocity reconstruction
used in QG models.

---

## Ghost-Cell Conventions

All operators write only to `[1:-1, 1:-1]` interior cells.  The one-cell ghost
ring is used as follows:

| Difference direction | Ghost consumed | Slice |
|----------------------|---------------|-------|
| Forward x (T → U) | East T ghost `T[:,  Nx−1]` | `h[1:-1, 2:]` at last col |
| Forward y (T → V) | North T ghost `T[Ny−1, :]` | `h[2:, 1:-1]` at last row |
| Backward x (U → T) | West U ghost `U[:, 0]` | `u[1:-1, :-2]` at first col |
| Backward y (V → T) | South V ghost `V[0, :]` | `v[:-2, 1:-1]` at first row |

:::{warning} Caller responsibility
Ghost cells for **backward** operators (west U-face, south V-face) are **not**
filled by any forward operator — the caller must set them via boundary
conditions before composing operators.  Leaving them at zero is equivalent to
imposing a homogeneous Dirichlet condition on the flux, which is correct for
closed-wall BCs but wrong for periodic or open boundaries.
:::

---

## 3-D Extension

On a 3-D grid the functions apply the same 2-D horizontal stencils to each
z-level.  Array shapes are `[Nz, Ny, Nx]` and only the interior
`[1:-1, 1:-1, 1:-1]` is written.

---

## Quick Usage

```python
import jax.numpy as jnp
from gridax import CartesianCGrid2D, GridOps

grid = CartesianCGrid2D.from_interior(ny=64, nx=64, Ly=1e6, Lx=1e6)
ops = GridOps(grid)

h  = jnp.ones((grid.Ny, grid.Nx))  # T-point field

# Forward difference: h at T-points → pressure gradient at U-points
dh_dx = ops.difference(h, axis="x", src="T", dst="U")  # shape [Ny, Nx], values at U-points

# Backward difference: u at U-points → divergence contribution at T-points
u = jnp.zeros((grid.Ny, grid.Nx))
v = jnp.zeros((grid.Ny, grid.Nx))
div = ops.divergence(u, v)       # shape [Ny, Nx], values at T-points

# Curl: relative vorticity at X-points
zeta = ops.curl(u, v)            # shape [Ny, Nx], values at X-points

# Laplacian: ∇²h at T-points
lap_h = ops.laplacian(h)         # shape [Ny, Nx], values at T-points

# Geostrophic velocity from streamfunction
psi = jnp.zeros((grid.Ny, grid.Nx))
u_geo, v_geo = ops.grad_perp(psi, at="T")  # u at U-points, v at V-points
```

---

## Decision Guide

```
What derivative do you need?
│
├── ∂(·)/∂x from T-points → difference(axis="x", src="T", dst="U")  (result at U-points)
├── ∂(·)/∂y from T-points → difference(axis="y", src="T", dst="V")  (result at V-points)
├── ∂(·)/∂y from U-points → difference(axis="y", src="U", dst="X")  (result at X-points)
├── ∂(·)/∂x from V-points → difference(axis="x", src="V", dst="X")  (result at X-points)
│
├── ∂(·)/∂x from U-points → difference(axis="x", src="U", dst="T")  (result at T-points)
├── ∂(·)/∂y from V-points → difference(axis="y", src="V", dst="T")  (result at T-points)
├── ∂(·)/∂y from X-points → difference(axis="y", src="X", dst="U")  (result at U-points)
├── ∂(·)/∂x from X-points → difference(axis="x", src="X", dst="V")  (result at V-points)
│
├── ∇·(u, v) at T-points  → divergence(u, v, grid)
├── ζ = ∇×(u, v) at X-pts → curl(u, v, grid)
├── ∇²h at T-points       → laplacian(h, grid)
└── geostrophic u, v      → grad_perp(psi, grid, at="T")
```

The functions `gridax.divergence`, `curl`, `laplacian`, `gradient` and `grad_perp` take any grid. On a [collocated grid](https://jejjohnson.github.io/gridax/reference/grids/) (every component on the same points, as in model output on a lat-lon grid) they run the collocated stencils, so one call serves both. The Layer 0 coefficients of `gridax.stencils` are underneath:

| Quantity | On a C-grid | On a collocated grid |
|---|---|---|
| ∂h/∂x | `difference(h, grid, axis="x", src="T", dst="U")` | `difference(h, grid, axis="x")` |
| ∇·(u, v) | `divergence(u, v, grid)` (at T) | `divergence(u, v, grid)` |
| ∇×(u, v) | `curl(u, v, grid)` (at X) | `curl(u, v, grid)` (at T) |
| ∇²h | `laplacian(h, grid)` (at T) | `laplacian(h, grid)` |
| Fourth-order ∂h/∂x | `difference(h, grid, axis="x", src="T", dst="U", scheme=Centred(4))` | `difference(h, grid, axis="x", scheme=Centred(4))` |

Without a grid object, the derivative along one axis of a collocated array is `gridax.analysis.difference(h, axis=-1, step=dx)`, and the C-grid divergence is `interior(diff_x_bwd_2d(u) / dx + diff_y_bwd_2d(v) / dy, u)`.

---

## References

- Arakawa & Lamb (1977) — Staggered C-grid layout and half-index notation
- Sadourny (1975) — Discretisation of the shallow-water equations on a C-grid
- Veros ocean model — C-grid operator conventions
