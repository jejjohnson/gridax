# Multilayer vs. 3D Discretization

This page explains the crucial distinction between *multilayer* operators
(via `jax.vmap` over independent 2D layers) and *true 3D* operators in
`finitevolX`, and shows how to use each correctly.

---

## Two Different Meanings of a Leading Axis

When you have an array of shape `[N, Ny, Nx]` in ocean or atmosphere
modelling, the leading axis `N` can mean one of two fundamentally different
things:

| Use case | Leading axis meaning | Example models |
|---|---|---|
| **Multilayer / baroclinic / modal** | Layer or mode index | QG *n*-layer, baroclinic SW |
| **True 3D discretization** | Vertical spatial dimension *z* | Primitive equations |

`finitevolX` provides explicit support for both, but **they must not be
confused** because their ghost-cell conventions are opposite.

---

## Multilayer: `jax.vmap`

### Concept

In a multilayer model (e.g., *n*-layer quasi-geostrophic or baroclinic
shallow water), the fluid column is divided into `nl` discrete layers.
Each layer is a complete, physically real 2D horizontal field with its own
dynamics.  There are **no "ghost" layers** — every slice `k = 0 … nl-1`
holds a real physical field.

The JAX-idiomatic way to apply any 2D horizontal operator to all layers at
once is `jax.vmap`, which vectorises the call over the leading axis without
any overhead.  (`sesmax_core.vertical.multilayer` is a thin `eqx.filter_vmap` wrapper over the same pattern.)

```python
import jax
import jax.numpy as jnp
import gridax as fvx

grid = fvx.CartesianCGrid2D.from_interior(64, 64, 1e4, 1e4)
ops = fvx.GridOps(grid)

nl = 3                                     # number of layers
h = jnp.ones((nl, grid.Ny, grid.Nx))      # [nl, Ny, Nx]

# dh/dx at the U points of every layer in one vectorised call
dh_dx = jax.vmap(lambda h: ops.difference(h, axis="x", src="T", dst="U"))(h)
# shape [nl, Ny, Nx]
```

For operators that take multiple arguments (e.g., `divergence`), pass the
method of `GridOps` directly — `jax.vmap` batches each positional argument
over the leading axis independently:

```python
u = jnp.ones((nl, grid.Ny, grid.Nx))
v = jnp.ones((nl, grid.Ny, grid.Nx))

div = jax.vmap(ops.divergence)(u, v)
```

### What `jax.vmap` does here

- Batches `fn` over the **first** axis.
- The 2D stencil (ghost-cell ring, interior writes) is completely unchanged.
- **All** `nl` slices are processed; there is no "outer ghost layer" concept.

---

## True 3D: a 3-D grid

### Concept

In a true 3D primitive-equation model, the leading axis is a *spatial*
vertical coordinate.  The discretization uses the same ghost-cell convention
as the horizontal dimensions: the **top** (`k = 0`) and **bottom**
(`k = Nz-1`) layers are ghost shells that must be filled by vertical boundary
conditions before chaining operators.  Only the interior `k = 1 … Nz-2`
layers are written by the stencil.

```python
grid3d = fvx.CartesianCGrid3D.from_interior(64, 64, 10, 1e4, 1e4, 50.0)

# h has shape [Nz, Ny, Nx]; Nz = 12 total (10 interior + 2 ghost)
h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))

dh_dx = fvx.difference(h, grid3d, axis="x", src="T", dst="U")
# dh_dx[0]  = 0   ← ghost layer (k=0), not written
# dh_dx[1:-1] ≠ 0 ← interior layers, written
# dh_dx[-1] = 0   ← ghost layer (k=Nz-1), not written
```

---

## Key Differences at a Glance

| Property | `jax.vmap(fn)` | a 3-D grid |
|---|---|---|
| Leading axis meaning | Layer / mode index | Vertical *z* dimension |
| Ghost layers at k=0, k=N-1 | ❌ None — all slices are real | ✅ Required — top/bottom are ghost shells |
| Which slices are written | All `k = 0 … nl-1` | Only interior `k = 1 … Nz-2` |
| Underlying mechanism | `jax.vmap(fn)` over axis 0 | Explicit `[1:-1, …]` slice writes |
| Correct for | Baroclinic / QG layered models | Primitive-equation 3D models |

---

## Equivalence in the Interior

Although the two approaches differ at the boundary layers, they produce
**identical results in the horizontal interior** for the shared interior
z-levels:

```python
import jax
import jax.numpy as jnp
import numpy as np
import gridax as fvx

grid2d = fvx.CartesianCGrid2D.from_interior(8, 8, 1.0, 1.0)
grid3d = fvx.CartesianCGrid3D.from_interior(8, 8, 4, 1.0, 1.0, 1.0)

def dh_dx(h, grid):
    return fvx.difference(h, grid, axis="x", src="T", dst="U")


Nz = grid3d.Nz   # 6 total (4 interior + 2 ghost)
h = jnp.arange(Nz * grid3d.Ny * grid3d.Nx, dtype=float).reshape(
    Nz, grid3d.Ny, grid3d.Nx
)

ml_result = jax.vmap(lambda layer: dh_dx(layer, grid2d))(h)
d3_result = dh_dx(h, grid3d)

# Interior z-levels AND horizontal interior agree exactly
np.testing.assert_allclose(
    ml_result[1:-1, 1:-1, 1:-1],
    d3_result[1:-1, 1:-1, 1:-1],
)

# But the boundary layers differ:
# the 3-D grid: k=0, k=5 are zero (ghost shells)
print(d3_result[0].max())    # 0.0
print(d3_result[-1].max())   # 0.0

# vmap over layers: k=0, k=5 are real (non-zero) layers
print(ml_result[0].max())    # non-zero
print(ml_result[-1].max())   # non-zero
```

---

## Summary

- Use **`jax.vmap(fn)`** when your leading axis indexes independent
  physical layers or modes.  Every layer is real; `jax.vmap` is the
  correct, efficient way to batch over them.
- Use a **3-D grid** when your leading axis is a spatial vertical
  coordinate that requires top/bottom ghost shells for boundary conditions.
- The two are **not interchangeable**.  Passing a true
  multilayer field with a 3-D grid silently zeros out the top and bottom layers, which are
  physically real in that context.
