# Boundary Conditions

This page covers the theory behind the boundary condition (BC) system in
finitevolX, explains the ghost-cell approach used on the Arakawa C-grid, and
provides guidance for composing and applying BCs.

---

## The Problem

All spatial operators in finitevolX write only to interior cells `[1:-1, 1:-1]`
and read the one-cell ghost ring to compute stencils at the boundary.  The
physical boundary condition is therefore **encoded in the ghost cells**: the
BC layer sets ghost values such that the interior stencil evaluates to the
correct physical quantity at the domain wall.

This design cleanly separates the operator code (which never checks for
boundaries) from the boundary logic (which only writes to ghost cells).  It
also allows arbitrary BCs to be applied by simply overwriting ghost cells
before the next operator call.

---

## The Ghost-Cell Method

For a field with interior values $\phi_1, \phi_2, \ldots$ and ghost cell
$\phi_0$, the physical boundary is modelled as lying exactly **between** the
first interior cell and the ghost cell.  This gives the standard ghost-cell
relationships:

**Dirichlet** ($\phi_{\text{wall}} = g$):

$$
\phi_{\text{wall}} = \tfrac{1}{2}(\phi_1 + \phi_0) = g
\implies \phi_0 = 2g - \phi_1
$$

**Neumann** ($\partial\phi/\partial n\big|_{\text{wall}} = g$, outward normal):

$$
\frac{\phi_0 - \phi_1}{\Delta n} = g
\implies \phi_0 = \phi_1 + g\,\Delta n
$$

where $\phi_0$ is the ghost cell and $\phi_1$ the adjacent interior cell. The
ghost cell lies outward of the interior on every face, so the same formula
holds on all four faces: $g$ is the true gradient along the outward normal
(e.g. $g = -\partial\phi/\partial y$ on the south wall).

**Periodic** ($\phi_0 = \phi_{N-1}$, $\phi_N = \phi_1$):

The ghost cell simply copies the opposite interior edge.

---

## Boundary Condition Types

A condition is an object with no face and no point: [BC](xref:api#gridax.BC) says which face it sits on, and [fill_ghosts](xref:api#gridax.fill_ghosts) says which point the field lives at and what the grid spacing is.  So one condition describes the physics, and the same one serves a T, a U and a V field.

### `Dirichlet`

[Dirichlet](xref:api#gridax.Dirichlet)`(value)` fixes the **value** of the field at the domain wall.  The ghost cell is set to the reflection that makes the average equal to the target value:

$$
\phi_{\text{ghost}} = 2\,v_{\text{target}} - \phi_{\text{interior}}
$$

**Physical use cases:**
- No-slip walls for the tangential velocity: $v = 0$ on a west / east wall.
- No normal flow through a wall: $u = 0$ on a west / east wall, for a U-field (see below).
- Fixed surface elevation: $\eta = 0$ on rigid lid.
- Streamfunction: $\psi = 0$ on closed coastline (used by DST Poisson solver).

### Staggered fields: the point of the field

The formula above places the wall half a cell outside the first interior
node, which is right for every **cell-centred** direction: `T` on all four
faces, `V` on the west / east faces, `U` on the south / north faces.  With
same-index staggering the other pairs have a node **on** the wall:
`U[:, 0]` and `U[:, Nx-2]` are the west and east wall faces, `V[0, :]` and
`V[Ny-2, :]` the south and north ones, and an `X` field has a node on all
four walls (`U[:, Nx-1]`, `V[Ny-1, :]` are outside the domain).

The point comes from the field: a [CField](xref:api#gridax.CField) carries its own, and for a plain array you pass `stagger="U"` (T by default) to `fill_ghosts`.  For a node on the wall the condition is imposed on that node: `Dirichlet` writes `value` to it, `Neumann` / `Robin` use the one-sided gradient between it and the first interior node, `Slip` writes `a * interior`, and the outside slot on the east / north side gets the odd mirror $2\phi_{\text{wall}} - \phi_{\text{adjacent}}$.  `Periodic` needs no special case: wrapping by one period is right for every stagger (`U[:, 0]` is the same face as `U[:, Nx-2]`).

:::{warning} Normal velocity needs its own point
Filling a U-field as if it were at T points (the default of a plain array) sets the ghost column `U[:, Nx-1] = -U[:, Nx-2]` and leaves the east wall face `U[:, Nx-2]` untouched, so mass flows through a "closed" wall.  Pass `stagger="U"` (or give `fill_ghosts` a U `CField`) so the wall node itself is written.
:::

A closed basin is one `BC` with a Dirichlet zero on every wall, filled once per velocity component at its own point:

```python
import numpy as np
import gridax

grid = gridax.CartesianCGrid2D.from_interior(8, 8, 1.0, 1.0)
u = np.ones(grid.shape)
v = np.ones(grid.shape)

walls = gridax.BC(x=gridax.Dirichlet(0.0), y=gridax.Dirichlet(0.0))
u = gridax.fill_ghosts(u, grid, walls, stagger="U")  # u = 0 on the W/E faces, no-slip on S/N
v = gridax.fill_ghosts(v, grid, walls, stagger="V")  # v = 0 on the S/N faces, no-slip on W/E
print(u[3, 0], u[3, -2], v[0, 3], v[-2, 3])
# Difference2D(grid).divergence(u, v) now integrates to zero over the basin.
```

### `Neumann`

[Neumann](xref:api#gridax.Neumann)`(gradient=0.0)` fixes the **outward-normal gradient** $\partial\phi/\partial n = g$:

$$
\phi_{\text{ghost}} = \phi_{\text{interior}} + g\,\Delta n
$$

**Physical use cases:**
- Free-slip walls: $\partial u_{\parallel}/\partial n = 0$.
- No-flux tracer BC: $\partial T/\partial n = 0$ on insulating walls.
- Pressure BC: $\partial p/\partial n = 0$ on solid walls.

### `Periodic`

[Periodic](xref:api#gridax.Periodic)`()` wraps the field: the ghost cell on one face copies the last interior cell on
the opposite face.  Both faces of an axis must be periodic.

**Physical use cases:**
- Doubly-periodic channels and basins.
- Zonal re-entrant channels (periodic in x, closed in y).

:::{note} enforce_periodic helper
For fully periodic domains the helper `enforce_periodic(field, ...)` is
available as a thin wrapper that wraps both axes at once.  Prefer `BC` for per-face control.
:::

### `Reflective`

[Reflective](xref:api#gridax.Reflective)`()` reflects the field about the boundary:

$$
\phi_{\text{ghost}} = \phi_{\text{interior}} \quad (\text{even reflection})
$$

This is Neumann with $g = 0$.  Used for **free-slip** walls where the
normal velocity is zero and the tangential velocity has zero normal gradient.

### `Slip`

[Slip](xref:api#gridax.Slip)`(coefficient=1.0)` is the general slip boundary condition controlled by a scalar `coefficient = a`:

$$
\phi_{\text{ghost}} = (2a - 1)\,\phi_{\text{interior}}
$$

where:

- `a = 1.0` (default) gives **free-slip** (even reflection),
- `a = 0.0` gives **no-slip** (odd reflection, equivalent to Dirichlet with $g=0$),
- `0 < a < 1` gives **partial-slip**, interpolating smoothly between free-slip and no-slip.

This BC is typically used for velocity components at solid walls, with the
choice of `coefficient` controlling how strongly the tangential velocity is
damped at the boundary.

### `Outflow`

[Outflow](xref:api#gridax.Outflow)`()` is the zero-gradient outflow condition (equivalent to Neumann with $g=0$):

$$
\phi_{\text{ghost}} = \phi_{\text{interior}}
$$

Used at **open boundaries** where the flow leaves the domain and the outward
gradient is assumed zero.

### `Extrapolation`

[Extrapolation](xref:api#gridax.Extrapolation)`(order=1)` extrapolates the polynomial of degree `order` through the innermost interior cells.  With the default order 1 this is the linear extrapolation from the two innermost cells:

$$
\phi_{\text{ghost}} = 2\,\phi_1 - \phi_2
$$

More accurate than zero-gradient for smoothly varying fields at open
boundaries.  Avoids spurious reflections from the domain edge.  The order runs from 1 to 5 (order 0 is `Outflow`), and the axis needs `order + 1` interior cells.

### `Robin`

[Robin](xref:api#gridax.Robin)`(a, b, c)` is the mixed (Robin / third-kind) condition: a linear combination of value and
gradient:

$$
a\,\phi_{\text{wall}} + b\,\frac{\partial\phi}{\partial n}\bigg|_{\text{wall}} = c
$$

With $\phi_{\text{wall}} = (\phi_{\text{ghost}} + \phi_{\text{interior}})/2$ and
$\partial\phi/\partial n = (\phi_{\text{ghost}} - \phi_{\text{interior}})/\Delta n$
(outward normal, same on every face), the ghost value is

$$
\phi_{\text{ghost}} = \frac{c - (a/2 - s)\,\phi_{\text{interior}}}{a/2 + s},
\qquad s = \frac{b}{\Delta n}
$$

With $b = 0$ it is a Dirichlet condition with value $c/a$, and with $a = 0$ a Neumann condition with gradient $c/b$.  Used for **radiation** boundary conditions in more
advanced setups.

### `Sponge`

[Sponge](xref:api#gridax.Sponge)`(background, weight)` relaxes the ghost cell towards a background value $\phi^*$ with a weight $w \in [0, 1]$:

$$
\phi_{\text{ghost}} = (1 - w)\,\phi_{\text{interior}} + w\,\phi^*
$$

Used at **open boundaries** to absorb outgoing waves and prevent reflections.  It acts on the ghost ring only: a sponge layer inside the domain is a tendency of the model.

---

## Composing Boundary Conditions

### `BC`

[BC](xref:api#gridax.BC) names the condition on each face of the grid.  A face is given by its axis (`x=`, `y=`, `z=`) with one condition for both sides or a `(low, high)` pair, or by its side (`west=`, `east=`, `south=`, `north=`, `bottom=`, `top=`) with one condition.  A face left out is not written.  Every face is processed per axis, so `fill_ghosts` works the same on a 1-D, 2-D or 3-D grid, Cartesian or spherical; on a sphere the spacing $\Delta n$ is the one of each row.

```python
import numpy as np
import gridax

grid = gridax.CartesianCGrid2D.from_interior(8, 8, 1.0, 1.0)
psi = np.ones(grid.shape)

# Closed basin: Dirichlet psi=0 on all walls
walls = gridax.BC(x=gridax.Dirichlet(0.0), y=gridax.Dirichlet(0.0))
psi = gridax.fill_ghosts(psi, grid, walls)

# A re-entrant channel: periodic in x, a closed south wall, a zero-gradient north wall
channel = gridax.BC(x=gridax.Periodic(), south=gridax.Dirichlet(0.0), north=gridax.Neumann())
h = gridax.fill_ghosts(np.ones(grid.shape), grid, channel)
print(psi[0, 3], h[0, 3], h[-1, 3])
```

The same `BC` can describe the boundary of an **elliptic solve**: the one-call inversions `streamfunction_from_vorticity`, `pressure_from_divergence` and `pv_inversion` take it as `bc=`: all faces Dirichlet select the DST, zero Neumann the DCT and Periodic the FFT.  Those wrappers moved to sesmax-core (`sesmax_core.gfd`); see the [elliptic inversions guide](https://jejjohnson.github.io/sesmax/elliptic-inversions/) for how a `BC` selects the transform and imposes non-zero `Dirichlet` values on the wall nodes.

When time-stepping, a `Dirichlet` value sits on the wall face, half a
cell outside the first wet T cell.  ψ lives at X points, and the X nodes
`0` and `N-2` lie **on** the walls, so a solve pins those wall nodes (the
coast ring of the ψ mask) to the face value: the same wall.  For a
T-point pressure solve the face is half a cell from any cell
centre, so non-zero `Dirichlet` faces are not supported there.  See
[Known Boundary Values](elliptic_solvers_usage.md#known-boundary-values-inhomogeneous-dirichlet)
for arbitrary per-cell values and interior observations.

### Several fields

A state is filled in one call. Give `fill_ghosts` a pytree of `BC` with the structure of the state (a dict, a NamedTuple, an `eqx.Module`), one per field: each [CField](xref:api#gridax.CField) brings its own point, and `None` leaves a field alone. A `BC` holds no point, so the same one can serve several fields.

```python
import numpy as np
import gridax

grid = gridax.CartesianCGrid2D.from_interior(8, 8, 1.0, 1.0)
state = {
    "u": gridax.CField(np.ones(grid.shape), ("y", "x"), loc="U"),
    "h": gridax.CField(np.ones(grid.shape), ("y", "x"), loc="T"),
}

# Free-slip velocity: zero normal flow on the W/E walls, zero tangential gradient on S/N
bcs = {
    "u": gridax.BC(x=gridax.Dirichlet(0.0), y=gridax.Neumann()),
    "h": gridax.BC(x=gridax.Outflow(), y=gridax.Outflow()),
}
state = gridax.fill_ghosts(state, grid, bcs)
```

A state of plain arrays passes its points as a pytree too: `stagger={"u": "U", "h": "T"}`.

### Corners

Where two faces meet, the ghost corner depends on the order the faces are applied in.  `BC(corners="average")` (the default) fills the corner with the mean over the orders; `corners="priority"` applies the axes in the grid's order, so the later axis (x over y) wins.  Off the corners every order agrees.  With $\phi = 3$ in the interior, a Dirichlet value 1 in x and 0 in y, the two orders give 5 and 1 at the corner, so the default is 3:

```python
import numpy as np
import gridax

grid = gridax.CartesianCGrid2D.from_interior(8, 8, 1.0, 1.0)
t = np.full(grid.shape, 3.0)
faces = dict(x=gridax.Dirichlet(1.0), y=gridax.Dirichlet(0.0))
mean = gridax.fill_ghosts(t, grid, gridax.BC(**faces))
first = gridax.fill_ghosts(t, grid, gridax.BC(corners="priority", **faces))
print(mean[0, 0], first[0, 0], mean[0, 4] == first[0, 4])  # 3.0 5.0 True
```

### Face values from a parent model

The numbers of a condition (`value`, `gradient`, `background`, `weight`, `a`, `b`, `c`, `coefficient`) can be arrays of shape `[*batch, N_face]` instead of scalars, so a nested model can take its boundary values from a parent model: one value per face point, and one per batch member.  They are pytree leaves of the `BC`, so a loss that depends on the filled field can be differentiated with respect to them.  Note that `Dirichlet` writes the ghost cell $2v - \phi_1$, so the value sits on the wall:

```python
import numpy as np
import gridax

grid = gridax.CartesianCGrid2D.from_interior(8, 8, 1.0, 1.0)
parent = np.linspace(0.0, 1.0, grid.shape[0])  # the parent's value at each west-face row
bc = gridax.BC(west=gridax.Dirichlet(parent), east=gridax.Neumann())
w = gridax.fill_ghosts(np.zeros(grid.shape), grid, bc)
print(w[:, 0])  # 2 * parent, as the interior is zero
```

### Ghost layers: `width`

A grid has a ghost ring of `grid.ghost` cells on each side (1 by default), and `fill_ghosts` fills all of them.  A scheme that reads more cells, such as a wide stencil, needs a wider ring: build the grid with `ghost=`, or pass `width=` for an array that has that many layers on each side.  For a cell-centred direction, layer $k$ outward lies the same distance outside the wall as the interior node $\phi_k$ lies inside it, so the condition holds on the wall for every layer.  For `Neumann` this gives $\phi_{-k} = \phi_k + (2k - 1)\,g\,\Delta n$:

```python
import numpy as np
import gridax

grid = gridax.CartesianCGrid1D(Nx=8 + 4, Lx=1.0, ghost=2)
x = np.arange(12.0)
print(gridax.fill_ghosts(x, grid, gridax.BC(x=gridax.Neumann())))
```

---

### Wide stencils: `bc=` on an operator

A field keeps one ghost ring, and `Centred(4)` or `WENO(5)` reads more than that next to a wall. Without more to go on, `difference` and `interpolate` shift such a stencil inwards at the edge, and `reconstruct` and `advect` narrow it. Give the operator the boundary condition instead, and it fills as many ghost layers as its scheme reads, computes, and returns a field with the usual ring:

```python
import gridax
import numpy as np
from gridax.schemes import WENO, Centred

grid = gridax.CartesianCGrid2D.from_interior(16, 16, 1.0, 1.0)
h = np.random.default_rng(0).normal(size=grid.shape)
u, v = np.ones(grid.shape), np.zeros(grid.shape)

around = gridax.BC(x=gridax.Periodic(), y=gridax.Periodic())
dhdx = gridax.difference(h, grid, axis="x", src="T", dst="U", scheme=Centred(4), bc=around)
dhdt = gridax.advect(h, u, v, grid, scheme=WENO(5), bc=around)
print(abs(dhdt.sum()) < 1e-10)  # nothing is lost across the seam
# True
```

The operator reads nothing of the ring of the field itself, so there is no need to fill it first. `advect` with `bc=` writes every interior cell, edges included, and reads the velocity on the walls.

### Coasts inside the domain: `mask`

A coastline is a wall too. With `mask=`, `fill_ghosts` also writes the dry points behind every coast with the mirror rules above: the nearest wet point is $\phi_1$, and a coast on the east side of the water takes the condition of the east face. A straight coast inside the domain and a domain edge with the same condition therefore hold the same values.

```python
import gridax
import numpy as np

grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
wet = np.ones(grid.shape, dtype=bool)
wet[2:4, 3:5] = False  # an island
mask = gridax.Mask2D.from_grid(grid, wet)

bc = gridax.BC(x=gridax.Periodic(), y=gridax.Neumann(), coast=gridax.BC(x=gridax.Neumann()))
h = np.arange(36.0).reshape(grid.shape)
h = gridax.fill_ghosts(h, grid, bc, mask=mask)
print(h[2, 3])  # the mean of the copies of h[2, 2] and h[1, 3]
# 11.5
```

- `width` layers of dry points are written behind each coast, and the nodes on it.
- A dry point with water on several sides holds the mean over them, and the nearest water wins over water further away.
- A periodic face says nothing about a coast, and neither does a face left out. `coast=` gives the conditions of the coasts, axis by axis, where the faces do not.
- No wet point is written. The first layer reads wet points only, so land may hold `NaN`; layer $k$ reads $\phi_k$, and `Extrapolation` `order + 1` points, which a narrow strait may not have.

## Helpers

### `pad_interior`

Pads a field array *from* an interior-only view *to* the full `[Ny, Nx]`
ghost-ring layout.  Useful when you have a field defined only on interior
points and need to apply ghost-cell BCs:

```python
from gridax import pad_interior

q_interior = jnp.ones((Ny - 2, Nx - 2))   # interior only
q_full     = pad_interior(q_interior, mode="constant", constant_values=0.0)
```

Supported modes match `jnp.pad`: `"constant"`, `"edge"`, `"reflect"`, `"wrap"`.

### `enforce_periodic`

Convenience wrapper: applies periodic BCs in x, y, or both directions:

```python
from gridax import enforce_periodic

q = enforce_periodic(q, x=True, y=True)   # both directions
```

---

## Decision Guide

```
What physical condition applies?
│
├── Fixed value at wall (ψ=0, T=T_wall, tangential velocity)
│   └── Dirichlet(value)
│
├── No normal flow through a wall (u on W/E, v on S/N)
│   └── Dirichlet(0.0), with stagger="U" / "V"
│
├── Zero normal gradient (free-slip, insulating)
│   └── Reflective()  or  Neumann()
│
├── Specified normal gradient (heat flux, wind stress)
│   └── Neumann(g)
│
├── No-slip (u=0, v=0 at wall)
│   └── Slip(0.0)  or  Dirichlet(0.0)
│
├── Periodic domain
│   └── Periodic()  or  enforce_periodic(field)
│
├── Open boundary, outflow
│   ├── Zero gradient   → Outflow()  or  Reflective()
│   └── Extrapolation   → Extrapolation(order)
│
└── Open boundary with wave absorption
    └── Sponge(background, weight)  or  Robin(a, b, c)
```

---

## References

- Haidvogel & Beckmann (1999) — *Numerical Ocean Circulation Modelling*, Ch. 4
- Griffies (2004) — *Fundamentals of Ocean Climate Models*, Ch. 3
- Durran (2010) — *Numerical Methods for Fluid Dynamics*, Ch. 8 (open BCs)
