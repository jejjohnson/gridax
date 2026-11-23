# Grid Classes

Arakawa C-grid spatial discretization containers for 1-D, 2-D, and 3-D domains.

Every grid names its axes (`grid.axes`, in array order) and is read through one accessor, at any point P of the staggering (`"T"`, `"U"`, `"V"`, `"X"`, and `"W"` in 3-D):

| Call | Cartesian | Spherical |
|---|---|---|
| `spacing(axis, at=P)` | `L / N_interior`, 0-d | `R·cos(lat_P)·dlon` along x, `R·dlat` along y |
| `coords(axis, at=P)` | `i·Δ`, or `(i + 1/2)·Δ` where P is staggered along the axis | longitude and latitude in radians |
| `area(at=P)` | `Δx·Δy`, `[Ny, Nx]` | `2R²·dlon·sin(dlat/2)·cos(lat_P)`, `[Ny, Nx]` |

The sizes, the ghost-ring width `ghost` and the axis names are static pytree metadata. The extents (`Lx`, …, and `R`, `dlon`, `dlat` on the sphere) are 0-d float64 array leaves, so one compiled function serves every grid of a shape. The operators cast a metric to the dtype of the field it scales; when you scale a float32 field by `grid.spacing("x")` yourself with x64 enabled, cast it first.

## 1-D Grid

::: gridax.ArakawaCGrid1D

## 2-D Grid

::: gridax.ArakawaCGrid2D

## 3-D Grid

::: gridax.ArakawaCGrid3D

## Spherical Grids

Concrete lat-lon C-grids. Their metric is the physical cell width, which shrinks towards the poles as `cos(lat)`:

| Call | Meaning |
|---|---|
| `spacing("x", at=P)` | zonal cell width at the points P, `R·cos(lat_P)·dlon` |
| `spacing("y")` | meridional cell width, `R·dlat` (uniform) |
| `coords("y", at=P)`, `coords("x", at=P)` | latitude and longitude of the points P (radians) |
| `area(at=P)` | exact area of the cell around P |
| `min_cell_width` | smallest interior cell width — the CFL-limiting length |
| `max_aspect` | largest interior Δy / Δx — polar cell anisotropy |

The zonal spacing is the raw metric and is not clamped, so a row at a
pole carries a degenerate width of roundoff size and arbitrary sign
(`cos(pi/2)` evaluates to `-4.4e-08` in float32 and `+6.1e-17` in
float64). The two reductions recognise both signs: `min_cell_width`
reports exactly zero for such a row and `max_aspect` reports `inf`. A
width is judged degenerate when it falls within a few ulps of the
equatorial width `R * dlon`, so a genuinely narrow cell is kept. Prefer the reductions when the value feeds a CFL
or resolution guard. Both exclude the ghost ring, and both return 0-d
arrays so they can be used inside `jax.jit`.

::: gridax.SphericalCGrid2D

::: gridax.SphericalCGrid3D
