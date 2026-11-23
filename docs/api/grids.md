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

::: gridax.CurvilinearGrid1D

::: gridax.CartesianCGrid1D

## 2-D Grid

::: gridax.ArakawaCGrid2D

::: gridax.CurvilinearGrid2D

::: gridax.CartesianCGrid2D

## 3-D Grid

::: gridax.ArakawaCGrid3D

::: gridax.CurvilinearGrid3D

::: gridax.CartesianCGrid3D

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

## Collocated grids

Every variable at T and no ghost ring, for fields that come with one value per cell: a raster scene, observations, an analysis product. Axes may be uniform or irregular, and uniform ones periodic. The steps are signed, so a derivative along `y` points north on a raster whose rows run south. The canonical functions (`gridax.difference`, the reductions) take these grids, and a mask built with `Mask2D.from_grid` has T points only.

| | `from_interior` | `from_affine` | `from_coords` |
|---|---|---|---|
| `CartesianAGrid1D` | uniform | | `x` |
| `CartesianAGrid2D` | uniform | a raster transform | `y`, `x` |
| `CartesianAGrid3D` | uniform | | `z`, `y`, `x` |
| `SphericalAGrid2D` | degrees | degrees | `lat`, `lon` in degrees |

::: gridax.agrid_from

::: gridax.CartesianAGrid1D

::: gridax.CartesianAGrid2D

::: gridax.CartesianAGrid3D

::: gridax.SphericalAGrid2D

::: gridax.CollocatedGrid1D

::: gridax.CollocatedGrid2D

::: gridax.CollocatedGrid3D

## Between the two families

A collocated grid of `Ny` by `Nx` points is the interior T points of a C-grid of `(Ny + 2)` by `(Nx + 2)`: the faces sit halfway between the points, and a ghost point is the mirror of the first one about the wall. `agrid.staggered()` and `cgrid.collocated()` give one from the other, and `agrid.staggered().collocated()` is `agrid`. An irregular Cartesian grid gives a `StretchedCGrid`, which the moves, the vector calculus, the Laplacian and the reductions take. An axis that decreases (the rows of a north-up raster) has to be flipped first, and an irregular spherical one has no C-grid.

`gridax.stagger` and `gridax.collocate` move the fields. A field at T moves exactly; any other point is interpolated, which damps short waves and warns with `LossyConversionWarning`, once per trace:

```python
import gridax
import numpy as np

agrid = gridax.CartesianAGrid2D.from_interior(8, 6, 8.0, 6.0, periodic=("x",))
cgrid = agrid.staggered()
ssh, u_geo = np.random.default_rng(0).normal(size=(2, *agrid.shape))

walls = gridax.BC.no_flux("y")  # the periodic axis needs no condition
h0 = gridax.stagger(ssh, agrid, dst="T", bc=walls)  # [8, 10], exact
u0 = gridax.stagger(u_geo, agrid, dst="U", bc=walls)  # (u_geo[j, i] + u_geo[j, i+1]) / 2
u_t = gridax.collocate(u0, cgrid, src="U")  # [6, 8], back at T, smoothed
```

There and back, `Centred(2)` multiplies a wave of wavenumber $k$ by $\cos^2(k \Delta x / 2)$, and removes the one of two cells. `scheme=Centred(4)` damps less.

::: gridax.stagger

::: gridax.collocate

::: gridax.LossyConversionWarning

::: gridax.StretchedCGrid1D

::: gridax.StretchedCGrid2D

::: gridax.StretchedCGrid3D
