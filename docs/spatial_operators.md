# Spatial Operators

This page covers the theory behind the higher-level spatial operators in
finitevolX: divergence, vorticity (relative and potential), diffusion,
Coriolis, momentum advection (vortex-force form), kinetic energy, and
diagnostic quantities.

---

## Overview

The spatial operators build on the primitive [difference](difference.md) and
[interpolation](interpolation.md) operators to compute physically meaningful
quantities on the Arakawa C-grid.  The table below lists the main operators
and what they compute:

| Function | Output location | Physical quantity |
|----------|-----------------|-------------------|
| `divergence` | T-points | $\nabla \cdot (u, v)$ |
| `curl` | X-points | $\zeta = \partial v/\partial x - \partial u/\partial y$ |
| `sw_potential_vorticity` | X-points | $q = (\zeta + f)/h$ |
| `pv_flux` | U/V-points | $(q\,Uh, q\,Vh)$ |
| `vortex_force` | U/V-points | $(+q\,Vh, -q\,Uh)$ |
| `diffuse`, `diffusive_fluxes` | T-points, U/V-points | $\nabla \cdot (\kappa\,\nabla h)$ and the face fluxes |
| `biharmonic` | T-points | $-\kappa\,\nabla^4 h$ |
| `coriolis` | U/V-points | $(+\overline{f_X\bar{v}}, -\overline{f_X\bar{u}})$, energy-conserving |
| `advect_momentum` | U/V-points | Vortex-force form |
| `kinetic_energy`, `bernoulli_potential` | T-points | $K$, $g\,h + K$ |
| `shear_strain`, `tensor_strain` | X-points, T-points | the two strains |
| `strain_rate_squared`, `okubo_weiss_parameter` | T-points | $\sigma^2$, Okubo–Weiss |
| `qg_potential_vorticity` | T-points | $q = \nabla^2\psi/f_0 + \beta(y - y_0)/f_0$ (− stretching) |
| `geostrophic_velocity`, `vertical_velocity` | U/V-points, W-points | $(u_g, v_g)$ (C-grid: on a sphere), $w$ from continuity |
| `arakawa_jacobian` | T-points | $J(f, g)$ |

Every one is `op(fields…, grid, *, mask=None, …)`: the type of the grid gives the rank and the geometry (Cartesian, spherical, stretched), so there is one name per operation. `GridOps(grid, mask)` has them all as methods with the grid and the mask filled in.

---

## Divergence

### Theory

The **flux-form divergence** of the horizontal velocity field at T-points:

$$
\delta_{j,i}
= \frac{u_{j,i+\frac{1}{2}} - u_{j,i-\frac{1}{2}}}{\Delta x}
+ \frac{v_{j+\frac{1}{2},i} - v_{j-\frac{1}{2},i}}{\Delta y}
$$

In the shallow-water model the continuity equation is
$\partial h/\partial t + \nabla \cdot (h\,\mathbf{u}) = 0$, so computing
the divergence of the **mass flux** $(hu, hv)$ gives the thickness tendency
$-\delta(hu, hv)$.

For a non-divergent velocity field ($\delta = 0$), the mass flux form and
the simpler advective form agree.  On a C-grid, the discrete divergence is
exactly zero for any velocity field derived from a streamfunction via
`grad_perp` — this is the key property that makes QG models mass-conservative.

### Usage

```python
from gridax import CartesianCGrid2D, diff_x_bwd_2d, diff_y_bwd_2d, divergence, interior

grid = CartesianCGrid2D.from_interior(64, 64, 1e6, 1e6)

delta = divergence(u, v, grid)    # T-points

# Without a grid object: the raw stencils and your own steps
delta = interior(diff_x_bwd_2d(u) / grid.spacing("x") + diff_y_bwd_2d(v) / grid.spacing("y"), u)
```

---

## Vorticity

### Relative Vorticity

The **relative vorticity** at X-points (NE corners):

$$
\zeta_{j+\frac{1}{2}, i+\frac{1}{2}}
= \frac{v_{j+\frac{1}{2},i+1} - v_{j+\frac{1}{2},i}}{\Delta x}
- \frac{u_{j+1,i+\frac{1}{2}} - u_{j,i+\frac{1}{2}}}{\Delta y}
$$

This is the discrete **curl** of the 2-D velocity field.  It is placed at
X-points because both of its differences land there: ∂/∂x from V and ∂/∂y
from U.

### Potential Vorticity (Shallow Water)

The **shallow-water potential vorticity** at X-points:

$$
q_{j+\frac{1}{2}, i+\frac{1}{2}}
= \frac{\zeta_{j+\frac{1}{2}, i+\frac{1}{2}} + f_{j+\frac{1}{2}, i+\frac{1}{2}}}{h_{j+\frac{1}{2}, i+\frac{1}{2}}}
$$

where $f$ and $h$ are interpolated from T-points to X-points.

:::{warning} Thin-layer singularity
$q$ is set to `NaN` where $h = 0$ to avoid division by zero.  Guard
against this with a minimum thickness clamp before calling the PV operator.
:::

### PV Fluxes

`pv_flux(q, Fu, Fv, grid, form=…)` combines $q$ (at X-points) with a pair of
face fields $(F_u, F_v)$ at U/V-points in one of three ways:

| `form=` | Form | U-point value |
|---|---|---|
| `"energy"` | multiply at the corners, then average | $\overline{q\,\overline{F_u}^{y}}^{y}$ |
| `"enstrophy"` | average $q$ to the face, then multiply | $\overline{q}^{y}\,F_u$ |
| `"blend"` | linear blend $\alpha E + (1-\alpha) Z$ | — |

The names refer to the **vortex-force** use (Sadourny 1975): when the
cross-face transport is used — $+q\,V$ in the $u$-equation, $-q\,U$ in the
$v$-equation, as `advect_momentum` forms it — the corner-product form
conserves total energy and the average-then-multiply form conserves
potential enstrophy for non-divergent flow.  Passed the same-component
fields, e.g. `(q, u*h, v*h)`, the function returns the PV *transport* flux
$(q\,Uh, q\,Vh)$ and neither label applies.

:::{warning} The blend is not the Arakawa–Lamb (1981) scheme
`form="blend"` is a linear blend of the two Sadourny forms and, for
$0 < \alpha < 1$, conserves neither invariant exactly.  The
Arakawa–Lamb (1981) scheme is a distinct stencil: `vortex_force(…, form="arakawa_lamb")` below.
:::

The corner products are formed on the full arrays, so the first interior
row and column read the south / west ghost corners of the caller's
ghost-filled $q$, $u$, $v$: fill them (periodic, free-slip, …) before calling.

### Usage

```python
from gridax import CartesianCGrid2D, curl, pv_flux, sw_potential_vorticity

grid = CartesianCGrid2D.from_interior(64, 64, 1e6, 1e6)

# Relative vorticity at X-points
zeta = curl(u, v, grid)

# Shallow-water PV at X-points
q = sw_potential_vorticity(u, v, h, f, grid)

# PV transport flux (q*Uh, q*Vh) at U/V-points
qu, qv = pv_flux(q, u*h, v*h, grid, form="energy")
```

---

## Diffusion

### Harmonic Diffusion

The **harmonic (Laplacian) diffusion** computes the tracer tendency
$\partial h/\partial t = \nabla \cdot (\kappa\,\nabla h)$ in **flux form**
using forward-then-backward differences:

1. **East-face flux** (forward T → U):
   $$
   F_{j,i+\frac{1}{2}}^x = \kappa_{j,i+\frac{1}{2}}\,\frac{h_{j,i+1} - h_{j,i}}{\Delta x}
   $$

2. **North-face flux** (forward T → V):
   $$
   F_{j+\frac{1}{2},i}^y = \kappa_{j+\frac{1}{2},i}\,\frac{h_{j+1,i} - h_{j,i}}{\Delta y}
   $$

3. **Tendency** (backward U/V → T):
   $$
   \frac{\partial h}{\partial t}\bigg|_{j,i}
   = \frac{F_{j,i+\frac{1}{2}}^x - F_{j,i-\frac{1}{2}}^x}{\Delta x}
   + \frac{F_{j+\frac{1}{2},i}^y - F_{j-\frac{1}{2},i}^y}{\Delta y}
   $$

For uniform $\kappa$ this reduces to $\kappa\,\nabla^2 h$.  For spatially
varying $\kappa$ (e.g. near coastlines or mixed-layer parameterisations) the
flux form ensures discrete conservation of the diffused quantity.  A
T-point $\kappa$ field is averaged to each face,
$\kappa_{j,i+\frac{1}{2}} = \tfrac{1}{2}(\kappa_{j,i} + \kappa_{j,i+1})$ (and
likewise in $y$), which keeps the operator symmetric and second-order
accurate; the spherical operators do the same.  With `wall="open"` the wall
faces average the edge cell with the ghost cell of $\kappa$, so fill the
$\kappa$ halo consistently with the tracer BC (wrapped for periodic, a copy
of the edge cell for zero gradient).

**Boundary conditions by default:** Face fluxes at domain walls are zero
(no-flux, closed-wall), giving a homogeneous Neumann BC $\partial h/\partial n = 0$.

### Biharmonic Diffusion

**Biharmonic diffusion** applies the harmonic operator twice:

$$
\frac{\partial h}{\partial t} = -\kappa\,\nabla^4 h
  = -\kappa\,\nabla^2(\nabla^2 h)
$$

It damps small scales much more strongly than large scales:
for a wavenumber $k$, harmonic damping scales as $\kappa k^2$
while biharmonic scales as $\kappa k^4$.  This is used to remove
grid-scale noise without excessively damping the resolved mesoscale.

:::{tip} Which diffusion to use
- **Harmonic** (`diffuse`): simple, physically motivated, appropriate for
  tracer diffusion when scale-selectivity is not critical.
- **Biharmonic** (`biharmonic`): preferred for **velocity**
  diffusion in eddy-resolving models because it acts mainly on unresolved
  scales.  Requires a larger $\kappa$ to achieve the same total dissipation.
:::

### Mask Support

`diffuse`, `diffusive_fluxes` and `biharmonic` take an optional `mask=` (`Mask2D` / `Mask3D`).
Fluxes through faces with a dry side are zeroed and land-cell tendencies are
zeroed, giving natural no-flux BCs at irregular coastlines without any
additional code.  The masked `biharmonic` applies this masked no-flux
Laplacian $D_m$ twice, $-\kappa\,D_m(D_m h)$, after zeroing the dry inputs:
the coast is no-flux for both $h$ and $\nabla^2 h$, land values (even `NaN`)
never enter the wet tendency, and the operator conserves $\sum_\text{wet} h$.

### Usage

```python
from gridax import CartesianCGrid2D, biharmonic, diffuse

grid = CartesianCGrid2D.from_interior(64, 64, 1e6, 1e6)

# Harmonic diffusion (unmasked)
dh_dt = diffuse(h, grid, kappa=100.0)

# With a mask: the fluxes through dry faces are zeroed before the
# divergence, and the tendency on dry cells after it.
dh_dt = diffuse(h, grid, kappa=100.0, mask=mask)

# Biharmonic (scale-selective)
dh_dt = biharmonic(h, grid, kappa4=1e9)
```

---

## Coriolis

### Theory

The Coriolis term is the vortex force with the planetary vorticity $f$.
`coriolis` uses the energy-conserving corner-product form of Sadourny
(1975), the same construction as the `"energy"` vortex force of
`advect_momentum`: $f$ (given at T-points) is averaged to the X-points,
multiplied there by the cross velocity averaged to the corner, and the
product is averaged back to the faces:

$$
\frac{\partial u}{\partial t}\bigg|_{\text{Cor}}
  = +\overline{f_X\,\overline{v}^{x}}^{y},
\qquad
\frac{\partial v}{\partial t}\bigg|_{\text{Cor}}
  = -\overline{f_X\,\overline{u}^{y}}^{x},
\qquad
f_X = \overline{f}^{xy}.
$$

Both tendencies come from the same corner products, so the kinetic-energy
tendency $\sum u\,\dot u + \sum v\,\dot v$ vanishes to round-off for **any**
$f$, including a $\beta$-plane, on a periodic domain or a closed basin (with
a `BC` of Dirichlet zero on every wall, filled at `stagger="U"` and `"V"`).  For
uniform $f$ it reduces to the familiar $+f\,\overline{v}^{xy}$,
$-f\,\overline{u}^{xy}$ (`interpolate` from V to U and from U to V).  The form
used before #289, $f_{\text{on-U}}\,\overline{v}_{\text{on-U}}$, conserves
energy only for uniform $f$.

### Usage

```python
from gridax import CartesianCGrid2D, coriolis

grid = CartesianCGrid2D.from_interior(64, 64, 1e6, 1e6)

# f at T-points
f = jnp.full((grid.Ny, grid.Nx), 1e-4)

du_cor, dv_cor = coriolis(u, v, grid, f=f)
```

---

## Momentum Advection (Vortex-Force Form)

### Theory

The **vector-invariant (vortex-force)** form of the horizontal momentum
advection separates the advection into a vorticity flux and a kinetic-energy
gradient:

$$
\frac{\partial u}{\partial t}\bigg|_{\text{adv}}
  = +(\zeta\,v)_{\text{on-U}} - \frac{\partial K}{\partial x}
$$

$$
\frac{\partial v}{\partial t}\bigg|_{\text{adv}}
  = -(\zeta\,u)_{\text{on-V}} - \frac{\partial K}{\partial y}
$$

where $\zeta = \partial v/\partial x - \partial u/\partial y$ is the relative
vorticity at X-points, $K = \tfrac{1}{2}(\bar{u}^2 + \bar{v}^2)$ is the
kinetic energy at T-points, and $(\zeta\,v)_{\text{on-U}}$ is the
vorticity-flux product interpolated to U-points.

This form has important discrete conservation properties depending on how
the vorticity-flux product is evaluated.  Four forms are available:

| `form=` | Discrete form of $(\zeta\,v)_{\text{on-U}}$ | Key property | Reference |
|--------|------|-------------|-----------|
| `"energy"` (default) | $\overline{\zeta\,\overline{v}^{xy}}^{y}$: product at the corners, then average | Conserves total KE for non-divergent flow | Sadourny (1975) E-scheme |
| `"enstrophy"` | $\overline{\zeta}^{y}\,\overline{v}^{xy}$: average to the face, then multiply | Conserves potential enstrophy | Sadourny (1975) Z-scheme |
| `"blend"` | $\tfrac13 E + \tfrac23 Z$ | Conserves neither exactly | linear blend, **not** Arakawa & Lamb (1981) |
| `"arakawa_lamb"` | six products for a face, with $K = \tfrac12(\overline{u^2}^x + \overline{v^2}^y)$ | Conserves energy and potential enstrophy | Arakawa & Lamb (1981) |

:::{tip} When to use each form
- **`"energy"`**: energy-conserving vortex force, the usual choice for
  long integrations.
- **`"enstrophy"`**: preferred when enstrophy conservation matters (e.g.
  turbulence spindown experiments).
- **`"blend"`**: a ⅓–⅔ blend of the two; it is not the Arakawa–Lamb
  (1981) scheme.
- **`"arakawa_lamb"`**: conserves both, at the cost of a wider stencil.
:::

### Usage

```python
from gridax import CartesianCGrid2D, advect_momentum

grid = CartesianCGrid2D.from_interior(64, 64, 1e6, 1e6)

# Energy-conserving (default)
du_adv, dv_adv = advect_momentum(u, v, grid)

# Enstrophy-conserving
du_adv, dv_adv = advect_momentum(u, v, grid, form="enstrophy")

# 1/3 energy + 2/3 enstrophy blend
du_adv, dv_adv = advect_momentum(u, v, grid, form="blend")
```

---

## Vortex Force

### Theory

The rotation term of the vector-invariant equations is a force at a right angle to the flow. With $q$ the vorticity at X-points and $(U, V)$ the transport at the faces,

$$
\frac{\partial u}{\partial t} = +(q\,V)_{\text{on-U}}, \qquad \frac{\partial v}{\partial t} = -(q\,U)_{\text{on-V}}.
$$

`vortex_force(q, u, v, grid, form=…)` has the two forms of Sadourny and the scheme of Arakawa and Lamb, which weighs the four cross transports and the two along-stream ones around a face with sums of $q$. With the mass fluxes $U = \overline{h}^x u$, $V = \overline{h}^y v$ and $q = (\zeta + f)/\overline{h}^{xy}$, in a periodic shallow-water box with divergence (`tests/test_vortex_force.py`):

| `form=` | Work of the force, $\sum U \dot u + V \dot v$ | $\frac{d}{dt}\sum \tfrac12 h q^2$ |
|---|---|---|
| `"energy"` | zero to rounding | not zero |
| `"enstrophy"` | not zero | zero to rounding |
| `"arakawa_lamb"` | zero to rounding | zero to rounding |

On a sphere the transports carry the lengths of the faces, so the work of `"arakawa_lamb"` is zero there too.

### Usage

```python
from gridax import bernoulli_potential, sw_potential_vorticity, vortex_force

q = sw_potential_vorticity(u, v, h, f, grid)
du, dv = vortex_force(q, hu, hv, grid, form="arakawa_lamb")  # hu, hv: the mass fluxes
```

---

## Arakawa Jacobian

### Theory

The **Arakawa (1966) Jacobian** $J(f, g) = \partial f/\partial x\,\partial g/\partial y - \partial f/\partial y\,\partial g/\partial x$ is the classical advection operator for QG models.  Unlike simple centred differences, the three-term Arakawa average

$$
J(f, g) = \tfrac{1}{3}(J^{++} + J^{+\times} + J^{\times+})
$$

conserves **energy**, **enstrophy**, and satisfies $J(f, f) = 0$ and
$\int J(f, g)\,\mathrm{d}A = 0$ exactly at the discrete level.

The three terms are:
- $J^{++}$: standard centred (advective) form
- $J^{+\times}$: flux form in one direction
- $J^{\times+}$: flux form in the other direction

:::{warning} Conservation under a mask
The discrete identities above hold for the unmasked operator with
suitable boundary conditions (e.g. $\psi$ constant on the boundary).
`arakawa_jacobian(f, g, grid, mask=mask)` zeroes land inputs and dry output
cells, so sums over a masked domain such as $\sum J$ or
$\sum q\,J$ are **not** guaranteed to vanish exactly.
:::

### Usage

```python
from gridax import CartesianCGrid2D, arakawa_jacobian

grid = CartesianCGrid2D.from_interior(64, 64, 1e6, 1e6)

# QG vorticity advection: J(psi, q)
# psi, q have shape [Ny, Nx] with one ghost cell on each side
Jpsi_q = arakawa_jacobian(psi, q, grid, mask=mask)  # mask optional
# Jpsi_q has shape [Ny, Nx]; ghost ring and dry cells are 0
```

---

## Decision Guide

```
What spatial operator do you need?
│
├── Divergence ∇·(u, v)                      → divergence
├── Relative vorticity ζ = ∂v/∂x − ∂u/∂y     → curl
├── Shallow-water PV q = (ζ+f)/h             → sw_potential_vorticity
│
├── Tracer diffusion ∂q/∂t = ∇·(κ∇q)
│   ├── Harmonic   → diffuse
│   └── Biharmonic → biharmonic  (scale-selective)
│
├── Coriolis tendency (f×v, −f×u)            → coriolis
│
├── Momentum advection ∂u/∂t|adv
│   ├── Conservative vortex-force form → advect_momentum, or vortex_force on (q, hU, hV)
│   └── QG vorticity advection         → arakawa_jacobian
│
└── Diagnostics (KE, APE, enstrophy, Okubo-Weiss, …)
    ├── Energy                → kinetic_energy, bernoulli_potential
    ├── Strain / Okubo-Weiss  → strain_rate_squared, okubo_weiss_parameter
    ├── Enstrophy             → enstrophy(curl(u, v, grid, mask=mask))
    ├── QG PV                 → qg_potential_vorticity
    └── See [Diagnostics API reference](api/diagnostics.md)
```

---

## References

- Sadourny (1975) — C-grid momentum advection, energy- and enstrophy-conserving schemes
- Arakawa & Lamb (1981) — Combined energy/enstrophy conserving scheme
- Arakawa (1966) — Energy/enstrophy-conserving Jacobian operator
- Smagorinsky (1963) — Horizontal diffusion parameterisation
- Haidvogel & Beckmann (1999) — Momentum advection in ocean models
