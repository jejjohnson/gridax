# Reconstruction Operators

High-order flux reconstruction methods (upwind, WENO) for advection terms.

## One reconstruction for every scheme and axis

`gridax.reconstruct(q, u, grid, axis=..., scheme=...)` gives the advective flux on the faces along any axis, for any [scheme](schemes.md). A scheme is a ladder of stencil widths, widest first, and each face takes the widest one that fits in the array (and, with `mask=`, on wet cells), down to first-order upwind. Every method of `Reconstruction1D/2D/3D` below is the same flux.

::: gridax.reconstruct

## The class forms

::: gridax.Reconstruction1D

::: gridax.Reconstruction2D

::: gridax.Reconstruction3D

## Point stencils

The cell averages around a face in, a face value out: `gridax.advection.kernels`.

::: gridax.advection.kernels

## Flux Limiters

::: gridax.minmod

::: gridax.mc

::: gridax.superbee

::: gridax.van_leer
