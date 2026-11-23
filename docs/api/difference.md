# Difference Operators

Finite-difference operators for computing spatial derivatives on staggered Arakawa C-grids.

The canonical form is one function on any C-grid, with the move named by its axis and points. The classes below bind a grid and a mask to it.

::: gridax.difference

## Bindings

::: gridax.Difference1D

::: gridax.Difference2D

::: gridax.Difference3D

## Spherical

`SphericalDifference2D` also provides `geostrophic_velocity(h, f, gravity)`,
the public (mask-aware) entry point for the spherical geostrophic velocity.

::: gridax.SphericalDifference2D
