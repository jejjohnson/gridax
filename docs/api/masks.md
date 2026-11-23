# Masks

Land/ocean masks with automatic stencil dispatch for Arakawa C-grids.

## Grid Masks

The three masks share their builders and what they derive from the wet cells.

::: gridax.Mask1D
    options:
      inherited_members: true

::: gridax.Mask2D
    options:
      inherited_members: true

::: gridax.Mask3D
    options:
      inherited_members: true

## Coastal ghosts

::: gridax.CoastGhosts

## Stencil Capability

::: gridax.StencilCapability1D

::: gridax.StencilCapability2D

::: gridax.StencilCapability3D

## Masked Sample Statistics

Mean and standard deviation over a leading sample axis (time or ensemble
member) that exclude dry cells. These are what a per-gridpoint
standardising transform needs as its `loc` and `scale`:

```python
from gridax import masked_moments

# samples: [N, Ny, Nx] stacked snapshots; mask: Mask2D
loc, scale = masked_moments(samples, mask)          # per gridpoint
standardised = (samples - loc) / scale              # land stays at 0
```

Three guarantees make the pair safe to divide by:

- **Land never contributes.** Dry cells are zeroed with
  `jnp.where(mask, x, 0.0)` before the reduction, so NaN/Inf sentinels
  stored on land cannot contaminate wet-cell statistics — the same
  guarantee `area_mean` gives. With a spatial reduction
  (`axis=(0, -2, -1)`) the divisor is the wet count, not the cell count.
- **Dry cells get an identity `(0, 1)`,** so a masked field round-trips
  unchanged through `(x - loc) / scale`.
- **The scale is floored at `eps`,** because a constant field has exactly
  zero sample variance. The floor is applied to the variance rather than
  to the standard deviation: the value is the same, but `sqrt` has an
  infinite derivative at zero, so flooring afterwards would return a NaN
  gradient for precisely the case the floor exists for.

Pick the point with `at=` (`"T"`, `"U"`, `"V"`, `"X"`, or `"W"` on `Mask3D`) so U- and V-point fields use their own C-grid mask.

::: gridax.masked_mean

::: gridax.masked_std

::: gridax.masked_moments
