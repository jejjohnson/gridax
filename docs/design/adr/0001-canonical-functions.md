# ADR 1: functions are the canonical form, classes are bindings

**Status:** accepted on 23 November 2026. **Epic:** [E5, #486](https://github.com/jejjohnson/gridax/issues/486). **Reverses:** the direction of [#456](https://github.com/jejjohnson/gridax/issues/456).

## Context

The derivative stack has four layers: coefficients, raw stencils, functional helpers and class operators. Layers 2 and 3 were split on the mask: Layer 2 was "metric-scaled, mask-free" and Layer 3 "carries a mask as a field". Because the mask was the dividing line, every masked operation needed a class and every class wanted a function, so one operation was public at several layers, under several names and with several signatures. A NumPy user had to build a class before computing one masked field.

| Operation | Public forms before E5 | What differs |
|---|---|---|
| Vorticity at X points | `diff_x_fwd_2d` with `diff_y_fwd_2d`; `Difference2D.curl`; `Vorticity2D.relative_vorticity`; `gridax.pipekit.Curl`; `relative_vorticity_cgrid` | the classes take a grid, Layer 1 takes nothing, the function takes `dx, dy` |
| Collocated curl | `gridax.analysis.curl_2d` and `adaptive_curl_2d(…, mask=)` | one name family, the arguments in the order `dy, dx` |
| Staggered difference | `diff_x_fwd_2d`; `Difference2D.diff_x_T_to_U`; `gridax.pipekit.Difference(axis="x", src="T", dst="U")` | three naming schemes |
| Interpolation | `avg_x_fwd_2d`; `Interpolation2D.T_to_U` and eleven more | `T_to_U` has no verb, `diff_x_T_to_U` has |
| Advective flux | `Advection2D(…, method="weno5")`; `reconstruct(…, scheme=WENO(5))`; the methods of `Reconstruction2D`; `weno_5pts`, `linear_4pts`, `upwind_3pt` | strings against scheme objects |
| Mask-aware collocated calculus | `gradient_2d` and `adaptive_gradient_2d`, and eleven more pairs | the mask is a second function |
| Diagnostics | `kinetic_energy(u, v)` and `Energetics2D.kinetic_energy` | the class adds only the mask |
| Reductions | `area_sum(field, grid, mask)` with `cartesian_area_sum`, `spherical_area_sum`, … | `area_sum` already dispatches on the grid |

#456 removed some of these by deprecating `divergence_2d` and `relative_vorticity_cgrid` in favour of `Difference2D.divergence` and `.curl`, under the rule that a C-grid Layer 2 helper exists only when it is more than one line of Layer 1.

## Decision

1. **One function per operation.** Layer 2 holds the canonical functions, `op(fields…, grid, *, scheme=…, mask=None, …)`. `grid` is positional wherever the operation moves values between points or uses the metric, never loose `dx, dy`, and the mask is a keyword. The rule that a Layer 2 helper must be more than one line of Layer 1 goes.
2. **The grid type dispatches.** The rank (1-D, 2-D, 3-D), the staggering and the geometry come from the type of `grid`, through `functools.singledispatch` on a private grid-first function. Dispatch happens on the Python type, which is static, so it costs nothing under `jit`, and the registry is data that a parity test reads.
3. **A move is named by its points.** `axis=`, `src=` and `dst=` name a staggered move, from one table of valid triples. A field that knows its point ([`CField`](../../fields.md)) supplies `src`.
4. **Schemes are objects**, never strings.
5. **Layer 3 holds bindings.** A binding holds a grid, a mask and caches derived from them, and each of its methods is one call of a Layer 2 function. `GridOps` and the pipekit operators are bindings. A test asserts that a binding equals the function, bit for bit.
6. **Pointwise formulas leave gridax.** A formula that neither moves values between points nor uses the metric (enstrophy, Okubo–Weiss, potential vorticity from its parts) belongs in geophysix.

## Alternatives that were rejected

**Classes are canonical.** This is where #456 was heading: every operation is a method of a class that holds a grid and a mask. It keeps 43 classes, one per family, rank and geometry, and a NumPy or notebook user builds an object to compute one field. The choice of rank and geometry is made by the class name, which a generic model cannot write.

**Keep both layers and document them.** It changes no code, and it leaves two signatures per operation to keep in step. The table above is what that produced.

A `match` statement on the grid type was also considered for the dispatch. It is closed to new grid types, depends on the order of its cases when one grid class derives from another, and leaves no registry for a test to read.

## Consequences

- The layers are described the same way in `AGENTS.md`, [the derivative stack](../../stencils.md) and [the mask page](../../masks.md).
- The functions arrive family by family: the registry and the reductions ([#488](https://github.com/jejjohnson/gridax/issues/488)), `difference` and `interpolate` ([#489](https://github.com/jejjohnson/gridax/issues/489)), the vector calculus ([#490](https://github.com/jejjohnson/gridax/issues/490)), then advection, diffusion, rotation and diagnostics ([#491](https://github.com/jejjohnson/gridax/issues/491)). The classes and the old functions are removed in [#493](https://github.com/jejjohnson/gridax/issues/493): the test suite keeps them as bindings in `tests/fixtures/bindings.py`, and `GridOps` is the one binding of the package.
- `divergence_2d` and `relative_vorticity_cgrid` go when `gridax.divergence(u, v, grid)` and `gridax.curl(u, v, grid)` land in #490. Names replaced inside gridax are removed, with no alias.
- The companion epics use the same functions: schemes ([E6, #494](https://github.com/jejjohnson/gridax/issues/494)), grids and masks ([E7, #500](https://github.com/jejjohnson/gridax/issues/500)), regions and boundary conditions ([E8, #507](https://github.com/jejjohnson/gridax/issues/507)) and located fields ([E9, #514](https://github.com/jejjohnson/gridax/issues/514)).
