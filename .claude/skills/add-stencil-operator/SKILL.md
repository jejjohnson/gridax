---
name: add-stencil-operator
description: Add a spatial operator to finitevolX — a raw stencil (Layer 1) or a mask-aware canonical function of a grid (Layer 2), which `GridOps` (Layer 3) then has as a method, such as a difference, interpolation, divergence / vorticity, diffusion, Coriolis or diagnostic, in 1-D / 2-D / 3-D and on the sphere — with its golden mask fixtures, scale-covariance case, docs and capability index. Use when asked to add, port or implement a finite-difference / finite-volume operator, stencil, diagnostic or spherical operator in gridax/_src/operators or gridax/_src/diffusion.
---

# Add a stencil operator

Read contracts 1–3 and 6–7 in `AGENTS.md` (the C-grid, the interior idiom
and the layers, masks, units, JAX numerics) first; this is the
step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/capabilities.md` for the quantity and its synonyms (curl =
  relative vorticity; Laplacian = harmonic diffusion with κ = 1; Okubo–Weiss
  = strain² − vorticity²). A composition of existing functions (a
  `difference` of an `interpolate`) needs no new code:
  say so instead.
- Check the theory page for the family (`docs/difference.md`,
  `docs/interpolation.md`, `docs/spatial_operators.md`, `docs/stencils.md`)
  and the module it would join.

## 2. Write it bottom-up

| Layer | Where | Rules |
|---|---|---|
| 0 — coefficients | `gridax/_src/stencils/coefficients.py` (public: `gridax.stencils`) | Offsets and exact coefficients from the Taylor conditions (`difference_offsets`, `difference_coeffs(..., data="point" \| "cell")`); a new stencil takes its weights from here instead of typing rationals, and a test pins it bit for bit |
| 1 — raw stencil | `gridax/_src/operators/stencils.py` (public with Layer 0 in `gridax.stencils`; wider staggered stencils are `staggered_difference` / `staggered_average` in `gridax/_src/stencils/staggered.py`), named `<op>_<axis>_<side>_<d>d` (a rank-`d` stencil reads the trailing `d` axes of `[*batch, *spatial]` and trims every spatial axis) | Pure slicing of the full array, interior-sized result (`[Ny-2, Nx-2]`), no metric, no padding; docstring with the `Δx h[i+½] = …` formula and the source → target stagger |
| 2 — canonical function | the family module (`operators/calculus.py`, `operators/diagnostics.py`, `operators/jacobian.py`, `diffusion/diffuse.py`, …) | One name for every grid: declared with `@grid_dispatch` over `@located`, listed in `SUPPORT` (`_src/api.py`), one implementation registered per grid type. Raw stencils ÷ the metric of the grid, padded with `interior(values, like)` from `operators/_ghost.py`; `mask=` a keyword, the output masked on the dry points of its stagger (`mask.wet(at=…)`, `kind="strict"` at X) behind `if mask is not None:`; array-API code (the AGENTS.md "Backends" rule); no physical constant inside an expression (default parameters only) |
| 3 — binding | nothing to write: `GridOps` gets the method from `OPERATIONS` (add the module to `_DECLARED_IN` of `gridops.py` if it is new) | |

- Comment every stencil with its half-index formula
  (`# zeta[j+1/2, i+1/2] = dv/dx - du/dy`) and the array shapes.
- Write region: `[1:-1, 1:-1]` only. Name the ghost cells the operator
  reads in the docstring; a backward operator reads the BC-owned west /
  south ghost, and the caller must fill it.
- Inputs that may hold `NaN` on land: zero them with `jnp.where` before the
  stencil (as `kinetic_energy` does), then mask the output
  with `jnp.where`.
- Spherical: take the metric from `SphericalCGrid2D/3D` (`cos_lat_*`,
  `dx_T`, …) and divide by `cos(lat)` through
  `operators/_utils._safe_div_cos`.
- 3-D: leave the `k = 0` / `k = Nz-1` ghost slabs zero; a multilayer use
  needs no new code (`jax.vmap`).
- NumPy-style docstring: formula, parameters and returns with shapes, an
  `Examples` section (doctests are not collected: run it by hand).

## 3. Export and document

- `packages/gridax/src/gridax/__init__.py`: import, and add the name to `__all__` in its
  logical group (e.g. under `# Difference stencils`, `# Diagnostics`).
- `::: gridax.NewName` on the matching `docs/api/*.md` page; a
  paragraph (formula, stagger, ghost cells read) on the theory page.
- `make capabilities`.

## 4. Tests

- `tests/test_<family>.py` (one test class per operator and dimension):
  analytic truth (constant → 0, linear field → its slope, a known
  trigonometric field), the output shape and zero ghost ring, the
  ghost-cell behaviour at the first / last interior cell.
- Raw stencils: `tests/test_stencils.py`. Difference operators: a
  convergence-order check in `tests/test_convergence.py`.
- `jit` / `vmap` agreement: `tests/test_jax_transforms.py`.
- Units: add a `Case(name, run, labels, order, grid=..., dim=...)` to
  `_cases()` in `tests/test_scale_covariance.py` (`order` = power of the
  inverse length: 0 interpolation, 1 gradient, 2 Laplacian, 4 biharmonic).
- Masks: register `unmasked` and `masked` entries in
  the family's `_*_entries` helper in `tests/fixtures/_gen_golden.py`,
  run `uv run python tests/fixtures/_gen_golden.py`, `git add` the new
  `tests/fixtures/golden/<Op>__<method>__*.npz`, and assert them in the
  family's `tests/test_*_masks.py` with `assert_matches_golden` (plus the
  all-ocean-equals-unmasked and dry-points-are-zero checks). The generator
  rewrites every golden: `git diff --stat tests/fixtures/golden/` must show
  only your new files.

## 5. Verify

```bash
uv run pytest -n 2 tests/test_<family>.py tests/test_<family>_masks.py \
  tests/test_stencils.py tests/test_scale_covariance.py tests/test_jax_transforms.py \
  tests/test_capabilities.py
```

(for a difference operator: `tests/test_difference.py
tests/test_difference_masks.py tests/test_convergence.py`; the interpolation,
divergence and vorticity goldens live in `tests/test_interp_div_vort_masks.py`,
the spherical ones in `tests/test_spherical_masks.py`). Then the
`pre-pr-check` skill.
