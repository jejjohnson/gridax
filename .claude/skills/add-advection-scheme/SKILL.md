---
name: add-advection-scheme
description: Add an advection scheme to finitevolX — a face-value reconstruction (upwind, linear, WENO / WENO-Z variant), a TVD flux limiter, or a face-flux function (Rusanov-style) — wired through gridax.schemes, the coastal stencil fallback, the scale-covariance lists and the golden fixtures. Use when asked to add, port or implement a reconstruction, WENO stencil, limiter or numerical flux in gridax/_src/advection.
---

# Add an advection scheme

Read contract 2 (the advection write region and `wall` modes), contract 3
(masks) and contract 6 (units) in `AGENTS.md` first, then
`docs/advection.md`.

## 1. Make sure it does not exist yet

Search `docs/capabilities.md` and `gridax.schemes`: `Centred(k)`, `Upwind(k)`,
`Biased(k, side)`, `WENO(k)` and `WENO(k, "z")`, `TVD` with `minmod` / `van_leer`
/ `superbee` / `mc`; the `upwind_*pt`, `weno_*pts*` and `linear_*pts*` stencil
functions of `gridax.advection.kernels`; `reconstruct`, `upwind_flux` and
`rusanov_flux`. A different limiter for TVD is a new limiter, not a new
scheme.

## 2. Where each piece goes (`packages/gridax/src/gridax/_src/advection/`)

Everything here runs on any array backend: `xp = namespace_of(...)` and the hooks in `pipekit_array.arrays`, never `jnp` or `.at[]`.

| Piece | File | Wire it into |
|---|---|---|
| Point stencil (`qm, q0, qp, … → q at the face`) | `weno.py` (nonlinear, left and `_right` variants), `linear.py` (polynomial) | `__init__.py` exports |
| Limiter `φ(r)` | `limiters.py` | `_LIMITERS` in `_src/schemes/nonlinear.py` (then `TVD(limiter)` takes it, in every function) |
| Scheme (a ladder of tiers) | `gridax/_src/schemes/linear.py` or `nonlinear.py`: a frozen dataclass with `order` and `tiers()`, widest first, the last one always fitting; `upwinded` and `linear` as they are | `gridax/schemes.py` exports; `advect`, `reconstruct` and the moves then take it as `scheme=`, masked or not |
| Coastal fallback | nothing: `face_flux` (`reconstruct.py`) picks the widest tier on wet cells, from `Tier.size` | nothing |

Rules:

- The face value is upwinded on the sign of the face velocity
  (`jnp.where(u >= 0, left_biased, right_biased)`), and the flux is
  `h_face * u`, single-valued at each face so the divergence conserves mass.
- Closed walls: `advect` writes the tendency on `[2:-2]` /
  `[2:-2, 2:-2]`; a stencil wider than the ghost ring must fall back near
  the walls (as `weno5` does) rather than read outside the array.
- No absolute epsilon on a dimensional quantity: floor the WENO weights
  through `weno._normalised` + `_WENO_EPS_REL` and TVD ratios through
  `schemes.nonlinear._slope_ratio`; add a nonlinear scheme to `WENO_METHODS` /
  `TVD_METHODS` in `tests/test_scale_covariance.py`
  (`TestReconstructionAmplitudeCovariance` holds them to rounding) and to
  `tests/test_relative_eps.py` (finite gradients on flat stencils, float32).
- Comment the stencil with its formula and cite the scheme in the
  docstring.

## 3. Export and document

`packages/gridax/src/gridax/__init__.py` (import + `__all__` group, e.g. `# WENO stencil
functions`, `# Flux limiters`), the `docs/api/advection.md` /
`docs/api/reconstruction.md` page, the list of schemes in `docs/advection.md` and `docs/schemes.md`, then `make capabilities`.

## 4. Tests

- Point stencils / limiters: `tests/test_weno.py` (constant fields, polynomial
  exactness, monotonicity near a step, symmetry), `tests/test_limiters.py`
  (TVD region, φ(1) = 1).
- On a grid: `tests/test_reconstruction.py`; the scheme in the `test_all_methods_run`
  lists of `tests/test_advection.py` and its name in `tests/_schemes.py`; conservation
  (`test_1d_conservation`).
- Units: add the method to `LINEAR_METHODS`, `WENO_METHODS` or `TVD_METHODS`
  in `tests/test_scale_covariance.py` (the probe adds a case per operator
  that supports it); if it carries an absolute epsilon, mark it as the WENO
  cases are and open an issue.
- Masked dispatch: golden entries in `_advection_entries`
  (`tests/fixtures/_gen_golden.py`) and `tests/test_advection_masks.py`;
  regenerate with `uv run python tests/fixtures/_gen_golden.py` and check
  that `git diff --stat tests/fixtures/golden/` lists only new files.
- Spherical: `tests/test_spherical_advection.py`.

## 5. Verify

```bash
uv run pytest -n 2 tests/test_weno.py tests/test_limiters.py tests/test_reconstruction.py \
  tests/test_reconstruction_pipeline.py tests/test_advection.py tests/test_advection_masks.py \
  tests/test_flux.py tests/test_spherical_advection.py tests/test_open_walls.py \
  tests/test_scale_covariance.py tests/test_capabilities.py
```

Then the `pre-pr-check` skill.
