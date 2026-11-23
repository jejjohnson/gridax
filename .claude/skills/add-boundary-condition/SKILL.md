---
name: add-boundary-condition
description: Add a boundary condition to finitevolX — an atom for gridax.BC (like Dirichlet, Neumann, Robin, Slip, Sponge) with its half-cell placement and its wall node, or a functional ghost-ring filler (like wall_boundaries) — with the Atom union, tests and docs. Use when asked to add, port or fix a boundary condition, ghost-cell fill, wall or open-boundary treatment in gridax/_src/boundary.
---

# Add a boundary condition

Read contract 1 (which ghost cells are BC-owned for each stagger) and
contract 4 (boundary conditions) in `AGENTS.md`, then
`docs/boundary_conditions.md`.

## 1. Make sure it does not exist yet

Search `docs/capabilities.md` (the `boundary` groups): `Dirichlet`, `Neumann`, `Robin` (a·u + b·∂u/∂n = c), `Periodic`, `Outflow`, `Extrapolation` (`order=`), `Reflective`, `Sponge`, `Slip`; `BC` and `fill_ghosts`; the fillers `enforce_periodic`, `zero_boundaries`, `zero_gradient_boundaries`, `no_slip_boundaries`, `free_slip_boundaries`, `no_flux_boundaries`, `wall_boundaries`, `fix_boundary_corners`. Many "new" conditions are a `Robin` with particular coefficients, or a `BC` of existing atoms.

## 2. Write it (`packages/gridax/src/gridax/_src/boundary/`)

- **An atom** goes in `atoms.py`: a `@pytree_dataclass` that subclasses `_Atom`, with its parameters as fields (a number or an array `[*batch, N_face]`, a pytree leaf; a setting that picks a code path is a `static_field()`), and validation in `__check_init__` (as `Sponge` checks `weight`). It knows neither its face nor its point: `BC` gives the face and `fill_ghosts` the point, and `_Atom.fill` handles both (`_Atom.coast` applies the same formulas behind a coast).
- Implement `_ghost(self, phi, k, step)`: layer `k` outside a wall half a cell from the first node, where `phi(m)` is the m-th interior node from the wall and `step` the normal spacing on the face. Place the condition **at the wall**: Dirichlet is `2 v - phi(k)`, Neumann `phi(k) + (2k - 1) step g`. Write that formula in the docstring and in a comment.
- Implement `_wall(self, phi, step)`: the value of the node that sits on the wall (`U` / `X` on the x faces, `V` / `X` on the y faces, `W` on the z faces), where the condition is imposed on that node and the gradient is one-sided.
- Implement `_outside(self, wall, phi, k, step)` only when the layers outside a node on the wall are not the odd mirror `2 wall - phi(k)` (as `Neumann`, `Outflow` and `Reflective` do). A condition that needs more than this (as `Periodic`, which copies from the far side) overrides `fill`.
- The atom runs on any array backend: use `xp = array_namespace(...)` and `set_at`, never `jnp` or `.at[]` (the AGENTS.md "Backends" rule).
- Add the class to the `Atom` union at the bottom of `atoms.py` (it types `BC`'s faces).
- **A functional filler** goes in `boundary.py` and should be a thin wrapper over `fill_ghosts` where possible (as `enforce_periodic` is).
- The elliptic inversions in sesmax-core (`sesmax_core.gfd`) take a `gridax.BC` as `bc=`: all faces Dirichlet select the DST, zero Neumann the DCT and Periodic the FFT. A new atom is rejected there with a clear message; don't extend that mapping unless the spectral solver has a matching transform.

## 3. Export and document

`packages/gridax/src/gridax/__init__.py` (import + `__all__`, next to `Dirichlet`), `::: gridax.NewBC` on `docs/api/boundary_conditions.md`, the formula on `docs/boundary_conditions.md`, `make capabilities`.

## 4. Tests

In `tests/test_fill_ghosts.py`: the ghost value on each face against a hand computation with `dx != dy`, for a cell-centred point and a point with the node on the wall, with `width` 1 and more, that the interior and the other faces are untouched, the validation error, and that the condition holds at the wall (e.g. `0.5 * (ghost + interior) == value`). `tests/test_boundary_helpers.py` for functional fillers. The elliptic wrappers that read a `BC` moved to `sesmax_core.gfd` with their tests (#475).

## 5. Verify

```bash
uv run pytest -n 2 tests/test_fill_ghosts.py tests/test_boundary_helpers.py \
  tests/test_open_walls.py tests/test_capabilities.py
```

Then the `pre-pr-check` skill.
