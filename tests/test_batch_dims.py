"""Leading (batch) axes on the 2-D Cartesian operators.

Every 2-D entry point takes ``[*batch, Ny, Nx]``: the stencils slice the
trailing two axes (``h[..., 1:-1, 2:]``) and ``interior`` pads only the
axes that lost a ghost ring. Each case is checked against a loop over the
batch members (bit for bit eager, to float64 rounding under ``jit``), with and without a
mask.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import inspect
import itertools

import equinox as eqx
import gridax as fvx
from gridax._src.api import OPERATIONS, SUPPORT
from gridax.stencils import interior
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.fixtures.inputs import make_grid_2d, make_mask_2d
from tests.test_gridops import CALLS

BATCH = (2, 3)
GRID = make_grid_2d()
MASK = make_mask_2d()
SCALAR = 0.7  # value for every float argument (dx, dy, g_prime, f0, …)

# Modules whose 2-D entry points are annotated ``*batch Ny Nx``. Every
# public function there that carries the annotation is tested below, so the
# annotation and the behaviour cannot drift apart. The canonical functions are
# tested through `GridOps`, masked and not.
MODULES = (
    "gridax._src.operators.stencils",
    "gridax._src.operators.vorticity",
    "gridax._src.operators.diagnostics",
)


def _batched(fn: Callable) -> bool:
    # 2-D entry points only: the fields built below are [*batch, Ny, Nx]. The
    # 1-D and 3-D raw stencils (annotated "*batch Nx", "*batch Nz Ny Nx") are
    # checked for batch axes in packages/gridax/tests/test_stencils_backends.py.
    return any("*batch Ny Nx" in str(p.annotation) for p in _params(fn))


def _params(fn: Callable) -> list[inspect.Parameter]:
    return [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.default is inspect.Parameter.empty and p.name != "self"
    ]


def _discover() -> list:
    functions = []
    for name in MODULES:
        mod = importlib.import_module(name)
        for attr, obj in vars(mod).items():
            if attr.startswith("_") or getattr(obj, "__module__", None) != name:
                continue
            if attr in OPERATIONS:
                continue
            if inspect.isfunction(obj) and _batched(obj):
                functions.append(obj)
    return functions


FUNCTIONS = _discover()

BCS = {
    "Dirichlet": fvx.BC(x=fvx.Dirichlet(1.5), y=fvx.Dirichlet(1.5)),
    "Neumann": fvx.BC(x=fvx.Neumann(0.25), y=fvx.Neumann(0.25)),
    "Periodic": fvx.BC(x=fvx.Periodic(), y=fvx.Periodic()),
    "Outflow": fvx.BC(x=fvx.Outflow(), y=fvx.Outflow()),
    "Reflective": fvx.BC(x=fvx.Reflective(), y=fvx.Reflective()),
    "Sponge": fvx.BC(x=fvx.Sponge(0.3, 0.4), y=fvx.Sponge(0.3, 0.4)),
    "Robin": fvx.BC(x=fvx.Robin(1.0, 0.5, 0.2), y=fvx.Robin(1.0, 0.5, 0.2)),
    "Extrapolation": fvx.BC(x=fvx.Extrapolation(2), y=fvx.Extrapolation(2)),
    "Slip": fvx.BC(x=fvx.Slip(0.5), y=fvx.Slip(0.5)),
    "mixed": fvx.BC(
        south=fvx.Dirichlet(1.5),
        north=fvx.Neumann(0.25),
        west=fvx.Slip(0.5),
        east=fvx.Outflow(),
    ),
}


class _Fields:
    """A function of ``n`` fields."""

    def __init__(self, n: int, fn: Callable) -> None:
        self.n, self.fn = n, fn


def _array_params(fn: Callable) -> int | None:
    """Number of required array arguments, or ``None`` if an argument is
    neither an array nor a float."""
    if isinstance(fn, _Fields):
        return fn.n
    n = 0
    for p in inspect.signature(fn).parameters.values():
        if p.default is not inspect.Parameter.empty or p.name == "self":
            continue
        ann = str(p.annotation)
        if ann.startswith("float"):  # float, or float | 0-d array: a scalar
            continue
        if "Array" in ann or p.annotation is inspect.Parameter.empty:
            n += 1
        else:
            return None
    return n


def _call(fn: Callable, fields: list) -> object:
    if isinstance(fn, _Fields):
        return fn.fn(*fields)
    it = iter(fields)
    args = []
    for p in inspect.signature(fn).parameters.values():
        if p.default is not inspect.Parameter.empty or p.name == "self":
            continue
        ann = str(p.annotation)
        is_array = not ann.startswith("float") and (
            "Array" in ann or p.annotation is inspect.Parameter.empty
        )
        args.append(next(it) if is_array else SCALAR)
    return fn(*args)


def _cases() -> list:
    cases = []
    for name, grids in SUPPORT.items():
        if "CartesianCGrid2D" not in grids:
            continue
        n, kwargs = CALLS[name]
        for masked in (False, True):
            ops = fvx.GridOps(GRID, MASK if masked else None)

            def call(*fields, ops=ops, name=name, kwargs=kwargs):
                return getattr(ops, name)(*fields, **kwargs)

            tag = f"{name}{'[mask]' if masked else ''}"
            cases.append(pytest.param(_Fields(n, call), id=tag))
    for fn in FUNCTIONS:
        assert _array_params(fn), fn.__name__
        cases.append(pytest.param(fn, id=fn.__name__))
    for (name, bc), point in itertools.product(BCS.items(), "TUVX"):
        cases.append(
            pytest.param(
                lambda f, b=bc, p=point: fvx.fill_ghosts(f, GRID, b, stagger=p),
                id=f"BC.{name}[{point}]",
            )
        )
    return cases


def _fields(n: int) -> list:
    shape = (*BATCH, GRID.Ny, GRID.Nx)
    return [jax.random.normal(jax.random.key(k), shape) for k in range(n)]


def _per_member(fn: Callable, fields: list) -> object:
    flat = [f.reshape(-1, *f.shape[len(BATCH) :]) for f in fields]
    outs = [_call(fn, [f[b] for f in flat]) for b in range(flat[0].shape[0])]
    return jax.tree.map(lambda *ys: jnp.stack(ys).reshape(*BATCH, *ys[0].shape), *outs)


@pytest.mark.parametrize("fn", _cases())
def test_leading_axes_match_a_loop_over_members(fn):
    fields = _fields(_array_params(fn) or 1)
    expected = _per_member(fn, fields)
    eager = _call(fn, fields)
    jitted = eqx.filter_jit(lambda *fs: _call(fn, list(fs)))(*fields)
    for got in (eager, jitted):
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
            assert a.shape == b.shape
            # Eager is bit-identical; under jit XLA may reassociate a fused
            # sum differently per shape, so allow float64 rounding on O(1)
            # fields (observed max |Δ| ≈ 3e-17).
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_interior_keeps_equal_axes_and_pads_ghost_axes():
    like = jnp.ones((4, 6, 5))
    out = interior(jnp.full((4, 4, 3), 2.0), like)
    assert out.shape == like.shape
    np.testing.assert_array_equal(out[:, 1:-1, 1:-1], 2.0)
    np.testing.assert_array_equal(out[:, 0], 0.0)
    # 3-D: every axis lost its ghost ring (z ghost slabs included)
    out3 = interior(jnp.full((2, 4, 3), 2.0), like)
    np.testing.assert_array_equal(out3[0], 0.0)


@pytest.mark.parametrize("shape", [(4, 5, 3), (4, 4)])
def test_interior_rejects_values_that_do_not_fit(shape):
    with pytest.raises(ValueError, match="interior"):
        interior(jnp.ones(shape), jnp.ones((4, 6, 5)))
