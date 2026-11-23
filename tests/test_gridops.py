"""`GridOps`: a method per canonical function, and nothing else (#492)."""

from __future__ import annotations

import inspect
from typing import Any

import gridax
from gridax import GridOps
from gridax._src.api import OPERATIONS, SUPPORT
from gridax.schemes import Upwind
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_registry_parity import GRIDS

RNG = np.random.default_rng(492)

#: How each function is called: its field arguments, and its other keywords.
CALLS: dict[str, tuple[int, dict[str, Any]]] = {
    "advect": (3, {"scheme": Upwind(3)}),
    "arakawa_jacobian": (2, {}),
    "area_mean": (1, {}),
    "area_sum": (1, {}),
    "biharmonic": (1, {"kappa4": 0.3}),
    "coriolis": (2, {"f": 1.2}),
    "curl": (2, {}),
    "difference": (1, {"axis": "x", "src": "T", "dst": "U"}),
    "diffuse": (1, {"kappa": 0.3}),
    "divergence": (2, {}),
    "grad_perp": (1, {}),
    "gradient": (1, {}),
    "interpolate": (1, {"src": "T", "dst": "U"}),
    "kinetic_energy": (2, {}),
    "laplacian": (1, {}),
    "reconstruct": (2, {"axis": "x", "scheme": Upwind(3)}),
    "shear_strain": (2, {}),
    "tensor_strain": (2, {}),
    "volume_mean": (1, {}),
    "volume_sum": (1, {}),
}
PAIRS = [(name, grid) for name, grids in SUPPORT.items() for grid in grids]
MASKS = {1: gridax.Mask1D, 2: gridax.Mask2D, 3: gridax.Mask3D}


def mask_of(grid: Any) -> Any:
    wet = RNG.random(grid.shape) > 0.1
    return MASKS[len(grid.axes)].from_grid(grid, wet)


def call(name: str, grid: Any, lead: tuple[int, ...] = ()) -> tuple[list[Any], Any]:
    n, kwargs = CALLS[name]
    fields: list[Any] = [
        jnp.asarray(RNG.normal(size=(*lead, *grid.shape))) for _ in range(n)
    ]
    if name == "advect" and len(grid.axes) == 1:
        fields[-1] = None
    if getattr(grid, "_collocated", False):
        kwargs = {k: v for k, v in kwargs.items() if k not in ("src", "dst")}
    return fields, dict(kwargs)


def same(got: Any, want: Any) -> None:
    assert jax.tree.structure(got) == jax.tree.structure(want)
    for x, y in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(x, y)


def test_every_function_is_called_here() -> None:
    assert set(CALLS) == set(OPERATIONS)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(("name", "grid_name"), PAIRS)
def test_a_method_is_its_function(name: str, grid_name: str, masked: bool) -> None:
    grid = GRIDS[grid_name]
    mask = mask_of(grid) if masked else None
    ops = GridOps(grid, mask)
    fields, kwargs = call(name, grid)
    op = getattr(gridax, name)
    same(getattr(ops, name)(*fields, **kwargs), op(*fields, grid, mask=mask, **kwargs))

    def method(ops: Any, *fields: Any) -> Any:
        return getattr(ops, name)(*fields, **kwargs)

    def function(grid: Any, mask: Any, *fields: Any) -> Any:
        return op(*fields, grid, mask=mask, **kwargs)

    # the binding crosses jit as an argument
    same(jax.jit(method)(ops, *fields), jax.jit(function)(grid, mask, *fields))


@pytest.mark.slow
@pytest.mark.parametrize(("name", "grid_name"), PAIRS)
def test_vmap_over_a_leading_axis(name: str, grid_name: str) -> None:
    grid = GRIDS[grid_name]
    ops = GridOps(grid, mask_of(grid))
    fields, kwargs = call(name, grid, (2,))
    axes = [None if f is None else 0 for f in fields]
    got = jax.vmap(lambda *x: getattr(ops, name)(*x, **kwargs), axes)(*fields)
    for k in range(2):
        one = [None if f is None else f[k] for f in fields]
        want = getattr(ops, name)(*one, **kwargs)
        for x, y in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            np.testing.assert_allclose(x[k], y, rtol=1e-12, atol=1e-12)


@pytest.mark.slow
def test_grad_through_a_coefficient_and_the_grid() -> None:
    grid = GRIDS["CartesianCGrid2D"]
    (f,), _ = call("diffuse", grid)

    def loss(ops: Any, kappa: Any) -> Any:
        return (ops.diffuse(f, kappa=kappa) ** 2).sum()

    ops = GridOps(grid, mask_of(grid))
    slope = jax.grad(loss, 1)(ops, 0.3)
    step = 1e-5
    central = (loss(ops, 0.3 + step) - loss(ops, 0.3 - step)) / (2 * step)
    np.testing.assert_allclose(slope, central, rtol=1e-6)
    per_grid = jax.grad(lambda dx: loss(GridOps(with_dx(grid, dx)), 0.3))(0.1)
    assert np.isfinite(per_grid) and per_grid != 0.0


def with_dx(grid: Any, dx: Any) -> Any:
    ny, nx = grid.shape
    return gridax.CartesianCGrid2D.from_interior(
        nx - 2, ny - 2, (nx - 2) * dx, (ny - 2) * 0.2
    )


@pytest.mark.parametrize("grid_name", sorted(GRIDS))
def test_it_has_the_methods_of_its_grid(grid_name: str) -> None:
    ops = GridOps(GRIDS[grid_name])
    defined = {name for name, grids in SUPPORT.items() if grid_name in grids}
    assert set(dir(ops)) & set(OPERATIONS) == defined
    for name in OPERATIONS:
        assert hasattr(ops, name) is (name in defined)
    for name in set(OPERATIONS) - defined:
        with pytest.raises(AttributeError, match=f"{name} is not defined on"):
            getattr(ops, name)


@pytest.mark.parametrize("name", sorted(OPERATIONS))
def test_names_docs_and_signatures(name: str) -> None:
    op = getattr(gridax, name)
    assert getattr(GridOps, name) is op
    grid = GRIDS[SUPPORT[name][0]]
    method = getattr(GridOps(grid), name)
    assert method.__doc__ == op.__doc__ and method.__name__ == name
    own = list(inspect.signature(method).parameters)
    assert own == [p for p in inspect.signature(op).parameters if p != "grid"]


def test_a_call_may_give_another_mask() -> None:
    grid = GRIDS["CartesianCGrid2D"]
    u, v = call("curl", grid)[0]
    mask = mask_of(grid)
    same(GridOps(grid).curl(u, v, mask=mask), GridOps(grid, mask).curl(u, v))
    same(GridOps(grid, mask).curl(u, v, mask=None), GridOps(grid).curl(u, v))


def test_fields_keep_their_points() -> None:
    grid = GRIDS["CartesianCGrid2D"]
    u, v = (
        gridax.CField(x, ("y", "x"), loc=point)
        for x, point in zip(call("curl", grid)[0], "UV", strict=True)
    )
    ops = GridOps(grid, mask_of(grid))
    zeta = ops.curl(u, v)
    assert zeta.stagger == (("x", "right"), ("y", "right"))
    same(zeta.data, ops.curl(u.data, v.data))


def test_it_holds_its_grid_and_its_mask_and_nothing_else() -> None:
    grid = GRIDS["CartesianCGrid2D"]
    mask = mask_of(grid)
    leaves = len(jax.tree.leaves(GridOps(grid, mask)))
    assert leaves == len(jax.tree.leaves((grid, mask)))
