"""Stencil, interpolation, boundary and mask operators of `gridax.pipekit`."""

from __future__ import annotations

import pytest

pytest.importorskip("pipekit_jax")

import equinox as eqx
import gridax as fvx
from gridax import pipekit as fp
from gridax.pipekit.testing import check_operator
import jax
import jax.numpy as jnp
import numpy as np
from pipekit_ops import Graph, Input

from tests.fixtures import bindings
from tests.fixtures._helpers import periodic
from tests.fixtures.inputs import make_grid_2d, make_mask_2d

GRID = make_grid_2d()
MASK = make_mask_2d()
D = bindings.Difference2D(grid=GRID)
I = bindings.Interpolation2D(grid=GRID)


def _field(seed: int) -> jax.Array:
    return jax.random.normal(jax.random.key(seed), (GRID.Ny, GRID.Nx))


def _f_beta(f0: float, beta: float) -> jax.Array:
    y = jnp.arange(GRID.Ny) * GRID.spacing("y")
    Y = jnp.broadcast_to(y[:, None], (GRID.Ny, GRID.Nx))
    return f0 + beta * (Y - jnp.mean(Y))


# (id, operator, Tier-A reference, number of input fields)
CASES = [
    *[
        (
            f"Difference-{a}-{s}{d}",
            fp.Difference(grid=GRID, axis=a, src=s, dst=d),
            getattr(D, f"diff_{a}_{s}_to_{d}"),
            1,
        )
        for a, s, d in fp.DIFFERENCES
    ],
    *[
        (
            f"Interp-{s}{d}",
            fp.Interp(grid=GRID, src=s, dst=d),
            getattr(I, f"{s}_to_{d}"),
            1,
        )
        for s, d in fp.INTERPOLATIONS
    ],
    ("Divergence", fp.Divergence(grid=GRID), D.divergence, 2),
    ("Curl", fp.Curl(grid=GRID), D.curl, 2),
    ("Laplacian", fp.Laplacian(grid=GRID), D.laplacian, 1),
    ("GradPerp", fp.GradPerp(grid=GRID), D.grad_perp, 1),
    (
        "Coriolis",
        fp.Coriolis(grid=GRID, f0=2.0, beta=0.3),
        lambda u, v: bindings.Coriolis2D(grid=GRID)(u, v, _f_beta(2.0, 0.3)),
        2,
    ),
    *[
        (
            f"FillGhosts-{k}-{s}",
            fp.FillGhosts(kind=k, stagger=s),
            lambda x, bc=bc, s=s: fvx.fill_ghosts(x, GRID, bc, stagger=s),
            1,
        )
        for k, bc in sorted(fp.GHOST_FILLERS.items())
        for s in "TUVX"
    ],
    *[
        (
            f"ApplyMask-{s}",
            fp.ApplyMask(mask=MASK, stagger=s),
            lambda x, s=s, k=k: jnp.where(MASK.wet(at=s, kind=k), x, 0.0),
            1,
        )
        for s, k in fp.MASK_KINDS.items()
    ],
    (
        "LinComb",
        fp.LinComb(coeffs=(2.0, -0.5)),
        lambda x, y: 2.0 * x - 0.5 * y,
        2,
    ),
]
IDS = [c[0] for c in CASES]


@pytest.mark.parametrize(("name", "op", "ref", "n"), CASES, ids=IDS)
@pytest.mark.slow  # compiles the operator several times (~2 s)
def test_contract(name, op, ref, n):
    check_operator(op, *[_field(k) for k in range(n)])


@pytest.mark.parametrize(("name", "op", "ref", "n"), CASES, ids=IDS)
def test_equals_tier_a(name, op, ref, n):
    xs = [_field(10 + k) for k in range(n)]
    got, want = op(*xs), ref(*xs)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(a, b)


def test_tables_cover_every_tier_a_method():
    methods = {
        tuple(name.removeprefix("diff_").replace("_to_", "_").split("_"))
        for name in dir(bindings.Difference2D)
        if name.startswith("diff_")
    }
    assert methods == set(fp.DIFFERENCES)
    pairs = {
        tuple(name.split("_to_"))
        for name in dir(bindings.Interpolation2D)
        if "_to_" in name and not name.startswith("_")
    }
    assert pairs == set(fp.INTERPOLATIONS)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: fp.Difference(grid=GRID, axis="x", src="T", dst="V"), "x/T/U"),
        (lambda: fp.Interp(grid=GRID, src="T", dst="T"), "T/U"),
        (lambda: fp.FillGhosts(kind="reflect"), "periodic"),
        (lambda: fp.ApplyMask(mask=MASK, stagger="W"), "T"),
    ],
)
def test_invalid_choices_list_the_valid_set(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_lincomb_checks_its_arity():
    with pytest.raises(ValueError, match="2 coefficients"):
        fp.LinComb(coeffs=(1.0, 2.0))(_field(0))


def test_pick_selects_a_tuple_output():
    u, v = fp.GradPerp(grid=GRID)(_field(0))
    np.testing.assert_array_equal(fp.Pick(index=1)((u, v)), v)


def test_coriolis_parameters_are_learnable():
    op = fp.Coriolis(grid=GRID, f0=2.0, beta=0.3)
    u, v = _field(0), _field(1)

    def loss(o):
        cu, cv = o(u, v)
        return jnp.sum(cu**2 + cv**2)

    grads = eqx.filter_grad(loss)(op)
    eps = 1e-6
    fd = (
        loss(fp.Coriolis(grid=GRID, f0=2.0 + eps, beta=0.3))
        - loss(fp.Coriolis(grid=GRID, f0=2.0 - eps, beta=0.3))
    ) / (2 * eps)
    np.testing.assert_allclose(grads.f0, fd, rtol=1e-6)


# ---------------------------------------------------------------- composition
H, G, F0 = 1.0, 9.81, 2.0


def sw_tendency() -> Graph:
    """Linear rotating shallow water on a doubly periodic C-grid.

    ∂ₜh = −H ∇·(u, v),  ∂ₜu = +f v̄ − g δₓh,  ∂ₜv = −f ū − g δᵧh
    """
    h, u, v = Input("h"), Input("u"), Input("v")
    hp, up, vp = (fp.FillGhosts(kind="periodic")(x) for x in (h, u, v))
    dh = fp.LinComb(coeffs=(-H,))(fp.Divergence(grid=GRID)(up, vp))
    cor = fp.Coriolis(grid=GRID, f0=F0)(up, vp)
    dx_h = fp.Difference(grid=GRID, axis="x", src="T", dst="U")(hp)
    dy_h = fp.Difference(grid=GRID, axis="y", src="T", dst="V")(hp)
    du = fp.LinComb(coeffs=(1.0, -G))(fp.Pick(index=0)(cor), dx_h)
    dv = fp.LinComb(coeffs=(1.0, -G))(fp.Pick(index=1)(cor), dy_h)
    return Graph(inputs={"h": h, "u": u, "v": v}, outputs={"h": dh, "u": du, "v": dv})


def sw_tendency_tier_a(h, u, v):
    h, u, v = (periodic(x) for x in (h, u, v))
    cu, cv = bindings.Coriolis2D(grid=GRID)(u, v, _f_beta(F0, 0.0))
    return {
        "h": -H * D.divergence(u, v),
        "u": cu - G * D.diff_x_T_to_U(h),
        "v": cv - G * D.diff_y_T_to_V(h),
    }


def test_shallow_water_graph_equals_tier_a():
    state = {
        k: _field(i)[None].repeat(3, axis=0) * (1 + jnp.arange(3.0))[:, None, None]
        for i, k in enumerate("huv")
    }  # [3, Ny, Nx]: a batch of states
    graph = sw_tendency()
    want = sw_tendency_tier_a(**state)
    eager = graph(**state)
    jitted = eqx.filter_jit(lambda g, s: g(**s))(graph, state)
    for k in "huv":
        np.testing.assert_allclose(eager[k], want[k], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(jitted[k], want[k], rtol=1e-12, atol=1e-12)
