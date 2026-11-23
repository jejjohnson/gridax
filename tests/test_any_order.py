"""`Upwind` and `WENO` of any odd order, generated from Layer 0 (#497)."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import gridax as g
from gridax._src.advection import linear, weno
from gridax.schemes import TVD, WENO, Upwind
import gridax.stencils as gs
import jax
import jax.numpy as jnp
import numpy as np
import pytest

RNG = np.random.default_rng(497)


@pytest.mark.parametrize("bias", ["left", "right"])
@pytest.mark.parametrize("order", [3, 5, 7, 9])
def test_generated_weno_is_the_hand_written_one(order: int, bias: str) -> None:
    q = RNG.normal(size=(order, 5000))
    suffix = "" if bias == "left" else "_right"
    want = getattr(weno, f"weno_{order}pts{suffix}")(*q)
    np.testing.assert_allclose(weno.weno_kernel(order, bias)(*q), want, atol=1e-14)


@pytest.mark.parametrize("order", [1, 3, 5])
def test_generated_upwind_is_the_hand_written_one(order: int) -> None:
    q = RNG.normal(size=(order, 5000))
    kernels: dict[int, tuple[Callable[..., Any], Callable[..., Any]]] = {
        1: (lambda q0: q0, lambda q0: q0),
        3: (linear.linear_3pts_left, linear.linear_3pts_right),
        5: (linear.linear_5pts_left, linear.linear_5pts_right),
    }
    weights = [float(w) for w in Upwind(order).weights(derivative=0, staggered=True)]
    for kernel, cells in zip(kernels[order], (q, q), strict=True):
        own = weights if kernel is kernels[order][0] else weights[::-1]
        np.testing.assert_allclose(
            sum(w * x for w, x in zip(own, cells, strict=True)),
            kernel(*cells),
            atol=1e-14,
        )


@pytest.mark.parametrize(
    "scheme", [Upwind(7), Upwind(13), WENO(11), WENO(13)], ids=repr
)
def test_the_mirror(scheme: Any) -> None:
    tier = scheme.tiers()[0]
    q = RNG.normal(size=(scheme.order, 200))
    np.testing.assert_allclose(tier.left(*q), tier.right(*q[::-1]), rtol=1e-10)


@pytest.mark.parametrize("order", [11, 13])
def test_smooth_data_takes_the_linear_weights(order: int) -> None:
    # a straight line: every sub-stencil and the linear weights give the face
    tier = WENO(order).tiers()[0]
    cells = np.arange(order) - order // 2
    np.testing.assert_allclose(tier.left(*(2.0 * cells + 5.0)), 6.0, rtol=1e-12)
    # the optimal weights combine the sub-stencils to the upwind value
    full, subs = gs.weno_stencils(order, bias="left")
    weights = gs.optimal_weights(full, subs)
    q = RNG.normal(size=order)
    faces = [
        sum(
            c * x
            for c, x in zip(
                gs.difference_coeffs(sub, 0, data="cell"),
                q[k : k + len(sub)],
                strict=True,
            )
        )
        for k, sub in enumerate(subs)
    ]
    want = Upwind(order).tiers()[0].left(*q)
    np.testing.assert_allclose(np.dot(weights, faces), want, rtol=1e-10)


@pytest.mark.parametrize(
    "scheme", [WENO(11), WENO(13), WENO(7, "z"), WENO(9, "z")], ids=repr
)
def test_a_step_stays_monotone(scheme: Any) -> None:
    tier = scheme.tiers()[0]
    n = scheme.order
    for jump in range(1, n):
        q = np.where(np.arange(n) < jump, 0.0, 1.0)
        for kernel in (tier.left, tier.right):
            assert -1e-2 < float(kernel(*q)) < 1.0 + 1e-2


@pytest.mark.parametrize(
    "scheme", [WENO(11), WENO(13), WENO(7, "z"), WENO(9, "z")], ids=repr
)
def test_amplitude_and_mean(scheme: Any) -> None:
    tier = scheme.tiers()[0]
    q = RNG.normal(size=(scheme.order, 300))
    base = tier.left(*q)
    for amplitude in (1e-9, 1e3):
        np.testing.assert_allclose(
            tier.left(*(amplitude * q)), amplitude * base, rtol=1e-9
        )
    # a large mean costs the digits of the sum, and no more
    np.testing.assert_allclose(tier.left(*(q + 1e4)) - 1e4, base, atol=1e-7)
    flat = tier.left(*np.full(scheme.order, 3.0))
    np.testing.assert_allclose(flat, 3.0, rtol=1e-12)


def _errors(scheme: Any, sizes: tuple[int, ...]) -> np.ndarray:
    """Mean error of the tendency of a periodic sine's cell averages."""
    out = []
    for n in sizes:
        grid = g.CartesianCGrid1D.from_interior(n, 2.0 * np.pi)
        dx = 2.0 * np.pi / n
        faces = np.arange(-1, n + 2) * dx
        q = -np.diff(np.cos(faces)) / dx
        got = g.advect(
            q, np.ones(n + 2), None, grid, scheme=scheme, bc=g.BC.periodic("x")
        )
        exact = -np.diff(np.sin(faces)) / dx
        out.append(np.abs(got - exact)[1:-1].mean())
    return np.asarray(out)


def _rate(scheme: Any, sizes: tuple[int, ...]) -> float:
    return float(np.polyfit(-np.log(sizes), np.log(_errors(scheme, sizes)), 1)[0])


@pytest.mark.slow
@pytest.mark.parametrize(
    ("scheme", "sizes"),
    [
        (Upwind(5), (16, 24, 32, 48)),
        (Upwind(7), (16, 24, 32, 48)),
        (Upwind(9), (12, 16, 24, 32)),
        (WENO(7, "z"), (24, 32, 48, 64)),
        (WENO(9, "z"), (16, 24, 32, 48)),
    ],
    ids=repr,
)
def test_measured_order(scheme: Any, sizes: tuple[int, ...]) -> None:
    assert abs(_rate(scheme, sizes) - scheme.order) < 0.3


@pytest.mark.slow
def test_measured_order_of_weno_11() -> None:
    # Jiang-Shu weights lose order at the crests: 10.2 to 10.4, not 11
    assert 10.0 < _rate(WENO(11), (16, 20, 24, 32, 40)) < 10.6
    # and so does the hand-written WENO-7, which converges at 6
    assert 5.9 < _rate(WENO(7), (32, 48, 64, 96)) < 6.2


def _taylor(m: int, cell: int) -> Fraction:
    """The average of ``x^m`` over the cell ``[cell - 1, cell]``."""
    return (Fraction(cell) ** (m + 1) - Fraction(cell - 1) ** (m + 1)) / (m + 1)


@pytest.mark.parametrize(("r", "order"), [(2, 3), (3, 5), (4, 7), (5, 8)])
def test_tau_vanishes_to_a_higher_order_than_the_indicators(r: int, order: int) -> None:
    tables = weno._weno_tables(2 * r - 1, "left")
    cells = range(1 - r, r)
    matrices = [
        [[Fraction(float(x)).limit_denominator(2_000_000) for x in row] for row in b]
        for b in tables.betas
    ]

    def term(m: int, n: int) -> Fraction:
        total = Fraction(0)
        for k, t in enumerate(weno._TAU[r]):
            qm = [_taylor(m, c) for c in cells[k : k + r]]
            qn = [_taylor(n, c) for c in cells[k : k + r]]
            total += t * sum(
                qm[i] * matrices[k][i][j] * qn[j] for i in range(r) for j in range(r)
            )
        return total

    orders = [
        m + n for m in range(order + 1) for n in range(order + 1) if term(m, n) != 0
    ]
    # beta is of order 2
    assert min(orders) == order


@pytest.mark.parametrize("scheme", [Upwind(7), WENO(11), WENO(9, "z")], ids=repr)
def test_masked_and_periodic(scheme: Any) -> None:
    grid = g.CartesianCGrid2D.from_interior(24, 20, 1.0, 1.0)
    wet = np.pad(RNG.random((20, 24)) > 0.05, 1, mode="wrap")
    mask = g.Mask2D.from_grid(grid, wet)
    q, u, v = (
        np.pad(x[1:-1, 1:-1], 1, "wrap") for x in RNG.normal(size=(3, *grid.shape))
    )
    q = np.where(wet, q, np.nan)
    got = g.advect(q, u, v, grid, scheme=scheme, mask=mask, bc=g.BC.periodic())
    assert np.isfinite(got).all()
    assert abs(got.sum()) < 1e-10 * np.abs(got).sum()


@pytest.mark.slow
@pytest.mark.parametrize("scheme", [WENO(11), WENO(9, "z")], ids=repr)
def test_jit_grad_and_dtype(scheme: Any) -> None:
    grid = g.CartesianCGrid1D.from_interior(24, 1.0)
    q = jnp.asarray(RNG.normal(size=26))
    u = jnp.ones(26)

    def loss(q: Any) -> Any:
        return (g.advect(q, u, None, grid, scheme=scheme) ** 2).sum()

    gradient = jax.jit(jax.grad(loss))
    assert np.isfinite(gradient(q)).all()
    assert np.isfinite(gradient(jnp.full(26, 2.0))).all()
    small = jax.eval_shape(
        lambda q, u: g.advect(q, u, None, grid, scheme=scheme),
        q.astype(jnp.float32),
        u.astype(jnp.float32),
    )
    assert small.dtype == jnp.float32


def test_bad_orders() -> None:
    for order in (0, 1, 2, 4, 10):
        with pytest.raises(ValueError, match="WENO order must be odd"):
            WENO(order)
    with pytest.raises(ValueError, match="WENO-Z goes up to order 9"):
        WENO(11, "z")
    with pytest.raises(ValueError, match="variant must be"):
        WENO(5, "m")
    for order in (0, 4, 6):
        with pytest.raises(ValueError, match="Upwind order must be odd"):
            Upwind(order)
    assert Upwind(2).order == 2


def test_ladders_and_halos() -> None:
    assert [t.size for t in WENO(11).tiers()] == [12, 10, 8, 6, 4, 2]
    assert WENO(11).halo(staggered=True) == 6
    assert WENO(9, "z").halo(staggered=True) == 5
    assert [t.size for t in TVD().tiers()] == [4, 2]
