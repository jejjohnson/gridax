"""Each tier is called once, and a mirrored one at the ends once for both sides (#550)."""

from __future__ import annotations

import gridax
from gridax._src.advection.reconstruct import _faces
from gridax.schemes import TVD, WENO, Upwind
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

SCHEMES = [WENO(o) for o in (3, 5, 7, 9, 11)] + [WENO(o, "z") for o in (3, 5, 7, 9)]


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
def test_mirrored_tiers_are_mirrors(scheme) -> None:
    rng = np.random.default_rng(0)
    for tier in scheme.tiers():
        cells = [
            rng.normal(size=200) * 10.0 ** rng.integers(-3, 3)
            for _ in range(tier.hi - tier.lo + 1)
        ]
        right = np.asarray(tier.right(*cells))
        left = np.asarray(tier.left(*cells[::-1]))
        np.testing.assert_allclose(
            left, right, rtol=0, atol=1e-14 * np.abs(right).max()
        )


@pytest.mark.parametrize("scheme", [*SCHEMES, Upwind(3), TVD()], ids=repr)
@pytest.mark.parametrize("size", [12, 26, 40])
def test_one_call_per_tier_is_a_call_per_face(scheme, size) -> None:
    rng = np.random.default_rng(1)
    q = rng.normal(size=(3, size))
    tiers = scheme.tiers()
    if size < max(t.hi - t.lo + 1 for t in tiers[:1]) + 2:
        pytest.skip("the widest tier does not fit")
    got = _faces(np, q, tiers, ("left", "right"))
    for side in ("left", "right"):
        for i in range(1, size - 1):
            # the first tier whose cells fit, called on this face alone
            for tier in tiers:
                lo, hi = (
                    (tier.lo, tier.hi) if side == "left" else (1 - tier.hi, 1 - tier.lo)
                )
                if i + lo >= 0 and i + hi <= size - 1:
                    fn = tier.left if side == "left" else tier.right
                    want = fn(*[q[:, i + o] for o in range(lo, hi + 1)])
                    break
            np.testing.assert_allclose(
                got[side][:, i - 1], want, rtol=1e-14, atol=1e-14, err_msg=f"{side} {i}"
            )


def test_the_gradient_of_weno9_is_one_kernel_per_tier() -> None:
    grid = gridax.CartesianCGrid1D.from_interior(24, 1.0)
    rng = np.random.default_rng(0)
    q, u = rng.random(26), rng.normal(size=26)

    def loss(q):
        return (gridax.advect(q, u, None, grid, scheme=WENO(9)) ** 2).sum()

    text = jax.jit(jax.grad(loss)).lower(q).as_text()
    # 8,732 lines with a kernel per run of faces of a tier and side; 4,857 now
    assert text.count("\n") < 6_000
