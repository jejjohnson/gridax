"""The schemes of ``gridax.schemes``: footprints, halos, exact weights (#495)."""

from __future__ import annotations

from fractions import Fraction
import itertools

import _reconstruction_cases as cases
import gridax
from gridax.schemes import TVD, WENO, Biased, Centred, Scheme, Upwind
import gridax.stencils as gs
import numpy as np
import pytest

F = Fraction

LINEAR = [
    *(
        Centred(order, data)
        for order in (2, 4, 6, 8)
        for data in (None, "point", "cell")
    ),
    *(Biased(order, side) for order in (1, 2, 3, 4, 5) for side in ("left", "right")),
    *(Upwind(order) for order in (1, 2, 3, 5, 7, 9)),
]
SCHEMES: list[Scheme] = [
    *LINEAR,
    *(WENO(order) for order in (3, 5, 7, 9)),
    *(WENO(order, "z") for order in (3, 5)),
    *(TVD(limiter) for limiter in ("minmod", "van_leer", "superbee", "mc")),
]


def _kernel_weights(kernel, n: int) -> tuple[Fraction, ...]:
    """The weights of a linear kernel, as rationals with small denominators."""
    return tuple(F(float(kernel(*row))).limit_denominator(10_000) for row in np.eye(n))


def test_weights_are_the_ones_of_the_stencils_they_replace() -> None:
    from gridax._src.advection import linear

    face = {"derivative": 0, "staggered": True}
    assert Centred(2).weights(**face) == (F(1, 2), F(1, 2))
    assert Centred(4, "cell").weights(**face) == _kernel_weights(linear.linear_4pts, 4)
    assert Centred(6, "cell").weights(**face) == _kernel_weights(linear.linear_6pts, 6)
    assert Centred(4, "point").weights(**face) == (
        F(-1, 16),
        F(9, 16),
        F(9, 16),
        F(-1, 16),
    )
    assert Upwind(3).weights(**face) == _kernel_weights(linear.linear_3pts_left, 3)
    assert Upwind(5).weights(**face) == _kernel_weights(linear.linear_5pts_left, 5)
    assert Upwind(2).weights(**face) == (F(-1, 2), F(3, 2))
    # the linear part of weno_5pts, as hand-typed in Shu (1998)
    assert WENO(5).weights(**face) == tuple(F(n, 60) for n in (2, -13, 47, 27, -3))
    # the fourth-order staggered difference
    slope = Centred(4, "point").weights(derivative=1, staggered=True)
    assert slope == tuple(F(n, 24) for n in (1, -27, 27, -1))


@pytest.mark.parametrize("scheme", LINEAR, ids=repr)
def test_every_linear_tier_is_its_weights(scheme: Scheme) -> None:
    # face values of cell averages, the widest tier, for either sign of the velocity
    tier = scheme.tiers()[0]
    n = tier.hi - tier.lo + 1
    data = getattr(scheme, "data", None) or "cell"
    offsets = [k - 0.5 for k in range(tier.lo, tier.hi + 1)]
    want = gs.difference_coeffs(tuple(offsets), 0, data=data)
    np.testing.assert_allclose([tier.left(*row) for row in np.eye(n)], want, rtol=1e-15)
    mirror = want[::-1] if scheme.upwinded else want
    np.testing.assert_allclose(
        [tier.right(*row) for row in np.eye(n)], mirror, rtol=1e-15
    )


@pytest.mark.parametrize("derivative", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("staggered", [True, False])
def test_centred_order_is_the_formal_order(
    derivative: int, order: int, staggered: bool
) -> None:
    # the first moment the stencil misses is the one of degree derivative + order
    scheme = Centred(order, "point")
    offsets = scheme.offsets(derivative=derivative, staggered=staggered)
    weights = scheme.weights(derivative=derivative, staggered=staggered)
    moments = [
        sum(c * F(s) ** m for c, s in zip(weights, offsets, strict=True))
        for m in range(derivative + order + 1)
    ]
    assert moments[derivative] == {1: 1, 2: 2, 3: 6}[derivative]
    assert [m for m, value in enumerate(moments) if value] == [
        derivative,
        derivative + order,
    ]


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_biased_is_the_forward_and_the_backward_difference(order: int) -> None:
    for derivative, (side, method) in itertools.product(
        (1, 2), (("right", "forward"), ("left", "backward"))
    ):
        offsets = gs.difference_offsets(
            derivative=derivative, accuracy=order, method=method
        )
        scheme = Biased(order, side)
        assert scheme.offsets(derivative=derivative, staggered=False) == offsets
        got = scheme.weights(derivative=derivative, staggered=False)
        assert tuple(map(float, got)) == gs.difference_coeffs(offsets, derivative)


@pytest.mark.parametrize("accuracy", [1, 2, 3, 4, 5, 6, 7])
def test_the_old_accuracies_are_orders(accuracy: int) -> None:
    # the table of #496: what accuracy= meant, as the formal order
    def same(scheme: Scheme, derivative: int, method) -> None:
        old = gs.difference_offsets(
            derivative=derivative, accuracy=accuracy, method=method
        )
        staggered = method == "staggered"
        assert scheme.offsets(derivative=derivative, staggered=staggered) == old
        weights = scheme.weights(derivative=derivative, staggered=staggered)
        assert tuple(map(float, weights)) == gs.difference_coeffs(old, derivative)

    same(Centred(2 * ((accuracy + 1) // 2)), 1, "central")
    same(Biased(accuracy, "right"), 1, "forward")
    same(Biased(accuracy, "left"), 1, "backward")
    same(Centred(2 * ((accuracy + 2) // 2), "point"), 1, "staggered")
    same(Centred(2 * ((accuracy + 1) // 2), "point"), 0, "staggered")


@pytest.mark.parametrize("order", [1, 2, 3, 5, 7])
def test_biased_is_upwind_with_a_fixed_side(order: int) -> None:
    rng = np.random.default_rng(order)
    q, u = rng.standard_normal((2, 8, 24))
    for side, speed in (("left", np.abs(u)), ("right", -np.abs(u))):
        got = cases.flux(q, speed, axis=-1, scheme=Biased(order, side))
        want = cases.flux(q, speed, axis=-1, scheme=Upwind(order))
        # at the edges Biased narrows one point at a time, Upwind two
        inner = (slice(None), slice(order, -order))
        np.testing.assert_allclose(got[inner], want[inner], rtol=1e-12)
        # the side does not follow the velocity
        against = cases.flux(q, -speed, axis=-1, scheme=Biased(order, side))
        np.testing.assert_allclose(against, -got, rtol=1e-12)


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
def test_sizes_halos_and_footprints_agree(scheme: Scheme) -> None:
    tiers = scheme.tiers()
    for tier in tiers:
        assert tier.size == 2 * tier.halo + 2
    sizes = [tier.size for tier in tiers]
    assert sizes == sorted(sizes, reverse=True)
    lo, hi = scheme.footprint(staggered=True)
    assert scheme.halo(staggered=True) == max(-lo, hi)
    if scheme.upwinded:
        # the cells of both flow directions, seen from cell i of the face i + 1/2
        assert (lo, hi) == (-tiers[0].halo, tiers[0].halo + 1)
    else:
        assert (lo, hi) == (tiers[0].lo, tiers[0].hi)


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
def test_the_footprint_is_what_reconstruct_reads(scheme: Scheme) -> None:
    n, face = 40, 20
    lo, hi = scheme.footprint(staggered=True)
    read = set()
    for sign in (1.0, -1.0):
        rng = np.random.default_rng(3)
        q = rng.standard_normal(n)
        u = np.full(n, sign)
        base = cases.flux(q, u, axis=-1, scheme=scheme)[face]
        for k in range(n):
            # two bumps, so that a limiter that happens to be flat still shows
            for bump in (0.37, -0.81):
                moved = q.copy()
                moved[k] += bump
                if cases.flux(moved, u, axis=-1, scheme=scheme)[face] != base:
                    read.add(k - face)
    assert read <= set(range(lo, hi + 1))
    if scheme.linear and scheme != Upwind(2):
        assert read == set(range(lo, hi + 1))


def test_the_table_of_the_design() -> None:
    assert Centred(2).footprint(staggered=True) == (0, 1)
    assert Centred(4).footprint(staggered=True) == (-1, 2)
    assert WENO(5).footprint(staggered=True) == (-2, 3)
    assert [s.halo(staggered=True) for s in (Centred(2), Centred(4), WENO(5))] == [
        1,
        2,
        3,
    ]
    assert Centred(4).footprint(staggered=False) == (-2, 2)
    assert Biased(2, "right").footprint(staggered=False) == (0, 2)


def test_degrades_on() -> None:
    cgrid = gridax.CartesianCGrid2D.from_interior(8, 6, 1.0, 1.0)
    wall = gridax.CartesianAGrid2D.from_interior(8, 6, 1.0, 1.0)
    half = gridax.CartesianAGrid2D.from_interior(8, 6, 1.0, 1.0, periodic=("x",))
    torus = gridax.CartesianAGrid2D.from_interior(8, 6, 1.0, 1.0, periodic=("x", "y"))
    for scheme, on in [
        (Centred(2), False),
        (Centred(4), True),
        (Upwind(1), False),
        (WENO(5), True),
    ]:
        assert scheme.degrades_on(cgrid) is on
    assert Centred(2).degrades_on(wall)
    assert Centred(2).degrades_on(half)
    assert not Centred(4).degrades_on(torus)


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
def test_repr_builds_the_scheme(scheme: Scheme) -> None:
    call, stencil = repr(scheme).split(" # ")
    names = {
        "TVD": TVD,
        "WENO": WENO,
        "Biased": Biased,
        "Centred": Centred,
        "Upwind": Upwind,
    }
    assert eval(call, names) == scheme
    assert eval(repr(scheme), names) == scheme
    assert stencil
    assert hash(scheme) == hash(eval(call, names))


def test_repr_names_the_stencil() -> None:
    assert repr(Centred(4, data="point")) == (
        "Centred(order=4, data='point') # (-1, 9, 9, -1)/16"
    )
    assert repr(Upwind(1)) == "Upwind(order=1) # (1)"


def test_flags() -> None:
    flags = {type(s).__name__: (s.linear, s.upwinded) for s in SCHEMES}
    assert flags == {
        "Centred": (True, False),
        "Biased": (True, False),
        "Upwind": (True, True),
        "WENO": (False, True),
        "TVD": (False, True),
    }


def test_errors() -> None:
    for build in (
        lambda: Centred(3),
        lambda: Centred(0),
        lambda: Centred(2, "face"),  # type: ignore[arg-type]
        lambda: Biased(2),
        lambda: Biased(2, "up"),  # type: ignore[arg-type]
        lambda: Biased(0, "left"),
        lambda: Upwind(4),
        lambda: Upwind(0),
        lambda: Centred(2).weights(derivative=0, staggered=False),
    ):
        with pytest.raises(ValueError):
            build()
    with pytest.raises(TypeError, match="face values only"):
        WENO(5).weights(derivative=1, staggered=True)
    with pytest.raises(TypeError, match="face values only"):
        Upwind(3).footprint(staggered=False)
    with pytest.raises(TypeError, match="no linear weights"):
        TVD().weights(derivative=0, staggered=True)


@pytest.mark.parametrize(
    "scheme", [Centred(4), Centred(6, "point"), Biased(3, "left"), Upwind(7)], ids=repr
)
def test_masked_faces_never_read_a_dry_cell(scheme: Scheme) -> None:
    wet = np.ones((14, 22), dtype=bool)
    wet[0, :] = wet[-1, :] = wet[:, 0] = wet[:, -1] = False
    wet[5:8, 8:11] = False
    mask = gridax.Mask2D.from_mask(wet)
    rng = np.random.default_rng(0)
    q, u = rng.standard_normal((2, 14, 22))
    u = u * mask.wet(at="U")
    q = np.where(wet, q, np.nan)
    flux = cases.flux(q, u, axis=-1, scheme=scheme, mask=mask)
    # a dry cell is NaN, and would show on any face that reads it
    assert np.isfinite(flux).all()
    # every interior face between two wet cells has a flux: none is closed
    faces = mask.wet(at="U")[1:-1, 1:-2].astype(bool)
    assert (flux[1:-1, 1:-2][faces] != 0.0).all()


def test_masked_centred_is_the_average_of_the_two_wet_cells() -> None:
    wet = np.ones((8, 14), dtype=bool)
    wet[0, :] = wet[-1, :] = wet[:, 0] = wet[:, -1] = False
    wet[4, 6] = False
    mask = gridax.Mask2D.from_mask(wet)
    rng = np.random.default_rng(1)
    q, u = rng.standard_normal((2, 8, 14))
    u = u * mask.wet(at="U")
    got = cases.flux(q, u, axis=-1, scheme=Centred(), mask=mask)
    want = cases.flux(q, u, axis=-1, scheme=Centred())
    np.testing.assert_array_equal(got, np.where(wet, want, 0.0) + 0.0)
