"""`Adaptive`: edge and coast policies of a scheme (#498)."""

from __future__ import annotations

from typing import Any

import gridax as g
from gridax._src.schemes.adaptive import policies
import gridax.analysis as ga
from gridax.schemes import TVD, WENO, Adaptive, Biased, Centred, Upwind
import gridax.stencils as gs
import jax
import numpy as np
import pytest

RNG = np.random.default_rng(498)
NX, NY = 16, 13
GRID = g.CartesianCGrid2D.from_interior(NX, NY, 8.0, 5.2)
SPHERE = g.SphericalCGrid2D.from_interior(NX, NY, (0.0, 40.0), (-20.0, 30.0))
WET = RNG.random(GRID.shape) > 0.12
MASK = g.Mask2D.from_grid(GRID, WET)
WIDE = [Centred(4), Centred(6), Biased(3, "left"), Biased(4, "right")]
FLUXES = [Upwind(5), WENO(5), WENO(9), TVD("mc"), Centred(4), Biased(3, "left")]
DIFFERENCES = [("x", "T", "U"), ("x", "U", "T"), ("y", "T", "V"), ("y", "X", "U")]
INTERPOLATIONS = [("T", "U"), ("V", "T"), ("T", "X"), ("U", "V")]
AROUND = g.BC(x=g.Periodic(), y=g.Periodic())


def field(grid: Any = GRID) -> np.ndarray:
    return RNG.normal(size=grid.shape)


# ----------------------------------------------------------------------
# The wrapper
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scheme", [*WIDE, *FLUXES], ids=repr)
def test_it_is_its_scheme(scheme: Any) -> None:
    adaptive = Adaptive(scheme)
    assert adaptive.order == scheme.order
    assert adaptive.linear is scheme.linear and adaptive.upwinded is scheme.upwinded
    assert [t.size for t in adaptive.tiers()] == [t.size for t in scheme.tiers()]
    for staggered in (True, False):
        if scheme.upwinded and not staggered:
            continue
        assert adaptive.halo(staggered=staggered) == scheme.halo(staggered=staggered)
    if not isinstance(scheme, TVD):
        kwargs = {"derivative": 0, "staggered": True}
        assert adaptive.weights(**kwargs) == scheme.weights(**kwargs)


def test_repr_and_equality() -> None:
    scheme = Adaptive(WENO(5), edges="periodic")
    assert repr(scheme) == "Adaptive(WENO(order=5, variant='js'), edges='periodic')"
    assert repr(Adaptive(Centred(4))) == "Adaptive(Centred(order=4))"
    assert scheme == Adaptive(WENO(5), edges="periodic")
    assert hash(scheme) == hash(Adaptive(WENO(5), edges="periodic"))


def test_bad_wrappers_raise() -> None:
    for bad in ("weno5", Adaptive(Centred(4))):
        with pytest.raises(TypeError, match="Adaptive wraps"):
            Adaptive(bad)
    with pytest.raises(ValueError, match="edges must be one of"):
        Adaptive(Centred(4), edges="wrap")
    with pytest.raises(ValueError, match="mask must be one of"):
        Adaptive(Centred(4), mask="extend")
    for scheme in (Upwind(3), WENO(5), TVD("mc")):
        for name in ("edges", "mask"):
            with pytest.raises(ValueError, match="read downwind cells"):
                Adaptive(scheme, **{name: "bias"})


def test_policies_and_bc_go_together() -> None:
    f = field()
    move = {"axis": "x", "src": "T", "dst": "U"}
    with pytest.raises(ValueError, match="give one"):
        g.difference(f, GRID, scheme=Adaptive(Centred(4), edges="extend"), **move)
    with pytest.raises(ValueError, match="give one"):
        scheme = Adaptive(Centred(4), mask="fill")
        g.difference(f, GRID, scheme=scheme, mask=MASK, **move)
    with pytest.raises(ValueError, match="leave one out"):
        scheme = Adaptive(Centred(4), edges="narrow")
        g.difference(f, GRID, scheme=scheme, bc=AROUND, **move)
    with pytest.raises(TypeError, match="WENO is nonlinear and upwinded"):
        g.difference(f, GRID, scheme=Adaptive(WENO(5)), **move)
    assert policies(Centred(4), AROUND, edges="bias", dry="narrow")[1] == "extend"


# ----------------------------------------------------------------------
# The defaults
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mask", [None, MASK], ids=["sea", "coast"])
@pytest.mark.parametrize("grid", [GRID, SPHERE], ids=["cartesian", "spherical"])
@pytest.mark.parametrize("scheme", [Centred(2), *WIDE], ids=repr)
def test_default_of_the_moves(scheme: Any, grid: Any, mask: Any) -> None:
    f = field()
    explicit = Adaptive(scheme, edges="bias", mask="narrow")
    for adaptive in (Adaptive(scheme), explicit):
        for axis, src, dst in DIFFERENCES:
            kwargs = {"axis": axis, "src": src, "dst": dst, "mask": mask}
            np.testing.assert_array_equal(
                g.difference(f, grid, scheme=adaptive, **kwargs),
                g.difference(f, grid, scheme=scheme, **kwargs),
            )
        for src, dst in INTERPOLATIONS:
            kwargs = {"src": src, "dst": dst, "mask": mask}
            np.testing.assert_array_equal(
                g.interpolate(f, grid, scheme=adaptive, **kwargs),
                g.interpolate(f, grid, scheme=scheme, **kwargs),
            )


@pytest.mark.parametrize("mask", [None, MASK], ids=["sea", "coast"])
@pytest.mark.parametrize("grid", [GRID, SPHERE], ids=["cartesian", "spherical"])
@pytest.mark.parametrize("scheme", FLUXES, ids=repr)
def test_default_of_the_fluxes(scheme: Any, grid: Any, mask: Any) -> None:
    q, u, v = field(), field(), field()
    explicit = Adaptive(scheme, edges="narrow", mask="narrow")
    for adaptive in (Adaptive(scheme), explicit):
        np.testing.assert_array_equal(
            g.advect(q, u, v, grid, scheme=adaptive, mask=mask),
            g.advect(q, u, v, grid, scheme=scheme, mask=mask),
        )
        np.testing.assert_array_equal(
            g.reconstruct(q, u, grid, axis="x", scheme=adaptive, mask=mask),
            g.reconstruct(q, u, grid, axis="x", scheme=scheme, mask=mask),
        )


AGRIDS = {
    "uniform": g.CartesianAGrid2D.from_interior(NX, NY, 4.0, 3.0),
    "irregular": g.CartesianAGrid2D.from_coords(
        y=np.sort(RNG.random(NY)), x=np.sort(RNG.random(NX))
    ),
    "sphere": g.SphericalAGrid2D.from_interior(
        NX, NY, (0.0, 360.0), (-60.0, 60.0), periodic=("x",)
    ),
}


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("name", list(AGRIDS))
@pytest.mark.parametrize(
    "scheme", [Centred(2), Centred(4), Centred(6), Biased(2, "left")], ids=repr
)
def test_default_of_the_collocated_difference(
    scheme: Any, name: str, masked: bool
) -> None:
    grid = AGRIDS[name]
    f = field(grid)
    mask = g.Mask2D.from_grid(grid, WET[1:-1, 1:-1]) if masked else None
    for axis in ("x", "y"):
        for dry in ("narrow", "bias", None):
            got = g.difference(
                f,
                grid,
                axis=axis,
                scheme=Adaptive(scheme, edges="bias", mask=dry),
                mask=mask,
            )
            want = g.difference(f, grid, axis=axis, scheme=scheme, mask=mask)
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_it_is_the_adaptive_difference(order: int) -> None:
    grid = AGRIDS["uniform"]
    f = field(grid)
    wet = WET[1:-1, 1:-1]
    for axis, position in (("x", -1), ("y", -2)):
        got = g.difference(
            f,
            grid,
            axis=axis,
            scheme=Adaptive(Centred(order), edges="bias", mask="narrow"),
            mask=g.Mask2D.from_grid(grid, wet),
        )
        want = ga.adaptive_difference(
            f,
            mask=~wet,
            axis=position,
            step=grid.spacing(axis),
            accuracy=order - 1,
        )
        np.testing.assert_array_equal(got, want)


# ----------------------------------------------------------------------
# The edges
# ----------------------------------------------------------------------


def line(n: int = 12) -> Any:
    return g.CartesianCGrid1D.from_interior(n, float(n))


def moved(f: Any, grid: Any, scheme: Any, **kwargs: Any) -> Any:
    return g.difference(f, grid, axis="x", src="T", dst="U", scheme=scheme, **kwargs)


def test_narrow_edges_take_the_next_rung() -> None:
    grid = line()
    f = RNG.normal(size=grid.shape)
    got = moved(f, grid, Adaptive(Centred(6), edges="narrow"))
    two, four, six = (moved(f, grid, Centred(k)) for k in (2, 4, 6))
    # face i + 1/2 reads cells i - 2 .. i + 3 of 0 .. 13
    np.testing.assert_array_equal(got[2:11], six[2:11])
    np.testing.assert_array_equal(got[[1, 11]], four[[1, 11]])
    np.testing.assert_array_equal(got[12], two[12])


@pytest.mark.parametrize(
    ("scheme", "zero"),
    [(Centred(4), [0, 12, 13]), (Centred(6), [0, 1, 11, 12, 13])],
    ids=repr,
)
def test_none_edges_are_zero(scheme: Any, zero: list[int]) -> None:
    grid = line()
    f = RNG.normal(size=grid.shape)
    full = moved(f, grid, scheme)
    mask = g.Mask1D.from_grid(grid, np.ones(grid.shape, dtype=bool))
    for kwargs in ({}, {"mask": mask}):
        got = moved(f, grid, Adaptive(scheme, edges="none"), **kwargs)
        assert np.flatnonzero(got == 0).tolist() == zero
        kept = [i for i in range(14) if i not in zero]
        np.testing.assert_array_equal(got[kept], full[kept])


@pytest.mark.parametrize("scheme", [Centred(2), *WIDE], ids=repr)
def test_periodic_edges_are_a_periodic_bc(scheme: Any) -> None:
    f = field()
    for axis, src, dst in DIFFERENCES:
        kwargs = {"axis": axis, "src": src, "dst": dst}
        np.testing.assert_array_equal(
            g.difference(f, GRID, scheme=Adaptive(scheme, edges="periodic"), **kwargs),
            g.difference(f, GRID, scheme=scheme, bc=AROUND, **kwargs),
        )
    for src, dst in INTERPOLATIONS:
        kwargs = {"src": src, "dst": dst}
        np.testing.assert_array_equal(
            g.interpolate(f, GRID, scheme=Adaptive(scheme, edges="periodic"), **kwargs),
            g.interpolate(f, GRID, scheme=scheme, bc=AROUND, **kwargs),
        )


@pytest.mark.parametrize("scheme", FLUXES, ids=repr)
def test_periodic_fluxes_conserve(scheme: Any) -> None:
    q, u, v = (np.pad(field()[1:-1, 1:-1], 1, mode="wrap") for _ in range(3))
    periodic = Adaptive(scheme, edges="periodic")
    got = g.advect(q, u, v, GRID, scheme=periodic)
    np.testing.assert_array_equal(
        got, g.advect(q, u, v, GRID, scheme=scheme, bc=AROUND)
    )
    assert abs(got.sum()) < 1e-11 * np.abs(got).sum()
    with pytest.raises(ValueError, match="drop wall="):
        g.advect(q, u, v, GRID, scheme=periodic, wall="open")


@pytest.mark.parametrize("order", [2, 4])
def test_extended_edges_keep_the_order(order: int) -> None:
    errors = []
    for n in (16, 32, 64):
        grid = g.CartesianCGrid1D.from_interior(n, 2.0 * np.pi)
        f = np.sin(grid.coords("x"))
        scheme = Adaptive(Centred(order), edges="extend")
        got = moved(f, grid, scheme, bc=g.BC(x=g.Periodic()))
        errors.append(np.abs(got - np.cos(grid.coords("x", at="U")))[1:-1].max())
    assert np.polyfit(range(3), -np.log2(errors), 1)[0] > order - 0.3


@pytest.mark.parametrize("scheme", FLUXES, ids=repr)
def test_flux_edges(scheme: Any) -> None:
    q, u = field(), field()
    kwargs = {"axis": "x"}
    full = g.reconstruct(q, u, GRID, scheme=scheme, **kwargs)
    none = g.reconstruct(q, u, GRID, scheme=Adaptive(scheme, edges="none"), **kwargs)
    lo, hi = scheme.footprint(staggered=True)
    for i in range(1, NX + 1):
        fits = i + lo >= 0 and i + hi <= NX + 1
        want = full[1:-1, i] if fits else 0.0
        np.testing.assert_array_equal(none[1:-1, i], want)


@pytest.mark.parametrize("scheme", [Centred(4), Centred(6), Biased(3, "left")])
def test_biased_flux_edges_keep_the_order(scheme: Any) -> None:
    # exact for cell averages of a polynomial of degree order - 1, at every face
    grid = line()
    faces = np.arange(-0.5, 14.0)
    degree = scheme.order
    q = np.diff(faces**degree) / degree  # averages of x^(degree - 1)
    u = np.ones(14)
    got = g.reconstruct(q, u, grid, axis="x", scheme=Adaptive(scheme, edges="bias"))
    np.testing.assert_allclose(got[1:-1], faces[2:-1] ** (degree - 1), rtol=1e-9)


# ----------------------------------------------------------------------
# The coast
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scheme", WIDE, ids=repr)
def test_biased_coasts_are_exact_for_polynomials(scheme: Any) -> None:
    # a strait wide enough for the shifted stencil: exact up to the coast
    n = 20
    grid = g.CartesianCGrid1D.from_interior(n, float(n))
    wet = np.zeros(n + 2, dtype=bool)
    wet[4:17] = True
    mask = g.Mask1D.from_grid(grid, wet)
    x = np.arange(n + 2.0)
    f = np.where(wet, x**scheme.order, np.nan)
    got = moved(f, grid, Adaptive(scheme, mask="bias"), mask=mask)
    want = scheme.order * (x + 0.5) ** (scheme.order - 1)
    np.testing.assert_allclose(got[4:16], want[4:16], rtol=1e-9)
    # a narrow stencil is not
    if scheme.order > 2:
        narrow = moved(f, grid, scheme, mask=mask)
        assert np.abs(narrow[4:16] - want[4:16]).max() > 1e-3


@pytest.mark.parametrize("scheme", WIDE, ids=repr)
def test_biased_coasts_read_wet_points_only(scheme: Any) -> None:
    f = np.where(WET, field(), np.nan)
    for axis, dst in (("x", "U"), ("y", "V")):
        got = g.difference(
            f,
            GRID,
            axis=axis,
            src="T",
            dst=dst,
            scheme=Adaptive(scheme, mask="bias"),
            mask=MASK,
        )
        on = MASK.wet(at=dst)
        assert np.isfinite(got[on]).all()
    got = g.interpolate(
        f, GRID, src="T", dst="X", scheme=Adaptive(scheme, mask="bias"), mask=MASK
    )
    assert np.isfinite(got[MASK.wet(at="X", kind="strict")]).all()


@pytest.mark.parametrize("scheme", [Centred(4), Centred(6)], ids=repr)
def test_no_coast_no_difference(scheme: Any) -> None:
    f = field()
    sea = g.Mask2D.from_grid(GRID, np.ones(GRID.shape, dtype=bool))
    for dry in ("narrow", "bias", "none"):
        got = g.difference(
            f,
            GRID,
            axis="x",
            src="T",
            dst="U",
            scheme=Adaptive(scheme, mask=dry),
            mask=sea,
        )
        want = moved(f, GRID, scheme)
        np.testing.assert_allclose(got[1:-1, 4:-4], want[1:-1, 4:-4], rtol=1e-12)


@pytest.mark.parametrize("scheme", WIDE, ids=repr)
def test_none_masks_the_output_only(scheme: Any) -> None:
    f = field()
    kwargs = {"axis": "x", "src": "T", "dst": "U"}
    got = g.difference(
        f, GRID, scheme=Adaptive(scheme, mask="none"), mask=MASK, **kwargs
    )
    want = g.difference(f, GRID, scheme=scheme, **kwargs) * MASK.wet(at="U")
    np.testing.assert_array_equal(got, want)


LAND = 4


def channel() -> tuple[Any, Any, Any]:
    edge = g.CartesianCGrid2D.from_interior(NX, NY, 8.0, 5.2)
    tall = NY + 2 * LAND
    land = g.CartesianCGrid2D.from_interior(NX, tall, 8.0, 0.4 * tall)
    wet = np.zeros(land.shape, dtype=bool)
    wet[1 + LAND : 1 + LAND + NY] = True
    return edge, land, g.Mask2D.from_grid(land, wet)


@pytest.mark.parametrize("scheme", [Centred(2), Centred(4), Centred(6)], ids=repr)
@pytest.mark.parametrize(
    "atom", [g.Dirichlet(0.4), g.Neumann(0.2), g.Slip(0.3)], ids=repr
)
def test_a_filled_coast_is_an_extended_edge(scheme: Any, atom: Any) -> None:
    edge, land, mask = channel()
    bc = g.BC(x=g.Periodic(), y=atom)
    f = field(edge)
    inland = np.full(land.shape, np.nan)
    inland[1 + LAND : 1 + LAND + NY] = f[1:-1]
    for src, dst in (("T", "V"), ("U", "X"), ("V", "T")):
        kwargs = {"axis": "y", "src": src, "dst": dst}
        want = g.difference(f, edge, scheme=scheme, bc=bc, **kwargs)
        got = g.difference(
            inland,
            land,
            scheme=Adaptive(scheme, edges="bias", mask="fill"),
            mask=mask,
            bc=bc,
            **kwargs,
        )
        # the node on the north wall is dry, and masked
        rows = NY if dst == "T" else NY - 1
        np.testing.assert_allclose(
            got[1 + LAND : 1 + LAND + rows, 1:-1],
            want[1 : 1 + rows, 1:-1],
            rtol=1e-11,
            atol=1e-11,
        )


@pytest.mark.parametrize("scheme", FLUXES, ids=repr)
def test_a_filled_coast_is_an_extended_edge_for_a_tendency(scheme: Any) -> None:
    edge, land, mask = channel()
    bc = g.BC(x=g.Periodic(), y=g.Neumann())
    q, u, v = (np.pad(field(edge)[:, 1:-1], ((0, 0), (1, 1)), "wrap") for _ in "quv")
    v[0] = v[-2] = 0.0
    want = g.advect(q, u, v, edge, scheme=scheme, bc=bc)

    def inland(f: np.ndarray, fill: float) -> np.ndarray:
        out = np.full(land.shape, fill)
        out[1 + LAND : 1 + LAND + NY] = f[1:-1]
        return out

    got = g.advect(
        inland(q, np.nan),
        inland(u, 0.0),
        inland(v, 0.0),
        land,
        scheme=Adaptive(scheme, mask="fill"),
        mask=mask,
        bc=bc,
    )
    np.testing.assert_allclose(
        got[1 + LAND : 1 + LAND + NY, 1:-1], want[1:-1, 1:-1], rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize(
    ("scheme", "dry"),
    [
        (scheme, dry)
        for scheme in FLUXES
        for dry in ("narrow", "bias", "fill", "none")
        if dry != "bias" or not scheme.upwinded
    ],
)
def test_every_coast_conserves(scheme: Any, dry: str) -> None:
    wet = np.pad(RNG.random((NY, NX)) > 0.12, 1, mode="wrap")
    mask = g.Mask2D.from_grid(GRID, wet)
    q, u, v = (np.pad(field()[1:-1, 1:-1], 1, mode="wrap") for _ in range(3))
    bc = g.BC(
        x=g.Periodic(),
        y=g.Periodic(),
        coast=g.BC(x=g.Neumann(), y=g.Neumann()),
    )
    got = g.advect(q, u, v, GRID, scheme=Adaptive(scheme, mask=dry), mask=mask, bc=bc)
    assert np.isfinite(got).all()
    assert abs(got[1:-1, 1:-1].sum()) < 1e-11 * np.abs(got).sum()
    assert (got[~wet] == 0).all()


# ----------------------------------------------------------------------
# Collocated edges, and whether a grid is too small
# ----------------------------------------------------------------------


def test_collocated_edges() -> None:
    grid = g.CartesianAGrid1D.from_interior(10, 10.0)
    f = RNG.normal(size=10)

    def diff(scheme: Any, **kwargs: Any) -> Any:
        return g.difference(f, grid, axis="x", scheme=scheme, **kwargs)

    none = diff(Adaptive(Centred(4), edges="none"))
    assert np.flatnonzero(np.isnan(none)).tolist() == [0, 1, 8, 9]
    np.testing.assert_array_equal(none[2:8], diff(Centred(4))[2:8])
    narrow = diff(Adaptive(Centred(4), edges="narrow"))
    np.testing.assert_array_equal(narrow[2:8], diff(Centred(4))[2:8])
    np.testing.assert_array_equal(narrow[[1, 8]], diff(Centred(2))[[1, 8]])
    np.testing.assert_array_equal(narrow[[0, 9]], np.diff(f)[[0, -1]])
    periodic = diff(Adaptive(Centred(2), edges="periodic"))
    np.testing.assert_allclose(periodic, (np.roll(f, -1) - np.roll(f, 1)) / 2)
    one_sided = diff(Adaptive(Biased(2, "right"), edges="narrow"))
    np.testing.assert_array_equal(one_sided[:8], diff(Biased(2, "right"))[:8])
    np.testing.assert_array_equal(one_sided[8:], np.diff(f)[[8, 8]])
    with pytest.raises(ValueError, match="give one"):
        diff(Adaptive(Centred(2), edges="extend"))


def test_collocated_masks() -> None:
    grid = g.CartesianAGrid1D.from_interior(10, 10.0)
    f = RNG.normal(size=10)
    wet = np.ones(10, dtype=bool)
    wet[4] = False
    mask = g.Mask1D.from_grid(grid, wet)
    got = g.difference(
        f, grid, axis="x", scheme=Adaptive(Centred(2), mask="none"), mask=mask
    )
    want = g.difference(f, grid, axis="x")
    np.testing.assert_array_equal(got[wet], want[wet])
    assert np.isnan(got[4])
    none = g.difference(
        f, grid, axis="x", scheme=Adaptive(Centred(2), edges="none"), mask=mask
    )
    assert np.isnan(none[[0, 4, 9]]).all() and np.isfinite(none[1:4]).all()


def test_an_irregular_axis_does_not_wrap() -> None:
    scheme = Adaptive(Centred(2), edges="periodic")
    with pytest.raises(ValueError, match="uniform x axis"):
        g.difference(
            field(AGRIDS["irregular"]), AGRIDS["irregular"], axis="x", scheme=scheme
        )


@pytest.mark.parametrize("edges", ["bias", "narrow", "none"])
def test_layer_one_takes_the_edge_policies(edges: str) -> None:
    grid = line()
    f = RNG.normal(size=grid.shape)
    scheme = Adaptive(Centred(4), edges=edges)
    got = gs.staggered_difference(f, axis="x", side="fwd", ndim=1, scheme=scheme)
    np.testing.assert_array_equal(got, moved(f, grid, scheme)[1:-1])
    with pytest.raises(ValueError, match="needs a grid"):
        gs.staggered_difference(
            f,
            axis="x",
            side="fwd",
            ndim=1,
            scheme=Adaptive(Centred(4), edges="periodic"),
        )


def test_degrades_on() -> None:
    sea = g.Mask2D.from_grid(GRID, np.ones(GRID.shape, dtype=bool))
    assert not Centred(2).degrades_on(GRID, sea)
    assert not Centred(2).degrades_on(GRID, MASK)
    assert Centred(4).degrades_on(GRID, sea)  # the edges
    for edges, degrades in {
        "narrow": True,
        "none": True,
        "bias": False,
        "periodic": False,
        "extend": False,
    }.items():
        scheme = Adaptive(Centred(4), edges=edges)
        assert scheme.degrades_on(GRID) is degrades
        assert scheme.degrades_on(GRID, sea) is degrades
        assert scheme.degrades_on(GRID, MASK)
    for dry in ("bias", "fill", "none"):
        assert not Adaptive(Centred(4), edges="bias", mask=dry).degrades_on(GRID, MASK)
    assert Adaptive(WENO(5), edges="periodic").degrades_on(GRID, MASK)
    agrid = AGRIDS["uniform"]
    amask = g.Mask2D.from_grid(agrid, WET[1:-1, 1:-1])
    assert Adaptive(Centred(2), edges="periodic").degrades_on(agrid, amask)


def test_jit_vmap_grad() -> None:
    f = jax.numpy.asarray(field())
    scheme = Adaptive(Centred(4), edges="narrow", mask="bias")

    @jax.jit
    def out(f: Any, mask: Any) -> Any:
        return g.difference(
            f, GRID, axis="x", src="T", dst="U", scheme=scheme, mask=mask
        )

    eager = g.difference(
        np.asarray(f), GRID, axis="x", src="T", dst="U", scheme=scheme, mask=MASK
    )
    np.testing.assert_allclose(out(f, MASK), eager, rtol=1e-12, atol=1e-14)
    stacked = jax.vmap(out, (0, None))(jax.numpy.stack([f, 2 * f]), MASK)
    np.testing.assert_allclose(stacked[1], 2 * eager, rtol=1e-12, atol=1e-14)
    grads = jax.grad(lambda f: (out(f, MASK) ** 2).sum())(f)
    assert np.isfinite(grads).all()
