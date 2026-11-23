"""``bc=`` on the operators that take a scheme: ghost layers as wide as it reads (#512)."""

from __future__ import annotations

from typing import Any

import gridax as g
from gridax.schemes import TVD, WENO, Biased, Centred, Upwind
import jax
import numpy as np
import pytest

RNG = np.random.default_rng(512)
NX, NY = 14, 12
SPAN = (0.0, 360.0)
GRIDS = {
    "cartesian": g.CartesianCGrid2D.from_interior(NX, NY, 7.0, 3.0),
    "spherical": g.SphericalCGrid2D.from_interior(NX, NY, SPAN, (-50.0, 40.0)),
}
LINEAR = [Centred(2), Centred(4), Centred(6), Biased(3, "left"), Biased(4, "right")]
FLUXES = [Upwind(1), Upwind(5), WENO(5), WENO(9), TVD("mc"), Centred(2), Centred(4)]
DIFFERENCES = [
    ("x", "T", "U"),
    ("x", "U", "T"),
    ("x", "V", "X"),
    ("y", "T", "V"),
    ("y", "V", "T"),
    ("y", "X", "U"),
]
INTERPOLATIONS = [("T", "U"), ("V", "T"), ("T", "X"), ("U", "V"), ("X", "T")]
PERIODIC = g.BC(x=g.Periodic(), y=g.Periodic())
PAD = 8


def wrapped(f: np.ndarray, extra: int = PAD) -> np.ndarray:
    """The periodic field on a ring of ``extra`` more columns and rows."""
    inner = f[1:-1, 1:-1]
    return np.pad(inner, 1 + extra, mode="wrap")


def wider(kind: str, extra: int = PAD) -> Any:
    """The grid with ``extra`` more cells on every side, so no edge is near."""
    if kind == "cartesian":
        return g.CartesianCGrid2D.from_interior(
            NX + 2 * extra,
            NY + 2 * extra,
            7.0 / NX * (NX + 2 * extra),
            3.0 / NY * (NY + 2 * extra),
        )
    raise AssertionError(kind)


def inner(f: Any, extra: int = PAD) -> np.ndarray:
    return np.asarray(f)[1 + extra : -1 - extra, 1 + extra : -1 - extra]


@pytest.mark.parametrize("scheme", LINEAR, ids=repr)
@pytest.mark.parametrize(("axis", "src", "dst"), DIFFERENCES)
def test_a_periodic_difference(scheme: Any, axis: str, src: str, dst: str) -> None:
    f = RNG.normal(size=GRIDS["cartesian"].shape)
    kwargs = {"axis": axis, "src": src, "dst": dst, "scheme": scheme}
    got = g.difference(f, GRIDS["cartesian"], bc=PERIODIC, **kwargs)
    want = g.difference(wrapped(f), wider("cartesian"), **kwargs)
    np.testing.assert_allclose(got[1:-1, 1:-1], inner(want), rtol=1e-12, atol=1e-12)
    assert (got[0] == 0).all() and (got[:, -1] == 0).all()


@pytest.mark.parametrize("scheme", LINEAR, ids=repr)
@pytest.mark.parametrize(("src", "dst"), INTERPOLATIONS)
def test_a_periodic_interpolation(scheme: Any, src: str, dst: str) -> None:
    f = RNG.normal(size=GRIDS["cartesian"].shape)
    kwargs = {"src": src, "dst": dst, "scheme": scheme}
    got = g.interpolate(f, GRIDS["cartesian"], bc=PERIODIC, **kwargs)
    want = g.interpolate(wrapped(f), wider("cartesian"), **kwargs)
    np.testing.assert_allclose(got[1:-1, 1:-1], inner(want), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("scheme", FLUXES, ids=repr)
def test_a_periodic_tendency(scheme: Any, masked: bool) -> None:
    grid = GRIDS["cartesian"]
    q, u, v = RNG.normal(size=(3, *grid.shape))
    q, u, v = (np.pad(x[1:-1, 1:-1], 1, mode="wrap") for x in (q, u, v))
    mask = wide_mask = None
    if masked:
        wet = np.pad(RNG.random((NY, NX)) > 0.15, 1, mode="wrap")
        mask = g.Mask2D.from_grid(grid, wet)
        wide_mask = g.Mask2D.from_grid(wider("cartesian"), wrapped(wet))
    got = g.advect(q, u, v, grid, scheme=scheme, mask=mask, bc=PERIODIC)
    want = g.advect(
        *(wrapped(x) for x in (q, u, v)),
        wider("cartesian"),
        scheme=scheme,
        mask=wide_mask,
    )
    np.testing.assert_allclose(got[1:-1, 1:-1], inner(want), rtol=1e-11, atol=1e-11)
    assert abs(float(np.sum(got))) < 1e-11 * float(np.sum(np.abs(got)))
    for axis in ("x", "y"):
        flux = g.reconstruct(
            q, u, grid, axis=axis, scheme=scheme, mask=mask, bc=PERIODIC
        )
        wide = g.reconstruct(
            wrapped(q),
            wrapped(u),
            wider("cartesian"),
            axis=axis,
            scheme=scheme,
            mask=wide_mask,
        )
        np.testing.assert_allclose(
            flux[1:-1, 1:-1], inner(wide), rtol=1e-11, atol=1e-11
        )


@pytest.mark.parametrize("scheme", FLUXES, ids=repr)
def test_a_zonally_periodic_sphere(scheme: Any) -> None:
    # the recipe of SphericalAdvection2D: a grid with extra wrapped columns
    grid = GRIDS["spherical"]
    q, u, v = (
        np.pad(x[:, 1:-1], ((0, 0), (1, 1)), mode="wrap")
        for x in RNG.normal(size=(3, *grid.shape))
    )
    v[0] = v[-2] = 0.0  # solid walls
    bc = g.BC(x=g.Periodic(), y=g.Neumann())
    got = g.advect(q, u, v, grid, scheme=scheme, bc=bc)
    d = 360.0 / NX
    big = g.SphericalCGrid2D.from_interior(
        NX + 2 * PAD, NY, (-PAD * d, 360.0 + PAD * d), (-50.0, 40.0)
    )
    columns = [
        np.pad(x[:, 1:-1], ((0, 0), (1 + PAD, 1 + PAD)), mode="wrap") for x in (q, u, v)
    ]
    want = g.advect(*columns, big, scheme=scheme, bc=g.BC(x=g.Outflow(), y=g.Neumann()))
    np.testing.assert_allclose(
        got[1:-1, 1:-1], np.asarray(want)[1:-1, 1 + PAD : -1 - PAD], rtol=1e-11
    )
    budget = float(g.area_sum(got, grid))
    assert abs(budget) < 1e-11 * float(g.area_sum(np.abs(got), grid))


def _rate(errors: list[float]) -> float:
    return float(np.polyfit(np.arange(len(errors)), -np.log2(errors), 1)[0])


@pytest.mark.parametrize(("order", "expected"), [(2, 2), (4, 4), (6, 6)])
def test_the_order_holds_up_to_a_dirichlet_wall(order: int, expected: int) -> None:
    errors = []
    for n in (16, 32, 64):
        grid = g.CartesianCGrid1D.from_interior(n, np.pi)
        # zero on both walls, half a cell outside the first and the last T point
        wall = 0.5 * (grid.coords("x")[0] + grid.coords("x")[1])
        f = np.sin(grid.coords("x") - wall)
        got = g.difference(
            f,
            grid,
            axis="x",
            src="T",
            dst="U",
            scheme=Centred(order),
            bc=g.BC(x=g.Dirichlet(0.0)),
        )
        exact = np.cos(grid.coords("x", at="U") - wall)
        errors.append(np.abs(got - exact)[1:-1].max())
    assert _rate(errors) > expected - 0.3


@pytest.mark.parametrize(("order", "expected"), [(2, 2), (4, 4)])
def test_the_order_holds_across_a_periodic_seam(order: int, expected: int) -> None:
    errors = []
    for n in (16, 32, 64):
        grid = g.CartesianCGrid1D.from_interior(n, 2.0 * np.pi)
        f = np.sin(grid.coords("x"))
        got = g.difference(
            f,
            grid,
            axis="x",
            src="T",
            dst="U",
            scheme=Centred(order),
            bc=g.BC(x=g.Periodic()),
        )
        errors.append(np.abs(got - np.cos(grid.coords("x", at="U")))[1:-1].max())
    assert _rate(errors) > expected - 0.3


def test_a_two_point_move_reads_the_ring_from_bc() -> None:
    grid = GRIDS["cartesian"]
    f = RNG.normal(size=grid.shape)
    bc = g.BC(x=g.Dirichlet(0.7))
    kwargs = {"axis": "x", "src": "T", "dst": "U"}
    np.testing.assert_array_equal(
        g.difference(f, grid, bc=bc, **kwargs),
        g.difference(g.fill_ghosts(f, grid, bc), grid, **kwargs),
    )


def test_a_neumann_wall_on_a_sphere_has_the_spacing_of_its_row() -> None:
    grid = GRIDS["spherical"]
    f = RNG.normal(size=grid.shape)
    bc = g.BC(x=g.Neumann(0.3))
    got = g.difference(f, grid, axis="x", src="T", dst="U", bc=bc, scheme=Centred(4))
    # Δf across the wall is Δx g, so the derivative there is g to second order
    two = g.difference(f, grid, axis="x", src="T", dst="U", bc=bc)
    np.testing.assert_allclose(two[1:-1, -2], 0.3, rtol=1e-12)
    assert np.isfinite(got).all()


def test_masked_moves_see_the_wet_points_across_the_seam() -> None:
    grid = GRIDS["cartesian"]
    f = RNG.normal(size=grid.shape)
    wet = np.ones(grid.shape, dtype=bool)
    mask = g.Mask2D.from_grid(grid, wet)
    kwargs = {"axis": "x", "src": "T", "dst": "U", "scheme": Centred(4)}
    np.testing.assert_array_equal(
        g.difference(f, grid, mask=mask, bc=PERIODIC, **kwargs)[:, :-1],
        g.difference(f, grid, bc=PERIODIC, **kwargs)[:, :-1],
    )


def test_fields_jit_and_grad() -> None:
    grid = GRIDS["cartesian"]
    data = jax.numpy.asarray(RNG.normal(size=grid.shape))
    h = g.CField(data, ("y", "x"), loc="T")

    @jax.jit
    def loss(data: Any, value: Any) -> Any:
        bc = g.BC(x=g.Dirichlet(value))
        out = g.difference(
            h.with_data(data), grid, axis="x", dst="U", scheme=Centred(4), bc=bc
        )
        assert out.stagger == (("x", "right"),)
        return (out.data**2).sum()

    df, dv = jax.grad(loss, (0, 1))(data, 0.4)
    assert np.isfinite(df).all() and abs(float(dv)) > 0


class TestErrors:
    GRID = GRIDS["cartesian"]
    F = np.zeros(GRID.shape)

    def test_a_missing_face(self) -> None:
        with pytest.raises(ValueError, match="no condition for the north face"):
            g.difference(
                self.F,
                self.GRID,
                axis="y",
                src="T",
                dst="V",
                bc=g.BC(x=g.Neumann(), south=g.Neumann()),
            )

    def test_a_face_the_move_does_not_read(self) -> None:
        bc = g.BC(x=g.Neumann())
        out = g.difference(self.F, self.GRID, axis="x", src="T", dst="U", bc=bc)
        assert out.shape == self.F.shape

    def test_an_unknown_axis(self) -> None:
        bc = g.BC(x=g.Neumann(), z=g.Neumann())
        with pytest.raises(ValueError, match="axes"):
            g.difference(self.F, self.GRID, axis="x", src="T", dst="U", bc=bc)

    def test_a_collocated_grid(self) -> None:
        grid = g.CartesianAGrid1D.from_interior(8, 4.0)
        with pytest.raises(NotImplementedError, match="no ghost ring"):
            g.difference(np.zeros(8), grid, axis="x", bc=g.BC(x=g.Neumann()))
        with pytest.raises(NotImplementedError, match="no ghost ring"):
            g.interpolate(np.zeros(8), grid, bc=g.BC(x=g.Neumann()))

    def test_an_open_wall(self) -> None:
        with pytest.raises(ValueError, match="drop wall="):
            g.advect(
                self.F,
                self.F,
                self.F,
                self.GRID,
                scheme=Upwind(1),
                wall="open",
                bc=PERIODIC,
            )
