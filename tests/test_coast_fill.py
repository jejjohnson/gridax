"""`fill_ghosts(..., mask=)`: the dry points behind a coast (#511)."""

from __future__ import annotations

from typing import Any

import gridax as g
from gridax._src.boundary.fill import _fill_coasts
from gridax._src.grid.points import point_stagger
from gridax.schemes import Centred, Upwind
import jax
import jax.numpy as jnp
import numpy as np
import pytest

RNG = np.random.default_rng(511)
NX, NY, LAND = 8, 6, 3

ATOMS = {
    "dirichlet": g.Dirichlet(0.4),
    "neumann": g.Neumann(-0.8),
    "outflow": g.Outflow(),
    "reflective": g.Reflective(),
    "sponge": g.Sponge(0.2, 0.3),
    "robin": g.Robin(1.5, 0.25, 0.6),
    "extrapolation2": g.Extrapolation(2),
    "slip": g.Slip(0.35),
}


def cartesian(ny: int, rows: float) -> Any:
    return g.CartesianCGrid2D.from_interior(NX, ny, 4.0, 0.5 * rows)


def spherical(ny: int, rows: float) -> Any:
    # rows of 2 degrees, the water always between 40N and 52N
    south = 40.0 - (rows - NY)
    return g.SphericalCGrid2D.from_interior(
        NX, ny, (0.0, 32.0), (south, south + 2.0 * rows)
    )


GRIDS = {"cartesian": cartesian, "spherical": spherical}


def channel(kind: str) -> tuple[Any, Any, Any]:
    """A periodic channel: alone, and between two bands of land, with its mask."""
    edge = GRIDS[kind](NY, NY)
    land = GRIDS[kind](NY + 2 * LAND, NY + 2 * LAND)
    wet = np.zeros(land.shape, dtype=bool)
    wet[1 + LAND : 1 + LAND + NY] = True
    return edge, land, g.Mask2D.from_grid(land, wet)


def embedded(f: np.ndarray, land: Any) -> np.ndarray:
    """The water of a channel field, with NaN on land."""
    out = np.full(land.shape, np.nan)
    out[1 + LAND : 1 + LAND + NY, 1:-1] = f[1 : 1 + NY, 1:-1]
    return out


def rows(f: Any) -> np.ndarray:
    return np.asarray(f)[LAND : LAND + NY + 2]


@pytest.mark.parametrize("point", ["T", "U", "V", "X"])
@pytest.mark.parametrize("atom", list(ATOMS))
@pytest.mark.parametrize("kind", list(GRIDS))
def test_a_coast_holds_what_an_edge_holds(kind: str, atom: str, point: str) -> None:
    edge, land, mask = channel(kind)
    f = RNG.normal(size=edge.shape)
    bc = g.BC(x=g.Periodic(), y=ATOMS[atom])
    at_edge = g.fill_ghosts(f, edge, bc, stagger=point)
    at_coast = g.fill_ghosts(embedded(f, land), land, bc, mask=mask, stagger=point)
    np.testing.assert_allclose(rows(at_coast), at_edge, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("atom", list(ATOMS))
def test_layers_of_a_line_of_water(atom: str, width: int) -> None:
    # a line of water, in an array with room for the layers, against a grid of
    # that water alone with the same number of ghost layers
    n = 7
    alone = g.CartesianCGrid1D.from_interior(n, 0.5 * n)
    f = RNG.normal(size=n + 2 * width)
    at_edge = g.fill_ghosts(f, alone, g.BC(x=ATOMS[atom]), width=width)
    wet = np.zeros(n + 2 * width + 2, dtype=bool)
    wet[1 + width : 1 + width + n] = True
    big = g.CartesianCGrid1D.from_interior(n + 2 * width, 0.5 * (n + 2 * width))
    field = np.where(wet, np.concatenate([[0.0], f, [0.0]]), np.nan)
    at_coast = _fill_coasts(
        field,
        big,
        g.BC(x=ATOMS[atom]),
        g.Mask1D.from_mask(wet),
        1,
        point_stagger("T"),
        layers=width,
    )
    np.testing.assert_allclose(at_coast[1:-1], at_edge, rtol=1e-13)


def _tendencies(op: str, grid: Any, fields: dict[str, Any]) -> list[Any]:
    u, v, q = fields["u"], fields["v"], fields["q"]
    if op == "diffuse":
        return [g.diffuse(q, grid, kappa=0.3, wall="open")]
    if op == "coriolis":
        return list(g.coriolis(u, v, grid, f=1.3))
    scheme = {"upwind": Upwind(1), "centred": Centred(2)}[op]
    return [g.advect(q, u, v, grid, scheme=scheme, wall="open")]


@pytest.mark.parametrize("slip", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    ("kind", "op"),
    [
        # coriolis is Cartesian, and spherical diffusion has closed edges only
        ("cartesian", "diffuse"),
        ("cartesian", "coriolis"),
        ("cartesian", "upwind"),
        ("cartesian", "centred"),
        ("spherical", "upwind"),
        ("spherical", "centred"),
    ],
)
def test_a_channel_in_land_is_a_channel(kind: str, op: str, slip: float) -> None:
    edge, land, mask = channel(kind)
    bcs = {
        "u": g.BC(x=g.Periodic(), y=g.Slip(slip)),
        "v": g.BC(x=g.Periodic(), y=g.Dirichlet(0.0)),
        "q": g.BC(x=g.Periodic(), y=g.Neumann(0.2)),
    }
    points = {"u": "U", "v": "V", "q": "T"}
    raw = {name: RNG.normal(size=edge.shape) for name in points}
    at_edge = {
        name: g.fill_ghosts(raw[name], edge, bcs[name], stagger=point)
        for name, point in points.items()
    }
    at_coast = {
        name: g.fill_ghosts(
            embedded(raw[name], land),
            land,
            bcs[name],
            mask=mask,
            stagger=point,
        )
        for name, point in points.items()
    }
    got = _tendencies(op, land, {k: np.nan_to_num(x) for k, x in at_coast.items()})
    want = _tendencies(op, edge, at_edge)
    for x, y in zip(got, want, strict=True):
        water = rows(x)[1:-1, 1:-1]
        wanted = np.asarray(y)[1:-1, 1:-1]
        if x is got[-1] and op == "coriolis":
            # the last V row is the wall node, which a model masks
            water, wanted = water[:-1], wanted[:-1]
        np.testing.assert_allclose(water, wanted, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", list(GRIDS))
def test_masked_diffusion_is_a_closed_edge(kind: str) -> None:
    # the masked operator and a filled coast agree: no flux through either
    edge, land, mask = channel(kind)
    bc = g.BC(x=g.Periodic(), y=g.Neumann())
    q = g.fill_ghosts(RNG.normal(size=edge.shape), edge, bc)
    inland = g.fill_ghosts(embedded(q, land), land, bc, mask=mask)
    got = g.diffuse(np.nan_to_num(inland), land, kappa=0.3, mask=mask)
    want = g.diffuse(q, edge, kappa=0.3)
    np.testing.assert_allclose(
        rows(got)[1:-1, 1:-1], np.asarray(want)[1:-1, 1:-1], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("point", ["T", "U", "V", "X"])
@pytest.mark.parametrize("width", [1, 2])
def test_no_wet_point_is_written(point: str, width: int) -> None:
    grid = g.CartesianCGrid2D.from_interior(20, 18, 2.0, 1.0)
    wet = RNG.random(grid.shape) > 0.2
    mask = g.Mask2D.from_grid(grid, wet)
    f = RNG.normal(size=grid.shape)
    bc = g.BC(x=g.Slip(0.3), y=g.Dirichlet(1.0))
    out = g.fill_ghosts(f, grid, bc, mask=mask, stagger=point, width=width)
    kind = "strict" if point == "X" else "default"
    inside = np.zeros(grid.shape, dtype=bool)
    inside[width : -1 - width, width : -1 - width] = True
    kept = mask.wet(at=point, kind=kind) & inside
    np.testing.assert_array_equal(out[kept], f[kept])
    # and some dry points are
    assert (out[~kept & inside] != f[~kept & inside]).any()


def test_a_dry_point_between_two_waters_holds_the_mean() -> None:
    grid = g.CartesianCGrid2D.from_interior(5, 5, 5.0, 5.0)
    wet = np.ones(grid.shape, dtype=bool)
    wet[3, 3] = False
    f = np.arange(49.0).reshape(grid.shape)
    out = g.fill_ghosts(
        f,
        grid,
        g.BC(x=g.Neumann(), y=g.Dirichlet(0.0)),
        mask=g.Mask2D.from_grid(grid, wet),
    )
    # copies of the two neighbours along x, minus those of the two along y
    assert out[3, 3] == (f[3, 2] + f[3, 4] - f[2, 3] - f[4, 3]) / 4


def test_the_nearest_layer_wins() -> None:
    # a wall of two cells: each is layer 1 of one basin and layer 2 of the other
    grid = g.CartesianCGrid1D.from_interior(8, 8.0)
    wet = np.ones(10, dtype=bool)
    wet[4:6] = False
    f = np.arange(10.0)
    out = g.fill_ghosts(
        f,
        grid,
        g.BC(x=g.Neumann()),
        mask=g.Mask1D.from_grid(grid, wet),
        width=2,
    )
    assert out[4:6].tolist() == [3.0, 6.0]


class TestCoast:
    GRID = g.CartesianCGrid2D.from_interior(6, 6, 6.0, 6.0)

    def mask(self) -> Any:
        wet = np.ones(self.GRID.shape, dtype=bool)
        wet[3:5, 3:5] = False
        return g.Mask2D.from_grid(self.GRID, wet)

    def test_periodic_faces_have_no_coast(self) -> None:
        f = RNG.normal(size=self.GRID.shape)
        bc = g.BC(x=g.Periodic(), y=g.Periodic())
        out = g.fill_ghosts(f, self.GRID, bc, mask=self.mask())
        np.testing.assert_array_equal(out, g.fill_ghosts(f, self.GRID, bc))

    def test_coast_gives_them_one(self) -> None:
        f = RNG.normal(size=self.GRID.shape)
        bc = g.BC(x=g.Periodic(), y=g.Periodic(), coast=g.BC(x=g.Neumann()))
        out = g.fill_ghosts(f, self.GRID, bc, mask=self.mask())
        np.testing.assert_array_equal(out[3:5, 3], f[3:5, 2])
        np.testing.assert_array_equal(out[3:5, 4], f[3:5, 5])

    def test_coast_of_a_periodic_image(self) -> None:
        # land in the first column: its water on the low side is the last column
        wet = np.ones(self.GRID.shape, dtype=bool)
        wet[:, (1, -1)] = False
        f = RNG.normal(size=self.GRID.shape)
        bc = g.BC(x=g.Periodic(), y=g.Periodic(), coast=g.BC(x=g.Neumann()))
        mask = g.Mask2D.from_grid(self.GRID, wet)
        out = g.fill_ghosts(f, self.GRID, bc, mask=mask)
        np.testing.assert_allclose(out[1:-1, 1], (f[1:-1, 6] + f[1:-1, 2]) / 2)
        np.testing.assert_array_equal(out[:, -1], out[:, 1])

    def test_face_values_run_along_the_coast(self) -> None:
        values = np.arange(8.0)
        f = np.zeros(self.GRID.shape)
        bc = g.BC(x=g.Dirichlet(values), y=g.Neumann())
        out = g.fill_ghosts(f, self.GRID, bc, mask=self.mask())
        # (2 v - 0) from the x side and a copy of 0 from the y side
        np.testing.assert_array_equal(out[3:5, 3], values[3:5])

    def test_coast_must_be_a_bc(self) -> None:
        with pytest.raises(TypeError, match="coast must be a BC"):
            g.BC(x=g.Neumann(), coast=g.Neumann())

    def test_coast_on_an_unknown_axis_raises(self) -> None:
        bc = g.BC(x=g.Neumann(), coast=g.BC(z=g.Neumann()))
        with pytest.raises(ValueError, match="axes"):
            g.fill_ghosts(np.zeros(self.GRID.shape), self.GRID, bc, mask=self.mask())


def test_fields_and_states() -> None:
    grid = TestCoast.GRID
    mask = TestCoast().mask()
    wall, flat = g.Dirichlet(0.0), g.Neumann()
    bcs = {"h": g.BC(x=flat, y=flat), "u": g.BC(x=wall, y=flat)}
    arrays = {k: jnp.asarray(RNG.normal(size=(2, *grid.shape))) for k in bcs}
    state = {
        k: g.CField(x, ("layer", "y", "x"), loc={"h": "T", "u": "U"}[k])
        for k, x in arrays.items()
    }
    out = g.fill_ghosts(state, grid, bcs, mask=mask)
    for k, point in (("h", "T"), ("u", "U")):
        expected = g.fill_ghosts(arrays[k], grid, bcs[k], mask=mask, stagger=point)
        np.testing.assert_array_equal(out[k].data, expected)
        for level in range(2):
            one = g.fill_ghosts(
                arrays[k][level], grid, bcs[k], mask=mask, stagger=point
            )
            np.testing.assert_array_equal(expected[level], one)


def test_three_dimensions() -> None:
    grid = g.CartesianCGrid3D.from_interior(4, 4, 3, 1.0, 1.0, 1.0)
    wet = np.ones(grid.shape, dtype=bool)
    wet[0] = wet[1] = False  # a flat bottom one cell up
    f = RNG.normal(size=grid.shape)
    bc = g.BC(z=(g.Dirichlet(0.0), g.Neumann()))
    out = g.fill_ghosts(f, grid, bc, mask=g.Mask3D.from_grid(grid, wet))
    np.testing.assert_array_equal(out[1], -f[2])


def test_jit_grad_and_nan_land() -> None:
    grid = TestCoast.GRID
    mask = TestCoast().mask()
    f = jnp.asarray(np.where(mask.wet(), RNG.normal(size=grid.shape), np.nan))

    @jax.jit
    def loss(f: Any, value: Any, mask: Any) -> Any:
        bc = g.BC(x=g.Dirichlet(value), y=g.Neumann())
        out = g.fill_ghosts(f, grid, bc, mask=mask)
        return jnp.sum(jnp.where(jnp.isnan(out), 0.0, out) ** 2)

    value, (df, dv) = jax.value_and_grad(loss, (0, 1))(f, 0.3, mask)
    assert np.isfinite(value) and np.isfinite(dv)
    assert np.isfinite(np.asarray(df)[mask.wet()]).all()
