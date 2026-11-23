"""The regions of a field: `interior`, `embed`, `select`, `apply_mask` (#508)."""

from __future__ import annotations

from typing import Any

import gridax as g
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_registry_parity import GRIDS

RNG = np.random.default_rng(508)
CGRIDS = {name: grid for name, grid in GRIDS.items() if grid.ghost}
GRID = g.CartesianCGrid2D.from_interior(5, 4, 5.0, 4.0)
FACES = {
    "west": ("x", 0),
    "east": ("x", 1),
    "south": ("y", 0),
    "north": ("y", 1),
    "bottom": ("z", 0),
    "top": ("z", 1),
}


def field(grid: Any = GRID, lead: tuple[int, ...] = ()) -> np.ndarray:
    return RNG.normal(size=(*lead, *grid.shape))


# ----------------------------------------------------------------------
# interior and embed
# ----------------------------------------------------------------------


@pytest.mark.parametrize("lead", [(), (2,), (2, 3)])
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_they_are_inverses(name: str, lead: tuple[int, ...]) -> None:
    grid = GRIDS[name]
    f = field(grid, lead)
    inner = g.interior(f, grid)
    assert inner.shape == (*lead, *(n - 2 * grid.ghost for n in grid.shape))
    zeroed = np.zeros_like(f)
    kept = (..., *(slice(grid.ghost, n - grid.ghost) for n in grid.shape))
    zeroed[kept] = f[kept]
    np.testing.assert_array_equal(g.embed(inner, grid), zeroed)
    np.testing.assert_array_equal(g.interior(g.embed(inner, grid), grid), inner)


@pytest.mark.parametrize("name", sorted(CGRIDS))
def test_embed_fills_the_ring(name: str) -> None:
    grid = CGRIDS[name]
    f = field(grid)
    out = g.embed(g.interior(f, grid), grid, fill=-3.5)
    inside = np.zeros(grid.shape, dtype=bool)
    inside[tuple(slice(1, -1) for _ in grid.shape)] = True
    np.testing.assert_array_equal(out[inside], f[inside])
    assert (out[~inside] == -3.5).all()


def test_embed_keeps_the_dtype() -> None:
    inner = np.ones((4, 5), dtype=np.float32)
    assert g.embed(inner, GRID).dtype == np.float32
    assert g.embed(inner, GRID, fill=0.5).dtype == np.float32


def test_a_wider_ring() -> None:
    import dataclasses

    grid = dataclasses.replace(GRID, Nx=9, Ny=8, ghost=2)
    f = RNG.normal(size=(8, 9))
    np.testing.assert_array_equal(g.interior(f, grid), f[2:-2, 2:-2])
    np.testing.assert_array_equal(g.select(f, grid, "north"), f[6])
    np.testing.assert_array_equal(g.select(f, grid, "west"), f[:, 1])
    np.testing.assert_array_equal(g.select(f, grid, "wall_south"), f[2, 2:-2])
    np.testing.assert_array_equal(
        g.select(f, grid, "corners"), [f[1, 1], f[1, 7], f[6, 1], f[6, 7]]
    )
    ring = np.ones((8, 9), dtype=bool)
    ring[2:-2, 2:-2] = False
    assert sorted(g.select(f, grid, "ghost")) == sorted(f[ring])


def test_fields() -> None:
    data = RNG.normal(size=(*GRID.shape, 2))
    u = g.CField(data, ("y", "x", "member"), {"member": ("a", "b")}, loc="U")
    inner = g.interior(u, GRID)
    assert inner.dims == u.dims and inner.stagger == u.stagger
    assert inner.labels == u.labels
    np.testing.assert_array_equal(inner.data, data[1:-1, 1:-1])
    back = g.embed(inner, GRID)
    assert back.dims == u.dims and back.stagger == u.stagger
    np.testing.assert_array_equal(back.data[1:-1, 1:-1], data[1:-1, 1:-1])
    np.testing.assert_array_equal(
        g.select(u, GRID, "north"), np.moveaxis(data, -1, 0)[:, -1]
    )


# ----------------------------------------------------------------------
# select
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CGRIDS))
def test_faces(name: str) -> None:
    grid = CGRIDS[name]
    f = field(grid, (2,))
    for face, (axis, side) in FACES.items():
        if axis not in grid.axes:
            with pytest.raises(ValueError, match="the grid has the axes"):
                g.select(f, grid, face)
            continue
        position = grid.axes.index(axis) - len(grid.axes)
        want = np.take(f, -1 if side else 0, axis=position)
        np.testing.assert_array_equal(g.select(f, grid, face), want)
        pair = (axis, ("low", "high")[side])
        np.testing.assert_array_equal(g.select(f, grid, pair), want)


@pytest.mark.parametrize("point", ["T", "U", "V", "X"])
def test_walls(point: str) -> None:
    f = field()
    ny, nx = GRID.shape
    on_x, on_y = point in "UX", point in "VX"
    want = {
        "wall_west": f[1:-1, 0 if on_x else 1],
        "wall_east": f[1:-1, nx - 2],
        "wall_south": f[0 if on_y else 1, 1:-1],
        "wall_north": f[ny - 2, 1:-1],
    }
    located = g.CField(f, ("y", "x"), loc=point)
    for region, values in want.items():
        np.testing.assert_array_equal(g.select(f, GRID, region, stagger=point), values)
        np.testing.assert_array_equal(g.select(located, GRID, region), values)


@pytest.mark.parametrize("point", ["T", "U", "V", "X"])
def test_a_wall_is_what_a_boundary_condition_holds_on(point: str) -> None:
    f = field()
    bc = g.BC(x=g.Dirichlet(0.7), y=g.Dirichlet(0.7))
    out = g.fill_ghosts(f, GRID, bc, stagger=point)
    for face in ("west", "east", "south", "north"):
        axis = FACES[face][0]
        inside = g.select(out, GRID, f"wall_{face}", stagger=point)
        if (axis == "x" and point in "UX") or (axis == "y" and point in "VX"):
            np.testing.assert_array_equal(inside, 0.7)
        else:
            ghost = g.select(out, GRID, face)[1:-1]
            np.testing.assert_allclose((inside + ghost) / 2, 0.7)


def test_ghost_and_corners() -> None:
    f = field()
    ghost = g.select(f, GRID, "ghost")
    ny, nx = GRID.shape
    assert ghost.shape == (2 * nx + 2 * (ny - 2),)
    want = np.concatenate([f[0], f[-1], f[1:-1, 0], f[1:-1, -1]])
    np.testing.assert_array_equal(ghost, want)
    np.testing.assert_array_equal(
        g.select(f, GRID, "corners"), [f[0, 0], f[0, -1], f[-1, 0], f[-1, -1]]
    )


@pytest.mark.parametrize("name", sorted(CGRIDS))
def test_interior_and_ghost_are_the_field(name: str) -> None:
    grid = CGRIDS[name]
    f = field(grid, (2,))
    parts = [g.select(f, grid, "interior").reshape(2, -1), g.select(f, grid, "ghost")]
    together = np.concatenate(parts, axis=-1)
    np.testing.assert_array_equal(
        np.sort(together, axis=-1), np.sort(f.reshape(2, -1), axis=-1)
    )


REGIONS = ["interior", "ghost", "corners", "south", "east", "wall_north", "wall_west"]


@pytest.mark.parametrize("region", REGIONS)
def test_select_under_transforms(region: str) -> None:
    f = jnp.asarray(field(GRID, (3,)))

    def pick(x: Any) -> Any:
        return g.select(x, GRID, region, stagger="U")

    np.testing.assert_array_equal(jax.jit(pick)(f), pick(np.asarray(f)))
    np.testing.assert_array_equal(jax.vmap(pick)(f), pick(f))
    text = str(jax.make_jaxpr(pick)(f))
    assert "gather" not in text and "dynamic" not in text


@pytest.mark.parametrize("fn", [g.interior, g.embed])
def test_interior_and_embed_under_transforms(fn: Any) -> None:
    x = jnp.asarray(field(GRID, (3,)))
    x = x if fn is g.interior else x[:, 1:-1, 1:-1]
    np.testing.assert_array_equal(jax.jit(lambda x: fn(x, GRID))(x), fn(x, GRID))
    np.testing.assert_array_equal(jax.vmap(lambda x: fn(x, GRID))(x), fn(x, GRID))


# ----------------------------------------------------------------------
# apply_mask
# ----------------------------------------------------------------------

MASK = g.Mask2D.from_grid(GRID, RNG.random(GRID.shape) > 0.2)


@pytest.mark.parametrize("fill", [0, -1.5, np.nan])
@pytest.mark.parametrize("point", ["T", "U", "V", "X"])
def test_apply_mask(point: str, fill: float) -> None:
    wet = MASK.wet(at=point, kind="strict" if point == "X" else "default")
    f = field(GRID, (2,))
    out = g.apply_mask(f, MASK, at=point, fill=fill)
    np.testing.assert_array_equal(out, np.where(wet, f, fill))
    located = g.CField(np.moveaxis(f, 0, -1), ("y", "x", "member"), loc=point)
    masked = g.apply_mask(located, MASK, fill=fill)
    assert masked.dims == located.dims and masked.stagger == located.stagger
    np.testing.assert_array_equal(np.moveaxis(masked.data, -1, 0), out)


def test_apply_mask_does_not_read_land() -> None:
    f = jnp.asarray(np.where(MASK.wet(), field(), np.nan))
    out, grad = jax.value_and_grad(
        lambda f: (g.apply_mask(f, MASK, at="T") ** 2).sum()
    )(f)
    assert np.isfinite(out) and np.isfinite(grad).all()
    assert (grad[~MASK.wet()] == 0).all()


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


def test_errors() -> None:
    f = field()
    agrid = GRIDS["CartesianAGrid2D"]
    with pytest.raises(TypeError, match=r"gridax\.stencils\.interior"):
        g.interior(f[1:-1, 1:-1], f)  # the Layer 1 call
    with pytest.raises(ValueError, match="is not on"):
        g.interior(f[1:], GRID)
    with pytest.raises(ValueError, match="the interior is"):
        g.embed(f, GRID)
    with pytest.raises(ValueError, match="unknown region"):
        g.select(f, GRID, "northeast")
    with pytest.raises(ValueError, match="unknown region"):
        g.select(f, GRID, ("x", "east"))
    with pytest.raises(ValueError, match="has no ghost ring"):
        g.select(field(agrid), agrid, "north")
    with pytest.raises(TypeError, match="at= is the point"):
        g.apply_mask(f, MASK)
    u = g.CField(f, ("y", "x"), loc="U")
    with pytest.raises(g.StaggerMismatchError):
        g.apply_mask(u, MASK, at="V")
    with pytest.raises(g.StaggerMismatchError):
        g.select(u, GRID, "wall_east", stagger="T")
    with pytest.raises(ValueError, match="has no dims"):
        g.interior(g.CField(f, ("lat", "x"), loc="T"), GRID)


def test_pad_interior_is_gone() -> None:
    assert not hasattr(g, "pad_interior")
