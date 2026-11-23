"""The presets of `BC`, against the helpers they replace (#510)."""

from __future__ import annotations

from typing import Any

import gridax as g
from gridax import BC
from gridax.schemes import WENO, Centred, Upwind
import jax
import jax.numpy as jnp
import numpy as np
import pytest

RNG = np.random.default_rng(510)
GRID = g.CartesianCGrid2D.from_interior(7, 6, 3.5, 2.4)
SPHERE = g.SphericalCGrid2D.from_interior(7, 6, (0.0, 30.0), (10.0, 40.0))
NO, FREE = (BC.wall_tangential(slip=s) for s in ("no", "free"))


def field() -> np.ndarray:
    return RNG.normal(size=GRID.shape)


def ring(f: np.ndarray) -> tuple[np.ndarray, ...]:
    """The four faces, without the corners: south, north, west, east."""
    return f[0, 1:-1], f[-1, 1:-1], f[1:-1, 0], f[1:-1, -1]


def same_ring(got: Any, want: np.ndarray) -> None:
    for x, y in zip(ring(np.asarray(got)), ring(want), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-15)


# ----------------------------------------------------------------------
# What the removed helpers wrote (gridax 0.3), where they were right
# ----------------------------------------------------------------------


def zero_boundaries(f: np.ndarray) -> np.ndarray:
    return np.pad(f[1:-1, 1:-1], 1)


def zero_gradient_boundaries(f: np.ndarray) -> np.ndarray:
    return np.pad(f[1:-1, 1:-1], 1, mode="edge")


def enforce_periodic(f: np.ndarray) -> np.ndarray:
    return np.pad(f[1:-1, 1:-1], 1, mode="wrap")


def slip_boundaries(f: np.ndarray, sign: float) -> np.ndarray:
    out = f.copy()
    out[0, 1:-1], out[-1, 1:-1] = sign * f[1, 1:-1], sign * f[-2, 1:-1]
    out[1:-1, 0], out[1:-1, -1] = sign * f[1:-1, 1], sign * f[1:-1, -2]
    return out


def fix_boundary_corners(f: np.ndarray) -> np.ndarray:
    out = f.copy()
    out[0, 0] = 0.5 * (f[0, 1] + f[1, 0])
    out[0, -1] = 0.5 * (f[0, -2] + f[1, -1])
    out[-1, 0] = 0.5 * (f[-1, 1] + f[-2, 0])
    out[-1, -1] = 0.5 * (f[-1, -2] + f[-2, -1])
    return out


def test_no_flux_is_zero_gradient() -> None:
    f = field()
    got = g.fill_ghosts(f, GRID, BC.no_flux())
    np.testing.assert_array_equal(got, zero_gradient_boundaries(f))


def test_no_flux_is_the_wall_of_a_tracer() -> None:
    # wall_boundaries(f, "h"): free-slip faces, averaged corners
    f = field()
    want = fix_boundary_corners(slip_boundaries(f, 1.0))
    got = g.fill_ghosts(f, GRID, BC.no_flux())
    same_ring(got, want)


def test_periodic() -> None:
    f = field()
    got = g.fill_ghosts(f, GRID, BC.periodic("x", "y"))
    np.testing.assert_array_equal(got, enforce_periodic(f))
    np.testing.assert_array_equal(g.fill_ghosts(f, GRID, BC.periodic()), got)


@pytest.mark.parametrize(("bc", "sign"), [(NO, -1.0), (FREE, 1.0)])
def test_tangential_walls_are_the_slip_helpers(bc: BC, sign: float) -> None:
    f = field()
    want = slip_boundaries(f, sign)
    # a tangential component: U along y, V along x
    u = g.fill_ghosts(f, GRID, bc, stagger="U")
    np.testing.assert_array_equal(u[0, 1:-1], want[0, 1:-1])
    np.testing.assert_array_equal(u[-1, 1:-1], want[-1, 1:-1])
    v = g.fill_ghosts(f, GRID, bc, stagger="V")
    np.testing.assert_array_equal(v[1:-1, 0], want[1:-1, 0])
    np.testing.assert_array_equal(v[1:-1, -1], want[1:-1, -1])


def test_walls_of_a_velocity_are_the_old_ones() -> None:
    # wall_boundaries(f, "u"): U = 0 on the W/E faces, free slip along S/N
    f = field()
    walls = BC.wall_normal() | FREE
    u = g.fill_ghosts(f, GRID, walls, stagger="U")
    assert (u[1:-1, 0] == 0).all() and (u[1:-1, -2] == 0).all()
    np.testing.assert_array_equal(u[1:-1, -1], -f[1:-1, -3])
    np.testing.assert_array_equal(u[0, 1:-2], f[1, 1:-2])
    np.testing.assert_array_equal(u[-1, 1:-2], f[-2, 1:-2])
    v = g.fill_ghosts(f, GRID, walls, stagger="V")
    assert (v[0, 1:-1] == 0).all() and (v[-2, 1:-1] == 0).all()
    np.testing.assert_array_equal(v[-1, 1:-1], -f[-3, 1:-1])
    np.testing.assert_array_equal(v[1:-2, 0], f[1:-2, 1])


def test_zero_is_zero_at_the_wall() -> None:
    # zero_boundaries wrote 0 on the ghosts, which is right for a node on a
    # wall: the W face of U and the S face of V
    f = field()
    want = zero_boundaries(f)
    u = g.fill_ghosts(f, GRID, BC.zero(), stagger="U")
    np.testing.assert_array_equal(u[1:-1, 0], want[1:-1, 0])
    v = g.fill_ghosts(f, GRID, BC.zero(), stagger="V")
    np.testing.assert_array_equal(v[0, 1:-1], want[0, 1:-1])
    # and at T the wall is half a cell away
    h = g.fill_ghosts(f, GRID, BC.zero())
    np.testing.assert_array_equal(h[0, 1:-1] + h[1, 1:-1], 0.0)


# ----------------------------------------------------------------------
# Roles
# ----------------------------------------------------------------------


@pytest.mark.parametrize("grid", [GRID, SPHERE], ids=["cartesian", "spherical"])
def test_wall_normal_leaves_the_other_faces_alone(grid: Any) -> None:
    f = field()
    u = g.fill_ghosts(f, grid, BC.wall_normal(), stagger="U")
    np.testing.assert_array_equal(u[[0, -1], 1:-2], f[[0, -1], 1:-2])
    v = g.fill_ghosts(f, grid, BC.wall_normal(), stagger="V")
    np.testing.assert_array_equal(v[1:-2, [0, -1]], f[1:-2, [0, -1]])
    # a tracer has no normal component
    np.testing.assert_array_equal(g.fill_ghosts(f, grid, BC.wall_normal()), f)


def test_one_wall_for_both_components() -> None:
    f = field()
    solid = BC.wall_normal() | NO
    u = g.fill_ghosts(f, GRID, solid, stagger="U")
    v = g.fill_ghosts(f, GRID, solid, stagger="V")
    for c, across in ((u, 0), (v, 1)):
        along = np.moveaxis(c, across, -1)
        # zero halfway to the ghost along a wall
        np.testing.assert_allclose(along[1:-2, 0] + along[1:-2, 1], 0.0, atol=1e-15)
        normal = np.moveaxis(c, 1 - across, -1)
        assert (normal[1:-1, 0] == 0).all() and (normal[1:-1, -2] == 0).all()
    state = {
        "u": g.CField(f, ("y", "x"), loc="U"),
        "v": g.CField(f, ("y", "x"), loc="V"),
    }
    filled = g.fill_ghosts(state, GRID, {"u": solid, "v": solid})
    np.testing.assert_array_equal(filled["u"].data, u)
    np.testing.assert_array_equal(filled["v"].data, v)


def test_axes() -> None:
    line = g.CartesianCGrid1D.from_interior(6, 3.0)
    f = RNG.normal(size=8)
    out = g.fill_ghosts(f, line, BC.no_flux("x"))
    assert (out[0], out[-1]) == (f[1], f[-2])
    box = g.CartesianCGrid3D.from_interior(3, 4, 5, 1.0, 1.0, 1.0)
    w = RNG.normal(size=box.shape)
    filled = g.fill_ghosts(w, box, BC.wall_normal("z"), stagger="W")
    assert (filled[0] == 0).all() and (filled[-2] == 0).all()
    channel = g.fill_ghosts(field(), GRID, BC.periodic("x") | BC.no_flux("y"))
    np.testing.assert_array_equal(channel[:, 0], channel[:, -2])
    np.testing.assert_array_equal(channel[0], channel[1])


def test_or() -> None:
    both = BC.wall_normal() | NO
    assert both == NO | BC.wall_normal()
    assert both == BC.wall_normal() | NO | FREE  # the first one given wins
    part = BC(west=g.Dirichlet(1.0)) | BC.no_flux()
    assert part.faces["x"] == (g.Dirichlet(1.0), g.Neumann(0.0))
    assert part.faces["y"] == (g.Neumann(0.0), g.Neumann(0.0))
    coast = BC(x=g.Periodic(), coast=BC.no_flux("x")) | BC(coast=BC.no_flux("y"))
    assert coast.coast == BC.no_flux()
    assert (BC(corners="priority") | BC.no_flux()).corners == "priority"
    with pytest.raises(TypeError):
        BC.no_flux() | g.Neumann()  # ty: ignore[unsupported-operator]


def test_bad_slip() -> None:
    with pytest.raises(ValueError, match="slip must be"):
        BC.wall_tangential(slip="partial")


def test_walls_at_a_coast() -> None:
    # an island in a basin: the same wall as the edge of a larger domain
    wet = np.ones(GRID.shape, dtype=bool)
    wet[3:5, 3:6] = False
    mask = g.Mask2D.from_grid(GRID, wet)
    f = field()
    solid = BC.wall_normal() | NO
    u = g.fill_ghosts(f, GRID, solid, mask=mask, stagger="U")
    # the U nodes on the west and east coasts of the island
    assert (u[3:5, 2] == 0).all() and (u[3:5, 5] == 0).all()
    # no slip along its south coast
    np.testing.assert_array_equal(u[3, 3:5], -f[2, 3:5])
    # on a periodic domain the coast needs a wall of its own
    around = BC(x=g.Periodic(), y=g.Periodic(), coast=solid)
    np.testing.assert_array_equal(
        g.fill_ghosts(f, GRID, around, mask=mask, stagger="U")[3:5, 2:6], u[3:5, 2:6]
    )


def test_wide_stencils_take_the_wall_of_the_component() -> None:
    f = field()
    solid = BC.wall_normal() | NO
    kwargs = {"axis": "y", "src": "U", "dst": "X", "scheme": Centred(4)}
    got = g.difference(f, GRID, bc=solid, **kwargs)
    want = g.difference(f, GRID, bc=BC(y=g.Dirichlet(0.0)), **kwargs)
    np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="no condition for the"):
        g.difference(f, GRID, bc=BC.wall_normal(), **kwargs)


def test_under_jit() -> None:
    f = jnp.asarray(field())
    solid = BC.wall_normal() | NO

    @jax.jit
    def fill(f: Any, bc: BC) -> Any:
        return g.fill_ghosts(f, GRID, bc, stagger="U")

    np.testing.assert_array_equal(
        fill(f, solid), g.fill_ghosts(f, GRID, solid, stagger="U")
    )


# ----------------------------------------------------------------------
# The bug class the roles close
# ----------------------------------------------------------------------


def _budget(bc: BC, steps: int, scheme: Any) -> tuple[float, float]:
    """∑ A q before and after ``steps`` of advection and diffusion in a basin."""
    grid = g.CartesianCGrid2D.from_interior(16, 14, 1.0, 0.8)
    y, x = jnp.meshgrid(
        grid.coords("y", at="V"), grid.coords("x", at="U"), indexing="ij"
    )
    lx, ly = grid.spacing("x"), grid.spacing("y")
    x, y = (x - lx / 2) / 1.0, (y - ly / 2) / 0.8
    psi = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)  # at X, zero on the walls
    u = jnp.zeros(grid.shape).at[1:].set(-(psi[1:] - psi[:-1]) / ly)
    v = jnp.zeros(grid.shape).at[:, 1:].set((psi[:, 1:] - psi[:, :-1]) / lx)
    walls = BC.wall_normal()
    u = g.fill_ghosts(u, grid, walls, stagger="U")
    v = g.fill_ghosts(v, grid, walls, stagger="V")
    q = jnp.asarray(np.random.default_rng(1).random(grid.shape))

    def step(q: Any, _: Any) -> Any:
        q = g.fill_ghosts(q, grid, bc)
        dq = g.advect(q, u, v, grid, scheme=scheme, bc=bc)
        dq = dq + g.diffuse(q, grid, kappa=1e-2, wall="open")
        return q + 2e-3 * dq, None

    total = jax.jit(lambda q: jax.lax.scan(step, q, length=steps)[0])(q)
    return float(g.area_sum(q, grid)), float(g.area_sum(total, grid))


@pytest.mark.slow
@pytest.mark.parametrize("scheme", [Upwind(1), WENO(5)], ids=repr)
def test_a_closed_basin_keeps_its_tracer(scheme: Any) -> None:
    before, after = _budget(BC.no_flux(), 1000, scheme)
    assert abs(after - before) < 1e-11 * before
    # zero at the wall is not a closed basin
    before, after = _budget(BC.zero(), 1000, scheme)
    assert abs(after - before) > 1e-2 * before
