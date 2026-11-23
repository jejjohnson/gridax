"""A periodic face of ``advect`` is one face: its velocity wraps too (#403)."""

from __future__ import annotations

from typing import Any

import gridax as g
from gridax.schemes import TVD, WENO, Adaptive, Centred, Scheme, Upwind
import jax
import jax.numpy as jnp
import numpy as np
import pytest

RNG = np.random.default_rng(403)
SCHEMES = [Upwind(1), Centred(4), TVD("van_leer"), WENO(5)]
PERIODIC = g.BC.periodic()
CHANNEL = g.BC(x=g.Periodic(), y=g.Neumann())
GRIDS = {
    "1d": (g.CartesianCGrid1D.from_interior(20, 1.0), g.BC.periodic("x")),
    "2d": (g.CartesianCGrid2D.from_interior(16, 12, 1.0, 2.0), PERIODIC),
    "channel": (g.CartesianCGrid2D.from_interior(16, 12, 1.0, 2.0), CHANNEL),
    "3d": (g.CartesianCGrid3D.from_interior(12, 10, 3, 1.0, 1.0, 1.0), PERIODIC),
    "sphere": (
        g.SphericalCGrid2D.from_interior(16, 12, (0.0, 360.0), (-60.0, 60.0)),
        CHANNEL,
    ),
}


def fields(grid: Any, bc: g.BC) -> tuple[Any, Any, Any]:
    """Random fields with unrelated ghost values; no flow through a wall."""
    q, u, v = RNG.normal(size=(3, *grid.shape))
    if len(grid.axes) == 1:
        return q, u, None
    if isinstance(bc.faces["y"][0], g.Neumann):
        v = g.fill_ghosts(v, grid, g.BC.wall_normal("y"), stagger="V")
    return q, u, v


def budget(tendency: Any, grid: Any) -> float:
    """``∑ A dq/dt`` over ``∑ A |dq/dt|``, on the interior."""
    area = g.volume_weights(grid) if len(grid.axes) == 3 else g.area_weights(grid)
    inner = tuple(slice(1, -1) for _ in grid.axes)
    terms = (np.asarray(tendency) * np.asarray(area))[inner]
    return abs(terms.sum()) / np.abs(terms).sum()


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
@pytest.mark.parametrize("name", GRIDS)
def test_the_velocity_needs_no_ghosts(name: str, scheme: Scheme) -> None:
    grid, bc = GRIDS[name]
    q, u, v = fields(grid, bc)
    got = g.advect(q, u, v, grid, scheme=scheme, bc=bc)
    if len(grid.axes) > 1:
        assert budget(got, grid) < 1e-13
    else:
        assert abs(got.sum()) < 1e-13 * np.abs(got).sum()
    # what the caller had to write before
    u = g.fill_ghosts(u, grid, g.BC(x=g.Periodic()), stagger="U")
    if v is not None and isinstance(bc.faces["y"][0], g.Periodic):
        v = g.fill_ghosts(v, grid, g.BC(y=g.Periodic()), stagger="V")
    np.testing.assert_array_equal(got, g.advect(q, u, v, grid, scheme=scheme, bc=bc))


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
def test_periodic_edges(scheme: Scheme) -> None:
    grid, bc = GRIDS["2d"]
    q, u, v = fields(grid, bc)
    np.testing.assert_array_equal(
        g.advect(q, u, v, grid, scheme=Adaptive(scheme, edges="periodic")),
        g.advect(q, u, v, grid, scheme=scheme, bc=bc),
    )


def test_interpolated_velocities() -> None:
    """The script of the issue: the faces come from an interpolation."""
    grid = g.CartesianCGrid2D.from_interior(64, 64, 1.0, 1.0)
    x = np.arange(grid.Nx) * grid.spacing("x")
    x = np.broadcast_to(x, grid.shape)
    tracer = g.fill_ghosts(np.exp(-(((x - 0.05) / 0.1) ** 2)) + 1.0, grid, PERIODIC)
    speed = g.fill_ghosts(1.0 + 0.5 * np.sin(2.0 * np.pi * x), grid, PERIODIC)
    u = g.interpolate(speed, grid, src="T", dst="U")
    assert (u[:, 0] == 0.0).all()  # no forward move fills it
    zero = np.zeros_like(u)
    out = g.advect(tracer, u, zero, grid, scheme=Upwind(1), bc=PERIODIC)
    assert abs(float(g.area_sum(out, grid))) < 1e-13
    # a wall that is open reads the zero
    out = g.advect(tracer, u, zero, grid, scheme=Upwind(1), wall="open")
    assert abs(float(g.area_sum(out, grid))) > 1.0


def test_a_wall_still_reads_the_velocity() -> None:
    grid, _ = GRIDS["channel"]
    q, u, v = RNG.normal(size=(3, *grid.shape))
    shut = g.fill_ghosts(v, grid, g.BC.wall_normal("y"), stagger="V")
    a, b = (g.advect(q, u, w, grid, scheme=Upwind(1), bc=CHANNEL) for w in (v, shut))
    assert (a[1] != b[1]).any()


def test_masked() -> None:
    grid, bc = GRIDS["2d"]
    wet = np.pad(RNG.random((12, 16)) > 0.2, 1, mode="wrap")
    mask = g.Mask2D.from_grid(grid, wet)
    q, u, v = fields(grid, bc)
    got = g.advect(q, u, v, grid, scheme=WENO(5), mask=mask, bc=bc)
    assert budget(got, grid) < 1e-13


def test_fields_jit_and_grad() -> None:
    grid, bc = GRIDS["2d"]
    q, u, v = (jnp.asarray(x) for x in fields(grid, bc))
    want = g.advect(q, u, v, grid, scheme=WENO(3), bc=bc)
    located = [g.CField(x, grid.axes, loc=p) for x, p in ((q, "T"), (u, "U"), (v, "V"))]
    np.testing.assert_array_equal(
        g.advect(*located, grid, scheme=WENO(3), bc=bc).data, want
    )

    def loss(u: Any) -> Any:
        return (g.advect(q, u, v, grid, scheme=WENO(3), bc=bc) ** 2).sum()

    gradient = jax.jit(jax.grad(loss))(u)
    assert np.isfinite(gradient).all()
    assert (gradient[:, 0] == 0.0).all()  # the ghost face is not read
