"""`vortex_force`, and the scheme of Arakawa and Lamb (1981) (#373)."""

from __future__ import annotations

from typing import Any

import gridax as g
import jax
import jax.numpy as jnp
import numpy as np
import pytest

RNG = np.random.default_rng(373)
PERIODIC = g.BC.periodic()
BOX = g.CartesianCGrid2D.from_interior(16, 12, 4.0, 2.0)  # dx = 0.25, dy = 1/6
BAND = g.SphericalCGrid2D.from_interior(16, 12, (0.0, 360.0), (-60.0, 60.0), R=2.0)
GRIDS = {"box": BOX, "band": BAND}
G = 9.8
INNER = (slice(1, -1), slice(1, -1))


def fill(f: Any, grid: Any, at: str) -> Any:
    """Periodic in x; in y periodic in the box, and walls in the band."""
    if grid is BOX:
        return g.fill_ghosts(f, grid, PERIODIC, stagger=at)
    f = g.fill_ghosts(f, grid, g.BC.periodic("x"), stagger=at)
    walls = g.BC(y=g.Dirichlet(0.0) if at in "VX" else g.Neumann())
    return g.fill_ghosts(f, grid, walls, stagger=at)


def metric(grid: Any) -> dict[str, Any]:
    """The lengths of the faces and the areas of the cells, on the full array."""

    def full(x: Any) -> Any:
        return np.broadcast_to(np.asarray(x), grid.shape)

    return {
        "dx_u": full(grid.spacing("x", at="U")),
        "dx_v": full(grid.spacing("x", at="V")),
        "dy": full(grid.spacing("y")),
        "area": full(grid.spacing("x")) * full(grid.spacing("y")),
    }


def shallow_water(grid: Any, f0: float) -> tuple[Any, ...]:
    """A random state, divergent, with its mass fluxes and its PV."""
    h = fill(2.0 + RNG.random(grid.shape), grid, "T")
    u = fill(RNG.normal(size=grid.shape), grid, "U")
    v = fill(RNG.normal(size=grid.shape), grid, "V")
    m = metric(grid)

    hu = np.zeros(grid.shape)
    hv = np.zeros(grid.shape)
    hu[:, :-1] = u[:, :-1] * 0.5 * (h[:, :-1] + h[:, 1:])
    hv[:-1] = v[:-1] * 0.5 * (h[:-1] + h[1:])
    hu, hv = fill(hu, grid, "U"), fill(hv, grid, "V")
    # the mass and the circulation around a corner, per area of four half cells
    mass = np.zeros(grid.shape)
    ha = h * m["area"]
    mass[:-1, :-1] = 0.25 * (ha[:-1, :-1] + ha[1:, :-1] + ha[:-1, 1:] + ha[1:, 1:])
    f = f0 * (1.0 + 0.3 * RNG.normal(size=grid.shape))
    q = np.zeros(grid.shape)
    q[:-1, :-1] = (circulation(u, v, m)[:-1, :-1] + f[:-1, :-1]) / mass[:-1, :-1]
    q = g.fill_ghosts(
        q, grid, PERIODIC if grid is BOX else g.BC.periodic("x"), stagger="X"
    )
    return h, u, v, hu, hv, q


def circulation(u: Any, v: Any, m: dict[str, Any]) -> Any:
    """``∮ u·dl`` around every corner."""
    out = np.zeros(u.shape)
    vy, ux = v * m["dy"], u * m["dx_u"]
    out[:-1, :-1] = (vy[:-1, 1:] - vy[:-1, :-1]) - (ux[1:, :-1] - ux[:-1, :-1])
    return out


def power(grid: Any, hu: Any, hv: Any, du: Any, dv: Any) -> tuple[Any, Any]:
    """The work of a force on the mass flux, over the sum of its parts."""
    m = metric(grid)
    a = (hu * du * m["dx_u"] * m["dy"])[INNER]
    b = (hv * dv * m["dx_v"] * m["dy"])[INNER]
    return a.sum() + b.sum(), np.abs(a).sum() + np.abs(b).sum()


@pytest.mark.parametrize("f0", [0.0, 2.0])
@pytest.mark.parametrize("name", GRIDS)
def test_the_force_does_no_work(name: str, f0: float) -> None:
    grid = GRIDS[name]
    _, _, _, hu, hv, q = shallow_water(grid, f0)
    du, dv = g.vortex_force(q, hu, hv, grid, form="arakawa_lamb")
    work, scale = power(grid, hu, hv, du, dv)
    assert abs(work) < 1e-13 * scale
    if grid is BOX and f0:
        # the two forms of Sadourny, for reference
        work, scale = power(grid, hu, hv, *g.vortex_force(q, hu, hv, grid))
        assert abs(work) < 1e-13 * scale
        z = g.vortex_force(q, hu, hv, grid, form="enstrophy")
        work, scale = power(grid, hu, hv, *z)
        assert abs(work) > 1e-4 * scale


def enstrophy_rate(form: str) -> tuple[float, float]:
    """``d/dt ∑ ½ h q²`` of the shallow-water equations in the box."""
    grid, m = BOX, metric(BOX)
    h, u, v, hu, hv, q = shallow_water(grid, 2.0)
    du, dv = g.vortex_force(q, hu, hv, grid, form=form)
    kinetic = fill(g.kinetic_energy(u, v, grid) + G * h * _ring(h), grid, "T")
    du = fill(du - g.difference(kinetic, grid, axis="x", src="T", dst="U"), grid, "U")
    dv = fill(dv - g.difference(kinetic, grid, axis="y", src="T", dst="V"), grid, "V")
    dh = fill(-g.divergence(hu, hv, grid), grid, "T")
    dmass = np.zeros(grid.shape)
    da = dh * m["area"]
    dmass[:-1, :-1] = 0.25 * (da[:-1, :-1] + da[1:, :-1] + da[:-1, 1:] + da[1:, 1:])
    # d(½ h q²) = q dζ - ½ q² dh
    rate = (q * circulation(du, dv, m) - 0.5 * q**2 * dmass)[INNER]
    return float(rate.sum()), float(np.abs(rate).sum())


def _ring(h: Any) -> Any:
    """One inside, zero on the ring, as the diagnostics write."""
    out = np.zeros(h.shape)
    out[INNER] = 1.0
    return out


def test_potential_enstrophy() -> None:
    """With the work above: the one form that keeps both, with divergence."""
    for form in ("arakawa_lamb", "enstrophy"):
        total, scale = enstrophy_rate(form)
        assert abs(total) < 1e-12 * scale
    total, scale = enstrophy_rate("energy")
    assert abs(total) > 1e-4 * scale


@pytest.mark.parametrize("name", GRIDS)
def test_uniform_vorticity(name: str) -> None:
    """Every form is ``q`` times the mean of the four cross velocities."""
    grid = GRIDS[name]
    u, v = RNG.normal(size=(2, *grid.shape))
    q = np.full(grid.shape, 1.7)
    want_u = 1.7 * g.interpolate(v, grid, src="V", dst="U")
    want_v = -1.7 * g.interpolate(u, grid, src="U", dst="V")
    forms = ("energy", "enstrophy", "arakawa_lamb") if grid is BOX else ("energy",)
    for form in forms:
        du, dv = g.vortex_force(q, u, v, grid, form=form)
        np.testing.assert_allclose(du, want_u, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(dv, want_v, rtol=1e-11, atol=1e-12)


def test_the_forms_of_advect_momentum() -> None:
    """The vortex force of the vorticity, less the gradient of K."""
    u, v = RNG.normal(size=(2, *BOX.shape))
    zeta = g.curl(u, v, BOX)
    inner = (slice(2, -2), slice(2, -2))
    for form in ("energy", "enstrophy"):
        u_t = g.interpolate(u, BOX, src="U", dst="T")
        v_t = g.interpolate(v, BOX, src="V", dst="T")
        kinetic = 0.5 * (u_t**2 + v_t**2)
        fu, fv = g.vortex_force(zeta, u, v, BOX, form=form)
        du, dv = g.advect_momentum(u, v, BOX, form=form)
        gx = g.difference(kinetic, BOX, axis="x", src="T", dst="U")
        gy = g.difference(kinetic, BOX, axis="y", src="T", dst="V")
        np.testing.assert_allclose(du[inner], (fu - gx)[inner], rtol=1e-10)
        np.testing.assert_allclose(dv[inner], (fv - gy)[inner], rtol=1e-10)
    du, dv = g.advect_momentum(u, v, BOX, form="arakawa_lamb")
    fu, fv = g.vortex_force(zeta, u, v, BOX, form="arakawa_lamb")
    kinetic = g.kinetic_energy(u, v, BOX)
    gx = g.difference(kinetic, BOX, axis="x", src="T", dst="U")
    np.testing.assert_array_equal(du[inner], (fu - gx)[inner])


def test_second_order() -> None:
    errors = []
    for n in (16, 32):
        grid = g.CartesianCGrid2D.from_interior(n, n, 2 * np.pi, 2 * np.pi)

        def at(point: str, grid: Any = grid) -> tuple[Any, Any]:
            return np.meshgrid(grid.coords("x", at=point), grid.coords("y", at=point))

        x, y = at("X")
        q = np.sin(x) * np.cos(y) + 2.0
        x, y = at("U")
        u = np.cos(x) * np.sin(y)
        want_u = (np.sin(x) * np.cos(y) + 2.0) * np.sin(x + y)
        x, y = at("V")
        v = np.sin(x + y)
        want_v = -(np.sin(x) * np.cos(y) + 2.0) * np.cos(x) * np.sin(y)
        du, dv = g.vortex_force(q, u, v, grid, form="arakawa_lamb")
        errors.append(
            max(np.abs(du - want_u)[INNER].max(), np.abs(dv - want_v)[INNER].max())
        )
    assert abs(np.log2(errors[0] / errors[1]) - 2.0) < 0.2


@pytest.mark.parametrize("form", ["energy", "arakawa_lamb"])
def test_a_basin_with_an_island(form: str) -> None:
    """The fluxes are zero on the coast, so the force does no work there either."""
    grid = g.CartesianCGrid2D.from_interior(14, 12, 1.0, 1.0)
    wet = np.zeros(grid.shape, dtype=bool)
    wet[2:-2, 2:-2] = RNG.random((10, 12)) > 0.15
    mask = g.Mask2D.from_grid(grid, wet)
    q, u, v = RNG.normal(size=(3, *grid.shape))
    q, u, v = (
        np.where(mask.wet(at=p), x, np.nan)
        for x, p in zip((q, u, v), "XUV", strict=True)
    )
    for x in (q, u, v):  # the ghost ring is the caller's
        x[[0, -1]] = x[:, [0, -1]] = 0.0
    du, dv = g.vortex_force(q, u, v, grid, form=form, mask=mask)
    assert np.isfinite(du).all() and np.isfinite(dv).all()
    a, b = (np.where(mask.wet(at=p), x, 0.0) for x, p in ((u, "U"), (v, "V")))
    work, scale = power(grid, a, b, du, dv)
    assert abs(work) < 1e-13 * scale


def test_levels_and_batches() -> None:
    grid = g.CartesianCGrid3D.from_interior(8, 7, 3, 1.0, 2.0, 1.0)
    flat = g.CartesianCGrid2D.from_interior(8, 7, 1.0, 2.0)
    q, u, v = RNG.normal(size=(3, *grid.shape))
    for form in ("energy", "enstrophy", "arakawa_lamb"):
        du, dv = g.vortex_force(q, u, v, grid, form=form)
        for k in (1, 2, 3):
            one = g.vortex_force(q[k], u[k], v[k], flat, form=form)
            np.testing.assert_allclose(du[k], one[0], rtol=1e-12)
            np.testing.assert_allclose(dv[k], one[1], rtol=1e-12)
        assert (du[[0, -1]] == 0).all()
        many = g.vortex_force(q, u, v, flat, form=form)  # levels as a batch
        np.testing.assert_allclose(many[0][1:-1], du[1:-1], rtol=1e-12)


def test_transforms_and_errors() -> None:
    q, u, v = (jnp.asarray(x) for x in RNG.normal(size=(3, *BOX.shape)))

    def loss(q: Any, u: Any, v: Any) -> Any:
        du, dv = g.vortex_force(q, u, v, BOX, form="arakawa_lamb")
        return (du**2 + dv**2).sum()

    for d in jax.jit(jax.grad(loss, (0, 1, 2)))(q, u, v):
        assert np.isfinite(d).all()
    small = [x.astype(jnp.float32) for x in (q, u, v)]
    for grid in (BOX, BAND):
        out = g.vortex_force(*small, grid, form="arakawa_lamb")
        assert all(x.dtype == jnp.float32 for x in out)
    with pytest.raises(ValueError, match="form must be 'energy', 'enstrophy' or"):
        g.vortex_force(q, u, v, BOX, form="blend")
    with pytest.raises(ValueError, match="form must be"):
        g.pv_flux(q, u, v, BOX, form="arakawa_lamb")
