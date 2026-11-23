"""`advect_vertical` and `diffuse_vertical`: the z terms of a tracer (#404)."""

from __future__ import annotations

from typing import Any

import gridax as g
from gridax.schemes import TVD, WENO, Centred, Scheme, Upwind
import jax
import jax.numpy as jnp
import numpy as np
import pytest

RNG = np.random.default_rng(404)
CARTESIAN = g.CartesianCGrid3D.from_interior(5, 4, 10, 1.0, 1.0, 2.0)
SPHERE = g.SphericalCGrid3D.from_interior(5, 4, 10, (0.0, 10.0), (0.0, 10.0), 2.0)
GRIDS = {"cartesian": CARTESIAN, "sphere": SPHERE}
STRETCHED = g.CartesianAGrid3D.from_coords(
    z=np.cumsum(0.5 + RNG.random(9)), y=np.arange(4.0), x=np.arange(5.0)
).staggered()
SCHEMES = [Upwind(1), Centred(2), TVD("van_leer"), WENO(5)]
INNER = (slice(1, -1),) * 3


def field(grid: Any) -> Any:
    return RNG.normal(size=grid.shape)


def along(grid: Any, values: Any) -> Any:
    """A profile of z on the whole grid."""
    return np.broadcast_to(np.asarray(values)[:, None, None], grid.shape)


def thickness(grid: Any) -> Any:
    step = np.asarray(grid.spacing("z"))
    return step[1:-1] if step.ndim else step


def plumax_advection(c: Any, w: Any, dz: float) -> Any:
    """plumax's kernel: ``w`` at T, averaged to the faces, upwind of first order."""
    face = 0.5 * (w[:-1] + w[1:])
    flux = face * np.where(face > 0.0, c[:-1], c[1:])
    out = np.zeros_like(c)
    out[1:-1] = -(flux[1:] - flux[:-1]) / dz
    return out


def plumax_diffusion(c: Any, kappa: Any, dz: float) -> Any:
    kappa = kappa if np.ndim(kappa) == 0 else 0.5 * (kappa[:-1] + kappa[1:])
    flux = kappa * (c[1:] - c[:-1]) / dz
    out = np.zeros_like(c)
    out[1:-1] = (flux[1:] - flux[:-1]) / dz
    return out


@pytest.mark.parametrize("name", GRIDS)
def test_it_is_plumax_on_its_layout(name: str) -> None:
    grid = GRIDS[name]
    c, w, kappa = field(grid), field(grid), np.abs(field(grid))
    dz = float(grid.spacing("z"))
    faces = np.zeros(grid.shape)  # w[k] on the top face of T[k], the floor too
    faces[:-1] = 0.5 * (w[:-1] + w[1:])
    np.testing.assert_allclose(
        g.advect_vertical(c, faces, grid, scheme=Upwind(1), wall="open")[INNER],
        plumax_advection(c, w, dz)[INNER],
        rtol=1e-12,
    )
    # the helper for a velocity at T, with the floor for the caller
    inside = g.interpolate(w, grid, src="T", dst="W")
    np.testing.assert_allclose(inside[INNER], faces[INNER], rtol=1e-14)
    for k in (0.3, kappa):
        np.testing.assert_allclose(
            g.diffuse_vertical(c, grid, kappa=k, wall="open")[INNER],
            plumax_diffusion(c, k, dz)[INNER],
            rtol=1e-11,
            atol=1e-12,
        )


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
@pytest.mark.parametrize("name", GRIDS)
def test_a_column_changes_by_what_crosses_its_ends(name: str, scheme: Scheme) -> None:
    grid = GRIDS[name]
    q, w, dz = field(grid), field(grid), float(grid.spacing("z"))

    def total(out: Any) -> Any:
        assert (out[0] == 0).all() and (out[-1] == 0).all()
        return (out * dz).sum(axis=0)[1:-1, 1:-1]

    closed = g.advect_vertical(q, w, grid, scheme=scheme)
    np.testing.assert_allclose(total(closed), 0.0, atol=1e-12)
    # an open end is first-order upwind of the ghost level
    lo = w[0] * np.where(w[0] >= 0, q[0], q[1])
    hi = w[-2] * np.where(w[-2] >= 0, q[-2], q[-1])
    out = g.advect_vertical(q, w, grid, scheme=scheme, wall="open")
    np.testing.assert_allclose(total(out), (lo - hi)[1:-1, 1:-1], atol=1e-12)
    # a periodic one returns what it takes
    out = g.advect_vertical(q, w, grid, scheme=scheme, bc=g.BC(z=g.Periodic()))
    np.testing.assert_allclose(total(out), 0.0, atol=1e-12)
    # and a wall that w does not cross is closed
    still = w.copy()
    still[0] = still[-2] = 0.0
    out = g.advect_vertical(q, still, grid, scheme=scheme, bc=g.BC(z=g.Neumann()))
    np.testing.assert_allclose(total(out), 0.0, atol=1e-12)


@pytest.mark.parametrize("grid", [CARTESIAN, SPHERE, STRETCHED], ids=lambda x: "")
def test_a_diffused_column_changes_by_what_crosses_its_ends(grid: Any) -> None:
    f, kappa = field(grid), np.abs(field(grid))
    for k in (0.4, kappa):
        closed = g.diffuse_vertical(f, grid, kappa=k)
        total = (closed[1:-1] * thickness(grid)).sum(axis=0)[1:-1, 1:-1]
        np.testing.assert_allclose(total, 0.0, atol=1e-12)
        out = g.diffuse_vertical(f, grid, kappa=k, wall="open")
        total = (out[1:-1] * thickness(grid)).sum(axis=0)
        face = k if np.ndim(k) == 0 else 0.5 * (k[1:] + k[:-1])
        gaps = np.asarray(grid.spacing("z", at="W"))
        gaps = gaps[:-1] if gaps.ndim else gaps
        flux = face * np.diff(f, axis=0) / gaps
        np.testing.assert_allclose(
            total[1:-1, 1:-1], (flux[-1] - flux[0])[1:-1, 1:-1], atol=1e-11
        )


@pytest.mark.parametrize("name", GRIDS)
def test_a_linear_profile(name: str) -> None:
    grid = GRIDS[name]
    q = along(grid, 2.0 * grid.coords("z"))
    for scheme in (Centred(2), Upwind(3), WENO(5)):
        out = g.advect_vertical(
            q,
            np.full(grid.shape, 3.0),
            grid,
            scheme=scheme,
            bc=g.BC(z=g.Extrapolation(1)),
        )
        np.testing.assert_allclose(out[INNER], -6.0, rtol=1e-9)
    z = along(grid, grid.coords("z"))
    out = g.diffuse_vertical(z**2, grid, kappa=0.5, wall="open")
    np.testing.assert_allclose(out[INNER], 1.0, rtol=1e-9)


def test_a_stretched_column() -> None:
    """The exact second difference on uneven levels: zero for a line."""
    z = along(STRETCHED, STRETCHED.coords("z"))
    out = g.diffuse_vertical(3.0 * z + 1.0, STRETCHED, kappa=0.5, wall="open")
    np.testing.assert_allclose(out, 0.0, atol=1e-11)
    with pytest.raises(NotImplementedError, match="advect_vertical"):
        g.advect_vertical(z, z, STRETCHED, scheme=Upwind(1))


def test_a_constant_in_a_flow_without_divergence() -> None:
    """The two parts cancel: w comes from continuity (`vertical_velocity`)."""
    grid = CARTESIAN
    u, v = field(grid), field(grid)
    w = g.vertical_velocity(u, v, grid)
    q = np.full(grid.shape, 7.0)
    flat = g.advect(q, u, v, grid, scheme=Upwind(1), wall="open")
    up = g.advect_vertical(q, w, grid, scheme=Upwind(1), wall="open")
    np.testing.assert_allclose((flat + up)[INNER], 0.0, atol=1e-10)


def test_observed_orders() -> None:
    errors: dict[int, list[float]] = {1: [], 5: []}
    for n in (16, 32):
        grid = g.CartesianCGrid3D.from_interior(2, 2, n, 1.0, 1.0, 1.0)
        dz = 1.0 / n
        faces = (np.arange(n + 3) - 1) * dz
        q = -np.diff(np.cos(2 * np.pi * faces)) / (2 * np.pi * dz)
        want = -np.diff(np.sin(2 * np.pi * faces)) / dz
        for order in errors:
            out = g.advect_vertical(
                along(grid, q),
                np.ones(grid.shape),
                grid,
                scheme=Upwind(order),
                bc=g.BC(z=g.Periodic()),
            )
            errors[order].append(np.abs(out[:, 1, 1] - want)[1:-1].max())
    for order, (a, b) in errors.items():
        assert abs(np.log2(a / b) - order) < 0.2


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
def test_topography(scheme: Scheme) -> None:
    grid = CARTESIAN
    depth = RNG.integers(1, 6, size=grid.shape[1:])
    wet = np.arange(grid.shape[0])[:, None, None] >= depth
    mask = g.Mask3D.from_grid(grid, wet)
    q, w = field(grid), field(grid)
    q = np.where(wet, q, np.nan)
    out = g.advect_vertical(q, w, grid, scheme=scheme, mask=mask)
    assert np.isfinite(out).all() and (out[~wet] == 0).all()
    np.testing.assert_allclose(out.sum(axis=0), 0.0, atol=1e-11)
    mixed = g.diffuse_vertical(q, grid, kappa=0.3, mask=mask)
    assert np.isfinite(mixed).all() and (mixed[~wet] == 0).all()
    np.testing.assert_allclose(mixed.sum(axis=0), 0.0, atol=1e-11)
    # a surface flux through an open lid over topography
    q = np.where(wet, 1.0, np.nan)
    q[-1] = 2.0
    mixed = g.diffuse_vertical(q, grid, kappa=0.3, mask=mask, wall="open")
    dz = float(grid.spacing("z"))
    np.testing.assert_allclose(mixed[-2, 1:-1, 1:-1], 0.3 / dz**2, rtol=1e-12)


def test_fields_batches_and_transforms() -> None:
    grid = CARTESIAN
    q, w = jnp.asarray(field(grid)), jnp.asarray(field(grid))
    want = g.advect_vertical(q, w, grid, scheme=WENO(3))
    c, cw = (g.CField(x, grid.axes, loc=p) for x, p in ((q, "T"), (w, "W")))
    out = g.advect_vertical(c, cw, grid, scheme=WENO(3))
    assert out.loc == "T"
    np.testing.assert_array_equal(out.data, want)
    with pytest.raises(g.StaggerMismatchError):
        g.advect_vertical(c, c, grid, scheme=WENO(3))
    many = jnp.stack([q, 2.0 * q])
    np.testing.assert_allclose(
        g.advect_vertical(many, jnp.stack([w, w]), grid, scheme=Upwind(1))[0],
        g.advect_vertical(q, w, grid, scheme=Upwind(1)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        g.diffuse_vertical(many, grid, kappa=0.2)[1],
        2.0 * g.diffuse_vertical(q, grid, kappa=0.2),
        rtol=1e-12,
    )

    def loss(q: Any, w: Any, k: Any) -> Any:
        a = g.advect_vertical(q, w, grid, scheme=WENO(3))
        return ((a + g.diffuse_vertical(q, grid, kappa=k)) ** 2).sum()

    for d in jax.jit(jax.grad(loss, (0, 1, 2)))(q, w, jnp.abs(q)):
        assert np.isfinite(d).all()
    small = g.diffuse_vertical(q.astype(jnp.float32), grid, kappa=jnp.abs(q))
    assert small.dtype == jnp.float32


def test_errors() -> None:
    q = field(CARTESIAN)
    with pytest.raises(ValueError, match="wall must be"):
        g.advect_vertical(q, q, CARTESIAN, scheme=Upwind(1), wall="shut")
    with pytest.raises(ValueError, match="wall must be"):
        g.diffuse_vertical(q, CARTESIAN, kappa=1.0, wall="shut")
    with pytest.raises(ValueError, match="drop wall="):
        g.advect_vertical(
            q, q, CARTESIAN, scheme=Upwind(1), wall="open", bc=g.BC(z=g.Periodic())
        )
    with pytest.raises(TypeError, match="scheme must be"):
        g.advect_vertical(q, q, CARTESIAN, scheme="upwind")
    with pytest.raises(NotImplementedError, match="diffuse_vertical"):
        g.diffuse_vertical(
            np.ones((6, 6)), g.CartesianCGrid2D.from_interior(4, 4, 1.0, 1.0), kappa=1.0
        )
