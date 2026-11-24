"""Strain, Okubo–Weiss, geostrophy and potential vorticity on collocated grids."""

from __future__ import annotations

import gridax as g
import jax
import jax.numpy as jnp
import numpy as np
import pytest

X = np.array([0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0])
Y = np.array([7.0, 5.0, 4.5, 3.0, 1.0, 0.0])
PLANES = {
    "uniform": g.CartesianAGrid2D.from_interior(7, 6, 3.5, 1.5),
    "periodic": g.CartesianAGrid2D.from_interior(7, 6, 3.5, 1.5, periodic=("x",)),
    "irregular": g.CartesianAGrid2D.from_coords(y=Y, x=X),
}
SPHERE = g.SphericalAGrid2D.from_coords(
    lat=[60.0, 30.0, 10.0, -20.0, -70.0, -85.0],
    lon=[0.0, 10.0, 25.0, 60.0, 80.0],
    R=2.0,
)
GRIDS = {**PLANES, "sphere": SPHERE}
VELOCITY_OPS = [
    "shear_strain",
    "tensor_strain",
    "strain_rate_squared",
    "okubo_weiss_parameter",
]
#: Off the edges, where the differences are centred.
INNER = (slice(1, -1), slice(1, -1))


def calls(grid, **kwargs):
    """Every diagnostic as a function of the fields ``u, v, h, f``."""
    out = {
        name: (lambda u, v, h, f, name=name: getattr(g, name)(u, v, grid, **kwargs))
        for name in VELOCITY_OPS
    }
    out["geostrophic_velocity"] = lambda u, v, h, f: g.geostrophic_velocity(
        h, grid, f=f, gravity=9.0, **kwargs
    )
    out["sw_potential_vorticity"] = lambda u, v, h, f: g.sw_potential_vorticity(
        u, v, h, f, grid, **kwargs
    )
    return out


ALL = list(calls(None))


def fields(grid, seed=0):
    rng = np.random.default_rng(seed)
    u, v, h, f = (rng.normal(size=grid.shape) for _ in range(4))
    return u, v, 5.0 + h, 3.0 + f


def rotation(step, axis, band=70.0):
    """A solid-body rotation of the unit sphere on a ``step``-degree grid.

    No strain, and ``ζ = 2 ω⃗ · r̂``.
    """
    grid = g.SphericalAGrid2D.from_interior(
        round(90 / step), round(2 * band / step), (0.0, 90.0), (-band, band), R=1.0
    )
    lam, phi = grid.coords("x")[None, :], grid.coords("y")[:, None]
    a, b, c = axis
    # the components of u⃗: u = ω⃗ · north, v = -ω⃗ · east
    u = c * np.cos(phi) - np.sin(phi) * (a * np.cos(lam) + b * np.sin(lam))
    v = np.broadcast_to(a * np.sin(lam) - b * np.cos(lam), grid.shape)
    up = np.cos(phi) * (a * np.cos(lam) + b * np.sin(lam)) + c * np.sin(phi)
    return grid, u, v, 2.0 * np.broadcast_to(up, grid.shape)


AXES = {
    "polar": (0.0, 0.0, 1.0),
    "tilted": (np.sin(0.65) * np.cos(0.3), np.sin(0.65) * np.sin(0.3), np.cos(0.65)),
}


class TestPlane:
    @pytest.mark.parametrize("name", ["uniform", "irregular"])
    def test_a_linear_flow_is_exact(self, name):
        grid = PLANES[name]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        u, v = 0.3 * x + 0.7 * y, 0.2 * x - 0.5 * y
        want = {
            "shear_strain": 0.2 + 0.7,
            "tensor_strain": 0.3 + 0.5,
            "strain_rate_squared": 0.9**2 + 0.8**2,
            "okubo_weiss_parameter": 0.9**2 + 0.8**2 - 0.5**2,
        }
        for op, value in want.items():
            np.testing.assert_allclose(getattr(g, op)(u, v, grid), value, rtol=1e-12)

    def test_pure_strain(self):
        grid = PLANES["irregular"]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        u, v = 0.4 * x, -0.4 * y
        np.testing.assert_allclose(g.tensor_strain(u, v, grid), 0.8, rtol=1e-12)
        np.testing.assert_allclose(g.shear_strain(u, v, grid), 0.0, atol=1e-12)
        out = g.okubo_weiss_parameter(u, v, grid)
        np.testing.assert_allclose(out, 4 * 0.4**2, rtol=1e-12)

    @pytest.mark.parametrize("name", ["uniform", "irregular"])
    def test_geostrophy_of_a_linear_height_is_exact(self, name):
        grid = PLANES[name]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        f = np.full(grid.shape, 2.0)
        u, v = g.geostrophic_velocity(0.3 * x - 0.5 * y, grid, f=f, gravity=9.0)
        np.testing.assert_allclose(u, 9.0 / 2.0 * 0.5, rtol=1e-12)
        np.testing.assert_allclose(v, 9.0 / 2.0 * 0.3, rtol=1e-12)

    def test_potential_vorticity(self):
        grid = PLANES["irregular"]
        u, v, h, f = fields(grid)
        want = (g.curl(u, v, grid) + f) / h
        np.testing.assert_allclose(
            g.sw_potential_vorticity(u, v, h, f, grid), want, rtol=1e-14
        )

    def test_the_strains_of_levels(self):
        grid = g.CartesianAGrid3D.from_coords(z=[0.0, 1.0, 3.0], y=Y, x=X)
        u, v, *_ = fields(grid)
        for op in ("shear_strain", "tensor_strain"):
            out = getattr(g, op)(u, v, grid)
            want = [getattr(g, op)(u[k], v[k], PLANES["irregular"]) for k in range(3)]
            np.testing.assert_allclose(out, np.stack(want), rtol=1e-14)

    def test_the_interior_of_the_c_grid_strain(self):
        """Averaged to T, a C-grid's shear strain is the collocated one of ū, v̄."""
        c = g.CartesianCGrid2D.from_interior(8, 8, 8.0, 8.0)
        x, y = np.meshgrid(np.arange(10) - 0.5, np.arange(10) - 0.5)
        # quadratic, so that both are exact away from the edges
        u, v = (x + 0.5) * y, x**2 - 0.5 * (y + 0.5) ** 2
        a = c.collocated()
        u_t, v_t = x * y, x**2 - 0.5 * y**2
        got = g.tensor_strain(u_t[1:-1, 1:-1], v_t[1:-1, 1:-1], a)[INNER]
        want = g.tensor_strain(u, v, c)[1:-1, 1:-1][INNER]
        np.testing.assert_allclose(got, want, atol=1e-12)


class TestSphere:
    @pytest.mark.parametrize("axis", sorted(AXES))
    def test_rotation_has_no_strain(self, axis):
        worst = []
        for step in (4.0, 2.0, 1.0):
            grid, u, v, _ = rotation(step, AXES[axis])
            worst.append(
                max(
                    np.abs(getattr(g, op)(u, v, grid)[INNER]).max()
                    for op in ("shear_strain", "tensor_strain")
                )
            )
        # second order, at 1 degree
        assert worst[-1] < 2e-4
        np.testing.assert_allclose(np.divide(worst[:-1], worst[1:]), 4.0, atol=0.4)

    @pytest.mark.parametrize("axis", sorted(AXES))
    def test_okubo_weiss_of_rotation(self, axis):
        worst = []
        for step in (4.0, 2.0, 1.0):
            grid, u, v, zeta = rotation(step, AXES[axis])
            error = g.okubo_weiss_parameter(u, v, grid) + zeta**2
            worst.append(np.abs(error[INNER]).max())
        np.testing.assert_allclose(np.divide(worst[:-1], worst[1:]), 4.0, atol=0.4)

    @pytest.mark.parametrize(("step", "tolerance"), [(1.0, 4.1e-4), (0.5, 1.1e-4)])
    def test_okubo_weiss_of_polar_rotation(self, step, tolerance):
        grid, u, v, zeta = rotation(step, AXES["polar"])
        ratio = g.okubo_weiss_parameter(u, v, grid) / -(zeta**2)
        spinning = np.abs(zeta) > 0.2
        np.testing.assert_allclose(ratio[INNER][spinning[INNER]], 1.0, atol=tolerance)

    def test_geostrophy_is_second_order(self):
        worst = []
        for step in (4.0, 2.0, 1.0):
            grid = g.SphericalAGrid2D.from_interior(
                round(90 / step), round(40 / step), (0.0, 90.0), (20.0, 60.0), R=2.0
            )
            lam, phi = np.meshgrid(grid.coords("x"), grid.coords("y"))
            f = 3.0 * np.sin(phi)
            u, v = g.geostrophic_velocity(
                np.cos(phi) * np.sin(2 * lam), grid, f=f, gravity=9.0
            )
            want_u = 9.0 / (f * 2.0) * np.sin(phi) * np.sin(2 * lam)
            want_v = 9.0 / (f * 2.0) * 2 * np.cos(2 * lam)
            worst.append(
                max(
                    np.abs((u - want_u)[INNER]).max(),
                    np.abs((v - want_v)[INNER]).max(),
                )
            )
        np.testing.assert_allclose(np.divide(worst[:-1], worst[1:]), 4.0, atol=0.4)

    def test_a_row_at_a_pole_is_nan(self):
        grid = g.SphericalAGrid2D.from_coords(
            lat=[-90.0, -45.0, 0.0, 45.0, 90.0], lon=[0.0, 10.0, 20.0, 30.0]
        )
        u, v, *_ = fields(grid)
        for op in VELOCITY_OPS:
            out = getattr(g, op)(u, v, grid)
            np.testing.assert_array_equal(
                np.isnan(out).any(axis=-1), [True, False, False, False, True]
            )


class TestZeroDivisors:
    @pytest.mark.parametrize("name", ["irregular", "sphere"])
    def test_no_coriolis_is_nan(self, name):
        grid = GRIDS[name]
        _, _, h, f = fields(grid)
        f[2] = 0.0
        for out in g.geostrophic_velocity(h, grid, f=f):
            np.testing.assert_array_equal(np.isnan(out), f == 0.0)

    @pytest.mark.parametrize("name", ["irregular", "sphere"])
    def test_no_thickness_is_nan(self, name):
        grid = GRIDS[name]
        u, v, h, f = fields(grid)
        h[1, 3] = 0.0
        out = g.sw_potential_vorticity(u, v, h, f, grid)
        np.testing.assert_array_equal(np.isnan(out), h == 0.0)

    def test_gradients_elsewhere_are_finite(self):
        grid = GRIDS["irregular"]
        u, v, h, f = fields(grid)
        f[2], h[1, 3] = 0.0, 0.0

        def speed(h):
            # what the caller keeps of it
            u, v = g.geostrophic_velocity(h, grid, f=f)
            return jnp.where(f == 0, 0.0, u**2 + v**2).sum()

        def vorticity(h):
            q = g.sw_potential_vorticity(u, v, h, f, grid)
            return jnp.where(h == 0, 0.0, q).sum()

        for loss in (speed, vorticity):
            assert np.isfinite(jax.grad(loss)(jnp.asarray(h))).all()


@pytest.mark.parametrize("op", ALL)
@pytest.mark.parametrize("name", sorted(GRIDS))
class TestEveryOne:
    def test_numpy_jax_and_jit_agree(self, name, op):
        grid = GRIDS[name]
        args = fields(grid)
        call = calls(grid)[op]
        want = call(*args)
        assert all(isinstance(x, np.ndarray) for x in jax.tree.leaves(want))
        on_jax = [jnp.asarray(x) for x in args]
        for got in (call(*on_jax), jax.jit(call)(*on_jax)):
            jax.tree.map(
                lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11),
                got,
                want,
            )

    def test_float32_stays_float32(self, name, op):
        grid = GRIDS[name]
        for xp in (np, jnp):
            args = [xp.asarray(x, dtype=xp.float32) for x in fields(grid)]
            for out in jax.tree.leaves(calls(grid)[op](*args)):
                assert out.dtype == np.float32

    def test_batches(self, name, op):
        grid = GRIDS[name]
        first, second = fields(grid, 1), fields(grid, 2)
        args = [np.stack(pair) for pair in zip(first, second, strict=True)]
        out = calls(grid)[op](*args)
        for k in range(2):
            want = calls(grid)[op](*(x[k] for x in args))
            jax.tree.map(
                lambda a, b, k=k: np.testing.assert_allclose(a[k], b, rtol=1e-13),
                out,
                want,
            )

    def test_is_the_method_of_gridops(self, name, op):
        grid = GRIDS[name]
        u, v, h, f = fields(grid)
        method = getattr(g.GridOps(grid), op)
        if op == "geostrophic_velocity":
            got = method(h, f=f, gravity=9.0)
        elif op == "sw_potential_vorticity":
            got = method(u, v, h, f)
        else:
            got = method(u, v)
        jax.tree.map(np.testing.assert_array_equal, got, calls(grid)[op](u, v, h, f))

    def test_an_all_wet_mask_changes_nothing(self, name, op):
        grid = GRIDS[name]
        mask = g.Mask2D.from_grid(grid, np.ones(grid.shape, dtype=bool))
        args = fields(grid)
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-13),
            calls(grid, mask=mask)[op](*args),
            calls(grid)[op](*args),
        )

    def test_dry_cells_are_nan_and_never_read(self, name, op):
        grid = GRIDS[name]
        wet = np.ones(grid.shape, dtype=bool)
        wet[2:4, 2] = False
        mask = g.Mask2D.from_grid(grid, wet)
        call = calls(grid, mask=mask)[op]
        outs = [
            call(*(np.where(wet, x, dry) for x in fields(grid)))
            for dry in (0.0, 1e3, np.nan)
        ]
        for out in jax.tree.leaves(outs):
            np.testing.assert_array_equal(np.isnan(out), ~wet)
        for other in outs[1:]:
            jax.tree.map(np.testing.assert_array_equal, other, outs[0])

    def test_gradients_are_finite(self, name, op):
        grid = GRIDS[name]
        wet = np.ones(grid.shape, dtype=bool)
        wet[2:4, 2] = False
        mask = g.Mask2D.from_grid(grid, wet)
        call = calls(grid, mask=mask)[op]

        def loss(*args):
            return sum(
                (jnp.where(wet, x, 0.0) ** 2).sum()
                for x in jax.tree.leaves(call(*args))
            )

        args = [jnp.asarray(np.where(wet, x, np.nan)) for x in fields(grid)]
        for gradient in jax.grad(loss, (0, 1, 2, 3))(*args):
            assert np.isfinite(np.asarray(gradient)[wet]).all()
