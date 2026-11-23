"""`divergence`, `curl`, `laplacian`, `gradient`, `grad_perp` on every grid type."""

from __future__ import annotations

import gridax as g
from gridax import CField, StaggerMismatchError, analysis
from gridax._src.grid.points import POINT_AXES
import jax
import jax.numpy as jnp
import numpy as np
import pytest

C2 = g.CartesianCGrid2D.from_interior(16, 14, 2.0, 3.0)
C3 = g.CartesianCGrid3D.from_interior(10, 8, 4, 2.0, 3.0, 5.0)
S2 = g.SphericalCGrid2D.from_interior(16, 14, (0.0, 40.0), (-60.0, 60.0), R=2.0)
S3 = g.SphericalCGrid3D.from_interior(10, 8, 4, (0.0, 40.0), (-60.0, 60.0), 5.0, R=2.0)
CGRIDS = {"c2": C2, "c3": C3, "s2": S2, "s3": S3}
Y = np.array([7.0, 5.0, 4.5, 3.0, 1.0, 0.0])
X = np.array([0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0])
AGRIDS = {
    "uniform": g.CartesianAGrid2D.from_interior(7, 6, 3.5, 1.5),
    "periodic": g.CartesianAGrid2D.from_interior(7, 6, 3.5, 1.5, periodic=("x",)),
}
MASKS = {2: g.Mask2D, 3: g.Mask3D}


def field(grid, seed=0, xp=jnp, lead=()):
    return xp.asarray(np.random.default_rng(seed).normal(size=(*lead, *grid.shape)))


def wet_mask(grid, seed=1):
    wet = np.random.default_rng(seed).random(grid.shape) > 0.2
    return MASKS[len(grid.axes)].from_grid(grid, wet)


def linear(grid, axis, at):
    shape = [1] * len(grid.axes)
    shape[grid.axes.index(axis)] = -1
    return np.broadcast_to(grid.coords(axis, at=at).reshape(shape), grid.shape)


class TestCGridIsExact:
    @pytest.mark.parametrize("name", ["c2", "c3"])
    def test_on_linear_fields(self, name):
        grid = CGRIDS[name]
        inner = (..., *(slice(1, -1),) * len(grid.axes))
        zero = np.zeros(grid.shape)

        def close(out, want):
            np.testing.assert_allclose(out[inner], want, atol=1e-11)

        close(g.divergence(linear(grid, "x", "U"), zero, grid), 1.0)
        close(g.divergence(zero, linear(grid, "y", "V"), grid), 1.0)
        close(g.curl(zero, linear(grid, "x", "V"), grid), 1.0)
        close(g.curl(linear(grid, "y", "U"), zero, grid), -1.0)
        dx, dy = g.gradient(2.0 * linear(grid, "x", "T") + linear(grid, "y", "T"), grid)
        close(dx, 2.0)
        close(dy, 1.0)
        for at in ("X", "T"):
            psi = 2.0 * linear(grid, "x", at) + linear(grid, "y", at)
            u, v = g.grad_perp(psi, grid, at=at)
            close(u, -1.0)
            close(v, 2.0)

    @pytest.mark.parametrize("at", ["T", "U", "V", "X"])
    def test_the_laplacian_of_a_quadratic_at_every_point(self, at):
        f = linear(C2, "x", at) ** 2 + 3.0 * linear(C2, "y", at) ** 2
        out = g.laplacian(f, C2, at=at)
        np.testing.assert_allclose(out[1:-1, 1:-1], 8.0, rtol=1e-9)

    @pytest.mark.parametrize("at", ["T", "U", "V", "X"])
    @pytest.mark.slow
    def test_the_spherical_laplacian_converges_at_every_point(self, at):
        """sin(φ) is an eigenfunction: ∇² sin φ = -2 sin φ / R²."""
        errors = []
        for n in (16, 32):
            grid = g.SphericalCGrid2D.from_interior(
                n, n, (0.0, 40.0), (-60.0, 60.0), R=2.0
            )
            f = np.sin(linear(grid, "y", at))
            out = g.laplacian(f, grid, at=at)
            errors.append(np.abs(out + 2.0 * f / 4.0)[1:-1, 1:-1].max())
        assert errors[0] / errors[1] > 3.5

    @pytest.mark.parametrize("name", sorted(CGRIDS))
    def test_the_velocity_of_a_streamfunction_is_non_divergent(self, name):
        grid = CGRIDS[name]
        psi = field(grid)
        u, v = g.grad_perp(psi, grid)
        div = g.divergence(u, v, grid)
        core = (..., *(slice(2, -2),) * 2)
        scale = float(jnp.abs(u).max()) / float(jnp.min(jnp.abs(grid.spacing("y"))))
        assert float(jnp.abs(div[core]).max()) < 1e-12 * scale

    @pytest.mark.parametrize("name", sorted(CGRIDS))
    def test_the_curl_of_a_gradient_is_zero(self, name):
        grid = CGRIDS[name]
        dx, dy = g.gradient(field(grid), grid)
        out = g.curl(dx, dy, grid)
        core = (..., *(slice(2, -2),) * 2)
        scale = float(jnp.abs(dx).max()) / float(jnp.min(jnp.abs(grid.spacing("y"))))
        assert float(jnp.abs(out[core]).max()) < 1e-12 * scale

    def test_summation_by_parts_on_a_closed_periodic_grid(self):
        """∑ h ∇·u A = -∑ u·∇h A."""
        wrap = g.BC(x=g.Periodic(), y=g.Periodic())
        h = g.fill_ghosts(field(C2, 0), C2, wrap)
        u = g.fill_ghosts(field(C2, 1), C2, wrap, stagger="U")
        v = g.fill_ghosts(field(C2, 2), C2, wrap, stagger="V")
        dx, dy = g.gradient(h, C2)
        left = g.area_sum(h * g.divergence(u, v, C2), C2)
        right = g.area_sum(u * dx + v * dy, C2)
        np.testing.assert_allclose(left, -right, rtol=1e-11)

    def test_the_laplacian_is_the_divergence_of_the_gradient(self):
        for grid in (C2, S2):
            h = field(grid)
            dx, dy = g.gradient(h, grid)
            core = (slice(2, -2), slice(2, -2))
            np.testing.assert_allclose(
                g.laplacian(h, grid)[core],
                g.divergence(dx, dy, grid)[core],
                rtol=1e-9,
                atol=1e-9,
            )


class TestBindings:
    """The classes are the functions, bit for bit, masked and not."""

    @pytest.mark.parametrize("masked", [False, True])
    def test_every_method(self, masked):
        for name, grid in CGRIDS.items():
            mask = wet_mask(grid) if masked else None
            f, u, v = (field(grid, seed) for seed in range(3))
            sphere, deep = name.startswith("s"), name.endswith("3")
            rank = "3D" if deep else "2D"
            pairs = []
            if sphere:
                pairs = [
                    (getattr(g, f"SphericalDivergence{rank}")(grid, mask), "div"),
                    (
                        getattr(g, f"SphericalVorticity{rank}")(
                            grid, mask
                        ).relative_vorticity,
                        "curl",
                    ),
                    (getattr(g, f"SphericalLaplacian{rank}")(grid, mask), "lap"),
                ]
            else:
                diff = getattr(g, f"Difference{rank}")(grid, mask)
                pairs = [
                    (diff.divergence, "div"),
                    (diff.laplacian, "lap"),
                    (
                        getattr(g, f"Vorticity{rank}")(grid, mask).relative_vorticity,
                        "curl",
                    ),
                ]
            want = {
                "div": (g.divergence(u, v, grid, mask=mask), (u, v)),
                "curl": (g.curl(u, v, grid, mask=mask), (u, v)),
                "lap": (g.laplacian(f, grid, mask=mask), (f,)),
            }
            for method, kind in pairs:
                out, args = want[kind]
                np.testing.assert_array_equal(method(*args), out, err_msg=name)

    def test_the_rest_of_difference_2d(self):
        diff = g.Difference2D(C2, wet_mask(C2))
        f, u, v = (field(C2, seed) for seed in range(3))
        np.testing.assert_array_equal(
            diff.curl(u, v), g.curl(u, v, C2, mask=wet_mask(C2))
        )
        for a, b in zip(
            diff.grad_perp(f),
            g.grad_perp(f, C2, at="T", mask=wet_mask(C2)),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
        one = g.CartesianCGrid1D.from_interior(8, 1.0)
        np.testing.assert_array_equal(
            g.Difference1D(one).laplacian(field(one)), g.laplacian(field(one), one)
        )

    @pytest.mark.parametrize("name", sorted(CGRIDS))
    def test_leading_axes_are_batch_axes(self, name):
        grid = CGRIDS[name]
        f, u, v = (field(grid, seed, lead=(2, 3)) for seed in range(3))
        for fn, args in (
            (g.divergence, (u, v)),
            (g.curl, (u, v)),
            (g.laplacian, (f,)),
        ):
            out = fn(*args, grid)
            one = fn(*(a[1, 2] for a in args), grid)
            np.testing.assert_allclose(out[1, 2], one, rtol=1e-12, atol=1e-12)


class TestCollocated:
    """The collocated calculus of ``gridax.analysis``, with max|Δ| = 0."""

    @pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("name", sorted(AGRIDS))
    def test_is_the_analysis_calculus(self, name, masked, xp):
        grid = AGRIDS[name]
        f, u, v = (field(grid, seed, xp) for seed in range(3))
        dy, dx = grid.spacing("y"), grid.spacing("x")
        kwargs = {"periodic_x": "x" in grid.periodic}
        mask, pre = None, ""
        if masked:
            wet = np.random.default_rng(3).random(grid.shape) > 0.2
            mask, pre = g.Mask2D.from_grid(grid, wet), "adaptive_"
            kwargs["mask"] = ~wet

        def old(what):
            return getattr(analysis, f"{pre}{what}_2d")

        np.testing.assert_array_equal(
            g.divergence(u, v, grid, mask=mask),
            old("divergence")(u, v, dy, dx, **kwargs),
        )
        np.testing.assert_array_equal(
            g.curl(u, v, grid, mask=mask), old("curl")(u, v, dy, dx, **kwargs)
        )
        np.testing.assert_array_equal(
            g.laplacian(f, grid, mask=mask), old("laplacian")(f, dy, dx, **kwargs)
        )
        d_y, d_x = old("gradient")(f, dy, dx, **kwargs)
        got = g.gradient(f, grid, mask=mask)
        np.testing.assert_array_equal(got[0], d_x)
        np.testing.assert_array_equal(got[1], d_y)
        u_psi, v_psi = g.grad_perp(f, grid, mask=mask)
        np.testing.assert_array_equal(u_psi, -d_y)
        np.testing.assert_array_equal(v_psi, d_x)

    @pytest.mark.parametrize("masked", [False, True])
    def test_the_laplacian_in_one_dimension(self, masked):
        for grid in (
            g.CartesianAGrid1D.from_interior(7, 3.5),
            g.CartesianAGrid1D.from_coords(X),
        ):
            f = field(grid, xp=np)
            kwargs = (
                {"coords": grid.coords("x")}
                if grid._step("x") is None
                else {"step": grid.spacing("x")}
            )
            if masked:
                wet = np.array([True, True, False, True, True, True, True])
                out = g.laplacian(f, grid, mask=g.Mask1D.from_grid(grid, wet))
                want = analysis.adaptive_laplacian_1d(f, mask=~wet, **kwargs)
            else:
                out = g.laplacian(f, grid)
                want = analysis.laplacian_1d(f, **kwargs)
            np.testing.assert_array_equal(out, want)

    def test_irregular_axes_and_levels(self):
        grid = g.CartesianAGrid3D.from_coords(z=[0.0, 10.0, 30.0], y=Y, x=X)
        x = np.broadcast_to(X, grid.shape)
        y = np.broadcast_to(Y[:, None], grid.shape)
        np.testing.assert_allclose(g.divergence(2.0 * x, 3.0 * y, grid), 5.0)
        np.testing.assert_allclose(g.curl(-y, x, grid), 2.0)
        np.testing.assert_allclose(g.laplacian(x**2 + y**2, grid), 4.0, rtol=1e-9)

    @pytest.mark.slow
    @pytest.mark.parametrize("irregular", [False, True])
    def test_the_sphere_converges(self, irregular):
        """Against the analytic calculus of sin φ cos λ, in second order inside."""
        errors = []
        for n in (32, 64):
            if irregular:
                t = np.linspace(-1.0, 1.0, n)
                grid = g.SphericalAGrid2D.from_coords(
                    lat=60.0 * (t + 0.1 * t**2),
                    lon=180.0 * (t + 0.1 * t**2),
                    R=2.0,
                )
            else:
                grid = g.SphericalAGrid2D.from_interior(
                    n, n, (0.0, 360.0), (-60.0, 60.0), R=2.0, periodic=("x",)
                )
            lam = grid.coords("x")[None, :]
            phi = grid.coords("y")[:, None]
            f = np.sin(phi) * np.cos(lam)
            u, v = np.cos(phi) * np.sin(lam), np.cos(phi) ** 2 * np.cos(lam)
            cos, R = np.cos(phi), 2.0
            want = {
                # 1/(R cos φ) [∂λ u + ∂φ(v cos φ)]
                "div": (cos * np.cos(lam) - 3.0 * cos**2 * np.sin(phi) * np.cos(lam))
                / (R * cos),
                # 1/(R cos φ) [∂λ v - ∂φ(u cos φ)]
                "curl": (
                    -(cos**2) * np.sin(lam) + 2.0 * cos * np.sin(phi) * np.sin(lam)
                )
                / (R * cos),
                # 1/(R² cos²φ) ∂λλ f + 1/R² (∂φφ f - tan φ ∂φ f)
                "lap": (-f / cos**2 - f - np.tan(phi) * cos * np.cos(lam)) / R**2,
            }
            got = {
                "div": g.divergence(u, v, grid),
                "curl": g.curl(u, v, grid),
                "lap": g.laplacian(f, grid),
            }
            # a fixed region, away from the one-sided edges
            core = (np.abs(phi) < np.deg2rad(40.0)) & (np.abs(lam - lam.mean()) < 2.0)
            errors.append({k: np.abs(got[k] - want[k])[core].max() for k in want})
        for k in errors[0]:
            order = np.log2(errors[0][k] / errors[1][k])
            assert order > 1.8, (k, order)

    def test_a_pole_is_nan(self):
        grid = g.SphericalAGrid2D.from_coords(
            lat=np.linspace(-90, 90, 7), lon=np.linspace(0, 315, 8), R=1.0
        )
        f = field(grid, xp=np)
        for out in (
            g.divergence(f, f, grid),
            g.curl(f, f, grid),
            g.laplacian(f, grid),
        ):
            assert np.all(np.isnan(out[[0, -1]]))
            assert np.all(np.isfinite(out[1:-1]))


class TestFields:
    def test_a_rotating_flow_by_its_points(self):
        psi = CField(field(C2), ("y", "x"), loc="X")
        u, v = g.grad_perp(psi, C2)
        assert (u.loc, v.loc) == ("U", "V")
        assert g.curl(u, v, C2).loc == "X"
        assert g.divergence(u, v, C2).loc == "T"
        assert g.laplacian(psi, C2).loc == "X"

    def test_a_streamfunction_at_t_takes_the_compact_stencil(self):
        psi = CField(field(C2), ("y", "x"), loc="T")
        for a, b in zip(
            g.grad_perp(psi, C2), g.grad_perp(psi.data, C2, at="T"), strict=True
        ):
            np.testing.assert_array_equal(a.data, b)

    def test_swapped_components_raise(self):
        u = CField(field(C2), ("y", "x"), loc="U")
        v = CField(field(C2), ("y", "x"), loc="V")
        for fn in (g.divergence, g.curl):
            with pytest.raises(StaggerMismatchError, match="u sits at V"):
                fn(v, u, C2)

    def test_a_plain_component_next_to_a_field_raises(self):
        u = CField(field(C2), ("y", "x"), loc="U")
        with pytest.raises(ValueError, match=r"f\.like"):
            g.divergence(u, field(C2), C2)

    def test_members_under_jit(self):
        dims = ("member", "y", "x")
        u = CField(field(C2, 0, lead=(3,)), dims, loc="U")
        v = CField(field(C2, 1, lead=(3,)), dims, loc="V")
        out = jax.jit(g.curl)(u, v, C2)
        assert (out.dims, out.loc) == (dims, "X")


class TestErrors:
    def test_a_streamfunction_at_a_face_raises(self):
        with pytest.raises(ValueError, match="X or T"):
            g.grad_perp(field(C2), C2, at="U")

    @pytest.mark.parametrize("at", sorted(set(POINT_AXES) - {"T", "W"}))
    def test_a_collocated_grid_has_t_points_only(self, at):
        grid = AGRIDS["uniform"]
        with pytest.raises(ValueError, match="T points only"):
            g.laplacian(field(grid), grid, at=at)

    @pytest.mark.parametrize("name", ["divergence_2d", "relative_vorticity_cgrid"])
    def test_the_old_functions_are_gone(self, name):
        assert not hasattr(g, name)
