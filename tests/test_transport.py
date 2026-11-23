"""The canonical transport, rotation and diagnostic functions (#491).

That every function equals the class it replaces is checked by the golden
fixtures through the classes, which call the functions.
"""

from __future__ import annotations

import gridax as g
from gridax.schemes import TVD, WENO, Biased, Centred, Upwind
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._schemes import SCHEMES

C2 = g.CartesianCGrid2D.from_interior(16, 14, 4.0, 3.0)
S2 = g.SphericalCGrid2D.from_interior(16, 14, (0.0, 40.0), (-60.0, 60.0), R=2.0)
C3 = g.CartesianCGrid3D.from_interior(12, 11, 4, 4.0, 3.0, 2.0)
S3 = g.SphericalCGrid3D.from_interior(12, 11, 4, (0.0, 40.0), (-60.0, 60.0), 2.0, R=2.0)
C1 = g.CartesianCGrid1D.from_interior(24, 4.0)
GRIDS = {"c1": C1, "c2": C2, "c3": C3, "s2": S2, "s3": S3}
MASKS = {1: g.Mask1D, 2: g.Mask2D, 3: g.Mask3D}
EXTRA = {"centred4": Centred(4), "biased3": Biased(3, "left"), "upwind7": Upwind(7)}
ALL = {**SCHEMES, **EXTRA}


def fields(grid, n, seed=0, lead=()):
    rng = np.random.default_rng(seed)
    return [jnp.asarray(rng.standard_normal((*lead, *grid.shape))) for _ in range(n)]


def island(grid):
    """The quickstart's basin: closed walls and an island."""
    wet = np.ones(grid.shape, dtype=bool)
    for axis in range(-min(len(grid.axes), 2), 0):
        edge = np.moveaxis(wet, axis, 0)
        edge[0] = edge[-1] = False
    if len(grid.axes) == 1:
        wet[9:11] = False
    else:
        wet[..., 5:8, 6:9] = False
    return MASKS[len(grid.axes)].from_mask(wet)


def velocities(grid, mask=None):
    u, v = fields(grid, 2, seed=1)
    if mask is not None:
        u = u * mask.wet(at="U")
        v = None if len(grid.axes) == 1 else v * mask.wet(at="V")
    return u, (None if len(grid.axes) == 1 else v)


class TestAdvect:
    @pytest.mark.parametrize("scheme", ALL.values(), ids=ALL)
    @pytest.mark.parametrize("name", GRIDS)
    def test_the_wet_tracer_budget_is_zero(self, name, scheme):
        # ∑ wet A dq/dt = 0: a closed wall leaves the ring next to the edge
        # unwritten, so nothing may flow between it and the written cells
        grid = GRIDS[name]
        mask = island(grid)
        (q,) = fields(grid, 1)
        ring = np.ones(grid.shape)
        ring[(..., *(slice(2, -2),) * min(len(grid.axes), 2))] = 0.0
        u, v = velocities(grid, mask)
        u = u * (1.0 - _touches(ring, -1))
        v = None if v is None else v * (1.0 - _touches(ring, -2))
        area = grid.area() if len(grid.axes) > 1 else 1.0
        budget = g.advect(q, u, v, grid, scheme=scheme, mask=mask) * area
        assert float(jnp.abs(budget).sum()) > 0.0
        assert abs(float(budget.sum())) <= 1e-12 * float(jnp.abs(budget).sum())

    @pytest.mark.parametrize("scheme", ALL.values(), ids=ALL)
    @pytest.mark.parametrize("name", ["c1", "c2", "c3", "s2"])
    def test_land_values_are_never_read(self, name, scheme):
        grid = GRIDS[name]
        mask = island(grid)
        (q,) = fields(grid, 1)
        u, v = velocities(grid, mask)
        dry = ~np.asarray(mask.wet(), dtype=bool)
        want = g.advect(q, u, v, grid, scheme=scheme, mask=mask)
        got = g.advect(jnp.where(dry, jnp.nan, q), u, v, grid, scheme=scheme, mask=mask)
        np.testing.assert_array_equal(got, want)
        assert (np.asarray(got)[dry] == 0.0).all()

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("name", GRIDS)
    def test_leading_axes_are_batch_axes(self, name, masked):
        grid = GRIDS[name]
        mask = island(grid) if masked else None
        q, u, v = fields(grid, 3, lead=(2, 3))
        if len(grid.axes) == 1:
            v = None
        for scheme in (WENO(5), Centred(4), TVD()):
            got = g.advect(q, u, v, grid, scheme=scheme, mask=mask)
            for b in np.ndindex(2, 3):
                vb = None if v is None else v[b]
                one = g.advect(q[b], u[b], vb, grid, scheme=scheme, mask=mask)
                # XLA rounds the end of a vectorised loop differently
                np.testing.assert_allclose(got[b], one, rtol=1e-13, atol=1e-13)

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("scheme", [Centred(2), Upwind(3), WENO(5)], ids=repr)
    def test_no_gather_and_no_scatter(self, scheme, masked):
        mask = island(C2) if masked else None
        q, u, v = fields(C2, 3)

        def fn(q, u, v):
            return g.advect(q, u, v, C2, scheme=scheme, mask=mask)

        text = jax.jit(fn).lower(q, u, v).compile().as_text()
        assert "gather(" not in text
        assert "scatter(" not in text

    @pytest.mark.parametrize("name", GRIDS)
    @pytest.mark.parametrize("masked", [False, True])
    def test_the_classes_are_the_function(self, name, masked):
        grid = GRIDS[name]
        mask = island(grid) if masked else None
        prefix = "Spherical" if name[0] == "s" else ""
        op = getattr(g, f"{prefix}Advection{len(grid.axes)}D")(grid, mask)
        q, u, v = fields(grid, 3)
        args = (q, u) if len(grid.axes) == 1 else (q, u, v)
        # Upwind(11) is wider than the cached masks
        for scheme in (*ALL.values(), Upwind(11)):
            want = g.advect(
                *args, *(None,) * (3 - len(args)), grid, scheme=scheme, mask=mask
            )
            np.testing.assert_array_equal(op(*args, scheme=scheme), want)

    def test_is_minus_the_divergence_of_the_fluxes(self):
        q, u, v = fields(C2, 3)
        scheme = WENO(5)
        fe = g.reconstruct(q, u, C2, axis="x", scheme=scheme)
        fn = g.reconstruct(q, v, C2, axis="y", scheme=scheme)
        out = g.advect(q, u, v, C2, scheme=scheme)
        want = -g.divergence(fe, fn, C2)
        np.testing.assert_allclose(out[2:-2, 2:-2], want[2:-2, 2:-2], rtol=1e-12)

    def test_a_string_is_not_a_scheme(self):
        q, u, v = fields(C2, 3)
        with pytest.raises(TypeError, match=r"gridax\.schemes"):
            g.advect(q, u, v, C2, scheme="weno5")

    def test_the_second_component(self):
        q, u, v = fields(C2, 3)
        with pytest.raises(TypeError, match="v is required"):
            g.advect(q, u, None, C2, scheme=Upwind(1))
        q, u, v = fields(C1, 3)
        with pytest.raises(TypeError, match="v is None"):
            g.advect(q, u, v, C1, scheme=Upwind(1))

    def test_the_wall(self):
        q, u, v = fields(C2, 3)
        with pytest.raises(ValueError, match="wall"):
            g.advect(q, u, v, C2, scheme=Upwind(1), wall="periodic")
        with pytest.raises(NotImplementedError, match="mask"):
            g.advect(q, u, v, C2, scheme=Upwind(1), wall="open", mask=island(C2))

    def test_a_field_gives_a_field(self):
        q, u, v = fields(C2, 3)
        dims = ("y", "x")
        out = g.advect(
            g.CField(q, dims, loc="T"),
            g.CField(u, dims, loc="U"),
            g.CField(v, dims, loc="V"),
            C2,
            scheme=WENO(5),
        )
        assert out.loc == "T"
        np.testing.assert_array_equal(out.data, g.advect(q, u, v, C2, scheme=WENO(5)))

    def test_the_point_of_a_flux_is_its_axis(self):
        q, u, _ = fields(C2, 3)
        dims = ("y", "x")
        field = g.CField(q, dims, loc="T")
        flux = g.reconstruct(
            field, g.CField(u, dims, loc="V"), C2, axis="y", scheme=Upwind(1)
        )
        assert flux.loc == "V"
        with pytest.raises(g.StaggerMismatchError):
            g.reconstruct(
                field, g.CField(u, dims, loc="U"), C2, axis="y", scheme=Upwind(1)
            )


def _touches(ring, axis):
    """1 on the faces along ``axis`` of a cell of ``ring``."""
    return np.maximum(ring, np.roll(ring, -1, axis=axis))


class TestDiffuse:
    @pytest.mark.parametrize("name", ["c2", "c3", "s2", "s3"])
    @pytest.mark.parametrize("masked", [False, True])
    def test_conserves(self, name, masked):
        grid = GRIDS[name]
        mask = island(grid) if masked else None
        f, k = fields(grid, 2)
        area = grid.area()
        for out in (
            g.diffuse(f, grid, kappa=0.3, mask=mask),
            g.diffuse(f, grid, kappa=jnp.abs(k), mask=mask),
            g.biharmonic(f, grid, kappa4=0.3, mask=mask),
        ):
            x = out * area
            assert abs(float(x.sum())) <= 1e-12 * float(jnp.abs(x).sum())

    @pytest.mark.parametrize("name", ["c2", "c3", "s2", "s3"])
    def test_leading_axes_are_batch_axes(self, name):
        grid = GRIDS[name]
        mask = island(grid)
        f, k = fields(grid, 2, lead=(3,))
        for fn in (
            lambda f, k: g.diffuse(f, grid, kappa=k, mask=mask),
            lambda f, k: g.biharmonic(f, grid, kappa4=0.1, mask=mask),
        ):
            got = fn(f, k)
            for b in range(3):
                np.testing.assert_allclose(got[b], fn(f[b], k[b]), rtol=1e-12)

    def test_a_diffusivity_of_one_is_the_laplacian(self):
        (f,) = fields(C2, 1)
        got = g.diffuse(f, C2, kappa=1.0)
        want = g.laplacian(f, C2, at="T")
        np.testing.assert_allclose(got[2:-2, 2:-2], want[2:-2, 2:-2], rtol=1e-11)

    def test_a_two_dimensional_diffusivity_on_levels(self):
        f, k = fields(C3, 2)
        plane = jnp.abs(k[2])
        got = g.diffuse(f, C3, kappa=plane)
        whole = g.diffuse(f, C3, kappa=jnp.broadcast_to(plane, f.shape))
        np.testing.assert_array_equal(got, whole)

    def test_an_open_wall_is_cartesian_and_unmasked(self):
        (f,) = fields(S2, 1)
        with pytest.raises(NotImplementedError, match="sphere"):
            g.diffuse(f, S2, kappa=1.0, wall="open")
        (f,) = fields(C2, 1)
        with pytest.raises(NotImplementedError, match="mask"):
            g.diffuse(f, C2, kappa=1.0, wall="open", mask=island(C2))

    def test_fields(self):
        f, k = fields(C2, 2)
        dims = ("y", "x")
        out = g.diffuse(
            g.CField(f, dims, loc="T"), C2, kappa=g.CField(k, dims, loc="T")
        )
        assert out.loc == "T"
        np.testing.assert_array_equal(out.data, g.diffuse(f, C2, kappa=k))
        with pytest.raises(g.StaggerMismatchError):
            g.diffuse(g.CField(f, dims, loc="U"), C2, kappa=1.0)


class TestCoriolis:
    @pytest.mark.parametrize("name", ["c2", "c3"])
    def test_does_no_work(self, name):
        grid = GRIDS[name]
        mask = island(grid)
        u, v = velocities(grid, mask)
        u, v = (c * jnp.zeros(grid.shape).at[..., 1:-1, 1:-1].set(1.0) for c in (u, v))
        (f,) = fields(C2 if name == "c2" else C3, 1)
        du, dv = g.coriolis(u, v, grid, f=f if name == "c2" else f[0], mask=mask)
        inner = (..., slice(1, -1), slice(1, -1))
        if name == "c3":
            inner = (slice(1, -1), slice(1, -1), slice(1, -1))
        work = (u * du + v * dv)[inner]
        assert abs(float(work.sum())) <= 1e-12 * float(jnp.abs(work).sum())

    def test_a_number_and_a_profile(self):
        u, v = fields(C2, 2)
        y = C2.coords("y")
        profile = jnp.asarray(1.0 + 0.1 * y)[:, None]
        whole = jnp.broadcast_to(profile, C2.shape)
        for a, b in [(0.5, jnp.full(C2.shape, 0.5)), (profile, whole)]:
            got, want = (g.coriolis(u, v, C2, f=f) for f in (a, b))
            np.testing.assert_array_equal(got[0], want[0])
            np.testing.assert_array_equal(got[1], want[1])

    def test_fields_and_a_plane_of_f(self):
        # f has no levels, and broadcasts over those of the velocity
        u, v = fields(C3, 2)
        (f,) = fields(C2, 1)
        f = f[:13, :14]
        zyx = ("z", "y", "x")
        du, dv = g.coriolis(
            g.CField(u, zyx, loc="U"),
            g.CField(v, zyx, loc="V"),
            C3,
            f=g.CField(f, ("y", "x"), loc="T"),
        )
        assert (du.loc, dv.loc) == ("U", "V")
        want = g.coriolis(u, v, C3, f=f)
        np.testing.assert_array_equal(du.data, want[0])
        np.testing.assert_array_equal(dv.data, want[1])


class TestDiagnostics:
    def test_kinetic_energy_on_levels(self):
        u, v = fields(C3, 2)
        got = g.kinetic_energy(u, v, C3)
        for k in range(1, C3.Nz - 1):
            np.testing.assert_array_equal(
                got[k], g.kinetic_energy(u[k], v[k], C3.horizontal_grid())
            )
        np.testing.assert_array_equal(got[0], 0.0)

    def test_kinetic_energy_on_cells(self):
        grid = g.CartesianAGrid2D.from_interior(6, 5, 1.0, 1.0)
        u, v = fields(grid, 2)
        np.testing.assert_array_equal(g.kinetic_energy(u, v, grid), 0.5 * (u**2 + v**2))
        wet = np.ones(grid.shape)
        wet[2, 3] = 0.0
        out = g.kinetic_energy(u, v, grid, mask=g.Mask2D.from_grid(grid, wet))
        assert np.isnan(out[2, 3]) and np.isfinite(out).sum() == out.size - 1

    def test_strain_on_levels(self):
        u, v = fields(C3, 2)
        for fn in (g.shear_strain, g.tensor_strain):
            got = fn(u, v, C3)
            for k in range(1, C3.Nz - 1):
                np.testing.assert_array_equal(
                    got[k], fn(u[k], v[k], C3.horizontal_grid())
                )

    def test_strain_and_rotation_split_the_gradient(self):
        u, v = fields(C2, 2)
        shear, vorticity = g.shear_strain(u, v, C2), g.curl(u, v, C2)
        dvdx = g.difference(v, C2, axis="x", src="V", dst="X")
        np.testing.assert_allclose(shear + vorticity, 2.0 * dvdx, atol=1e-12)

    def test_the_jacobian_conserves(self):
        a, b = (
            jnp.zeros(C2.shape).at[1:-1, 1:-1].set(x[1:-1, 1:-1]) for x in fields(C2, 2)
        )
        jac = g.arakawa_jacobian(a, b, C2)
        assert jac.shape == C2.shape
        for weight in (a, b):
            total = weight * jac
            assert abs(float(total.sum())) <= 1e-11 * float(jnp.abs(total).sum())

    @pytest.mark.parametrize(
        "name", ["diffusion_2d", "total_energy", "total_enstrophy", "weno_5pts"]
    )
    def test_the_old_names_are_gone(self, name):
        assert not hasattr(g, name)
