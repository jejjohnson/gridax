"""Collocated grids: their metric, their constructors and the calculus on them."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import gridax as g
from gridax import CField, StaggerMismatchError, analysis
from gridax._src.analysis.differences import adaptive_difference
import jax
import jax.numpy as jnp
import numpy as np
import pytest

X = np.array([0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0])
Y = np.array([7.0, 5.0, 4.5, 3.0, 1.0, 0.0])
Z = np.array([0.0, 10.0, 30.0, 100.0])
GRIDS = {
    "1d": g.CartesianAGrid1D.from_interior(7, 3.5, x0=1.0),
    "1d irregular": g.CartesianAGrid1D.from_coords(X),
    "2d": g.CartesianAGrid2D.from_interior(7, 6, 3.5, 1.5),
    "2d periodic": g.CartesianAGrid2D.from_interior(7, 6, 3.5, 1.5, periodic=("x",)),
    "2d irregular": g.CartesianAGrid2D.from_coords(y=Y, x=X),
    "2d mixed": g.CartesianAGrid2D.from_coords(y=Y, x=X),
    "3d": g.CartesianAGrid3D.from_interior(7, 6, 4, 3.5, 1.5, 100.0),
    "3d irregular": g.CartesianAGrid3D.from_coords(z=Z, y=Y, x=X),
    "sphere": g.SphericalAGrid2D.from_interior(
        8, 6, (0.0, 360.0), (-60.0, 60.0), R=2.0, periodic=("x",)
    ),
    "sphere irregular": g.SphericalAGrid2D.from_coords(
        lat=[60.0, 30.0, 10.0, -20.0, -70.0], lon=[0.0, 10.0, 25.0, 60.0], R=2.0
    ),
}
CASES = [(name, axis) for name, grid in GRIDS.items() for axis in grid.axes]


def field(grid, seed=0, xp=np):
    return xp.asarray(np.random.default_rng(seed).normal(size=grid.shape))


class TestMetric:
    @pytest.mark.parametrize(("name", "axis"), CASES)
    def test_coords_and_spacing(self, name, axis):
        grid = GRIDS[name]
        coords = grid.coords(axis)
        assert coords.shape == (grid.shape[grid.axes.index(axis)],)
        if name.startswith("sphere"):
            return
        want = np.gradient(coords, edge_order=1)
        if grid._step(axis) is None:
            # the distance between the faces, which sit at the midpoints
            faces = 0.5 * (coords[1:] + coords[:-1])
            first, last = 2 * coords[0] - faces[0], 2 * coords[-1] - faces[-1]
            want = np.diff(np.concatenate([[first], faces, [last]]))
        np.testing.assert_allclose(
            np.ravel(np.broadcast_to(np.ravel(grid.spacing(axis)), coords.shape)), want
        )

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_spacing_broadcasts_against_a_field(self, name):
        grid = GRIDS[name]
        for axis in grid.axes:
            out = np.broadcast_shapes(np.shape(grid.spacing(axis)), grid.shape)
            assert out == grid.shape

    @pytest.mark.parametrize("name", ["sphere", "sphere irregular"])
    def test_the_sphere_is_the_arc_length(self, name):
        grid = GRIDS[name]
        lat, lon = grid.coords("y"), grid.coords("x")
        np.testing.assert_allclose(
            grid.spacing("x")[:, 1], 2.0 * np.cos(lat) * 0.5 * (lon[2] - lon[0])
        )
        np.testing.assert_allclose(
            np.ravel(np.broadcast_to(np.ravel(grid.spacing("y")), lat.shape))[1:-1],
            2.0 * 0.5 * (lat[2:] - lat[:-2]),
        )

    def test_steps_are_signed_and_areas_are_not(self):
        grid = GRIDS["2d irregular"]
        assert np.all(grid.spacing("y") < 0)
        assert np.all(grid.area() > 0)

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_there_are_t_points_only(self, name):
        grid = GRIDS[name]
        assert grid.ghost == 0
        with pytest.raises(ValueError, match="T points only"):
            grid.spacing("x", at="U")
        with pytest.raises(ValueError, match="axes"):
            grid.coords("q")


class TestConstructors:
    def test_interior_points_are_those_of_the_c_grid(self):
        a = g.CartesianAGrid2D.from_interior(7, 6, 3.5, 1.5, x0=0.5, y0=0.25)
        c = g.CartesianCGrid2D.from_interior(7, 6, 3.5, 1.5)
        for axis in "xy":
            np.testing.assert_allclose(a.coords(axis), c.coords(axis)[1:-1])
            np.testing.assert_allclose(a.spacing(axis), c.spacing(axis))
        s = g.SphericalAGrid2D.from_interior(8, 6, (0.0, 40.0), (-60.0, 60.0))
        sc = g.SphericalCGrid2D.from_interior(8, 6, (0.0, 40.0), (-60.0, 60.0))
        for axis in "xy":
            np.testing.assert_allclose(s.coords(axis), sc.coords(axis)[1:-1])
            want = np.broadcast_to(sc.spacing(axis), sc.shape)[1:-1, 1:-1]
            np.testing.assert_allclose(
                np.broadcast_to(s.spacing(axis), s.shape), want, rtol=1e-12
            )

    def test_affine_gives_the_cell_centres(self):
        grid = g.CartesianAGrid2D.from_affine((30, 0, 600, 0, -30, 900), (4, 5))
        np.testing.assert_array_equal(grid.coords("x"), 615.0 + 30.0 * np.arange(5))
        np.testing.assert_array_equal(grid.coords("y"), 885.0 - 30.0 * np.arange(4))

    @pytest.mark.parametrize("cls", [g.CartesianAGrid2D, g.SphericalAGrid2D])
    def test_a_rotated_affine_raises(self, cls):
        with pytest.raises(ValueError, match="rotated"):
            cls.from_affine((30, 1, 600, 0, -30, 900), (4, 5))

    @pytest.mark.parametrize(
        ("crs", "cls"),
        [
            (SimpleNamespace(is_geographic=True), g.SphericalAGrid2D),  # EPSG:4326
            (SimpleNamespace(is_geographic=False), g.CartesianAGrid2D),  # UTM
            (None, g.CartesianAGrid2D),
        ],
    )
    @pytest.mark.parametrize("shape", [(4, 5), (3, 4, 5)])
    def test_agrid_from_a_raster(self, crs, cls, shape):
        stub = SimpleNamespace(
            transform=(0.1, 0.0, 10.0, 0.0, -0.1, 50.0, 0.0, 0.0, 1.0),
            shape=shape,
            crs=crs,
        )
        grid = g.agrid_from(stub)
        assert type(grid) is cls
        assert grid.shape == (4, 5)
        scale = np.pi / 180.0 if cls is g.SphericalAGrid2D else 1.0
        np.testing.assert_allclose(grid.coords("y")[0], scale * 49.95)

    def test_agrid_from_an_object_without_a_crs(self):
        stub = SimpleNamespace(transform=(1, 0, 0, 0, 1, 0), shape=(4, 5))
        assert type(g.agrid_from(stub)) is g.CartesianAGrid2D

    @pytest.mark.parametrize(
        ("make", "match"),
        [
            (lambda: g.CartesianAGrid1D.from_coords([0.0, 1.0, 1.0]), "strictly"),
            (lambda: g.CartesianAGrid1D.from_coords([1.0]), "2 points"),
            (
                lambda: g.CartesianAGrid1D.from_interior(4, 1.0, periodic=("y",)),
                "periodic",
            ),
            (
                lambda: dataclasses.replace(GRIDS["1d irregular"], periodic=("x",)),
                "uniform",
            ),
        ],
    )
    def test_invalid_grids_raise(self, make, match):
        with pytest.raises(ValueError, match=match):
            make()


class TestPytree:
    @pytest.mark.parametrize("name", sorted(GRIDS))
    @pytest.mark.parametrize("jit", [jax.jit, eqx.filter_jit])
    def test_a_grid_crosses_jit(self, name, jit):
        grid = GRIDS[name]
        f = field(grid, xp=jnp)
        out = jit(lambda f, grid: g.difference(f, grid, axis="x"))(f, grid)
        np.testing.assert_allclose(
            out, g.difference(f, grid, axis="x"), rtol=1e-12, atol=1e-12
        )

    def test_one_compile_for_two_spacings(self):
        traces = []

        @eqx.filter_jit
        def diff(f, grid):
            traces.append(1)
            return g.difference(f, grid, axis="x")

        for length in (1.0, 2.0):
            grid = g.CartesianAGrid1D.from_interior(7, length)
            diff(field(grid, xp=jnp), grid)
        assert len(traces) == 1


class TestDifference:
    """The collocated calculus of ``gridax.analysis``, with the grid's metric."""

    @pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
    @pytest.mark.parametrize(("name", "axis"), CASES)
    @pytest.mark.parametrize("masked", [False, True])
    def test_is_the_analysis_kernel(self, name, axis, xp, masked):
        grid = GRIDS[name]
        if name == "sphere irregular":
            pytest.skip("the analysis kernel takes no metric with coords=")
        f = field(grid, xp=xp)
        position = grid.axes.index(axis) - len(grid.axes)
        kwargs = {"axis": position}
        if grid._step(axis) is None:
            kwargs["coords"] = grid.coords(axis)
        else:
            kwargs["step"] = grid.spacing(axis)
            kwargs["periodic"] = axis in grid.periodic
        if masked:
            wet = np.random.default_rng(3).random(grid.shape) > 0.2
            cls = {1: g.Mask1D, 2: g.Mask2D, 3: g.Mask3D}[len(grid.axes)]
            out = g.difference(f, grid, axis=axis, mask=cls.from_grid(grid, wet))
            want = adaptive_difference(f, mask=~wet, **kwargs)
        else:
            out = g.difference(f, grid, axis=axis)
            want = analysis.difference(f, **kwargs)
        np.testing.assert_array_equal(out, want)

    @pytest.mark.parametrize(("name", "axis"), CASES)
    def test_is_exact_on_the_coordinate(self, name, axis):
        grid = GRIDS[name]
        if axis in grid.periodic:
            pytest.skip("a periodic axis has no linear field")
        shape = [1] * len(grid.axes)
        shape[grid.axes.index(axis)] = -1
        f = np.broadcast_to(grid.coords(axis).reshape(shape), grid.shape)
        out = g.difference(f, grid, axis=axis)
        want = 1.0
        if name.startswith("sphere"):
            lat = np.cos(grid.coords("y"))[:, None]
            want = 1.0 / (2.0 * (lat if axis == "x" else 1.0))
        np.testing.assert_allclose(out, np.broadcast_to(want, grid.shape), rtol=1e-10)

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_float32_stays_float32(self, name):
        grid = GRIDS[name]
        f = field(grid, xp=jnp).astype(jnp.float32)
        for axis in grid.axes:
            assert g.difference(f, grid, axis=axis).dtype == jnp.float32

    def test_the_old_numpy_call(self):
        """``difference(f, axis=-1, step=dx)`` and ``coords=x`` (#503)."""
        uniform, irregular = GRIDS["1d"], GRIDS["1d irregular"]
        f = field(uniform)
        np.testing.assert_array_equal(
            g.difference(f, uniform, axis="x"),
            analysis.difference(f, axis=-1, step=0.5),
        )
        np.testing.assert_array_equal(
            g.difference(f, irregular, axis="x"),
            analysis.difference(f, axis=-1, coords=X),
        )

    def test_staggered_points_raise(self):
        grid = GRIDS["2d"]
        for kwargs in ({"src": "U"}, {"dst": "U"}):
            with pytest.raises(ValueError, match="T points only"):
                g.difference(field(grid), grid, axis="x", **kwargs)
        out = g.difference(field(grid), grid, axis="x", src="T", dst="T")
        np.testing.assert_array_equal(out, g.difference(field(grid), grid, axis="x"))

    def test_fields(self):
        grid = GRIDS["2d"]
        f = CField(field(grid), ("y", "x"))
        assert g.difference(f, grid, axis="y").loc == "T"
        with pytest.raises(StaggerMismatchError):
            g.difference(CField(field(grid), ("y", "x"), loc="U"), grid, axis="x")


class TestMaskAndReductions:
    def test_the_mask_has_t_points_only(self):
        grid = GRIDS["2d"]
        wet = np.random.default_rng(0).random(grid.shape) > 0.3
        mask = g.Mask2D.from_grid(grid, wet)
        np.testing.assert_array_equal(mask.wet(), wet)
        for at in "UVX":
            with pytest.raises(ValueError, match="T points only"):
                mask.wet(at=at)

    @pytest.mark.parametrize(
        "name", ["2d", "2d irregular", "sphere", "sphere irregular"]
    )
    def test_area_reductions_take_every_cell(self, name):
        grid = GRIDS[name]
        f = field(grid)
        np.testing.assert_allclose(g.area_sum(f, grid), np.sum(f * grid.area()))
        np.testing.assert_allclose(
            g.area_mean(f, grid), np.sum(f * grid.area()) / np.sum(grid.area())
        )

    @pytest.mark.parametrize("name", ["3d", "3d irregular"])
    def test_volume_reductions_take_every_cell(self, name):
        grid = GRIDS[name]
        f = field(grid)
        volume = np.broadcast_to(grid.area() * abs(grid.spacing("z")), grid.shape)
        np.testing.assert_allclose(g.volume_sum(f, grid), np.sum(f * volume))

    def test_interpolate_returns_the_field(self):
        grid = GRIDS["2d"]
        f = field(grid)
        np.testing.assert_array_equal(g.interpolate(f, grid), f)
