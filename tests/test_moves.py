"""`difference` and `interpolate`: every staggered move, on every C-grid."""

from __future__ import annotations

import gridax as g
from gridax._src.grid.points import point_stagger
from gridax._src.operators.triples import PAIRS, TRIPLES
import jax
import jax.numpy as jnp
import numpy as np
import pytest

GRIDS = {
    "c1": g.CartesianCGrid1D.from_interior(8, 4.0),
    "c2": g.CartesianCGrid2D.from_interior(8, 6, 4.0, 3.0),
    "c3": g.CartesianCGrid3D.from_interior(8, 6, 5, 4.0, 3.0, 10.0),
    "s2": g.SphericalCGrid2D.from_interior(8, 6, (0.0, 40.0), (-60.0, 60.0), R=2.0),
    "s3": g.SphericalCGrid3D.from_interior(
        8, 6, 5, (0.0, 40.0), (-60.0, 60.0), 10.0, R=2.0
    ),
}
MASKS = {1: g.Mask1D, 2: g.Mask2D, 3: g.Mask3D}
DIFFERENCES = [
    pytest.param(name, *triple, id=f"{name}-{'-'.join(triple)}")
    for name, grid in GRIDS.items()
    for triple in sorted(TRIPLES[grid.axes])
]
AVERAGES = [
    pytest.param(name, *pair, id=f"{name}-{'-'.join(pair)}")
    for name, grid in GRIDS.items()
    for pair in sorted(PAIRS[grid.axes])
]


def field(grid, seed=0, xp=jnp, lead=()):
    return xp.asarray(np.random.default_rng(seed).normal(size=(*lead, *grid.shape)))


def wet_mask(grid, seed=1):
    wet = np.random.default_rng(seed).random(grid.shape) > 0.3
    return MASKS[len(grid.axes)].from_grid(grid, wet)


def linear(grid, axis, at):
    """The coordinate along ``axis`` at the points ``at``, on the whole grid."""
    shape = [1] * len(grid.axes)
    shape[grid.axes.index(axis)] = -1
    return np.broadcast_to(grid.coords(axis, at=at).reshape(shape), grid.shape)


def inner(grid):
    return (..., *(slice(1, -1),) * len(grid.axes))


class TestTables:
    def test_the_two_dimensional_moves(self):
        assert set(TRIPLES["y", "x"]) == {
            ("x", "T", "U"),
            ("x", "U", "T"),
            ("x", "V", "X"),
            ("x", "X", "V"),
            ("y", "T", "V"),
            ("y", "V", "T"),
            ("y", "U", "X"),
            ("y", "X", "U"),
        }
        assert len(PAIRS["y", "x"]) == 12

    def test_every_method_of_the_bindings_is_a_move(self):
        """Nothing the classes offer is missing from the tables."""
        for cls, axes in (
            (g.Difference1D, ("x",)),
            (g.Difference2D, ("y", "x")),
            (g.Difference3D, ("z", "y", "x")),
        ):
            for name in dir(cls):
                if name.startswith("diff_"):
                    _, axis, src, _, dst = name.split("_")
                    assert (axis, src, dst) in TRIPLES[axes]
        for cls, axes in (
            (g.Interpolation1D, ("x",)),
            (g.Interpolation2D, ("y", "x")),
            (g.Interpolation3D, ("z", "y", "x")),
        ):
            for name in dir(cls):
                if "_to_" in name and not name.startswith("_"):
                    assert tuple(name.split("_to_")) in PAIRS[axes]


class TestDifference:
    @pytest.mark.parametrize(("name", "axis", "src", "dst"), DIFFERENCES)
    def test_is_exact_on_the_coordinate(self, name, axis, src, dst):
        """δ of the coordinate is 1 in units of the spacing at dst."""
        grid = GRIDS[name]
        f = linear(grid, axis, src)
        out = g.difference(f, grid, axis=axis, src=src, dst=dst)
        step = f.take(1, grid.axes.index(axis)) - f.take(0, grid.axes.index(axis))
        want = np.broadcast_to(step.flat[0] / grid.spacing(axis, at=dst), grid.shape)
        np.testing.assert_allclose(out[inner(grid)], want[inner(grid)], rtol=1e-10)

    @pytest.mark.parametrize(("name", "axis", "src", "dst"), DIFFERENCES)
    def test_the_ghost_ring_and_the_dry_points_are_zero(self, name, axis, src, dst):
        grid = GRIDS[name]
        mask = wet_mask(grid)
        out = g.difference(field(grid), grid, axis=axis, src=src, dst=dst, mask=mask)
        wet = mask.wet(at=dst, kind="strict" if dst == "X" else "default")
        assert not np.any(np.asarray(out)[~np.asarray(wet)])
        ring = np.ones(grid.shape, dtype=bool)
        ring[inner(grid)] = False
        assert not np.any(np.asarray(out)[ring])

    @pytest.mark.parametrize(("name", "axis", "src", "dst"), DIFFERENCES)
    def test_numpy_and_jit_agree(self, name, axis, src, dst):
        grid = GRIDS[name]

        def fn(f):
            return g.difference(
                f, grid, axis=axis, src=src, dst=dst, mask=wet_mask(grid)
            )

        want = fn(field(grid, xp=np))
        np.testing.assert_allclose(fn(field(grid)), want, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(
            jax.jit(fn)(field(grid)), want, rtol=1e-14, atol=1e-14
        )

    def test_leading_axes_are_batch_axes(self):
        for name, grid in GRIDS.items():
            f = field(grid, lead=(2, 3))
            out = g.difference(f, grid, axis="x", src="T", dst="U")
            one = g.difference(f[1, 2], grid, axis="x", src="T", dst="U")
            # XLA vectorises the division of a batch differently: an ulp
            np.testing.assert_allclose(out[1, 2], one, rtol=4e-16, err_msg=name)

    def test_next_to_a_pole_a_zonal_difference_is_nan(self):
        grid = g.SphericalCGrid2D.from_interior(8, 6, (0.0, 360.0), (-90.0, 90.0))
        out = g.difference(field(grid), grid, axis="x", src="V", dst="X")
        assert bool(jnp.isnan(out[-2, 1:-1]).all())  # the X row on the north pole


class TestInterpolate:
    @pytest.mark.parametrize(("name", "src", "dst"), AVERAGES)
    def test_is_exact_on_the_coordinates(self, name, src, dst):
        grid = GRIDS[name]
        for axis in grid.axes:
            out = g.interpolate(linear(grid, axis, src), grid, src=src, dst=dst)
            np.testing.assert_allclose(
                out[inner(grid)],
                linear(grid, axis, dst)[inner(grid)],
                rtol=1e-12,
                atol=1e-12,
            )

    @pytest.mark.parametrize(("name", "src", "dst"), AVERAGES)
    def test_the_dry_points_are_zero(self, name, src, dst):
        grid = GRIDS[name]
        mask = wet_mask(grid)
        out = g.interpolate(field(grid), grid, src=src, dst=dst, mask=mask)
        wet = mask.wet(at=dst, kind="strict" if dst == "X" else "default")
        assert not np.any(np.asarray(out)[~np.asarray(wet)])

    @pytest.mark.parametrize(("src", "via", "dst"), [("U", "T", "V"), ("T", "U", "X")])
    def test_a_diagonal_move_is_two_moves(self, src, via, dst):
        grid = GRIDS["c2"]
        f = field(grid)
        first = g.interpolate(f, grid, src=src, dst=via)
        two = g.interpolate(first, grid, src=via, dst=dst)
        one = g.interpolate(f, grid, src=src, dst=dst)
        # the first move leaves the ghost ring at zero: compare inside it
        core = (slice(2, -2), slice(2, -2))
        np.testing.assert_allclose(one[core], two[core], rtol=1e-13)


class TestBindings:
    @pytest.mark.parametrize(("name", "axis", "src", "dst"), DIFFERENCES)
    @pytest.mark.parametrize("masked", [False, True])
    def test_a_difference_method_is_the_function(self, name, axis, src, dst, masked):
        grid = GRIDS[name]
        rank = len(grid.axes)
        mask = wet_mask(grid) if masked else None
        if name.startswith("s"):
            axis_name = {"x": "lon", "y": "lat"}.get(axis, axis)
            cls = {2: g.SphericalDifference2D, 3: g.SphericalDifference3D}[rank]
        else:
            axis_name = axis
            cls = {1: g.Difference1D, 2: g.Difference2D, 3: g.Difference3D}[rank]
        method = getattr(
            cls(grid=grid, mask=mask), f"diff_{axis_name}_{src}_to_{dst}", None
        )
        if method is None:
            pytest.skip("the binding has no such method")
        f = field(grid)
        np.testing.assert_array_equal(
            method(f), g.difference(f, grid, axis=axis, src=src, dst=dst, mask=mask)
        )

    @pytest.mark.parametrize(("name", "src", "dst"), AVERAGES)
    def test_an_interpolation_method_is_the_function(self, name, src, dst):
        grid = GRIDS[name]
        cls = {1: g.Interpolation1D, 2: g.Interpolation2D, 3: g.Interpolation3D}
        mask = wet_mask(grid)
        binding = cls[len(grid.axes)](grid=grid, mask=mask)
        method = getattr(binding, f"{src}_to_{dst}", None)
        if method is None:
            pytest.skip("the binding has no such method")
        f = field(grid)
        np.testing.assert_array_equal(
            method(f), g.interpolate(f, grid, src=src, dst=dst, mask=mask)
        )


class TestPoints:
    def test_a_stagger_is_a_point(self):
        grid = GRIDS["c2"]
        f = field(grid)
        for axis, src, dst in TRIPLES[grid.axes]:
            np.testing.assert_array_equal(
                g.difference(
                    f,
                    grid,
                    axis=axis,
                    src=point_stagger(src),
                    dst=point_stagger(dst),
                ),
                g.difference(f, grid, axis=axis, src=src, dst=dst),
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"axis": "x", "src": "T", "dst": "V"}, "x/T/U"),
            ({"axis": "z", "src": "T", "dst": "W"}, "x/T/U"),
            ({"axis": "x", "src": "T"}, "src= and dst="),
            ({"axis": "x", "src": "T", "dst": (("x", "left"),)}, "stagger"),
        ],
    )
    def test_a_move_that_does_not_exist_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            g.difference(field(GRIDS["c2"]), GRIDS["c2"], **kwargs)

    def test_an_average_that_does_not_exist_raises(self):
        grid = GRIDS["c3"]
        with pytest.raises(ValueError, match="U/W is not available"):
            g.interpolate(field(grid), grid, src="U", dst="W")
