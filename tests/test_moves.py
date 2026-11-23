"""`difference` and `interpolate`: every staggered move, on every C-grid."""

from __future__ import annotations

import gridax as g
from gridax._src.grid.points import point_stagger
from gridax._src.operators.triples import PAIRS, TRIPLES
import gridax.analysis as ga
from gridax.schemes import TVD, WENO, Biased, Centred, Upwind
import gridax.stencils as gs
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.fixtures import bindings

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
            (bindings.Difference1D, ("x",)),
            (bindings.Difference2D, ("y", "x")),
            (bindings.Difference3D, ("z", "y", "x")),
        ):
            for name in dir(cls):
                if name.startswith("diff_"):
                    _, axis, src, _, dst = name.split("_")
                    assert (axis, src, dst) in TRIPLES[axes]
        for cls, axes in (
            (bindings.Interpolation1D, ("x",)),
            (bindings.Interpolation2D, ("y", "x")),
            (bindings.Interpolation3D, ("z", "y", "x")),
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


WIDE = [Centred(4), Centred(6), Biased(2, "left"), Biased(3, "right")]


class TestSchemes:
    @pytest.mark.parametrize(("name", "axis", "src", "dst"), DIFFERENCES)
    def test_a_two_point_scheme_is_the_default(self, name, axis, src, dst):
        grid = GRIDS[name]
        f = field(grid)
        kw = {"axis": axis, "src": src, "dst": dst}
        np.testing.assert_array_equal(
            g.difference(f, grid, scheme=Centred(2, "point"), **kw),
            g.difference(f, grid, **kw),
        )

    def test_two_points_lower_to_the_same_program(self):
        grid = GRIDS["c2"]

        def lowered(**kw):
            fn = lambda f: g.difference(f, grid, axis="x", src="T", dst="U", **kw)
            return jax.jit(fn).lower(field(grid)).as_text()

        assert lowered(scheme=Centred(2)) == lowered()

    @pytest.mark.parametrize("scheme", WIDE, ids=repr)
    @pytest.mark.parametrize(("name", "axis", "src", "dst"), DIFFERENCES)
    def test_is_exact_on_a_polynomial(self, name, axis, src, dst, scheme):
        # of the degree of the order, on every face: the wall stencils shift
        grid = GRIDS[name]
        degree = min(scheme.order, 3)
        lo, hi = scheme.footprint(staggered=True)
        if hi - lo > grid.shape[grid.axes.index(axis)] - 1:
            pytest.skip("the axis is shorter than the stencil")
        x, y = (linear(grid, axis, at) for at in (src, dst))
        f, want = x**degree, degree * y ** (degree - 1)
        got = g.difference(f, grid, axis=axis, src=src, dst=dst, scheme=scheme)
        if isinstance(grid, g.SphericalCGrid2D | g.SphericalCGrid3D) and axis != "z":
            cos = np.cos(linear(grid, "y", dst)) if axis == "x" else 1.0
            want = want / (grid.R * cos)
        np.testing.assert_allclose(
            got[inner(grid)],
            np.broadcast_to(want, grid.shape)[inner(grid)],
            rtol=1e-8,
            atol=1e-9,
        )

    @pytest.mark.parametrize("scheme", WIDE, ids=repr)
    @pytest.mark.parametrize(("name", "src", "dst"), AVERAGES)
    def test_interpolation_is_exact_on_a_polynomial(self, name, src, dst, scheme):
        grid = GRIDS[name]
        degree = min(scheme.order - 1, 3)

        def poly(at):
            out = 1.0
            for axis in grid.axes:
                out = out * (1.0 + linear(grid, axis, at) ** degree)
            return out

        got = g.interpolate(poly(src), grid, src=src, dst=dst, scheme=scheme)
        np.testing.assert_allclose(
            got[inner(grid)], poly(dst)[inner(grid)], rtol=1e-8, atol=1e-9
        )

    @pytest.mark.parametrize("points", [32, 64])
    def test_fourth_order(self, points):
        # sin(kx) on one wavelength, faces 3 cells from an edge
        def error(n):
            grid = g.CartesianCGrid1D.from_interior(n, 2 * np.pi)
            f = np.sin(grid.coords("x"))
            got = g.difference(f, grid, axis="x", src="T", dst="U", scheme=Centred(4))
            want = np.cos(grid.coords("x", at="U"))
            return np.abs(got - want)[4:-4].max()

        # the faces left out move towards the edge as the grid refines, so the largest
        # value of the fifth derivative is in both
        assert error(points) / error(2 * points) == pytest.approx(16.0, rel=0.1)

    def test_a_diagonal_move_is_the_product_of_the_weights(self):
        grid = GRIDS["c2"]
        f = np.zeros(grid.shape)
        f[4, 4] = 1.0
        got = g.interpolate(f, grid, src="T", dst="X", scheme=Centred(4))
        line = np.array([-1.0, 9.0, 9.0, -1.0]) / 16.0
        np.testing.assert_allclose(got[2:6, 2:6], np.outer(line, line), rtol=1e-14)

    def test_is_the_layer_1_stencil(self):
        grid = GRIDS["c2"]
        f = field(grid, xp=np)
        got = g.difference(f, grid, axis="y", src="U", dst="X", scheme=Centred(4))
        raw = gs.staggered_difference(
            f, axis="y", side="fwd", ndim=2, scheme=Centred(4)
        )
        np.testing.assert_array_equal(
            got[1:-1, 1:-1], raw / float(grid.spacing("y", at="X"))
        )

    def test_the_data_choose_the_weights(self):
        grid = g.CartesianCGrid1D.from_interior(8, 8.0)
        f = np.zeros(10)
        f[4] = 1.0
        values = {
            data: g.interpolate(f, grid, src="T", dst="U", scheme=Centred(4, data))[4]
            for data in ("point", "cell", None)
        }
        assert values["point"] == values[None] == 9.0 / 16.0
        assert values["cell"] == pytest.approx(7.0 / 12.0)

    @pytest.mark.parametrize("scheme", [Centred(4), Biased(2, "left")], ids=repr)
    @pytest.mark.parametrize(
        ("src", "dst"), [("T", "U"), ("T", "V"), ("T", "X"), ("U", "T")]
    )
    def test_next_to_dry_points_the_stencil_narrows(self, scheme, src, dst):
        grid = g.CartesianCGrid2D.from_interior(16, 14, 4.0, 3.0)
        wet = np.ones(grid.shape)
        wet[6:8, 6:9] = 0.0
        mask = g.Mask2D.from_grid(grid, wet)
        dry = ~np.asarray(mask.wet(at=src), dtype=bool)
        f = field(grid, xp=np)
        # what a dry point holds cannot matter where a wider stencil was chosen
        calls = [
            lambda f: g.interpolate(f, grid, src=src, dst=dst, scheme=scheme, mask=mask)
        ]
        if dst != "X":
            axis = "y" if dst == "V" else "x"
            calls.append(
                lambda f: g.difference(
                    f, grid, axis=axis, src=src, dst=dst, scheme=scheme, mask=mask
                )
            )
        kind = "strict" if dst == "X" else "default"
        target = np.asarray(mask.wet(at=dst, kind=kind), dtype=bool)
        for call in calls:
            got = call(f)
            assert (got[~target] == 0.0).all()
            if src == "T":
                # every wet U, V and (strict) X point has wet T cells around it
                moved = call(np.where(dry, 1e6, f))
                np.testing.assert_array_equal(moved, got)
            far = np.zeros(grid.shape, dtype=bool)
            far[3:4, 4:-4] = True
            wide = (
                g.interpolate(f, grid, src=src, dst=dst, scheme=scheme)
                if call is calls[0]
                else g.difference(f, grid, axis=axis, src=src, dst=dst, scheme=scheme)
            )
            np.testing.assert_array_equal(got[far], wide[far])

    @pytest.mark.parametrize("scheme", [Upwind(3), WENO(5), TVD()], ids=repr)
    @pytest.mark.parametrize("name", ["c2", "a2"])
    def test_only_linear_schemes_without_upwinding(self, name, scheme):
        grid = GRIDS.get(name) or g.CartesianAGrid2D.from_interior(8, 6, 4.0, 3.0)
        f = field(grid)
        for call in (
            lambda: g.difference(f, grid, axis="x", src="T", dst="U", scheme=scheme),
            lambda: g.interpolate(f, grid, src="T", dst="U", scheme=scheme),
        ):
            with pytest.raises(TypeError, match=r"use gridax\.reconstruct"):
                call()
        with pytest.raises(TypeError, match=r"gridax\.schemes"):
            g.interpolate(f, grid, src="T", dst="U", scheme="centred")

    def test_jit_and_grad(self):
        grid = GRIDS["c2"]
        f = field(grid)

        def fn(f):
            return g.interpolate(f, grid, src="T", dst="X", scheme=Centred(4))

        np.testing.assert_allclose(jax.jit(fn)(f), fn(f), rtol=1e-12, atol=1e-13)
        assert np.isfinite(jax.grad(lambda f: (fn(f) ** 2).sum())(f)).all()


class TestCollocatedSchemes:
    @pytest.mark.parametrize("accuracy", [1, 3, 5])
    @pytest.mark.parametrize("periodic", [(), ("x",)])
    def test_centred_is_the_central_difference(self, accuracy, periodic):
        grid = g.CartesianAGrid2D.from_interior(16, 12, 4.0, 3.0, periodic=periodic)
        f = field(grid, xp=np)
        got = g.difference(f, grid, axis="x", scheme=Centred(accuracy + 1))
        want = ga.difference(
            f,
            step=float(grid.spacing("x")),
            accuracy=accuracy,
            periodic=bool(periodic),
        )
        np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize(
        ("side", "method"), [("right", "forward"), ("left", "backward")]
    )
    def test_biased_is_the_one_sided_difference(self, order, side, method):
        grid = g.CartesianAGrid2D.from_interior(16, 12, 4.0, 3.0)
        f = field(grid, xp=np)
        got = g.difference(f, grid, axis="y", scheme=Biased(order, side))
        want = ga.difference(
            f, axis=-2, step=float(grid.spacing("y")), accuracy=order, method=method
        )
        np.testing.assert_array_equal(got, want)

    def test_biased_with_a_mask_keeps_its_side_where_it_fits(self):
        grid = g.CartesianAGrid1D.from_interior(12, 12.0)
        wet = np.ones(12)
        wet[6] = 0.0
        mask = g.Mask1D.from_grid(grid, wet)
        f = field(grid, xp=np)
        got = g.difference(f, grid, axis="x", scheme=Biased(1, "right"), mask=mask)
        want = np.diff(f)
        assert np.isnan(got[6])
        kept = [i for i in range(11) if i not in (5, 6)]
        np.testing.assert_allclose(got[kept], want[kept], rtol=1e-14)
        # the cell before the dry one has no right neighbour: backward there
        assert got[5] == pytest.approx(f[5] - f[4])

    def test_cell_averages_are_not_available(self):
        grid = g.CartesianAGrid1D.from_interior(12, 12.0)
        with pytest.raises(NotImplementedError, match="cell"):
            g.difference(field(grid), grid, axis="x", scheme=Centred(4, "cell"))


class TestBindings:
    @pytest.mark.parametrize(("name", "axis", "src", "dst"), DIFFERENCES)
    @pytest.mark.parametrize("masked", [False, True])
    def test_a_difference_method_is_the_function(self, name, axis, src, dst, masked):
        grid = GRIDS[name]
        rank = len(grid.axes)
        mask = wet_mask(grid) if masked else None
        if name.startswith("s"):
            axis_name = {"x": "lon", "y": "lat"}.get(axis, axis)
            cls = {
                2: bindings.SphericalDifference2D,
                3: bindings.SphericalDifference3D,
            }[rank]
        else:
            axis_name = axis
            cls = {
                1: bindings.Difference1D,
                2: bindings.Difference2D,
                3: bindings.Difference3D,
            }[rank]
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
        cls = {
            1: bindings.Interpolation1D,
            2: bindings.Interpolation2D,
            3: bindings.Interpolation3D,
        }
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
