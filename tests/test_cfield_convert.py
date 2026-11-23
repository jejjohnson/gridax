"""`CField` to and from xarray (xgcm's positions), DimArray, and left faces."""

from __future__ import annotations

import gridax as g
from gridax import CField, to_dimarray, to_right
import jax.numpy as jnp
import numpy as np
import pytest

xr = pytest.importorskip("xarray")

GRIDS = {
    "cartesian": g.CartesianCGrid2D.from_interior(8, 6, 4.0, 3.0),
    "spherical": g.SphericalCGrid2D.from_interior(8, 6, (0.0, 40.0), (-30.0, 30.0)),
}
GRID = GRIDS["cartesian"]
DIMS = ("member", "y", "x")


def field(loc, grid=GRID, **kwargs):
    data = np.random.default_rng(0).normal(size=(2, *grid.shape))
    return CField(data, DIMS, loc=loc, **kwargs)


class TestXarray:
    @pytest.mark.parametrize("grid", sorted(GRIDS))
    @pytest.mark.parametrize("loc", ["T", "U", "V", "X"])
    def test_coordinates_are_the_grids(self, loc, grid):
        grid = GRIDS[grid]
        ds = field(loc, grid).to_xarray(grid, "q")
        want = tuple(
            f"{axis}_g" if axis in dict(field(loc).stagger) else f"{axis}_c"
            for axis in "yx"
        )
        assert ds["q"].dims == ("member", *want)
        for axis in "yx":
            np.testing.assert_array_equal(ds[f"{axis}_c"], grid.coords(axis))
            point = {"x": "U", "y": "V"}[axis]
            np.testing.assert_array_equal(ds[f"{axis}_g"], grid.coords(axis, at=point))
            assert ds[f"{axis}_c"].attrs == {"axis": axis.upper()}
            assert ds[f"{axis}_g"].attrs["c_grid_axis_shift"] == 0.5

    @pytest.mark.parametrize("loc", ["T", "U", "V", "X"])
    def test_round_trip(self, loc):
        f = field(loc, labels={"member": ("a", "b")})
        back = CField.from_xarray(f.to_xarray(GRID, "q")["q"])
        assert (back.dims, back.stagger, back.labels) == (f.dims, f.stagger, f.labels)
        np.testing.assert_array_equal(back.data, f.data)

    def test_a_jax_field(self):
        f = CField(jnp.ones(GRID.shape), ("y", "x"), loc="U")
        assert isinstance(f.to_xarray(GRID)["field"].data, np.ndarray)

    def test_a_reduced_field_writes_the_axes_left(self):
        f = CField(np.ones(GRID.Ny), ("y",), loc="V")
        ds = f.to_xarray(GRID)
        assert set(ds.coords) == {"y_c", "y_g"}

    def test_a_left_face_reads_as_left_and_does_not_write(self):
        da = xr.DataArray(
            np.ones(GRID.shape),
            dims=("YC", "XG"),
            coords={
                "YC": ("YC", np.arange(8.0), {"axis": "Y"}),
                "XG": ("XG", np.arange(10.0), {"axis": "X", "c_grid_axis_shift": -0.5}),
            },
        )
        f = CField.from_xarray(da)
        assert (f.dims, f.stagger) == (("y", "x"), (("x", "left"),))
        with pytest.raises(ValueError, match="to_right"):
            f.to_xarray(GRID)

    def test_an_unknown_shift_raises(self):
        coords = {"x": ("x", np.arange(3.0), {"axis": "X", "c_grid_axis_shift": 0.25})}
        with pytest.raises(ValueError, match="c_grid_axis_shift"):
            CField.from_xarray(xr.DataArray(np.ones(3), dims=("x",), coords=coords))


class TestXgcm:
    """xgcm's grid recovers the positions, and interpolates as gridax does."""

    xgcm = pytest.importorskip("xgcm")

    @pytest.mark.parametrize("loc", ["T", "U", "V", "X"])
    def test_the_positions(self, loc):
        grid = self.xgcm.Grid(field(loc).to_xarray(GRID))
        for axis in "XY":
            assert grid.axes[axis].coords == {
                "center": f"{axis.lower()}_c",
                "right": f"{axis.lower()}_g",
            }

    @pytest.mark.parametrize(
        ("loc", "axis", "method"),
        [
            ("T", "X", "T_to_U"),
            ("T", "Y", "T_to_V"),
            ("U", "X", "U_to_T"),
            ("V", "Y", "V_to_T"),
            ("U", "Y", "U_to_X"),
            ("V", "X", "V_to_X"),
        ],
    )
    def test_interpolation(self, loc, axis, method):
        f = field(loc)
        ds = f.to_xarray(GRID, "q")
        grid = self.xgcm.Grid(ds)
        theirs = grid.interp(ds["q"], axis, padding="fill", fill_value=0.0)
        ours = getattr(g.Interpolation2D(grid=GRID), method)(f.data)
        dst = method.rsplit("_", 1)[1]
        back = CField.from_xarray(theirs)
        assert back.loc == dst
        inner = (slice(None), slice(1, -1), slice(1, -1))
        np.testing.assert_allclose(back.data[inner], ours[inner], rtol=1e-14)


class TestToDimArray:
    def test_drops_the_stagger(self):
        pa = pytest.importorskip("pipekit_array")
        f = field("U", labels={"member": ("a", "b")})
        d = to_dimarray(f)
        assert type(d) is pa.DimArray
        assert (d.dims, d.labels) == (f.dims, f.labels)
        assert d.data is f.data


class TestToRight:
    """An MITgcm-style field (U on the west face) equals the gridax one."""

    BCS = g.BC(x=g.Periodic(), y=g.Periodic())

    @staticmethod
    def velocity(x, y):
        return np.sin(np.pi * x / 2.0) * np.cos(2.0 * np.pi * y / 3.0)

    @pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
    @pytest.mark.parametrize("left", [("x",), ("y",), ("x", "y")])
    def test_the_same_velocity(self, left, xp):
        def sample(sign):
            coords = {
                a: GRID.coords(a) + (sign * 0.5 * GRID.spacing(a) if a in left else 0.0)
                for a in "xy"
            }
            values = self.velocity(coords["x"][None, :], coords["y"][:, None])
            return xp.asarray(values)

        west = CField(sample(-1), ("y", "x"), stagger=dict.fromkeys(left, "left"))
        out = to_right(west, GRID, self.BCS)
        assert dict(out.stagger) == dict.fromkeys(left, "right")
        np.testing.assert_allclose(out.data, sample(+1), atol=1e-14)

    def test_the_values_inside_are_moved_not_changed(self):
        data = np.random.default_rng(1).normal(size=GRID.shape)
        west = CField(data, ("y", "x"), stagger={"x": "left"})
        out = to_right(west, GRID, g.BC(x=g.Dirichlet(0.0)))
        np.testing.assert_array_equal(out.data[:, :-1], data[:, 1:])
        # the face outside the domain is the odd mirror about the wall
        np.testing.assert_array_equal(out.data[:, -1], -out.data[:, -3])

    def test_a_right_field_is_returned_as_it_is(self):
        u = field("U")
        assert to_right(u, GRID, self.BCS) is u

    def test_a_field_with_members_in_any_dim_order(self):
        data = np.random.default_rng(1).normal(size=(10, 3, 8))
        f = CField(data, ("x", "member", "y"), stagger={"x": "left"})
        out = to_right(f, GRID, self.BCS)
        np.testing.assert_array_equal(out.data[:-1], data[1:])
        assert out.dims == f.dims

    def test_errors(self):
        outer = CField(np.ones(GRID.shape), ("y", "x"), stagger={"x": "outer"})
        with pytest.raises(ValueError, match="outer"):
            to_right(outer, GRID, self.BCS)
        west = CField(np.ones(GRID.shape), ("y", "x"), stagger={"x": "left"})
        with pytest.raises(ValueError, match="high x face"):
            to_right(west, GRID, g.BC(y=g.Periodic()))
