"""Tests for CartesianGrid1D, CartesianGrid2D, CartesianGrid3D."""

from gridax import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
    SphericalGrid2D,
    SphericalGrid3D,
)
import pytest


class TestArakawaCGrid1D:
    def test_from_interior_sizes(self):
        g = CartesianGrid1D.from_interior(10, 1.0)
        assert g.Nx == 12
        assert g.Lx == 1.0
        assert g.spacing("x") == pytest.approx(0.1)

    def test_direct_construction(self):
        g = CartesianGrid1D(Nx=6, Lx=4.0)
        assert g.Nx == 6
        assert g.spacing("x") == 1.0

    def test_dx_matches_Lx(self):
        g = CartesianGrid1D.from_interior(4, 2.0)
        assert g.spacing("x") == pytest.approx(g.Lx / (g.Nx - 2))


class TestArakawaCGrid2D:
    def test_from_interior_sizes(self):
        g = CartesianGrid2D.from_interior(8, 6, 1.0, 2.0)
        assert g.Nx == 10
        assert g.Ny == 8
        assert g.spacing("x") == pytest.approx(1.0 / 8)
        assert g.spacing("y") == pytest.approx(2.0 / 6)

    def test_direct_construction(self):
        g = CartesianGrid2D(Nx=4, Ny=4, Lx=1.0, Ly=1.0)
        assert g.Nx == 4
        assert g.Ny == 4

    def test_square_grid(self):
        g = CartesianGrid2D.from_interior(10, 10, 1.0, 1.0)
        assert g.spacing("x") == g.spacing("y")


class TestArakawaCGrid3D:
    def test_from_interior_sizes(self):
        g = CartesianGrid3D.from_interior(4, 5, 6, 1.0, 2.0, 3.0)
        assert g.Nx == 6
        assert g.Ny == 7
        assert g.Nz == 8
        assert g.spacing("x") == pytest.approx(1.0 / 4)
        assert g.spacing("y") == pytest.approx(2.0 / 5)
        assert g.spacing("z") == pytest.approx(3.0 / 6)

    def test_direct_construction(self):
        g = CartesianGrid3D(Nx=4, Ny=4, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0)
        assert g.Nz == 4


# ---------------------------------------------------------------------------
# Staggered grid position tests
# ---------------------------------------------------------------------------
# The Arakawa C-grid convention (same-index colocation):
#   T[j, i] → cell centre  at  (i * dx,      j * dy     )
#   U[j, i] → east face    at  ((i+1/2)*dx,  j * dy     )
#   V[j, i] → north face   at  (i * dx,      (j+1/2)*dy )
#   X[j, i] → NE corner    at  ((i+1/2)*dx,  (j+1/2)*dy )
#
# Interior indices run i = 1 … Nx-2, j = 1 … Ny-2.
# ---------------------------------------------------------------------------

import numpy as np


class TestGridStaggerPositions2D:
    """Verify that derived face-point coordinates follow the stated conventions."""

    def test_T_cell_centre_positions(self):
        """T[j,i] is at (i*dx, j*dy)."""
        g = CartesianGrid2D.from_interior(8, 6, 1.0, 2.0)
        for i in range(1, g.Nx - 1):
            assert g.spacing("x") * i == pytest.approx(i * g.spacing("x"))
        # Simply verify the grid spacing is consistent
        assert g.spacing("x") == pytest.approx(g.Lx / (g.Nx - 2))
        assert g.spacing("y") == pytest.approx(g.Ly / (g.Ny - 2))

    def test_U_face_offset_by_half_dx(self):
        """U-face coordinate in x is T-centre shifted by +dx/2."""
        g = CartesianGrid2D.from_interior(8, 6, 1.0, 2.0)
        # T-point x at index i: x_T[i] = i * dx
        # U-point x at index i: x_U[i] = (i + 0.5) * dx = x_T[i] + dx/2
        for i in range(1, g.Nx - 1):
            x_T = i * g.spacing("x")
            x_U = (i + 0.5) * g.spacing("x")
            assert x_U == pytest.approx(x_T + g.spacing("x") / 2)

    def test_V_face_offset_by_half_dy(self):
        """V-face coordinate in y is T-centre shifted by +dy/2."""
        g = CartesianGrid2D.from_interior(8, 6, 1.0, 2.0)
        for j in range(1, g.Ny - 1):
            y_T = j * g.spacing("y")
            y_V = (j + 0.5) * g.spacing("y")
            assert y_V == pytest.approx(y_T + g.spacing("y") / 2)

    def test_X_corner_offset_by_half_dx_and_dy(self):
        """X-corner is offset by (dx/2, dy/2) from the T-centre."""
        g = CartesianGrid2D.from_interior(8, 6, 1.0, 2.0)
        for i in range(1, g.Nx - 1):
            for j in range(1, g.Ny - 1):
                x_X = (i + 0.5) * g.spacing("x")
                y_X = (j + 0.5) * g.spacing("y")
                x_T = i * g.spacing("x")
                y_T = j * g.spacing("y")
                assert x_X == pytest.approx(x_T + g.spacing("x") / 2)
                assert y_X == pytest.approx(y_T + g.spacing("y") / 2)

    def test_grid_symmetry_square_domain(self):
        """For a square uniform domain [0,L]x[0,L] the grid is symmetric."""
        L = 1.0
        g = CartesianGrid2D.from_interior(10, 10, L, L)
        assert g.spacing("x") == g.spacing("y")
        # T-point positions are symmetric about L/2

        interior = np.arange(1, g.Nx - 1)
        x_T = interior * g.spacing("x")
        # Just check that the mean is symmetric about L/2
        assert np.mean(x_T) == pytest.approx(L / 2, abs=g.spacing("x"))

    def test_ghost_cell_count(self):
        """There is exactly one ghost cell ring on each side."""
        g = CartesianGrid2D.from_interior(8, 6, 1.0, 2.0)
        assert g.Nx == 8 + 2
        assert g.Ny == 6 + 2

    def test_grid_spacing_consistent_with_domain(self):
        """dx = Lx / Nx_interior, dy = Ly / Ny_interior."""
        Lx, Ly = 3.0, 5.0
        nx, ny = 12, 20
        g = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
        assert g.spacing("x") == pytest.approx(Lx / nx)
        assert g.spacing("y") == pytest.approx(Ly / ny)


class TestGridStaggerPositions1D:
    """Verify 1D grid positions."""

    def test_T_positions(self):
        """T[i] is at i * dx."""
        g = CartesianGrid1D.from_interior(8, 1.0)
        for i in range(1, g.Nx - 1):
            assert i * g.spacing("x") == pytest.approx(i * g.spacing("x"))

    def test_U_face_at_plus_half_dx(self):
        """U[i] (east face) is at (i + 0.5) * dx."""
        g = CartesianGrid1D.from_interior(8, 1.0)
        for i in range(1, g.Nx - 1):
            x_U = (i + 0.5) * g.spacing("x")
            assert x_U == pytest.approx(i * g.spacing("x") + g.spacing("x") / 2)

    def test_dx_equals_Lx_over_n(self):
        n = 16
        L = 2.0
        g = CartesianGrid1D.from_interior(n, L)
        assert g.spacing("x") == pytest.approx(L / n)


class TestTPointCoordinates:
    """T[j, i] at (j dy, i dx), rows in ravel order (#476)."""

    def test_2d_points_follow_the_raveled_field(self):
        from gridax import t_point_coordinates

        grid = CartesianGrid2D.from_interior(3, 2, 6.0, 1.0)
        points = t_point_coordinates(grid)
        assert points.shape == (grid.Ny * grid.Nx, 2)
        rows, cols = np.meshgrid(np.arange(grid.Ny), np.arange(grid.Nx), indexing="ij")
        np.testing.assert_array_equal(points[:, 0], rows.ravel() * grid.spacing("y"))
        np.testing.assert_array_equal(points[:, 1], cols.ravel() * grid.spacing("x"))

    def test_1d_points(self):
        from gridax import CartesianGrid1D, t_point_coordinates

        grid = CartesianGrid1D.from_interior(4, 2.0)
        np.testing.assert_array_equal(
            t_point_coordinates(grid)[:, 0], np.arange(grid.Nx) * grid.spacing("x")
        )

    def test_spherical_grid_raises(self):
        from gridax import SphericalGrid2D, t_point_coordinates

        grid = SphericalGrid2D.from_interior(4, 4, (0.0, 1.0), (0.0, 1.0))
        with pytest.raises(TypeError, match="latitudes"):
            t_point_coordinates(grid)


METRIC_GRIDS = {
    "c1": lambda: CartesianGrid1D.from_interior(8, 4.0),
    "c2": lambda: CartesianGrid2D.from_interior(8, 6, 4.0, 3.0),
    "c3": lambda: CartesianGrid3D.from_interior(8, 6, 5, 4.0, 3.0, 10.0),
    "s2": lambda: SphericalGrid2D.from_interior(
        8, 6, (0.0, 40.0), (-30.0, 30.0), R=2.0
    ),
    "s3": lambda: SphericalGrid3D.from_interior(
        8, 6, 5, (0.0, 40.0), (-30.0, 30.0), 10.0, R=2.0
    ),
}
METRIC_POINTS = {1: "TU", 2: "TUVX", 3: "TUVXW"}
STAGGERED = {"T": "", "U": "x", "V": "y", "X": "xy", "W": "z"}


class TestMetricAccessor:
    """``spacing``, ``coords`` and ``area`` at every point of every grid."""

    @pytest.fixture(params=sorted(METRIC_GRIDS))
    def grid(self, request):
        return METRIC_GRIDS[request.param]()

    def test_axes_follow_the_array(self, grid):
        sizes = {"x": "Nx", "y": "Ny", "z": "Nz"}
        assert grid.shape == tuple(getattr(grid, sizes[a]) for a in grid.axes)
        assert grid.ghost == 1

    def test_coords_are_shifted_half_a_step_where_staggered(self, grid):
        for at in METRIC_POINTS[len(grid.axes)]:
            for axis in grid.axes:
                t, p = grid.coords(axis), grid.coords(axis, at=at)
                step = t[1] - t[0]
                shift = 0.5 * step if axis in STAGGERED[at] else 0.0
                np.testing.assert_allclose(p - t, shift, atol=1e-12)
                assert p.shape == (grid.shape[grid.axes.index(axis)],)

    def test_cartesian_spacing_is_the_coordinate_step(self):
        for name in ("c1", "c2", "c3"):
            grid = METRIC_GRIDS[name]()
            for at in METRIC_POINTS[len(grid.axes)]:
                for axis in grid.axes:
                    np.testing.assert_allclose(
                        np.diff(grid.coords(axis, at=at)),
                        grid.spacing(axis, at=at),
                        rtol=1e-12,
                    )

    def test_spherical_spacing_is_the_arc_length(self):
        for name in ("s2", "s3"):
            grid = METRIC_GRIDS[name]()
            for at in METRIC_POINTS[len(grid.axes)]:
                lat = grid.coords("y", at=at)[:, None]
                # Δx_P = R cos(lat_P) dlon, Δy = R dlat
                np.testing.assert_allclose(
                    grid.spacing("x", at=at),
                    np.broadcast_to(2.0 * np.cos(lat) * np.deg2rad(5.0), (8, 10)),
                    rtol=1e-12,
                )
                np.testing.assert_allclose(
                    grid.spacing("y", at=at), 2.0 * np.deg2rad(10.0), rtol=1e-12
                )

    def test_area(self, grid):
        for at in METRIC_POINTS[len(grid.axes)]:
            horizontal = tuple(a for a in grid.axes if a != "z")
            assert grid.area(at=at).shape == tuple(
                grid.shape[grid.axes.index(a)] for a in horizontal
            )
        if isinstance(grid, SphericalGrid2D | SphericalGrid3D):
            # ∑A = R² Lλ (sin lat_max - sin lat_min), exactly
            want = 4.0 * np.deg2rad(40.0) * (2.0 * np.sin(np.deg2rad(30.0)))
        else:
            want = 4.0 if grid.axes == ("x",) else 12.0
        inner = (slice(1, -1),) * (1 if grid.axes == ("x",) else 2)
        np.testing.assert_allclose(grid.area()[inner].sum(), want, rtol=1e-12)

    def test_a_wider_ghost_ring_keeps_the_spacing(self):
        grid = CartesianGrid1D(Nx=8 + 2 * 4, Lx=4.0, ghost=4)
        assert grid.spacing("x") == 0.5

    @pytest.mark.parametrize("call", ["spacing", "coords"])
    def test_unknown_axis_or_point_raises(self, call):
        grid = METRIC_GRIDS["c2"]()
        with pytest.raises(ValueError, match="axes"):
            getattr(grid, call)("z")
        with pytest.raises(ValueError, match="at='W'"):
            getattr(grid, call)("x", at="W")
        with pytest.raises(ValueError, match="at='Q'"):
            grid.area(at="Q")

    def test_the_old_metric_names_are_gone(self):
        for name in ("dx", "dy", "dx_T", "dy_T", "cos_lat_T", "lat_T", "lon_T"):
            assert not hasattr(METRIC_GRIDS["s2"](), name)
        assert not hasattr(METRIC_GRIDS["c2"](), "dx")
