"""`fill_ghosts` and `BC`: ghost layers from boundary conditions, point and grid."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import gridax as g
from gridax import BC, CField, StaggerMismatchError, fill_ghosts
from gridax._src.boundary.fill import FACES
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.fixtures import bindings

GOLDEN = np.load(Path(__file__).parent / "fixtures" / "golden" / "boundary_atoms.npz")
DX, DY = (float(d) for d in GOLDEN["spacing"])
GRID = g.CartesianCGrid2D.from_interior(7, 7, 7 * DX, 7 * DY)

ATOMS = {
    "dirichlet": g.Dirichlet(0.4),
    "neumann": g.Neumann(-0.8),
    "periodic": g.Periodic(),
    "outflow": g.Outflow(),
    "reflective": g.Reflective(),
    "sponge": g.Sponge(0.2, 0.3),
    "robin": g.Robin(1.5, 0.25, 0.6),
    "extrapolation2": g.Extrapolation(2),
    "extrapolation5": g.Extrapolation(5),
    "slip": g.Slip(0.35),
}
MIRRORS = [name for name in ATOMS if name not in ("periodic", "extrapolation5")]


def field(shape=GRID.shape, seed=0):
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape))


class TestParity:
    """One ghost layer is what the one-face atoms wrote before (max|Δ| = 0).

    ``boundary_atoms.npz`` holds the output of ``Dirichlet1D`` … ``Slip1D``
    and ``BoundaryConditionSet`` at gridax 0.2, for every atom, face and
    point.
    """

    @pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
    @pytest.mark.parametrize("atom", sorted(ATOMS))
    def test_every_face_and_point(self, atom, xp):
        f = xp.asarray(GOLDEN["field"])
        seen = 0
        for name, want in zip(GOLDEN["names"], GOLDEN["values"], strict=True):
            kind, face, *point = str(name).split("/")
            if kind != atom:
                continue
            seen += 1
            axis, side = FACES[face]
            out = ATOMS[atom].fill(f, GRID, axis, side, point[0], 1)
            # The golden is NumPy's; XLA's division differs by an ulp (as the
            # old atoms did on JAX).
            rtol = 0.0 if xp is np else 4e-16
            np.testing.assert_allclose(out, want, rtol=rtol, err_msg=str(name))
            if atom != "periodic":
                bc = BC(**{face: ATOMS[atom]})
                np.testing.assert_array_equal(
                    fill_ghosts(f, GRID, bc, stagger=point[0]), out
                )
        assert seen == 16

    @pytest.mark.parametrize("point", ["T", "U", "V", "X"])
    def test_priority_corners_are_the_old_order(self, point):
        bc = BC(
            corners="priority",
            south=ATOMS["dirichlet"],
            north=ATOMS["neumann"],
            west=ATOMS["robin"],
            east=ATOMS["extrapolation2"],
        )
        names = [str(n) for n in GOLDEN["names"]]
        want = GOLDEN["values"][names.index(f"set/{point}")]
        out = fill_ghosts(GOLDEN["field"], GRID, bc, stagger=point)
        np.testing.assert_array_equal(out, want)


class TestLayers:
    """``width`` layers satisfy the discrete condition at the wall."""

    WIDTHS = (1, 2, 3)

    @staticmethod
    def line(width, n=8):
        grid = g.CartesianCGrid1D(Nx=n + 2 * width, Lx=2.0, ghost=width)
        return grid, field((n + 2 * width,))

    @staticmethod
    def sides(f, width):
        """``(φ_k, φ_{-k})`` for k = 1.. from the low and the high wall."""
        n = f.shape[0]
        low = [(f[width - 1 + k], f[width - k]) for k in range(1, width + 1)]
        high = [(f[n - width - k], f[n - width - 1 + k]) for k in range(1, width + 1)]
        return low, high

    @pytest.mark.parametrize("width", WIDTHS)
    @pytest.mark.parametrize("side", [0, 1])
    def test_dirichlet(self, width, side):
        grid, f = self.line(width)
        out = fill_ghosts(f, grid, BC(x=g.Dirichlet(0.4)))
        for inner, ghost in self.sides(out, width)[side]:
            np.testing.assert_allclose(0.5 * (inner + ghost), 0.4, rtol=1e-14)

    @pytest.mark.parametrize("width", WIDTHS)
    @pytest.mark.parametrize("side", [0, 1])
    def test_neumann_is_the_outward_gradient(self, width, side):
        grid, f = self.line(width)
        out = fill_ghosts(f, grid, BC(x=g.Neumann(-0.8)))
        for k, (inner, ghost) in enumerate(self.sides(out, width)[side], start=1):
            gradient = (ghost - inner) / ((2 * k - 1) * grid.spacing("x"))
            np.testing.assert_allclose(gradient, -0.8, rtol=1e-12)

    @pytest.mark.parametrize("width", WIDTHS)
    @pytest.mark.parametrize("side", [0, 1])
    def test_robin(self, width, side):
        grid, f = self.line(width)
        a, b, c = 1.5, 0.25, 0.6
        out = fill_ghosts(f, grid, BC(x=g.Robin(a, b, c)))
        for k, (inner, ghost) in enumerate(self.sides(out, width)[side], start=1):
            gradient = (ghost - inner) / ((2 * k - 1) * grid.spacing("x"))
            np.testing.assert_allclose(
                a * 0.5 * (ghost + inner) + b * gradient, c, rtol=1e-12
            )

    @pytest.mark.parametrize("width", WIDTHS)
    @pytest.mark.parametrize("side", [0, 1])
    def test_reflective_outflow_and_slip(self, width, side):
        grid, f = self.line(width)
        even = fill_ghosts(f, grid, BC(x=g.Reflective()))
        flat = fill_ghosts(f, grid, BC(x=g.Outflow()))
        slip = fill_ghosts(f, grid, BC(x=g.Slip(0.35)))
        first = self.sides(flat, width)[side][0][0]
        for inner, ghost in self.sides(even, width)[side]:
            assert inner == ghost
        for _, ghost in self.sides(flat, width)[side]:
            assert ghost == first
        for inner, ghost in self.sides(slip, width)[side]:
            np.testing.assert_allclose(0.5 * (inner + ghost), 0.35 * inner, rtol=1e-12)

    @pytest.mark.parametrize("width", WIDTHS)
    @pytest.mark.parametrize("point", ["T", "U"])
    def test_periodic_wraps_every_layer(self, width, point):
        grid, f = self.line(width)
        out = fill_ghosts(f, grid, BC(x=g.Periodic()), stagger=point)
        core = np.asarray(f)[width:-width]
        np.testing.assert_array_equal(out, np.pad(core, width, mode="wrap"))

    @pytest.mark.parametrize("width", WIDTHS)
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_extrapolation_is_exact_on_a_polynomial(self, width, order):
        grid, f = self.line(width)
        x = np.arange(f.shape[0]) / 4.0
        poly = x**order - x
        holed = np.array(poly)
        holed[:width] = holed[-width:] = np.nan
        out = fill_ghosts(holed, grid, BC(x=g.Extrapolation(order)))
        np.testing.assert_allclose(out, poly, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("width", WIDTHS)
    def test_a_node_on_the_wall(self, width):
        """U: the wall nodes hold the value, with odd mirrors outside."""
        grid, f = self.line(width)
        out = np.asarray(fill_ghosts(f, grid, BC(x=g.Dirichlet(0.4)), stagger="U"))
        west, east = width - 1, f.shape[0] - width - 1
        for wall, inwards in ((west, 1), (east, -1)):
            assert out[wall] == 0.4
            for k in range(1, (wall if inwards == 1 else width) + 1):
                np.testing.assert_allclose(
                    0.5 * (out[wall + inwards * k] + out[wall - inwards * k]),
                    0.4,
                    rtol=1e-14,
                )

    @pytest.mark.parametrize("width", WIDTHS)
    def test_neumann_on_a_wall_node_is_an_even_mirror(self, width):
        grid, f = self.line(width)
        out = np.asarray(fill_ghosts(f, grid, BC(x=g.Neumann(0.7)), stagger="U"))
        wall = f.shape[0] - width - 1
        for k in range(1, width + 1):
            gradient = (out[wall + k] - out[wall - k]) / (2 * k * grid.spacing("x"))
            np.testing.assert_allclose(gradient, 0.7, rtol=1e-12)

    @pytest.mark.parametrize("width", WIDTHS)
    @pytest.mark.parametrize("name", MIRRORS)
    def test_the_interior_is_untouched(self, width, name):
        grid, f = self.line(width)
        out = fill_ghosts(f, grid, BC(x=ATOMS[name]))
        np.testing.assert_array_equal(out[width:-width], f[width:-width])

    def test_the_width_defaults_to_the_grids_ring(self):
        grid, f = self.line(3)
        bc = BC(x=g.Dirichlet(0.4))
        np.testing.assert_array_equal(
            fill_ghosts(f, grid, bc), fill_ghosts(f, grid, bc, width=3)
        )


class TestGrids:
    def test_neumann_on_a_sphere_uses_each_rows_spacing(self):
        """The centred difference across the west wall is g on every row."""
        grid = g.SphericalCGrid2D.from_interior(
            8, 10, (0.0, 40.0), (-60.0, 60.0), R=2.0
        )
        out = fill_ghosts(
            field(grid.shape), grid, BC(x=(g.Neumann(0.3), g.Neumann(-0.2)))
        )
        dx = grid.spacing("x", at="U")[:, 0]
        np.testing.assert_allclose((out[:, 0] - out[:, 1]) / dx, 0.3, rtol=1e-10)
        np.testing.assert_allclose((out[:, -1] - out[:, -2]) / dx, -0.2, rtol=1e-10)

    @pytest.mark.parametrize("width", [1, 2])
    def test_a_sphere_takes_wider_rings(self, width):
        grid = g.SphericalCGrid2D.from_interior(8, 10, (0.0, 40.0), (-60.0, 60.0))
        f = field((10 + 2 * width, 8 + 2 * width))
        out = fill_ghosts(f, grid, BC(x=g.Neumann(1e-6)), width=width)
        dx = np.asarray(grid.spacing("x"))[1:-1, 0]
        inner = slice(width, -width)
        np.testing.assert_allclose(
            (out[inner, width - 1] - out[inner, width]) / dx, 1e-6, rtol=1e-8
        )

    def test_three_dimensions(self):
        grid = g.CartesianCGrid3D.from_interior(6, 5, 4, 1.0, 1.0, 2.0)
        f = field((2, *grid.shape))
        bc = BC(
            x=g.Periodic(),
            y=g.Dirichlet(0.0),
            z=(g.Neumann(0.5), g.Dirichlet(1.0)),
        )
        out = fill_ghosts(f, grid, bc)
        inner = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
        np.testing.assert_array_equal(out[inner], f[inner])
        np.testing.assert_array_equal(out[:, 2, 2, 0], f[:, 2, 2, -2])
        np.testing.assert_allclose(out[:, 2, 0, 2], -f[:, 2, 1, 2])
        np.testing.assert_allclose(
            (out[:, 0, 2, 2] - out[:, 1, 2, 2]) / grid.spacing("z"), 0.5, rtol=1e-10
        )
        np.testing.assert_allclose(0.5 * (out[:, -1, 2, 2] + out[:, -2, 2, 2]), 1.0)

    def test_w_has_a_node_on_the_top_and_bottom(self):
        grid = g.CartesianCGrid3D.from_interior(6, 5, 4, 1.0, 1.0, 2.0)
        bc = BC(z=g.Dirichlet(0.0))
        out = fill_ghosts(field(grid.shape), grid, bc, stagger="W")
        assert float(jnp.abs(out[0]).max()) == float(jnp.abs(out[-2]).max()) == 0.0

    def test_float32_stays_float32(self):
        f = field().astype(jnp.float32)
        bc = BC(
            x=(g.Neumann(0.3), g.Robin(1.0, 0.2, 0.1)),
            y=(g.Dirichlet(np.ones(9)), g.Sponge(0.1, 0.5)),
        )
        assert fill_ghosts(f, GRID, bc).dtype == jnp.float32


class TestCorners:
    def test_average_is_the_mean_of_the_two_orders(self):
        f = field()
        faces = {"x": g.Dirichlet(0.4), "y": g.Extrapolation(2)}
        x_wins = fill_ghosts(f, GRID, BC(corners="priority", **faces))
        y_first = fill_ghosts(f, GRID, BC(x=faces["x"]))
        y_wins = fill_ghosts(y_first, GRID, BC(y=faces["y"]))
        out = fill_ghosts(f, GRID, BC(**faces))
        for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            np.testing.assert_allclose(
                out[corner], 0.5 * (x_wins[corner] + y_wins[corner]), rtol=1e-13
            )

    @pytest.mark.parametrize("point", ["T", "U", "V", "X"])
    def test_the_rules_agree_off_the_corners(self, point):
        faces = {"x": g.Robin(1.0, 0.2, 0.1), "y": g.Neumann(0.3)}
        a = fill_ghosts(field(), GRID, BC(**faces), stagger=point)
        b = fill_ghosts(field(), GRID, BC(corners="priority", **faces), stagger=point)
        assert bool(jnp.all((a == b)[2:-2, :])) and bool(jnp.all((a == b)[:, 2:-2]))

    def test_periodic_corners_wrap(self):
        out = fill_ghosts(field(), GRID, BC(x=g.Periodic(), y=g.Periodic()))
        np.testing.assert_array_equal(out, np.pad(field()[1:-1, 1:-1], 1, mode="wrap"))


class TestPoint:
    def test_a_closed_basin_has_zero_net_mass_flux(self):
        walls = BC(x=g.Dirichlet(0.0), y=g.Dirichlet(0.0))
        ones = jnp.ones(GRID.shape)
        u = fill_ghosts(ones, GRID, walls, stagger="U")
        v = fill_ghosts(ones, GRID, walls, stagger="V")
        div = bindings.Difference2D(GRID).divergence(u, v)[1:-1, 1:-1]
        assert abs(float(div.sum())) < 1e-12

    def test_the_point_of_a_field_is_its_own(self):
        bc = BC(x=g.Dirichlet(0.4), y=g.Neumann())
        for point in "TUVX":
            f = CField(field(), ("y", "x"), loc=point)
            out = fill_ghosts(f, GRID, bc)
            assert (out.dims, out.loc) == (f.dims, point)
            np.testing.assert_array_equal(
                out.data, fill_ghosts(f.data, GRID, bc, stagger=point)
            )

    def test_the_grid_axes_of_a_field_are_found_by_name(self):
        bc = BC(x=g.Dirichlet(0.4), y=g.Neumann(0.1))
        data = field((9, 3, 9))
        f = CField(data, ("x", "member", "y"), loc="U")
        out = fill_ghosts(f, GRID, bc)
        assert out.dims == f.dims
        want = fill_ghosts(jnp.moveaxis(data, 0, -1), GRID, bc, stagger="U")
        np.testing.assert_array_equal(out.data, jnp.moveaxis(want, -1, 0))

    def test_a_stagger_that_contradicts_the_field_raises(self):
        f = CField(field(), ("y", "x"), loc="U")
        with pytest.raises(StaggerMismatchError):
            fill_ghosts(f, GRID, BC(x=g.Neumann()), stagger="V")

    @pytest.mark.parametrize("stagger", ["Q", (("x", "left"),)])
    def test_a_point_gridax_does_not_compute_on_raises(self, stagger):
        with pytest.raises(ValueError, match=r"point|not one of"):
            fill_ghosts(field(), GRID, BC(x=g.Neumann()), stagger=stagger)

    def test_a_field_without_a_grid_axis_raises(self):
        with pytest.raises(ValueError, match="no dims"):
            fill_ghosts(CField(field((9,)), ("y",)), GRID, BC(y=g.Neumann()))


class TestValues:
    def test_a_value_per_face_point_and_member(self):
        u_parent = field((3, 9), seed=7)
        out = fill_ghosts(
            field((3, 9, 9)), GRID, BC(x=(None, g.Dirichlet(u_parent))), stagger="U"
        )
        np.testing.assert_array_equal(out[:, :, -2], u_parent)

    def test_gradient_with_respect_to_the_face_values(self):
        f, parent = field(), field((9,), seed=3)

        def loss(values):
            bc = BC(x=(None, g.Dirichlet(values)))
            return jnp.sum(fill_ghosts(f, GRID, bc, stagger="U")[:, -2] ** 2)

        np.testing.assert_allclose(jax.grad(loss)(parent), 2.0 * parent)

    def test_a_bc_is_a_pytree_of_its_values(self):
        bc = BC(x=(g.Dirichlet(0.4), g.Neumann(0.1)), y=g.Periodic())
        assert jax.tree.leaves(bc) == [0.4, 0.1]
        doubled = jax.tree.map(lambda v: 2 * v, bc)
        assert doubled.faces["x"][0].value == 0.8
        assert doubled.corners == "average"

    @pytest.mark.parametrize("jit", [jax.jit, eqx.filter_jit])
    def test_under_jit_and_vmap(self, jit):
        bc = BC(x=g.Dirichlet(0.4), y=(g.Neumann(0.1), g.Slip(0.5)))
        f = field((3, *GRID.shape))
        want = fill_ghosts(f, GRID, bc)
        got = jit(fill_ghosts)(f, GRID, bc)
        np.testing.assert_allclose(got, want, rtol=1e-14)
        mapped = jax.vmap(lambda f: fill_ghosts(f, GRID, bc))(f)
        np.testing.assert_array_equal(mapped, want)

    @pytest.mark.parametrize(
        "make",
        [
            g.Slip,
            lambda v: g.Sponge(0.1, v),
            lambda v: g.Robin(v, 0.1, 0.2),
        ],
        ids=["slip", "sponge", "robin"],
    )
    def test_a_coefficient_may_be_traced(self, make):
        """#321: construction under ``vmap`` does not branch on the value."""
        values = jnp.array([0.2, 0.8])
        out = jax.vmap(lambda v: fill_ghosts(field(), GRID, BC(x=make(v))))(values)
        for i, v in enumerate((0.2, 0.8)):
            np.testing.assert_allclose(
                out[i], fill_ghosts(field(), GRID, BC(x=make(v))), rtol=1e-14
            )


class State(eqx.Module):
    h: CField
    u: CField
    v: CField


class Triple(NamedTuple):
    h: object
    u: object
    v: object


def state_of(make, lead=()):
    """A state of T, U and V fields, built with ``make``."""
    dims = (*("member",)[: len(lead)], "y", "x")
    return make(
        **{
            name: CField(field((*lead, *GRID.shape), seed), dims, loc=loc)
            for seed, (name, loc) in enumerate((("h", "T"), ("u", "U"), ("v", "V")))
        }
    )


WALL, FLAT = g.Dirichlet(0.0), g.Neumann()
STATE_BCS = {
    "h": BC(x=FLAT, y=FLAT),
    "u": BC(x=WALL, y=FLAT),
    "v": BC(x=FLAT, y=WALL),
}


class TestState:
    """One pytree of `BC` per state, matched leaf by leaf."""

    @pytest.mark.parametrize("make", [dict, Triple, State])
    def test_every_field_is_filled_at_its_point(self, make):
        state = state_of(make)
        out = fill_ghosts(state, GRID, make(**STATE_BCS))
        assert type(out) is type(state)
        fields = state if isinstance(state, dict) else vars_of(state)
        filled = out if isinstance(out, dict) else vars_of(out)
        for name, f in fields.items():
            want = fill_ghosts(f, GRID, STATE_BCS[name])
            assert filled[name].loc == f.loc
            np.testing.assert_array_equal(filled[name].data, want.data)

    def test_plain_arrays_take_their_points_from_stagger(self):
        state = {name: f.data for name, f in state_of(dict).items()}
        points = {"h": "T", "u": "U", "v": "V"}
        out = fill_ghosts(state, GRID, STATE_BCS, stagger=points)
        for name, f in state_of(dict).items():
            np.testing.assert_array_equal(
                out[name], fill_ghosts(f, GRID, STATE_BCS[name]).data
            )

    def test_none_leaves_a_field_alone(self):
        state = state_of(dict)
        out = fill_ghosts(state, GRID, {**STATE_BCS, "u": None})
        assert out["u"] is state["u"]

    def test_a_nested_state(self):
        state = {"ocean": state_of(State), "tracers": [state_of(dict)["h"]]}
        bcs = {"ocean": State(**STATE_BCS), "tracers": [STATE_BCS["h"]]}
        out = fill_ghosts(state, GRID, bcs)
        np.testing.assert_array_equal(
            out["tracers"][0].data,
            fill_ghosts(state["tracers"][0], GRID, STATE_BCS["h"]).data,
        )

    @pytest.mark.parametrize("jit", [jax.jit, eqx.filter_jit])
    def test_an_equinox_state_under_jit(self, jit):
        state = state_of(State)
        got = jit(fill_ghosts)(state, GRID, State(**STATE_BCS))
        want = fill_ghosts(state, GRID, State(**STATE_BCS))
        assert jax.tree.structure(got) == jax.tree.structure(want)
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-14)

    def test_an_equinox_state_under_vmap_over_members(self):
        state = state_of(State, (3,))
        bcs = State(**STATE_BCS)
        mapped = jax.vmap(lambda s: fill_ghosts(s, GRID, bcs))(state)
        named = fill_ghosts(state, GRID, bcs)
        for a, b in zip(jax.tree.leaves(mapped), jax.tree.leaves(named), strict=True):
            np.testing.assert_array_equal(a, b)

    def test_another_structure_raises(self):
        with pytest.raises(ValueError):
            fill_ghosts(state_of(dict), GRID, {"h": STATE_BCS["h"]})

    @pytest.mark.parametrize("bcs", [{**STATE_BCS, "u": 0.0}, {**STATE_BCS, "u": WALL}])
    def test_a_leaf_that_is_not_a_bc_raises(self, bcs):
        with pytest.raises(TypeError, match="BC or None"):
            fill_ghosts(state_of(dict), GRID, bcs)

    def test_one_field_needs_a_bc(self):
        with pytest.raises(TypeError, match="must be a BC"):
            fill_ghosts(field(), GRID, {"x": WALL})


def vars_of(state):
    return {name: getattr(state, name) for name in ("h", "u", "v")}


class TestErrors:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: g.Sponge(0.0, 1.5),
            lambda: g.Slip(-0.1),
            lambda: g.Robin(0.0, 0.0, 1.0),
            lambda: g.Extrapolation(0),
            lambda: g.Extrapolation(6),
            lambda: BC(corners="mean"),
            lambda: BC(q=g.Neumann()),
            lambda: BC(x=g.Neumann(), west=g.Dirichlet(0.0)),
            lambda: BC(x=(g.Neumann(),)),
            lambda: BC(x=(g.Periodic(), g.Neumann())),
            lambda: BC(west=g.Periodic()),
        ],
    )
    def test_invalid_conditions_raise(self, make):
        with pytest.raises(ValueError):
            make()

    def test_a_condition_of_the_wrong_type_raises(self):
        with pytest.raises(TypeError, match="boundary condition"):
            BC(x=0.0)

    def test_faces_the_grid_has_not_raise(self):
        with pytest.raises(ValueError, match="z"):
            fill_ghosts(field(), GRID, BC(z=g.Neumann()))

    def test_no_room_for_the_ghost_layers_raises(self):
        with pytest.raises(ValueError, match="no room"):
            fill_ghosts(field((4, 4)), GRID, BC(x=g.Neumann()), width=2)

    @pytest.mark.parametrize(
        "name",
        [
            "Dirichlet1D",
            "Neumann1D",
            "Periodic1D",
            "Outflow1D",
            "Reflective1D",
            "Sponge1D",
            "Robin1D",
            "Extrapolation1D",
            "Slip1D",
            "BoundaryConditionSet",
            "FieldBCSet",
        ],
    )
    def test_the_old_names_are_gone(self, name):
        assert not hasattr(g, name)
