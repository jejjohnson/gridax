"""`CField`, `combine` and `gridax.fields`: located fields and element-wise maths."""

from __future__ import annotations

import equinox as eqx
import gridax
from gridax import CField, StaggerMismatchError, combine, fields
import jax
import jax.numpy as jnp
import numpy as np
import pytest

NL, NY, NX = 3, 6, 8
DIMS = ("layer", "y", "x")


def make(loc="T", xp=jnp, seed=0, dims=DIMS, **kwargs):
    shape = {"layer": NL, "y": NY, "x": NX, "z": 4, "member": 2}
    data = np.random.default_rng(seed).normal(size=[shape[d] for d in dims])
    return CField(xp.asarray(data), dims=dims, loc=loc, **kwargs)


class TestConstruction:
    @pytest.mark.parametrize(
        ("loc", "stagger"),
        [
            ("T", ()),
            ("U", (("x", "right"),)),
            ("V", (("y", "right"),)),
            ("X", (("x", "right"), ("y", "right"))),
        ],
    )
    def test_loc_is_a_shorthand_for_the_stagger(self, loc, stagger):
        f = make(loc)
        assert f.stagger == stagger
        assert f.loc == loc
        same = CField(f.data, DIMS, stagger=dict(stagger))
        assert same.stagger == stagger

    def test_w_is_staggered_along_z(self):
        assert make("W", dims=("z", "y", "x")).stagger == (("z", "right"),)

    def test_the_stagger_is_normalised(self):
        f = CField(
            np.zeros((NY, NX)),
            ("y", "x"),
            stagger=(("y", "right"), ("x", "center")),
        )
        assert f.stagger == (("y", "right"),)

    @pytest.mark.parametrize("position", ["left", "outer", "inner"])
    def test_other_positions_are_described_but_have_no_name(self, position):
        f = CField(np.zeros((NY, NX)), ("y", "x"), stagger={"x": position})
        assert f.loc is None
        assert dict(f.stagger) == {"x": position}

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dims": ("y",)}, "data of shape"),
            ({"dims": ("x", "x")}, "unique"),
            ({"loc": "W"}, "not in dims"),
            ({"loc": "Q"}, "not a point"),
            ({"stagger": {"x": "east"}}, "position"),
            ({"stagger": (("x", "left"), ("x", "right"))}, "repeats"),
            ({"stagger": {"x": "left"}, "loc": "U"}, "not both"),
            ({"labels": {"t": (0,)}}, "not in dims"),
            ({"labels": {"y": (0,)}}, "labels for dim"),
            ({"labels": {"x": range(NX)}, "loc": "U"}, "staggered axis"),
        ],
    )
    def test_invalid_metadata_raises(self, kwargs, match):
        kwargs = {"dims": ("y", "x"), **kwargs}
        with pytest.raises(ValueError, match=match):
            CField(np.zeros((NY, NX)), **kwargs)

    def test_like_asserts_the_point_of_a_full_array(self):
        u = make("U", labels={"layer": ("a", "b", "c")})
        g = u.like(np.ones(u.shape))
        assert (g.dims, g.stagger, g.labels) == (u.dims, u.stagger, u.labels)
        with pytest.raises(ValueError, match="not like"):
            u.like(np.ones((NY, NX)))
        with pytest.raises(ValueError, match="plain array"):
            u.like(make("T"))

    def test_with_data_may_change_the_sizes(self):
        assert make("U").with_data(np.zeros((1, 4, 4))).loc == "U"

    def test_transpose_keeps_the_metadata(self):
        u = make("U", labels={"layer": ("a", "b", "c")})
        t = u.transpose(("x", "layer", "y"))
        assert t.sizes == u.sizes
        assert (t.stagger, t.labels) == (u.stagger, u.labels)
        np.testing.assert_array_equal(t.data, jnp.transpose(u.data, (2, 0, 1)))
        with pytest.raises(ValueError, match="order"):
            u.transpose(("x", "y"))

    @pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
    def test_arithmetic_operators_raise(self, xp):
        u, h = make("U", xp), make("T", xp)
        for op in (
            lambda: u * h,
            lambda: u + u,
            lambda: 2.0 * u,
            lambda: xp.ones(u.shape) * u,
        ):
            with pytest.raises(TypeError):
                op()


class TestCombine:
    @pytest.mark.parametrize("jit", [lambda f: f, jax.jit], ids=["eager", "jit"])
    def test_fields_on_different_points_raise(self, jit):
        """U × T raises at trace time."""
        u, h = make("U"), make("T")
        with pytest.raises(StaggerMismatchError, match="T, not at U"):
            jit(lambda a, b: combine(jnp.multiply, a, b))(u, h)

    @pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
    def test_fields_on_one_point_combine(self, xp):
        a, b = make("U", xp, 0), make("U", xp, 1)
        out = combine(lambda p, q: p - 2.0 * q, a, b)
        assert (out.dims, out.stagger) == (a.dims, a.stagger)
        np.testing.assert_array_equal(out.data, a.data - 2.0 * b.data)

    def test_fields_are_aligned_by_name(self):
        a, b = make("V", seed=0), make("V", seed=1)
        out = combine(jnp.add, a, b.transpose(("x", "layer", "y")))
        assert out.dims == a.dims
        np.testing.assert_array_equal(out.data, a.data + b.data)

    def test_fields_with_other_dims_sizes_or_labels_raise(self):
        a = make("T", labels={"layer": ("a", "b", "c")})
        with pytest.raises(ValueError, match="dims"):
            combine(jnp.add, a, make("T", dims=("y", "x")))
        with pytest.raises(ValueError, match="sizes"):
            combine(jnp.add, a, CField(jnp.ones((NL, NY, 1)), DIMS))
        with pytest.raises(ValueError, match="labels"):
            combine(jnp.add, a, make("T", labels={"layer": ("a", "b", "z")}))

    def test_labels_of_every_operand_are_kept(self):
        a, b = make("T"), make("T", labels={"layer": ("a", "b", "c")})
        assert combine(jnp.add, a, b).labels == b.labels

    @pytest.mark.parametrize("loc", ["T", "U", "V", "X"])
    @pytest.mark.parametrize(
        "shape", [(), (NL, 1, 1), (1, 1)], ids=["scalar", "column", "ones"]
    )
    def test_plain_arrays_may_broadcast(self, loc, shape):
        f = make(loc)
        c = jnp.ones(shape) * 2.0
        np.testing.assert_array_equal(
            combine(jnp.multiply, f, c).data, f.data * c, strict=True
        )

    @pytest.mark.parametrize("loc", ["T", "U", "V", "X"])
    @pytest.mark.parametrize("shape", [(NL, NY, NX), (NY, NX)])
    def test_a_full_plain_array_raises_and_names_like(self, loc, shape):
        f = make(loc)
        with pytest.raises(ValueError, match=r"f\.like"):
            combine(jnp.multiply, f, jnp.ones(shape))
        if shape == f.shape:
            combine(jnp.multiply, f, f.like(jnp.ones(shape)))

    def test_a_profile_along_an_unstaggered_axis_is_allowed(self):
        """f(y) on U rows is the latitude of U; on V rows it could be T's."""
        profile = jnp.arange(NY, dtype=float)[:, None]
        combine(jnp.multiply, make("U"), profile)
        with pytest.raises(ValueError, match=r"\['y'\]"):
            combine(jnp.multiply, make("V"), profile)

    def test_plain_arrays_that_do_not_broadcast_raise(self):
        with pytest.raises(ValueError, match="does not broadcast"):
            combine(jnp.add, make("U"), jnp.ones((5, 1, 1)))
        with pytest.raises(ValueError, match="more axes"):
            combine(jnp.add, make("U"), jnp.ones((2, NL, 1, 1)))

    def test_a_reduction_is_not_element_wise(self):
        with pytest.raises(ValueError, match="element-wise"):
            combine(lambda a: jnp.sum(a, axis=0), make("T"))

    def test_without_a_field_raises(self):
        with pytest.raises(TypeError, match="CField"):
            combine(jnp.add, jnp.ones(3), 1.0)

    def test_the_program_is_the_one_on_bare_arrays(self):
        a, b = make("U", seed=0), make("U", seed=1)

        def fn(p, q):
            return 0.5 * p - jnp.sin(q)

        located = jax.make_jaxpr(lambda p, q: combine(fn, a.like(p), b.like(q)).data)
        assert str(located(a.data, b.data)) == str(jax.make_jaxpr(fn)(a.data, b.data))


class TestElementwise:
    @pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
    @pytest.mark.parametrize("name", ["abs", "cos", "exp", "sign", "sin", "tanh"])
    def test_unary(self, name, xp):
        f = make("V", xp)
        out = getattr(fields, name)(f)
        assert out.loc == "V"
        np.testing.assert_array_equal(out.data, getattr(xp, name)(f.data))

    @pytest.mark.parametrize("name", ["sqrt", "log"])
    def test_unary_of_a_positive_field(self, name):
        f = fields.exp(make("U"))
        np.testing.assert_array_equal(
            getattr(fields, name)(f).data, getattr(jnp, name)(f.data)
        )

    def test_binary_and_keywords(self):
        a, b = make("U", seed=0), make("U", seed=1)
        np.testing.assert_array_equal(
            fields.maximum(a, b).data, jnp.maximum(a.data, b.data)
        )
        np.testing.assert_array_equal(
            fields.minimum(a, 0.0).data, jnp.minimum(a.data, 0.0)
        )
        np.testing.assert_array_equal(
            fields.clip(a, min=-0.1, max=0.1).data, jnp.clip(a.data, -0.1, 0.1)
        )
        wet = a.like(a.data > 0)
        np.testing.assert_array_equal(
            fields.where(wet, a, 0.0).data, jnp.where(a.data > 0, a.data, 0.0)
        )

    def test_points_are_checked(self):
        with pytest.raises(StaggerMismatchError):
            fields.maximum(make("U"), make("V"))

    def test_without_a_field_raises(self):
        with pytest.raises(TypeError, match=r"fields\.sqrt"):
            fields.sqrt(jnp.ones(3))

    def test_every_listed_function_is_public(self):
        from gridax._src.fields.combine import ELEMENTWISE

        assert set(ELEMENTWISE) <= set(fields.__all__)
        assert all(callable(getattr(fields, name)) for name in ELEMENTWISE)


class TestPytree:
    def test_the_data_is_the_only_leaf(self):
        f = make("U", labels={"layer": ("a", "b", "c")})
        leaves, treedef = jax.tree.flatten(f)
        assert len(leaves) == 1
        assert hash(treedef) == hash(jax.tree.structure(f))
        g = jax.tree.unflatten(treedef, leaves)
        assert (g.dims, g.labels, g.stagger) == (f.dims, f.labels, f.stagger)

    @pytest.mark.parametrize("jit", [jax.jit, eqx.filter_jit])
    def test_one_compile_per_point(self, jit):
        traces = []

        @jit
        def double(f):
            traces.append(f.loc)
            return combine(lambda a: 2.0 * a, f)

        for loc in ("U", "U", "V"):
            assert double(make(loc)).loc == loc
        assert traces == ["U", "V"]

    def test_a_stack_has_the_tree_structure_of_one_field(self):
        one = CField(jnp.ones((4, 5)), ("y", "x"), loc="U")
        stack = jax.tree.map(lambda x: jnp.zeros((3, *x.shape)), one)
        assert (stack.dims, stack.loc) == (("batch", "y", "x"), "U")
        assert jax.tree.structure(stack) == jax.tree.structure(one)
        # what a solver does with its output buffer
        written = jax.tree.map(lambda buffer, x: buffer.at[1].set(x), stack, one)
        assert float(written.data.sum()) == 20.0
        twice = jax.tree.map(lambda x: x[None], stack)
        assert twice.dims == ("batch", "batch_1", "y", "x")
        assert jax.tree.structure(twice) == jax.tree.structure(one)

    def test_dims_of_another_name_are_part_of_the_structure(self):
        members = CField(jnp.ones((3, 4)), ("member", "x"))
        times = CField(jnp.ones((3, 4)), ("time", "x"))
        assert jax.tree.structure(members) != jax.tree.structure(times)
        leaves, structure = jax.tree.flatten(members)
        assert structure.unflatten(leaves).dims == ("member", "x")

    def test_diffrax_saves_located_states(self):
        dfx = pytest.importorskip("diffrax")
        y0 = {"u": CField(jnp.ones((4, 5)), ("y", "x"), loc="U")}
        plain = {"u": y0["u"].data}

        def solve(y):
            return dfx.diffeqsolve(
                dfx.ODETerm(lambda t, y, args: jax.tree.map(jnp.negative, y)),
                dfx.Tsit5(),
                0.0,
                1.0,
                0.1,
                y,
                saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 1.0, 4)),
            ).ys["u"]

        out = solve(y0)
        assert (out.dims, out.loc) == (("batch", "y", "x"), "U")
        np.testing.assert_array_equal(out.data, solve(plain))

    def test_vmap_drops_the_first_dim_and_returns_it_as_batch(self):
        u = make("U", dims=("member", "y", "x"))
        seen = []

        def inner(f):
            seen.append((f.dims, f.loc))
            return combine(lambda a: 2.0 * a, f)

        out = jax.vmap(inner)(u)
        assert seen == [(("y", "x"), "U")]
        assert (out.dims, out.loc) == (("batch", "y", "x"), "U")
        np.testing.assert_array_equal(out.data, 2.0 * u.data)

    def test_scan_carries_a_field(self):
        u = make("U")
        out, _ = jax.lax.scan(
            lambda f, _: (combine(lambda a: 0.5 * a, f), None), u, length=3
        )
        assert out.loc == "U"
        np.testing.assert_array_equal(out.data, u.data / 8.0)

    def test_gradient_is_a_field_on_the_same_point(self):
        u = make("U")
        g = jax.grad(lambda f: jnp.sum(combine(jnp.square, f).data))(u)
        assert g.loc == "U"
        np.testing.assert_allclose(g.data, 2.0 * u.data)

    def test_in_an_equinox_state(self):
        class State(eqx.Module):
            h: CField
            u: CField

        s = State(h=make("T"), u=make("U"))
        out = eqx.filter_jit(lambda s: jax.tree.map(lambda a: 2.0 * a, s))(s)
        assert (out.h.loc, out.u.loc) == ("T", "U")


class TestGridAcceptsAStagger:
    def test_spacing_and_coords_at_a_fields_point(self):
        grid = gridax.SphericalCGrid2D.from_interior(
            NX - 2, NY - 2, (0.0, 40.0), (-30.0, 30.0)
        )
        for loc in "TUVX":
            f = make(loc)
            np.testing.assert_array_equal(
                grid.spacing("x", at=f.stagger), grid.spacing("x", at=loc)
            )
            np.testing.assert_array_equal(
                grid.coords("y", at=f.stagger), grid.coords("y", at=loc)
            )

    def test_a_left_stagger_raises(self):
        grid = gridax.CartesianCGrid2D.from_interior(4, 4, 1.0, 1.0)
        with pytest.raises(ValueError, match="left"):
            grid.spacing("x", at=(("x", "left"),))
