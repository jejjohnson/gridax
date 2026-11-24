"""`CFieldAdapter`: pipekit's dim functions on located fields."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pipekit_array")

from gridax import CField, StaggerMismatchError, combine
from gridax._src.fields.adapter import CFieldAdapter
import jax
import jax.numpy as jnp
import numpy as np
import pipekit_array as pa
from pipekit_array.testing import check_adapter
from pipekit_ops import BackendError, get_adapter, guard, signature_of

DIMS = ("time", "y", "x")


def make(loc="T", seed=0, xp=jnp, **kwargs):
    data = np.random.default_rng(seed).normal(size=(3, 4, 5))
    return CField(xp.asarray(data), DIMS, loc=loc, **kwargs)


@pytest.mark.parametrize("loc", ["T", "U", "V", "X"])
@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
def test_the_adapter_contract(loc, xp):
    check_adapter(
        CFieldAdapter(),
        lambda data, dims, labels: CField(xp.asarray(data), dims, labels, loc=loc),
    )


@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
class TestDemotion:
    def test_a_v_field_averaged_over_x_stays_staggered_in_y(self, xp):
        out = pa.reduce(make("V", xp=xp), "mean", "x")
        assert isinstance(out, CField)
        assert (out.dims, out.stagger) == (("time", "y"), (("y", "right"),))

    def test_an_x_point_field_keeps_the_axis_left(self, xp):
        assert pa.reduce(make("X", xp=xp), "mean", "y").stagger == (("x", "right"),)

    def test_without_a_grid_axis_it_is_a_dimarray(self, xp):
        out = pa.reduce(make("V", xp=xp), "mean", ("y", "x"))
        assert type(out) is pa.DimArray
        assert out.dims == ("time",)

    def test_isel_slices_keep_the_position_and_ints_drop_the_axis(self, xp):
        u = make("U", xp=xp, labels={"time": ("a", "b", "c")})
        assert pa.isel(u, x=slice(1, 3)).loc == "U"
        assert pa.isel(u, x=0).loc == "T"
        assert pa.isel(u, time=0).dims == ("y", "x")
        assert pa.isel(u, time=slice(1, None)).labels == {"time": ("b", "c")}


@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
class TestValuesOffTheirPoint:
    """`diff` and `coarsen` never claim a point their values are not on (#563)."""

    @pytest.mark.parametrize("loc", ["T", "U", "V", "X"])
    @pytest.mark.parametrize("dim", ["y", "x"])
    def test_diff_along_a_grid_axis_has_no_point(self, xp, loc, dim):
        out = pa.diff(make(loc, xp=xp), dim)
        assert type(out) is pa.DimArray
        want = np.diff(np.asarray(make(loc).data), axis=DIMS.index(dim))
        np.testing.assert_allclose(out.data, want, rtol=1e-6)

    def test_diff_along_another_dim_keeps_it(self, xp):
        out = pa.diff(make("U", xp=xp, labels={"time": ("a", "b", "c")}), "time")
        assert (out.loc, out.labels) == ("U", {"time": ("b", "c")})
        assert pa.diff(make("U", xp=xp), "x", 0).loc == "U"

    def test_a_difference_cannot_join_a_field_of_its_old_point(self, xp):
        u = make("U", xp=xp)
        with pytest.raises(ValueError):
            combine(jnp.add, pa.isel(u, x=slice(1, None)), pa.diff(u, "x"))

    def test_coarsen_along_a_staggered_axis_has_no_point(self, xp):
        u = pa.isel(make("U", xp=xp), x=slice(0, 4))
        assert type(pa.coarsen(u, "mean", x=2)) is pa.DimArray
        assert type(pa.coarsen(u, "mean", y=2, x=2)) is pa.DimArray

    def test_coarsen_along_other_axes_keeps_it(self, xp):
        assert pa.coarsen(make("U", xp=xp), "mean", y=2).loc == "U"
        assert pa.coarsen(make("U", xp=xp), "mean", time=3).loc == "U"
        assert pa.coarsen(make("T", xp=xp), "mean", y=2).loc == "T"

    def test_moves_by_whole_cells_keep_it(self, xp):
        u = make("U", xp=xp)
        for out in (pa.roll(u, x=1), pa.shift(u, x=1), pa.pad(u, x=1)):
            assert out.loc == "U"
        assert pa.window(u, x=2).loc == "U"


class TestPoints:
    def test_combine_is_gridaxs(self):
        a, b = make("U"), make("U", seed=1)
        got, want = pa.combine(jnp.subtract, a, b), combine(jnp.subtract, a, b)
        assert (got.dims, got.stagger) == (want.dims, want.stagger)
        np.testing.assert_array_equal(got.data, want.data)

    @pytest.mark.parametrize(
        "fn",
        [
            lambda a, b: pa.combine(jnp.add, a, b),
            lambda a, b: pa.stack([a, b], "run"),
            lambda a, b: pa.concat([a, b], "time"),
        ],
        ids=["combine", "stack", "concat"],
    )
    def test_fields_on_different_points_raise(self, fn):
        with pytest.raises(StaggerMismatchError):
            fn(make("U"), make("T"))

    def test_signature_names_the_dims(self):
        assert signature_of(make("U")).matches("time y x")


class TestMapOver:
    @pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
    @pytest.mark.parametrize("dim", ["time", "y"])
    def test_maps_over_a_named_dim(self, xp, dim):
        f = make("U", xp=xp, labels={"time": ("a", "b", "c")})
        seen = []

        def fn(item):
            seen.append((item.dims, item.loc))
            return combine(lambda a: 2.0 * a, item)

        out = pa.map_over(f, fn, dim)
        rest = tuple(d for d in DIMS if d != dim)
        assert set(seen) == {(rest, "U")}
        assert (out.dims, out.loc, out.labels) == ((dim, *rest), "U", f.labels)
        np.testing.assert_array_equal(
            out.data, 2.0 * xp.moveaxis(f.data, DIMS.index(dim), 0)
        )

    def test_is_one_trace_on_jax(self):
        traces = []
        pa.map_over(make(), lambda item: traces.append(1) or item, "time")
        assert len(traces) == 1

    def test_the_function_must_return_a_carrier(self):
        with pytest.raises(TypeError, match="named carrier"):
            pa.map_over(make(), lambda item: item.data, "time")

    def test_an_unknown_dim_raises(self):
        with pytest.raises(KeyError, match="unknown dim"):
            pa.map_over(make(), lambda item: item, "member")

    def test_a_function_that_cannot_be_traced_is_looped_over(self):
        def fn(item):
            return pa.reduce(item, "max" if float(item.data.sum()) else "min", "x")

        out = pa.map_over(make("V"), fn, "time")
        assert (out.dims, out.loc) == (("time", "y"), "V")
        np.testing.assert_array_equal(out.data, make("V").data.max(axis=-1))


class TestGuard:
    """`gridax.combine` joins NumPy to JAX, and pipekit's guard reports it."""

    def test_a_mix_is_on_jax_and_reported(self):
        a, b = make("U", xp=np), make("U", seed=1)
        out = combine(jnp.add, a, b)
        assert isinstance(out.data, jax.Array)
        np.testing.assert_allclose(out.data, np.asarray(a.data) + b.data, rtol=1e-6)
        with guard("raise"):
            with pytest.raises(BackendError, match="combine: NumPy data joined jax"):
                combine(jnp.add, a, b)

    def test_numbers_and_constants_of_a_trace_are_not_moves(self):
        a, b = make("U", xp=np), make("U", seed=1)
        with guard("raise"):
            combine(lambda v: 2.0 * v, b)
            combine(jnp.multiply, b, 2.0)
            combine(np.add, a, a)
            jax.jit(lambda f: combine(jnp.add, a, f))(b)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "imports",
    ["import gridax, pipekit_ops", "import pipekit_ops, gridax"],
    ids=["gridax first", "pipekit first"],
)
def test_registered_whichever_is_imported_first(imports):
    res = _run(
        f"""
        {imports}
        import numpy as np

        f = gridax.CField(np.zeros((2, 3)), ("y", "x"), loc="U")
        assert type(pipekit_ops.get_adapter(f)).__name__ == "CFieldAdapter"
        """
    )
    assert res.returncode == 0, res.stderr


def test_import_gridax_imports_only_the_light_half_of_pipekit():
    res = _run(
        """
        import sys, gridax

        heads = {m.split(".")[0] for m in sys.modules}
        assert not heads & {"pipekit_jax", "pipekit_cycle", "xarray", "sklearn", "einx"}
        """
    )
    assert res.returncode == 0, res.stderr


def test_under_jit():
    f = make("V")
    out = jax.jit(lambda f: pa.reduce(f, "mean", "x"))(f)
    assert out.stagger == (("y", "right"),)
    assert get_adapter(out) is get_adapter(f)
