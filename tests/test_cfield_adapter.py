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
from pipekit_ops import get_adapter, signature_of

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
        with pytest.raises(ValueError, match="named carrier"):
            pa.map_over(make(), lambda item: item.data, "time")

    def test_an_unknown_dim_raises(self):
        with pytest.raises(ValueError, match="not a dim"):
            pa.map_over(make(), lambda item: item, "member")


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


def test_import_gridax_does_not_import_pipekit():
    res = _run(
        """
        import sys, gridax

        heads = {m.split(".")[0] for m in sys.modules}
        assert not heads & {"pipekit_ops", "pipekit_array", "pipekit_jax"}
        assert "gridax._src.fields.adapter" not in sys.modules
        """
    )
    assert res.returncode == 0, res.stderr


def test_under_jit():
    f = make("V")
    out = jax.jit(lambda f: pa.reduce(f, "mean", "x"))(f)
    assert out.stagger == (("y", "right"),)
    assert get_adapter(out) is get_adapter(f)
