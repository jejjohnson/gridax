"""Scaffolding of the `gridax.pipekit` operator layer (Tier B)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("pipekit_jax")

import equinox as eqx
import gridax as fvx
from gridax.pipekit import (
    GridOperator,
    over_leading,
)
from gridax.pipekit.testing import check_operator, compiled_hlo
import jax
import jax.numpy as jnp
import numpy as np
from pipekit_ops import Operator

SPEC = (8, 6, 2.0, 1.5)
GRID = fvx.CartesianCGrid2D.from_interior(*SPEC)


class ScaledLaplacian(GridOperator):
    """h ↦ κ ∇²h at T-points (κ learnable): a stand-in Tier-B operator."""

    kappa: jax.Array = eqx.field(converter=jnp.asarray, default=1.0)

    def _apply(self, h):
        return self.kappa * fvx.Difference2D(grid=self.grid).laplacian(h)


def _field(seed: int = 0) -> jax.Array:
    g = GRID
    return jax.random.normal(jax.random.key(seed), (g.Ny, g.Nx))


class TestGridField:
    def test_takes_a_grid_or_its_config(self):
        for grid in (GRID, SPEC, list(SPEC)):
            assert ScaledLaplacian(grid=grid).grid == GRID

    @pytest.mark.parametrize("bad", [(8, 6, 2.0), (0, 6, 2.0, 1.5)])
    def test_rejects_bad_specs(self, bad):
        with pytest.raises(ValueError, match="grid"):
            ScaledLaplacian(grid=bad)

    def test_grid_round_trips_json(self):
        op = ScaledLaplacian(grid=GRID, kappa=0.5)
        state = json.loads(json.dumps(op.state))
        assert state["config"]["grid"] == list(SPEC)
        rebuilt = Operator.from_state(state)
        assert rebuilt.grid == GRID

    def test_one_compile_for_grids_of_one_shape(self):
        traces = []

        @eqx.filter_jit
        def run(op, h):
            traces.append(1)
            return op(h)

        for length in (1.0, 2.0):
            grid = fvx.CartesianCGrid2D.from_interior(8, 6, length, 1.5)
            run(ScaledLaplacian(grid=grid), _field())
        assert len(traces) == 1


class TestOverLeading:
    @pytest.mark.parametrize("lead", [(), (3,), (2, 3)])
    def test_lifts_a_2d_function(self, lead):
        def f(x):  # 2-D only: indexes axis 0 positionally
            return x[1:-1] - x[:-2]

        x = jax.random.normal(jax.random.key(1), (*lead, 5, 4))
        out = over_leading(f)(x)
        flat = x.reshape(-1, 5, 4)
        ref = jnp.stack([f(xi) for xi in flat]).reshape(*lead, 3, 4)
        np.testing.assert_array_equal(out, ref)

    def test_tuple_outputs_keep_the_leading_shape(self):
        def f(x, y):
            return x + y, (x * y).sum()

        x = jnp.ones((2, 3, 5, 4))
        a, s = over_leading(f)(x, 2 * x)
        assert a.shape == (2, 3, 5, 4)
        assert s.shape == (2, 3)

    def test_rejects_mismatched_leading_shapes(self):
        with pytest.raises(ValueError, match="leading shapes"):
            over_leading(lambda x, y: x + y)(jnp.ones((2, 5, 4)), jnp.ones((3, 5, 4)))


class TestCheckOperator:
    @pytest.mark.slow  # compiles the operator several times (~2 s)
    def test_passes_for_a_tier_b_operator(self):
        check_operator(ScaledLaplacian(grid=SPEC, kappa=0.5), _field())

    def test_matches_tier_a(self):
        h = _field(2)
        out = ScaledLaplacian(grid=SPEC, kappa=0.5)(h)
        ref = 0.5 * fvx.Difference2D(grid=GRID).laplacian(h)
        np.testing.assert_array_equal(out, ref)

    @pytest.mark.slow  # compiles the operator several times (~2 s)
    def test_flags_a_gather(self):
        class Gathers(GridOperator):
            def _apply(self, h):
                # per-row lookup with a data-dependent index: a gather
                w = jnp.linspace(0.0, 1.0, h.shape[-1])
                return h * w[jnp.argmax(h, axis=-1)][..., None]

        with pytest.raises(AssertionError, match="gather"):
            check_operator(Gathers(grid=SPEC), _field())
        assert "gather(" in compiled_hlo(Gathers(grid=SPEC), _field())


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_import_gridax_does_not_import_pipekit():
    res = _run(
        "import sys, gridax, gridax.solvers; "
        "assert not any(m.split('.')[0] in ('pipekit', 'pipekit_jax') "
        "for m in sys.modules), 'pipekit imported'"
    )
    assert res.returncode == 0, res.stderr


def test_missing_extra_names_it():
    res = _run("import sys; sys.modules['pipekit_jax'] = None; import gridax.pipekit")
    assert res.returncode != 0
    assert "gridax[pipekit]" in res.stderr
