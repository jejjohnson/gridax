"""`.halo` of the pipekit operators and `graph_halo` (#499)."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("pipekit_jax")

import gridax as g
import gridax.pipekit as gp
from gridax.schemes import WENO, Adaptive, Biased, Centred
import jax
import jax.numpy as jnp
import numpy as np
from pipekit_ops import Graph, Input, Operator, Sequential

DX, DY = 0.25, 0.2
NX, NY = 40, 36
SPEC = (NX, NY, NX * DX, NY * DY)
C4, C6 = Centred(4), Centred(6)

#: ``operator -> number of inputs``, on the grid it is given.
MAKERS: dict[str, Any] = {
    "difference": lambda G: (gp.Difference(grid=G, axis="x", src="T", dst="U"), 1),
    "difference4": lambda G: (
        gp.Difference(grid=G, axis="y", src="V", dst="T", scheme=C4),
        1,
    ),
    "difference-biased": lambda G: (
        gp.Difference(grid=G, axis="x", src="T", dst="U", scheme=Biased(3, "left")),
        1,
    ),
    "difference6-narrow": lambda G: (
        gp.Difference(
            grid=G, axis="x", src="U", dst="T", scheme=Adaptive(C6, edges="narrow")
        ),
        1,
    ),
    "interp": lambda G: (gp.Interp(grid=G, src="T", dst="X"), 1),
    "interp4": lambda G: (gp.Interp(grid=G, src="U", dst="V", scheme=C4), 1),
    "divergence": lambda G: (gp.Divergence(grid=G), 2),
    "curl": lambda G: (gp.Curl(grid=G), 2),
    "laplacian": lambda G: (gp.Laplacian(grid=G), 1),
    "gradperp": lambda G: (gp.GradPerp(grid=G), 1),
    "coriolis": lambda G: (gp.Coriolis(grid=G, f0=1.2), 2),
    "fill": lambda G: (gp.FillGhosts(kind="periodic"), 1),
    "vorticity": lambda G: (gp.RelativeVorticity(grid=G), 2),
    "pv": lambda G: (gp.PotentialVorticity(grid=G), 4),
    "enstrophy": lambda G: (gp.Enstrophy(grid=G), 2),
    "ke": lambda G: (gp.KineticEnergy(grid=G), 2),
    "ape": lambda G: (gp.AvailablePE(grid=G, g_prime=0.3, H=0.1), 1),
    "shear": lambda G: (gp.Strain(grid=G, component="shear"), 2),
    "tensor": lambda G: (gp.Strain(grid=G, component="tensor"), 2),
    "magnitude": lambda G: (gp.Strain(grid=G, component="magnitude_squared"), 2),
    "okubo": lambda G: (gp.OkuboWeiss(grid=G), 2),
    "lincomb": lambda G: (gp.LinComb(coeffs=[0.3, 0.7]), 2),
}


def graph(G: Any) -> Graph:
    """A diamond: vorticity of a fourth-order gradient, and a shortcut."""
    h = Input("h")
    u = gp.Difference(grid=G, axis="y", src="T", dst="V", scheme=C4)(h)
    v = gp.Difference(grid=G, axis="x", src="T", dst="U", scheme=C4)(h)
    back = gp.Divergence(grid=G)(v, u)
    total = gp.LinComb(coeffs=[1.0, 0.5])(back, h)
    out = gp.Interp(grid=G, src="T", dst="X", scheme=C6)(total)
    return Graph(inputs={"h": h}, outputs={"out": out, "grad": v})


def fields(n: int) -> list[Any]:
    rng = np.random.default_rng(499)
    return [jnp.asarray(1.0 + rng.random((NY + 2, NX + 2))) for _ in range(n)]


def on_patch(make: Any, n: int, halo: int, inputs: list[Any]) -> tuple[Any, Any]:
    """The result on a patch padded by ``halo``, and that of the whole domain."""
    whole = make(SPEC)(*inputs)
    j0, j1, i0, i1 = 12, 25, 14, 30
    cut = (slice(j0 - halo, j1 + halo), slice(i0 - halo, i1 + halo))
    ny, nx = j1 - j0 + 2 * halo, i1 - i0 + 2 * halo
    spec = (nx - 2, ny - 2, (nx - 2) * DX, (ny - 2) * DY)
    part = make(spec)(*(f[cut] for f in inputs))
    inner = slice(halo, -halo) if halo else slice(None)
    return (
        jax.tree.map(lambda x: x[inner, inner], part),
        jax.tree.map(lambda x: x[j0:j1, i0:i1], whole),
    )


def agrees(make: Any, n: int, halo: int) -> bool:
    got, want = on_patch(make, n, halo, fields(n))
    return all(
        np.allclose(x, y, rtol=1e-12, atol=1e-12)
        for x, y in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True)
    )


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_the_halo_is_enough_and_needed(name: str) -> None:
    op, n = MAKERS[name](SPEC)
    assert isinstance(op.halo, int)

    def make(G: Any) -> Any:
        return MAKERS[name](G)[0]

    assert agrees(make, n, op.halo)
    if op.halo > 0:
        # one cell less and the patch differs from the domain
        assert not agrees(make, n, op.halo - 1)


@pytest.mark.parametrize(
    ("scheme", "halo"),
    [(Centred(2), 1), (C4, 2), (C6, 3), (Biased(3, "left"), 2)],
)
def test_halo_of_a_scheme(scheme: Any, halo: int) -> None:
    assert gp.Difference(grid=SPEC, axis="x", src="T", dst="U", scheme=scheme).halo == (
        halo
    )
    wrapped = Adaptive(scheme, edges="narrow")
    assert gp.Interp(grid=SPEC, src="U", dst="T", scheme=wrapped).halo == halo


def test_a_sequence_sums() -> None:
    to_u = gp.Interp(grid=SPEC, src="T", dst="U", scheme=C4)
    back = gp.Difference(grid=SPEC, axis="x", src="U", dst="T", scheme=C4)
    assert gp.graph_halo(to_u | back) == 4
    assert gp.graph_halo(Sequential([to_u, back, gp.FillGhosts(kind="periodic")])) == 5
    assert gp.graph_halo(Sequential([])) == 0
    assert agrees(
        lambda G: Sequential(
            [
                gp.Interp(grid=G, src="T", dst="U", scheme=C4),
                gp.Difference(grid=G, axis="x", src="U", dst="T", scheme=C4),
            ]
        ),
        1,
        4,
    )


def test_a_graph_takes_its_longest_path() -> None:
    # difference 2, divergence 1, combination 0, interpolation 3
    assert gp.graph_halo(graph(SPEC)) == 6
    assert agrees(graph, 1, 6)


def test_graphs_nest() -> None:
    h = Input("h")
    inner = Graph(inputs={"h": h}, outputs={"u": gp.Laplacian(grid=SPEC)(h)})
    x = Input("x")
    outer = Graph(
        inputs={"x": x},
        outputs={"y": inner(gp.Interp(grid=SPEC, src="U", dst="T")(x)), "x": x},
    )
    assert gp.graph_halo(outer) == 2
    assert gp.graph_halo(Graph(inputs={"x": x}, outputs={"x": x})) == 0


@pytest.mark.parametrize(
    ("scheme", "stages"), [("euler", 1), ("heun", 2), ("rk3_ssp", 3), ("rk4", 4)]
)
def test_a_step_reads_once_per_stage(scheme: str, stages: int) -> None:
    def make(G: Any) -> Any:
        h = Input("h")
        rhs = Graph(inputs={"h": h}, outputs={"h": gp.Laplacian(grid=G)(h)})
        step = gp.ExplicitStep(rhs=rhs, dt=1e-3, scheme=scheme)
        return lambda f: step({"h": f})

    h = Input("h")
    rhs = Graph(inputs={"h": h}, outputs={"h": gp.Laplacian(grid=SPEC)(h)})
    step = gp.ExplicitStep(rhs=rhs, dt=1e-3, scheme=scheme)
    assert gp.graph_halo(step) == stages
    assert agrees(make, 1, stages)


@pytest.mark.parametrize("cls", [gp.TotalEnergy, gp.DomainIntegral, gp.DomainMean])
def test_a_reduction_has_no_halo(cls: Any) -> None:
    with pytest.raises(TypeError, match="has no halo"):
        gp.graph_halo(gp.Laplacian(grid=SPEC) | cls(grid=SPEC))


def test_an_unknown_operator_has_no_halo() -> None:
    class Elsewhere(Operator):
        def _apply(self, x: Any) -> Any:
            return x

    with pytest.raises(TypeError, match="Elsewhere has no halo"):
        gp.graph_halo(Elsewhere())


@pytest.mark.parametrize(
    "scheme",
    [Centred(2), Centred(4, "cell"), Biased(3, "left"), Adaptive(C4, edges="periodic")],
)
def test_a_scheme_round_trips_json(scheme: Any) -> None:
    op = gp.Difference(grid=SPEC, axis="x", src="T", dst="U", scheme=scheme)
    state = json.loads(json.dumps(op.state))
    rebuilt: Any = Operator.from_state(state)
    assert rebuilt.scheme == scheme and rebuilt.grid == op.grid
    f = fields(1)[0]
    np.testing.assert_array_equal(rebuilt(f), op(f))
    np.testing.assert_array_equal(
        op(f), g.difference(f, op.grid, axis="x", src="T", dst="U", scheme=scheme)
    )


def test_a_config_without_a_scheme_is_the_two_point_one() -> None:
    op = gp.Interp(grid=SPEC, src="T", dst="U")
    assert op.scheme == Centred(2)


def test_a_flux_scheme_is_not_a_move() -> None:
    with pytest.raises(TypeError, match="WENO is nonlinear and upwinded"):
        gp.Difference(grid=SPEC, axis="x", src="T", dst="U", scheme=WENO(5))
