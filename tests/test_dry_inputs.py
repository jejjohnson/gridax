"""No operation with a mask reads a dry value (#205).

Whatever a field holds on land, NaN too, the result is the same, and so is its
gradient. The ghost ring is the caller's and is read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gridax as g
from gridax._src.api import OPERATIONS
from gridax._src.gridops import defined_on
from gridax._src.operators.triples import PAIRS, TRIPLES
from gridax.schemes import Adaptive, Centred, Upwind
import jax
import jax.numpy as jnp
import numpy as np
import pytest

RNG = np.random.default_rng(205)
GRIDS = {
    "1d": g.CartesianCGrid1D.from_interior(14, 1.0),
    "2d": g.CartesianCGrid2D.from_interior(12, 10, 2.0, 3.0),
    "3d": g.CartesianCGrid3D.from_interior(10, 8, 4, 2.0, 3.0, 5.0),
    "sphere": g.SphericalCGrid2D.from_interior(
        12, 10, (0.0, 40.0), (-50.0, 50.0), R=2.0
    ),
    "sphere3d": g.SphericalCGrid3D.from_interior(
        10, 8, 4, (0.0, 40.0), (-50.0, 50.0), 5.0, R=2.0
    ),
    "stretched": g.CartesianAGrid2D.from_coords(
        y=np.cumsum(1.0 + RNG.random(10)), x=np.cumsum(1.0 + RNG.random(12))
    ).staggered(),
}
MASKS = {1: g.Mask1D, 2: g.Mask2D, 3: g.Mask3D}

#: The points of the fields of an operation, and its keywords.
CALLS: dict[str, tuple[str, dict[str, Any]]] = {
    "gradient": ("T", {}),
    "divergence": ("UV", {}),
    "curl": ("UV", {}),
    "grad_perp-X": ("X", {"at": "X"}),
    "grad_perp-T": ("T", {"at": "T"}),
    "coriolis": ("UV", {"f": 1.0}),
    "advect_momentum": ("UV", {"form": "blend"}),
    "pv_flux": ("XUV", {"form": "blend"}),
    "kinetic_energy": ("UV", {}),
    "bernoulli_potential": ("TUV", {"gravity": 9.0}),
    "shear_strain": ("UV", {}),
    "tensor_strain": ("UV", {}),
    "strain_rate_squared": ("UV", {}),
    "okubo_weiss_parameter": ("UV", {}),
    "sw_potential_vorticity": ("UVTT", {}),
    "qg_potential_vorticity": ("T", {"f0": 1.0, "beta": 0.1}),
    "vertical_velocity": ("UV", {}),
    "vortex_force-energy": ("XUV", {"form": "energy"}),
    "vortex_force-enstrophy": ("XUV", {"form": "enstrophy"}),
    "vortex_force-lamb": ("XUV", {"form": "arakawa_lamb"}),
    "advect_momentum-lamb": ("UV", {"form": "arakawa_lamb"}),
    "arakawa_jacobian": ("TT", {}),
    "diffuse": ("T", {"kappa": 1.0}),
    "diffusive_fluxes": ("T", {"kappa": 1.0}),
    "biharmonic": ("T", {"kappa4": 1.0}),
    "advect": ("TUV", {"scheme": Upwind(3)}),
    "advect_vertical": ("TW", {"scheme": Upwind(3)}),
    "diffuse_vertical": ("T", {"kappa": 1.0}),
    "area_sum": ("T", {}),
    "volume_sum": ("T", {}),
}


def masked(grid: Any) -> Any:
    return MASKS[len(grid.axes)].from_grid(grid, RNG.random(grid.shape) > 0.2)


def filled(f: Any, mask: Any, at: str, value: float) -> Any:
    """``f`` with ``value`` on its dry points off the ghost ring."""
    keep = np.ones(f.shape, dtype=bool)
    inner = (slice(1, -1),) * f.ndim
    keep[inner] = np.asarray(mask.wet(at=at))[inner]
    return np.where(keep, f, value)


def check(grid: Any, points: str, call: Callable[..., Any]) -> None:
    mask = masked(grid)
    fields = [RNG.normal(size=grid.shape) + 3.0 for _ in points]
    out = [
        jax.tree.leaves(
            call(
                *(filled(f, mask, p, v) for f, p in zip(fields, points, strict=True)),
                mask=mask,
            )
        )
        for v in (0.0, 1e3, np.nan)
    ]
    for other in out[1:]:
        for a, b in zip(out[0], other, strict=True):
            assert np.isfinite(a).all()
            np.testing.assert_array_equal(a, b)

    def loss(*args: Any) -> Any:
        return sum((x**2).sum() for x in jax.tree.leaves(call(*args, mask=mask)))

    args = [
        jnp.asarray(filled(f, mask, p, np.nan))
        for f, p in zip(fields, points, strict=True)
    ]
    for gradient in jax.grad(loss, tuple(range(len(args))))(*args):
        assert np.isfinite(gradient).all()


CASES = [
    (name, call)
    for name, grid in GRIDS.items()
    for call in CALLS
    if defined_on(OPERATIONS[call.split("-")[0]], grid)
    and not (name == "1d" and len(CALLS[call][0]) > 1)
    and not (name == "stretched" and call == "grad_perp-T")
]


@pytest.mark.parametrize(("name", "call"), CASES)
def test_operations(name: str, call: str) -> None:
    grid = GRIDS[name]
    points, kwargs = CALLS[call]
    op = getattr(g, call.split("-")[0])
    check(grid, points, lambda *f, mask: op(*f, grid, mask=mask, **kwargs))


@pytest.mark.parametrize("name", ["2d", "sphere3d"])
def test_a_field_of_diffusivity(name: str) -> None:
    grid = GRIDS[name]
    check(
        grid,
        "TT",
        lambda f, kappa, mask: g.diffuse(f, grid, kappa=kappa, mask=mask),
    )


def test_every_operation_is_here() -> None:
    covered = {c.split("-")[0] for c in CALLS}
    elsewhere = {"difference", "interpolate", "laplacian", "geostrophic_velocity"}
    # on a flux, whose dry faces `advect` zeroes; and quotients of the sums
    others = {"reconstruct", "area_mean", "volume_mean"}
    assert covered | elsewhere | others == set(OPERATIONS)


@pytest.mark.slow
@pytest.mark.parametrize("scheme", [Centred(2), Centred(4)], ids=repr)
@pytest.mark.parametrize("name", ["1d", "2d", "sphere3d"])
def test_moves(name: str, scheme: Any) -> None:
    grid = GRIDS[name]
    for axis, src, dst in TRIPLES[grid.axes]:
        check(
            grid,
            src,
            lambda f, mask, a=axis, s=src, d=dst: g.difference(
                f, grid, axis=a, src=s, dst=d, scheme=scheme, mask=mask
            ),
        )
    for src, dst in PAIRS[grid.axes]:
        check(
            grid,
            src,
            lambda f, mask, s=src, d=dst: g.interpolate(
                f, grid, src=s, dst=d, scheme=scheme, mask=mask
            ),
        )


def test_a_move_can_read_land_if_asked_to() -> None:
    grid, mask = GRIDS["2d"], masked(GRIDS["2d"])
    u = filled(np.ones(grid.shape), mask, "U", 1e3)
    scheme = Adaptive(Centred(2), mask="none")
    out = g.interpolate(u, grid, src="U", dst="T", scheme=scheme, mask=mask)
    assert out.max() > 100.0


@pytest.mark.parametrize("name", list(GRIDS))
def test_laplacian(name: str) -> None:
    grid = GRIDS[name]
    points = ["T"] if name in ("1d", "stretched") else ["T", "U", "V", "X"]
    for at in points:
        check(grid, at, lambda f, mask, at=at: g.laplacian(f, grid, at=at, mask=mask))


@pytest.mark.parametrize("name", ["2d", "sphere", "3d"])
def test_the_laplacian_has_no_flux_through_a_coast(name: str) -> None:
    """Away from the walls it is ``diffuse`` with κ = 1."""
    grid, mask = GRIDS[name], masked(GRIDS[name])
    f = RNG.normal(size=grid.shape)
    inner = (
        *(slice(1, -1),) * (len(grid.axes) - 2),
        slice(2, -2),
        slice(2, -2),
    )
    np.testing.assert_allclose(
        g.laplacian(f, grid, mask=mask)[inner],
        g.diffuse(f, grid, kappa=1.0, mask=mask)[inner],
        rtol=1e-10,
        atol=1e-10,
    )


def test_a_constant_has_no_laplacian_next_to_land() -> None:
    grid, mask = GRIDS["stretched"], masked(GRIDS["stretched"])
    f = filled(np.full(grid.shape, 5.0), mask, "T", 0.0)
    np.testing.assert_allclose(g.laplacian(f, grid, mask=mask), 0.0, atol=1e-12)


def test_geostrophic_velocity() -> None:
    grid = GRIDS["sphere"]
    f = np.full(grid.shape, 1e-4)
    check(
        grid,
        "T",
        lambda h, mask: g.geostrophic_velocity(h, grid, f=f, gravity=1e-4, mask=mask),
    )
    # a plane keeps its slope next to land
    mask = masked(grid)
    y = np.broadcast_to(grid.coords("y")[:, None], grid.shape)
    h = filled(y, mask, "T", 1e3)
    u, v = g.geostrophic_velocity(h, grid, f=f, gravity=1e-4, mask=mask)
    # a face with no wet difference beside it has no slope
    wet, rows = np.asarray(mask.wet()), np.ones(grid.shape, bool)
    rows[1:-1, 1:-1] = wet[1:-1, 1:-1]
    beside = rows[1:] & rows[:-1]  # [j+1/2, i]
    some = beside[1:, 1:-1] | beside[:-1, 1:-1] | beside[1:, 2:] | beside[:-1, 2:]
    at_u = np.asarray(mask.wet(at="U"))[1:-1, 1:-1] & some
    np.testing.assert_allclose(u[1:-1, 1:-1][at_u] * 2.0, -1.0, rtol=1e-9)
    np.testing.assert_allclose(v, 0.0, atol=1e-12)


def test_the_velocity_of_a_streamfunction_is_non_divergent_next_to_land() -> None:
    grid, mask = GRIDS["2d"], masked(GRIDS["2d"])
    psi = filled(RNG.normal(size=grid.shape), mask, "T", np.nan)
    u, v = g.grad_perp(psi, grid, at="T", mask=mask)
    div = g.divergence(u, v, grid)
    # the cells with no dry face
    ok = np.asarray(mask.wet())[1:-1, 1:-1]
    at_u, at_v = (np.asarray(mask.wet(at=p)) for p in "UV")
    ok = ok & at_u[1:-1, 1:-1] & at_u[1:-1, :-2] & at_v[1:-1, 1:-1] & at_v[:-2, 1:-1]
    inner = div[1:-1, 1:-1][1:, 1:][ok[1:, 1:]]
    np.testing.assert_allclose(inner, 0.0, atol=1e-11)


@pytest.mark.parametrize("name", list(GRIDS))
def test_no_land_no_difference(name: str) -> None:
    grid = GRIDS[name]
    sea = MASKS[len(grid.axes)].from_grid(grid, np.ones(grid.shape, dtype=bool))
    for call, (points, kwargs) in CALLS.items():
        op = OPERATIONS[call.split("-")[0]]
        if (name, call) not in CASES:
            continue
        fields = [RNG.normal(size=grid.shape) + 3.0 for _ in points]
        for a, b in zip(
            jax.tree.leaves(op(*fields, grid, mask=sea, **kwargs)),
            jax.tree.leaves(op(*fields, grid, **kwargs)),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
    f = RNG.normal(size=grid.shape)
    np.testing.assert_array_equal(g.laplacian(f, grid, mask=sea), g.laplacian(f, grid))
