"""A mask's ghost ring is read wrapped along a periodic axis (#551)."""

from __future__ import annotations

import gridax
from gridax import BC, Mask1D, Mask2D, Mask3D, Periodic
from gridax._src.boundary.fill import wrapped_mask
from gridax.schemes import WENO, Adaptive, Centred, Upwind
import jax
import numpy as np
import pytest

RINGS = {
    "wrap": {"mode": "wrap"},
    "dry": {"mode": "constant", "constant_values": False},
    "wet": {"mode": "constant", "constant_values": True},
    "edge": {"mode": "edge"},
}


def _wrap(h: np.ndarray, positions: list[int]) -> np.ndarray:
    for p in positions:
        h = np.moveaxis(h, p, -1)
        h = np.concatenate([h[..., -2:-1], h[..., 1:-1], h[..., 1:2]], axis=-1)
        h = np.moveaxis(h, -1, p)
    return h


@pytest.fixture
def basin():
    rng = np.random.default_rng(0)
    grid = gridax.CartesianCGrid2D.from_interior(16, 12, 1.0, 1.0)
    inner = rng.random((12, 16)) > 0.2
    q, u, v = (rng.normal(size=grid.shape) for _ in range(3))
    return grid, inner, q, u, v


GRIDS = {
    Mask1D: gridax.CartesianCGrid1D.from_interior(7, 1.0),
    Mask2D: gridax.CartesianCGrid2D.from_interior(7, 5, 1.0, 1.0),
    Mask3D: gridax.CartesianCGrid3D.from_interior(7, 5, 2, 1.0, 1.0, 1.0),
}


def _wrapped(mask, grid, axes):
    return wrapped_mask(
        mask, grid, axes, BC(**dict.fromkeys(axes, Periodic())), periodic=False
    )


@pytest.mark.parametrize("cls", [Mask1D, Mask2D, Mask3D], ids=lambda c: c.__name__)
def test_wrapped_is_the_mask_of_the_wrapped_cells(cls) -> None:
    rng = np.random.default_rng(1)
    grid = GRIDS[cls]
    axes = cls._axes
    horizontal = [a for a in axes if a != "z"]
    for _ in range(10):
        h = rng.random(grid.shape) > 0.4
        mask = cls.from_mask(h)
        for periodic in ([axes[-1]], horizontal):
            expected = cls.from_mask(_wrap(h, [axes.index(a) for a in periodic]))
            for got in (
                _wrapped(mask, grid, periodic),
                jax.jit(lambda m, p=periodic: _wrapped(m, grid, p))(mask),
            ):
                pairs = zip(
                    jax.tree_util.tree_leaves(got),
                    jax.tree_util.tree_leaves(expected),
                    strict=True,
                )
                for a, b in pairs:
                    np.testing.assert_array_equal(np.asarray(a), b)
                    assert np.asarray(a).dtype == b.dtype


def test_the_counts_saturate_as_from_mask_does() -> None:
    h = np.ones(300, dtype=bool)
    h[7] = False
    grid = gridax.CartesianCGrid1D.from_interior(298, 1.0)
    got = _wrapped(Mask1D.from_mask(h), grid, ["x"]).stencil_capability
    expected = Mask1D.from_mask(_wrap(h, [0])).stencil_capability
    np.testing.assert_array_equal(np.asarray(got.x_pos), expected.x_pos)
    np.testing.assert_array_equal(np.asarray(got.x_neg), expected.x_neg)


@pytest.mark.parametrize("scheme", [Upwind(1), WENO(5), Centred(4)], ids=repr)
def test_advect_reads_any_ring_as_wrapped(basin, scheme) -> None:
    grid, inner, q, u, v = basin
    out = {
        name: gridax.advect(
            q,
            u,
            v,
            grid,
            scheme=scheme,
            mask=Mask2D.from_grid(grid, np.pad(inner, 1, **pad)),
            bc=BC.periodic(),
        )
        for name, pad in RINGS.items()
    }
    for name in RINGS:
        np.testing.assert_array_equal(out[name], out["wrap"], err_msg=name)
    # the tendency of a periodic, masked domain sums to zero
    total = np.asarray(out["wet"])[1:-1, 1:-1]
    assert abs(total.sum()) < 1e-12 * np.abs(total).sum()


def test_advect_wraps_only_the_periodic_axis(basin) -> None:
    grid, inner, q, u, v = basin
    channel = BC(x=Periodic(), y=gridax.Neumann())
    wet = np.pad(inner, 1, mode="constant", constant_values=True)
    got = gridax.advect(
        q, u, v, grid, scheme=WENO(5), mask=Mask2D.from_grid(grid, wet), bc=channel
    )
    # the ring of the y faces is the caller's; only the x ring wraps
    expected = gridax.advect(
        q,
        u,
        v,
        grid,
        scheme=WENO(5),
        mask=Mask2D.from_grid(grid, _wrap(wet, [1])),
        bc=channel,
    )
    np.testing.assert_array_equal(got, expected)


def test_periodic_edges_wrap_the_mask_too(basin) -> None:
    grid, inner, q, u, v = basin
    scheme = Adaptive(WENO(5), edges="periodic")
    out = [
        gridax.advect(
            q,
            u,
            v,
            grid,
            scheme=scheme,
            mask=Mask2D.from_grid(grid, np.pad(inner, 1, **pad)),
        )
        for pad in RINGS.values()
    ]
    for other in out[1:]:
        np.testing.assert_array_equal(other, out[0])


@pytest.mark.parametrize(
    ("move", "src", "dst"),
    [("difference", "T", "U"), ("interpolate", "T", "U"), ("difference", "U", "T")],
)
def test_the_moves_read_any_ring_as_wrapped(basin, move, src, dst) -> None:
    grid, inner, q, _, _ = basin
    function = getattr(gridax, move)
    kwargs = {"axis": "x"} if move == "difference" else {}
    out = [
        function(
            q,
            grid,
            src=src,
            dst=dst,
            scheme=Centred(4),
            mask=Mask2D.from_grid(grid, np.pad(inner, 1, **pad)),
            bc=BC.periodic(),
            **kwargs,
        )
        for pad in RINGS.values()
    ]
    for other in out[1:]:
        np.testing.assert_array_equal(other, out[0])
