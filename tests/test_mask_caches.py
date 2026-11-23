"""The tiers, the coastal ghosts and the kinds derived from a mask (#506)."""

from __future__ import annotations

from typing import Any

import gridax
from gridax import Mask1D, Mask2D, Mask3D, count_contiguous
from gridax.schemes import TVD, WENO, Centred, Upwind
import jax
import numpy as np
import pytest
from scipy.ndimage import binary_dilation

SIZES = (2, 4, 6, 8, 10)
RNG = np.random.default_rng(506)


def _random_wet(shape: tuple[int, ...], share: float = 0.85) -> np.ndarray:
    return RNG.random(shape) < share


MASKS: dict[str, Any] = {
    "1d": Mask1D.from_mask(_random_wet((60,))),
    "1d-ocean": Mask1D.from_dimensions(40),
    "2d": Mask2D.from_mask(_random_wet((30, 34))),
    "2d-ocean": Mask2D.from_dimensions(24, 24),
    "2d-dry": Mask2D.from_mask(np.zeros((8, 8), dtype=bool)),
    "3d": Mask3D.from_mask(_random_wet((9, 14, 16), 0.95)),
}
AXES = {1: ("x",), 2: ("y", "x"), 3: ("z", "y", "x")}

#: (at, kind) of each of the sources of the removed get_adaptive_masks
SOURCES = {
    "h": ("T", "default"),
    "u": ("U", "default"),
    "v": ("V", "default"),
    "xy_corner": ("X", "default"),
    "xy_corner_strict": ("X", "strict"),
}
POINTS = {1: ("h", "u"), 2: tuple(SOURCES), 3: tuple(SOURCES)}


def get_adaptive_masks(
    wet: np.ndarray, axis: int, stencil_sizes: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """``Mask*.get_adaptive_masks`` before #506, on the wet points of any source."""
    pos = np.minimum(count_contiguous(wet, axis=axis, forward=True), 127)
    neg = np.minimum(count_contiguous(wet, axis=axis, forward=False), 127)
    max_s = np.zeros(pos.shape, dtype=np.int32)
    for s in sorted(stencil_sizes):
        hw = s // 2
        max_s = np.where((pos >= hw) & (neg >= hw), s, max_s)
    return {s: (max_s == s) for s in stencil_sizes}


CASES = [
    (name, axis, source)
    for name, mask in MASKS.items()
    for axis in AXES[mask.wet().ndim]
    for source in POINTS[mask.wet().ndim]
]


@pytest.mark.parametrize(("name", "axis", "source"), CASES)
@pytest.mark.parametrize("sizes", [SIZES, (2,), (2, 6), (6, 2, 4), (4, 12, 16)])
def test_tiers_are_the_old_adaptive_masks(
    name: str, axis: str, source: str, sizes: tuple[int, ...]
) -> None:
    mask = MASKS[name]
    at, kind = SOURCES[source]
    position = AXES[mask.wet().ndim].index(axis)
    expected = get_adaptive_masks(mask.wet(at=at, kind=kind), position, sizes)
    got = mask.tiers(axis, at=at, kind=kind, sizes=sizes)
    assert list(got) == list(sizes)
    for size in sizes:
        np.testing.assert_array_equal(got[size], expected[size], err_msg=str(size))
        assert got[size].dtype == bool


@pytest.mark.parametrize("source", list(SOURCES))
def test_tiers_of_a_traced_mask(source: str) -> None:
    mask = MASKS["2d"]
    at, kind = SOURCES[source]
    eager = mask.tiers("y", at=at, kind=kind)
    traced = jax.jit(lambda m: m.tiers("y", at=at, kind=kind))(mask)
    for size in SIZES:
        np.testing.assert_array_equal(traced[size], eager[size])


def test_tiers_takes_a_stagger() -> None:
    mask = MASKS["2d"]
    got = mask.tiers("x", at=(("x", "right"),))
    for size, points in mask.tiers("x", at="U").items():
        np.testing.assert_array_equal(got[size], points)


@pytest.mark.parametrize("sizes", [(), (3,), (0, 2)])
def test_bad_sizes_raise(sizes: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="sizes must be even"):
        MASKS["2d"].tiers("x", sizes=sizes)


def test_unknown_axis_raises() -> None:
    with pytest.raises(ValueError, match="axis must be one of"):
        MASKS["2d"].tiers("z")


@pytest.mark.parametrize("scheme", [Upwind(5), WENO(9), TVD("mc"), Centred(4)])
def test_the_binding_is_the_function(scheme: Any) -> None:
    grid = gridax.CartesianCGrid2D.from_interior(32, 28, 1.0, 1.0)
    mask = MASKS["2d"]
    q, u, v = RNG.normal(size=(3, *grid.shape))
    expected = gridax.advect(q, u, v, grid, scheme=scheme, mask=mask)
    got = gridax.Advection2D(grid, mask)(q, u, v, scheme=scheme)
    np.testing.assert_array_equal(got, expected)


# ----------------------------------------------------------------------
# Coastal ghosts
# ----------------------------------------------------------------------


@pytest.mark.parametrize("layer", [1, 2, 3])
@pytest.mark.parametrize("side", ["low", "high"])
@pytest.mark.parametrize(("name", "axis", "source"), CASES)
def test_coast_ghosts_by_hand(
    name: str, axis: str, source: str, side: str, layer: int
) -> None:
    mask = MASKS[name]
    at, kind = SOURCES[source]
    wet = np.moveaxis(mask.wet(at=at, kind=kind), AXES[mask.wet().ndim].index(axis), -1)
    ghosts = mask.coast_ghosts(axis, side, at=at, kind=kind, layer=layer)
    step = 1 if side == "low" else -1
    assert ghosts.offset == step * layer
    expected = np.zeros(wet.shape, dtype=bool)
    n = wet.shape[-1]
    for i in range(n):
        source_ = i + step * layer
        if not 0 <= source_ < n:
            continue
        between = wet[..., [i + step * k for k in range(layer)]]
        expected[..., i] = wet[..., source_] & ~between.any(axis=-1)
    position = AXES[mask.wet().ndim].index(axis)
    np.testing.assert_array_equal(np.moveaxis(ghosts.points, position, -1), expected)


def test_a_coast_on_the_edge_is_the_ghost_ring() -> None:
    # a dry ring round the water: its points are the ghost ring of the grid
    wet = np.zeros((6, 7), dtype=bool)
    wet[1:-1, 1:-1] = True
    mask = Mask2D.from_mask(wet)
    east = mask.coast_ghosts("x", "high").points
    assert east[1:-1, -1].all() and east.sum() == 4
    # the U point on the wall is the first dry one, not the ring's
    east = mask.coast_ghosts("x", "high", at="U").points
    assert east[1:-1, -2].all() and east.sum() == 4


def test_coast_ghosts_of_a_traced_mask() -> None:
    mask = MASKS["2d"]
    eager = mask.coast_ghosts("x", "low", at="V", layer=2)
    traced = jax.jit(lambda m: m.coast_ghosts("x", "low", at="V", layer=2))(mask)
    np.testing.assert_array_equal(traced.points, eager.points)
    assert traced.offset == eager.offset == 2  # static, so a Python int


@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis": "z", "side": "low"},
        {"axis": "x", "side": "east"},
        {"axis": "x", "side": "low", "layer": 0},
        {"axis": "x", "side": "low", "at": "W"},
    ],
)
def test_bad_coasts_raise(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"axis|side|layer|at="):
        MASKS["2d"].coast_ghosts(**kwargs)


# ----------------------------------------------------------------------
# What is derived instead of stored
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(MASKS))
def test_classification_is_the_dilation_of_the_land(name: str) -> None:
    mask = MASKS[name]
    h = mask.wet()
    cross = np.zeros((3,) * h.ndim, dtype=bool)
    centre = (1,) * h.ndim
    cross[centre] = True
    for axis in range(h.ndim):
        for k in (0, 2):
            cross[(*centre[:axis], k, *centre[axis + 1 :])] = True
    d1 = binary_dilation(~h, structure=cross)
    d2 = binary_dilation(d1, structure=cross)
    expected = np.zeros(h.shape, dtype=np.int8)
    expected[h & d1] = 1
    expected[h & d2 & ~d1] = 2
    expected[h & ~d2] = 3
    np.testing.assert_array_equal(mask.classification, expected)
    assert mask.classification.dtype == np.int8


@pytest.mark.parametrize("name", ["2d", "3d"])
@pytest.mark.parametrize("kind", ["y_wall", "x_wall", "convex", "valid"])
def test_kinds_of_a_traced_mask(name: str, kind: str) -> None:
    mask = MASKS[name]
    traced = jax.jit(lambda m: m.wet(at="X", kind=kind))(mask)
    np.testing.assert_array_equal(traced, mask.wet(at="X", kind=kind))


def test_a_mask_stores_nine_bytes_per_cell() -> None:
    # T, U, V, two kinds of X and four int8 counts; 14 before #506
    leaves = jax.tree.leaves(Mask2D.from_dimensions(64, 64))
    assert len(leaves) == 9
    assert sum(leaf.nbytes for leaf in leaves) == 9 * 64 * 64


def test_a_binding_stores_its_mask_once() -> None:
    grid = gridax.CartesianCGrid2D.from_interior(32, 28, 1.0, 1.0)
    op = gridax.Advection2D(grid, MASKS["2d"])
    extra = len(jax.tree.leaves(op)) - len(jax.tree.leaves(grid))
    assert extra == len(jax.tree.leaves(MASKS["2d"]))
