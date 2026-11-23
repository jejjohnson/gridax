"""The mask classes are frozen NumPy dataclasses and JAX pytrees.

Masks are built and stored with NumPy, so the mask modules import without
JAX; when JAX is importable each class is a pytree. The fields that were
kept match the masks the Equinox classes built, value for value
(``tests/fixtures/mask_fields_pre_419.npz`` was written by the old classes
from the canonical fixture masks).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import ClassVar

import equinox as eqx
from gridax import (
    Advection1D,
    Advection2D,
    Advection3D,
    CartesianCGrid1D,
    CartesianCGrid2D,
    CartesianCGrid3D,
    Mask1D,
    Mask2D,
    Mask3D,
    SphericalCGrid2D,
)
from gridax.schemes import WENO
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.fixtures import inputs

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "fixtures" / "mask_fields_pre_419.npz"

MASKS = {
    "1d": inputs.make_mask_1d,
    "2d": inputs.make_mask_2d,
    "3d": inputs.make_mask_3d,
}


@pytest.fixture(params=sorted(MASKS))
def mask(request):
    return MASKS[request.param]()


class TestDataclass:
    def test_is_frozen_numpy_dataclass(self, mask):
        assert dataclasses.is_dataclass(mask)
        assert not isinstance(mask, eqx.Module)
        with pytest.raises(dataclasses.FrozenInstanceError):
            mask._h = mask.wet()
        for leaf in jax.tree_util.tree_leaves(mask):
            assert isinstance(leaf, np.ndarray)

    def test_removed_fields_are_gone(self, mask):
        names = {f.name for f in dataclasses.fields(mask)}
        assert "sponge" not in names
        assert not any("irrbound" in n or n.startswith("not_") for n in names)

    def test_narrow_integer_storage(self, mask):
        assert mask.classification.dtype == np.int8
        for field in dataclasses.fields(mask.stencil_capability):
            assert getattr(mask.stencil_capability, field.name).dtype == np.int8


#: The arguments of ``wet`` that give each of the masks' old attributes.
WET = {
    "h": {},
    "u": {"at": "U"},
    "v": {"at": "V"},
    "w": {"at": "W"},
    "xy_corner": {"at": "X"},
    **{
        f"xy_corner_{kind}": {"at": "X", "kind": kind}
        for kind in ("strict", "y_wall", "x_wall", "convex", "valid")
    },
}


@pytest.mark.parametrize("dim", sorted(MASKS))
def test_wet_takes_a_stagger(dim):
    m = MASKS[dim]()
    points = {
        "U": (("x", "right"),),
        "V": (("y", "right"),),
        "X": (("x", "right"), ("y", "right")),
        "W": (("z", "right"),),
    }
    for point in {"1d": "U", "2d": "UVX", "3d": "UVXW"}[dim]:
        np.testing.assert_array_equal(m.wet(at=points[point]), m.wet(at=point))
    np.testing.assert_array_equal(m.wet(at=()), m.wet())


@pytest.mark.parametrize("dim", sorted(MASKS))
def test_the_old_attributes_are_gone(dim):
    m = MASKS[dim]()
    for name in WET:
        assert not hasattr(m, name)
        assert not hasattr(m, f"not_{name}")


@pytest.mark.parametrize(
    ("dim", "kwargs"),
    [
        ("1d", {"at": "V"}),
        ("1d", {"at": "X"}),
        ("2d", {"at": "W"}),
        ("2d", {"at": "Q"}),
        ("2d", {"at": (("x", "left"),)}),
        ("2d", {"at": "X", "kind": "lenient"}),
        ("2d", {"at": "U", "kind": "strict"}),
    ],
)
def test_unknown_points_and_kinds_raise(dim, kwargs):
    with pytest.raises(ValueError, match=r"at=|kind"):
        MASKS[dim]().wet(**kwargs)


class TestFromGrid:
    GRIDS: ClassVar = {
        "1d": CartesianCGrid1D.from_interior(6, 1.0),
        "2d": CartesianCGrid2D.from_interior(6, 5, 1.0, 1.0),
        "sphere": SphericalCGrid2D.from_interior(6, 5, (0.0, 30.0), (0.0, 30.0)),
        "3d": CartesianCGrid3D.from_interior(6, 5, 4, 1.0, 1.0, 1.0),
    }
    CLASSES: ClassVar = {"1d": Mask1D, "2d": Mask2D, "sphere": Mask2D, "3d": Mask3D}

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_is_from_mask_on_the_grids_shape(self, name):
        grid, cls = self.GRIDS[name], self.CLASSES[name]
        wet = np.random.default_rng(0).random(grid.shape) > 0.2
        assert cls.from_grid(grid, wet) == cls.from_mask(wet)

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_another_shape_raises(self, name):
        grid, cls = self.GRIDS[name], self.CLASSES[name]
        with pytest.raises(ValueError, match="shape"):
            cls.from_grid(grid, np.ones([n - 1 for n in grid.shape], dtype=bool))


def test_kept_fields_match_the_equinox_masks():
    ref = np.load(REFERENCE)
    for key in ref.files:
        dim, name = key.split("/")
        m = MASKS[dim]()
        if name.startswith("adaptive_x."):
            size = int(name.split(".")[1])
            value = m.tiers("x")[size]
        elif name.startswith("stencil_capability."):
            value = getattr(m.stencil_capability, name.split(".")[1])
        else:
            inverse = name.startswith("not_")
            point = name.removeprefix("not_")
            value = m.wet(**WET[point]) if point in WET else getattr(m, name)
            value = ~value if inverse else value
        np.testing.assert_array_equal(np.asarray(value), ref[key], err_msg=key)


def test_stencil_counts_saturate_at_127():
    """Counts are int8; a 300-cell wet run reports 127, never wraps."""
    m = Mask1D.from_mask(np.ones(300, dtype=bool))
    counts = m.stencil_capability.x_pos
    assert counts.max() == 127 and counts.min() == 1
    tiers = m.tiers("x", sizes=(2, 4, 6, 8, 10))
    assert tiers[10][5:-5].all()


def test_mask2d_pytree_size_at_512():
    """#325: the 512² pytree was 9.96 MB with sponge, not_* and int32 counts."""
    m = Mask2D.from_dimensions(512, 512)
    nbytes = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(m))
    assert nbytes <= 4_000_000, nbytes


class TestPytree:
    def test_round_trip(self, mask):
        leaves, treedef = jax.tree_util.tree_flatten(mask)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(rebuilt) is type(mask)
        assert rebuilt == mask

    def test_tiers_under_jit(self, mask):
        eager = mask.tiers("x", sizes=(2, 4, 6))
        traced = jax.jit(lambda m: m.tiers("x", sizes=(2, 4, 6)))(mask)
        for size in (2, 4, 6):
            np.testing.assert_array_equal(traced[size], eager[size])


def _advection_case(dim):
    if dim == "1d":
        grid = CartesianCGrid1D.from_interior(inputs.NX - 2, 1.0)
        h = jnp.cos(jnp.linspace(0.0, 3.0, inputs.NX))
        return grid, (h, jnp.ones_like(h)), Advection1D
    if dim == "2d":
        grid = CartesianCGrid2D.from_interior(inputs.NX - 2, inputs.NY - 2, 1.0, 1.0)
        h = inputs.make_h_field_2d()
        return grid, (h, jnp.ones_like(h), 0.5 * jnp.ones_like(h)), Advection2D
    grid = CartesianCGrid3D.from_interior(
        inputs.NX - 2, inputs.NY - 2, inputs.NZ - 2, 1.0, 1.0, 1.0
    )
    h = jnp.broadcast_to(inputs.make_h_field_2d(), (inputs.NZ, inputs.NY, inputs.NX))
    return grid, (h, jnp.ones_like(h), 0.5 * jnp.ones_like(h)), Advection3D


@pytest.mark.parametrize("dim", sorted(MASKS))
def test_masked_advection_jits_with_mask_as_argument(dim):
    grid, fields, cls = _advection_case(dim)
    mask = MASKS[dim]()

    def tendency(mask, *fields):
        return cls(grid=grid, mask=mask)(*fields, scheme=WENO(5))

    np.testing.assert_allclose(
        eqx.filter_jit(tendency)(mask, *fields), tendency(mask, *fields), rtol=1e-12
    )


def test_from_bathymetry_equals_from_mask_of_the_threshold():
    rng = np.random.default_rng(0)
    depth = rng.uniform(-50.0, 200.0, size=(20, 24))
    depth[3, 4] = np.nan
    m = Mask2D.from_bathymetry(depth, min_depth=10.0)
    with np.errstate(invalid="ignore"):
        expected = Mask2D.from_mask(depth > 10.0)
    assert m == expected
    assert not m.wet()[3, 4]


def test_from_bathymetry_drops_small_wet_regions():
    depth = np.full((12, 12), -1.0)
    depth[1:9, 1:9] = 100.0  # basin of 64 cells
    depth[10, 10] = 100.0  # isolated one-cell pond
    assert Mask2D.from_bathymetry(depth).wet()[10, 10]
    m = Mask2D.from_bathymetry(depth, min_area=2)
    assert not m.wet()[10, 10]
    assert m.wet()[1:9, 1:9].all()
