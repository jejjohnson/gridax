"""The canonical functions, their registries and the support matrix agree.

Every function of ``gridax._src.api.OPERATIONS`` is called on every grid
type: a pair that ``SUPPORT`` lists must have an implementation, and any other
must raise ``NotImplementedError`` naming the operation and the grid.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

import gridax
from gridax._src.api import OPERATIONS, SUPPORT, grid_dispatch
import jax
import jax.numpy as jnp
import numpy as np
import pytest

GRIDS = {
    "CartesianCGrid1D": gridax.CartesianCGrid1D.from_interior(6, 1.0),
    "CartesianCGrid2D": gridax.CartesianCGrid2D.from_interior(6, 5, 1.0, 1.0),
    "CartesianCGrid3D": gridax.CartesianCGrid3D.from_interior(6, 5, 4, 1.0, 1.0, 1.0),
    "SphericalCGrid2D": gridax.SphericalCGrid2D.from_interior(
        6, 5, (0.0, 30.0), (0.0, 30.0)
    ),
    "CartesianAGrid1D": gridax.CartesianAGrid1D.from_interior(8, 4.0),
    "CartesianAGrid2D": gridax.CartesianAGrid2D.from_coords(
        y=[0.0, 1.0, 3.0, 4.5, 5.0, 7.0], x=[0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0]
    ),
    "CartesianAGrid3D": gridax.CartesianAGrid3D.from_interior(8, 6, 5, 4.0, 3.0, 10.0),
    "SphericalAGrid2D": gridax.SphericalAGrid2D.from_interior(
        8, 6, (0.0, 360.0), (-60.0, 60.0), R=2.0, periodic=("x",)
    ),
    "StretchedCGrid1D": gridax.CartesianAGrid1D.from_coords(
        x=[0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0]
    ).staggered(),
    "StretchedCGrid2D": gridax.CartesianAGrid2D.from_coords(
        y=[0.0, 1.0, 3.0, 4.5, 5.0, 7.0], x=[0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0]
    ).staggered(),
    "StretchedCGrid3D": gridax.CartesianAGrid3D.from_coords(
        z=[0.0, 1.0, 3.0, 4.0],
        y=[0.0, 1.0, 3.0, 4.5, 5.0, 7.0],
        x=[0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0],
    ).staggered(),
    "SphericalCGrid3D": gridax.SphericalCGrid3D.from_interior(
        6, 5, 4, (0.0, 30.0), (0.0, 30.0), 1.0
    ),
}


def test_every_concrete_grid_is_here():
    concrete = {
        name
        for name in gridax.__all__
        if re.fullmatch(r"(Cartesian|Spherical|Stretched)[AC]Grid\dD", name)
    }
    assert concrete == set(GRIDS)


def test_every_operation_is_in_the_matrix_and_public():
    assert set(OPERATIONS) == set(SUPPORT)
    for name, op in OPERATIONS.items():
        assert getattr(gridax, name) is op


def test_the_matrix_names_known_grids():
    assert {g for grids in SUPPORT.values() for g in grids} <= set(GRIDS)


@pytest.mark.parametrize("grid", sorted(GRIDS))
@pytest.mark.parametrize("name", sorted(SUPPORT))
def test_support(name, grid):
    op = OPERATIONS[name]
    chosen = op._impl.dispatch(type(GRIDS[grid]))
    default = op._impl.dispatch(object)
    if grid in SUPPORT[name]:
        assert chosen is not default, f"{name} has no implementation for {grid}"
        return
    assert chosen is default, f"{name} is defined on {grid}, not in SUPPORT"
    with pytest.raises(NotImplementedError, match=f"{name}.*{grid}"):
        op._impl(GRIDS[grid])


def test_a_grid_type_may_be_added_from_outside():
    @grid_dispatch
    def double(x, grid):
        return double._impl(grid, x)

    class Line:
        pass

    @double.register(Line)
    def _(grid, x):
        return 2 * x

    try:
        assert double(3, Line()) == 6
        with pytest.raises(NotImplementedError, match="double is not defined on"):
            double(3, GRIDS["CartesianCGrid1D"])
    finally:
        OPERATIONS.pop("double")


@pytest.mark.parametrize("grid", ["CartesianCGrid2D", "SphericalCGrid2D"])
def test_dispatch_is_resolved_at_trace_time(grid):
    """No control flow in the program, and one trace per grid type."""
    grid = GRIDS[grid]
    x = jnp.ones(grid.shape)
    text = str(jax.make_jaxpr(gridax.area_sum)(x, grid))
    assert "cond" not in text and "switch" not in text
    np.testing.assert_allclose(
        jax.jit(gridax.area_sum)(x, grid), gridax.area_sum(x, grid), rtol=1e-14
    )


@pytest.mark.parametrize(
    "name",
    [
        f"{geometry}_{measure}_{what}"
        for geometry in ("cartesian", "spherical")
        for measure in ("area", "volume")
        for what in ("sum", "mean", "weights")
    ],
)
def test_the_per_geometry_names_are_gone(name):
    assert not hasattr(gridax, name)


#: What takes a grid beside the canonical functions: the weights and the
#: coordinates of a grid, the ghost ring and the regions of a field, and the
#: conversions between two grids or two conventions.
GRID_HELPERS = {
    "agrid_from",
    "area_weights",
    "collocate",
    "embed",
    "fill_ghosts",
    "interior",
    "select",
    "stagger",
    "t_point_coordinates",
    "to_right",
    "volume_weights",
}


def test_one_public_name_per_operation():
    """No class but `GridOps` holds a grid, and no function takes spacings (#493)."""
    holders, takers = set(), set()
    for name in gridax.__all__:
        obj = getattr(gridax, name)
        if inspect.isclass(obj):
            if dataclasses.is_dataclass(obj):
                if any(field.name == "grid" for field in dataclasses.fields(obj)):
                    holders.add(name)
        elif callable(obj):
            parameters = set(inspect.signature(obj).parameters)
            assert not parameters & {"dx", "dy", "dz"}, name
            if "grid" in parameters:
                takers.add(name)
    assert holders == {"GridOps"}
    assert takers == set(OPERATIONS) | GRID_HELPERS
