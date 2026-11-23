"""Every canonical function accepts `CField`: the point is read, checked and set."""

from __future__ import annotations

import gridax as g
from gridax import CField, StaggerMismatchError, combine
from gridax._src.api import OPERATIONS, SUPPORT
from gridax._src.grid.points import POINT_AXES, point_stagger
from gridax._src.operators.triples import PAIRS, TRIPLES
import jax
import jax.numpy as jnp
import numpy as np
import pytest

GRIDS = {
    "CartesianCGrid1D": g.CartesianCGrid1D.from_interior(8, 4.0),
    "CartesianCGrid2D": g.CartesianCGrid2D.from_interior(8, 6, 4.0, 3.0),
    "CartesianCGrid3D": g.CartesianCGrid3D.from_interior(8, 6, 5, 4.0, 3.0, 10.0),
    "SphericalCGrid2D": g.SphericalCGrid2D.from_interior(
        8, 6, (0.0, 40.0), (-60.0, 60.0), R=2.0
    ),
    "CartesianAGrid1D": g.CartesianAGrid1D.from_interior(8, 4.0),
    "CartesianAGrid2D": g.CartesianAGrid2D.from_coords(
        y=[0.0, 1.0, 3.0, 4.5, 5.0, 7.0], x=[0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0]
    ),
    "CartesianAGrid3D": g.CartesianAGrid3D.from_interior(8, 6, 5, 4.0, 3.0, 10.0),
    "SphericalAGrid2D": g.SphericalAGrid2D.from_interior(
        8, 6, (0.0, 360.0), (-60.0, 60.0), R=2.0, periodic=("x",)
    ),
    "SphericalCGrid3D": g.SphericalCGrid3D.from_interior(
        8, 6, 5, (0.0, 40.0), (-60.0, 60.0), 10.0, R=2.0
    ),
}
C2 = GRIDS["CartesianCGrid2D"]
MASKS = {1: g.Mask1D, 2: g.Mask2D, 3: g.Mask3D}


def array(grid, seed=0, lead=()):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(*lead, *grid.shape)))


def wet_mask(grid, seed=1):
    wet = np.random.default_rng(seed).random(grid.shape) > 0.3
    return MASKS[len(grid.axes)].from_grid(grid, wet)


def keywords(name, grid):
    """The keyword arguments of each call an operation is checked on."""
    if getattr(grid, "_collocated", False):
        axes = [{"axis": axis} for axis in grid.axes]
        return axes if name == "difference" else [{}]
    if name == "laplacian":
        points = [p for p, along in POINT_AXES.items() if set(along) <= set(grid.axes)]
        return [{"at": p} for p in points]
    if name == "grad_perp":
        return [{"at": "X"}, {"at": "T"}]
    if name == "difference":
        return [
            {"axis": a, "src": s, "dst": d} for a, s, d in sorted(TRIPLES[grid.axes])
        ]
    if name == "interpolate":
        return [{"src": s, "dst": d} for s, d in sorted(PAIRS[grid.axes])]
    return [{}]


def resolve(point, kwargs, grid):
    """A declared point: a name, or the keyword that holds it; T if collocated."""
    if getattr(grid, "_collocated", False):
        return "T"
    return point if point in POINT_AXES else kwargs[point]


CALLS = [
    pytest.param(name, grid, kwargs, id=f"{name}-{grid}-{'-'.join(kwargs.values())}")
    for name in sorted(OPERATIONS)
    for grid in SUPPORT[name]
    for kwargs in keywords(name, GRIDS[grid])
]


@pytest.mark.parametrize("name", sorted(OPERATIONS))
def test_every_operation_declares_its_points(name):
    """A canonical function without ``@located`` fails here."""
    out, points = OPERATIONS[name].__located__
    assert points, name
    for point in (*points.values(), *(out if isinstance(out, tuple) else (out,))):
        assert point is None or point in POINT_AXES or point in ("src", "dst", "at")


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(("name", "grid", "kwargs"), CALLS)
def test_the_field_path_is_the_array_path(name, grid, kwargs, masked):
    """``op(CField…).data`` equals ``op(array…, src=…)``, with max|Δ| = 0."""
    grid = GRIDS[grid]
    op = OPERATIONS[name]
    out, points = op.__located__
    mask = wet_mask(grid) if masked else None
    arrays = {n: array(grid, seed) for seed, n in enumerate(points)}
    fields = {
        n: CField(a, grid.axes, loc=resolve(points[n], kwargs, grid))
        for n, a in arrays.items()
    }
    # the keywords that only name the point of an operand are inferred
    inferred = {k: v for k, v in kwargs.items() if k not in points.values()}
    want = op(**arrays, grid=grid, mask=mask, **kwargs)
    got = op(**fields, grid=grid, mask=mask, **inferred)
    if out is None:
        assert not isinstance(got, CField)
        np.testing.assert_array_equal(got, want)
        return
    outs = out if isinstance(out, tuple) else (out,)
    gots, wants = (x if isinstance(x, tuple) else (x,) for x in (got, want))
    for point, a, b in zip(outs, gots, wants, strict=True):
        assert a.dims == grid.axes
        assert a.stagger == point_stagger(resolve(point, kwargs, grid))
        np.testing.assert_array_equal(a.data, b)


@pytest.mark.parametrize(("name", "grid", "kwargs"), CALLS)
def test_a_field_on_another_point_raises(name, grid, kwargs):
    grid = GRIDS[grid]
    op = OPERATIONS[name]
    _, points = op.__located__
    for wrong in points:
        fields = {}
        for n, point in points.items():
            at = resolve(point, kwargs, grid)
            if n == wrong:
                at = "U" if at == "T" else "T"
            fields[n] = CField(array(grid), grid.axes, loc=at)
        with pytest.raises(StaggerMismatchError, match=name):
            op(**fields, grid=grid, **kwargs)


class TestPoints:
    def test_the_point_is_inferred_and_set(self):
        h = CField(array(C2), ("y", "x"), loc="T")
        assert g.difference(h, C2, axis="x", dst="U").loc == "U"
        assert g.interpolate(h, C2, dst="X").loc == "X"

    def test_a_src_that_contradicts_the_field_raises(self):
        h = CField(array(C2), ("y", "x"), loc="T")
        with pytest.raises(StaggerMismatchError, match="T, not at U"):
            g.difference(h, C2, axis="x", src="U", dst="T")
        with pytest.raises(StaggerMismatchError):
            jax.jit(lambda f: g.interpolate(f, C2, src="V", dst="T"))(h)

    def test_the_same_src_is_fine(self):
        h = CField(array(C2), ("y", "x"), loc="T")
        assert g.interpolate(h, C2, src="T", dst="U").loc == "U"

    def test_u_times_h_needs_the_interpolation(self):
        """The mistake `CField` exists to catch, and its correct form (#515)."""
        u = CField(array(C2, 1), ("y", "x"), loc="U")
        h = CField(array(C2, 2), ("y", "x"), loc="T")
        with pytest.raises(StaggerMismatchError):
            combine(jnp.multiply, u, h)
        flux = combine(jnp.multiply, u, g.interpolate(h, C2, dst="U"))
        assert flux.loc == "U"
        assert g.difference(flux, C2, axis="x", dst="T").loc == "T"


class TestDims:
    def test_the_grid_axes_are_found_by_name(self):
        data = array(C2, lead=(3,))
        f = CField(jnp.moveaxis(data, 0, 1), ("y", "member", "x"), loc="T")
        out = g.difference(f, C2, axis="x", dst="U")
        want = g.difference(data, C2, axis="x", src="T", dst="U")
        assert out.dims == f.dims
        np.testing.assert_array_equal(out.data, jnp.moveaxis(want, 0, 1))

    def test_labels_are_kept(self):
        labels = {"member": ("a", "b")}
        f = CField(array(C2, lead=(2,)), ("member", "y", "x"), labels, loc="T")
        assert g.interpolate(f, C2, dst="U").labels == labels

    def test_a_field_without_a_grid_axis_raises(self):
        f = CField(jnp.ones((C2.Ny, 3)), ("y", "member"))
        with pytest.raises(ValueError, match="no dims"):
            g.interpolate(f, C2, dst="V")

    def test_under_vmap_over_members(self):
        f = CField(array(C2, lead=(3,)), ("member", "y", "x"), loc="T")
        mapped = jax.vmap(lambda f: g.difference(f, C2, axis="x", dst="U"))(f)
        named = g.difference(f, C2, axis="x", dst="U")
        assert mapped.loc == "U"
        np.testing.assert_allclose(mapped.data, named.data, rtol=1e-15)

    def test_the_program_is_the_one_on_arrays(self):
        h = CField(array(C2), ("y", "x"), loc="T")
        located = jax.make_jaxpr(
            lambda a: g.difference(h.like(a), C2, axis="x", dst="U").data
        )(h.data)
        bare = jax.make_jaxpr(
            lambda a: g.difference(a, C2, axis="x", src="T", dst="U")
        )(h.data)
        assert str(located) == str(bare)


class TestReductions:
    def test_without_other_dims_the_result_is_a_plain_array(self):
        h = CField(array(C2), ("y", "x"), loc="T")
        out = g.area_sum(h, C2)
        assert out.shape == ()
        assert isinstance(out, jax.Array)

    def test_the_other_dims_are_left_as_a_dimarray(self):
        pa = pytest.importorskip("pipekit_array")
        data = array(C2, lead=(3,))
        h = CField(data, ("member", "y", "x"), {"member": ("a", "b", "c")}, loc="T")
        out = g.area_mean(h, C2, mask=wet_mask(C2))
        assert type(out) is pa.DimArray
        assert (out.dims, out.labels) == (("member",), h.labels)
        for i in range(3):
            np.testing.assert_allclose(
                out.data[i], g.area_mean(data[i], C2, mask=wet_mask(C2)), rtol=1e-12
            )

    def test_a_pipekit_reduction_keeps_the_axis_left(self):
        pa = pytest.importorskip("pipekit_array")
        v = CField(array(C2), ("y", "x"), loc="V")
        assert pa.reduce(v, "mean", "x").stagger == (("y", "right"),)
        assert type(pa.reduce(v, "mean", ("y", "x"))) is pa.DimArray
