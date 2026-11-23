"""gridax.analysis differences: the geoops kernel tests, on NumPy and JAX.

Ported from geotoolz-ops (``tests/test_gradient.py``, ``tests/test_calculus.py``),
where they ran against the kernels moved here (#423); the carrier and
Operator tests stay there. finitediffx's numbers come from
``data/finitediffx_reference.npz``, written once with finitediffx 0.1.0
(the generator script is quoted in that file's test below), so gridax does
not depend on it. Tests that need JAX skip without it: the
``gridax-numpy-only`` CI job runs the rest.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import gridax
from gridax._src.analysis.differences import adaptive_difference
from gridax.analysis import difference
from gridax.stencils import difference_coeffs, difference_offsets, nonuniform_coeffs
import numpy as np
import pytest

try:
    import jax
except ImportError:  # the gridax-numpy-only CI job
    jax = None

HAVE_JAX = jax is not None
needs_jax = pytest.mark.skipif(not HAVE_JAX, reason="needs JAX")
REFERENCE = Path(__file__).parent / "data" / "finitediffx_reference.npz"

CASES = [
    (derivative, accuracy, method)
    for derivative in (1, 2)
    for accuracy in (1, 2, 3, 4)
    for method in ("central", "forward", "backward")
]
DEPTHS = np.array([0.0, 5.0, 15.0, 30.0, 50.0, 80.0])
LEVELS = np.array([0.0, 5.0, 15.0, 30.0, 50.0, 75.0, 105.0, 140.0, 180.0])


def _jax():
    if not HAVE_JAX:
        pytest.skip("needs JAX")
    return jax


@pytest.fixture(autouse=True)
def _x64():
    if not HAVE_JAX:
        yield
        return
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@functools.cache
def _reference() -> dict[str, np.ndarray]:
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


# finitediffx reference values (geotoolz-ops' finitediffx tests)


@pytest.mark.parametrize(("derivative", "accuracy", "method"), CASES)
def test_difference_equals_finitediffx(
    derivative: int, accuracy: int, method: str
) -> None:
    ref = _reference()
    key = f"difference/{derivative}/{accuracy}/{method}"
    f = ref[key + "/f"]
    for axis, step in ((0, 0.3), (1, -2.0)):
        got = difference(
            f,
            axis=axis,
            step=step,
            derivative=derivative,
            accuracy=accuracy,
            method=method,
        )
        np.testing.assert_allclose(got, ref[f"{key}/axis{axis}"], rtol=1e-9, atol=1e-9)


def _grid(f: np.ndarray, *steps: float) -> Any:
    """A collocated grid with the steps of the reference, slowest axis first."""
    ny, nx = f.shape
    dy, dx = steps
    return gridax.CartesianAGrid2D.from_interior(nx, ny, nx * dx, ny * dy)


def test_2d_calculus_equals_finitediffx() -> None:
    # accuracy 1, the central differences of the canonical functions
    ref = _reference()
    u, v = ref["calculus2d/1/u"], ref["calculus2d/1/v"]
    grid = _grid(u, 0.3, -2.0)
    for got, key in (
        (gridax.divergence(u, v, grid), "div"),
        (gridax.curl(u, v, grid), "curl"),
        (gridax.laplacian(u, grid), "lap"),
    ):
        np.testing.assert_allclose(got, ref[f"calculus2d/1/{key}"], atol=1e-9)


# dtypes (#421): the caller decides


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_input_dtype_is_kept(dtype: Any) -> None:
    y, x = np.mgrid[0:6, 0:8].astype(dtype)
    grid = _grid(x, 1.0, 1.0)
    wet = np.ones(x.shape, dtype=bool)
    wet[2, 3] = False
    for mask in (None, gridax.Mask2D.from_grid(grid, wet)):
        for out in (
            *gridax.gradient(x * y, grid, mask=mask),
            gridax.divergence(x, y, grid, mask=mask),
            gridax.curl(x, y, grid, mask=mask),
            gridax.laplacian(x * y, grid, mask=mask),
        ):
            assert isinstance(out, np.ndarray) and out.dtype == dtype


def test_second_order_inside() -> None:
    errors = []
    for n in (32, 64):
        x = np.linspace(0.0, 1.0, n)
        field = np.broadcast_to(np.sin(2 * np.pi * x), (4, n))
        grid = _grid(field, 1.0, x[1] - x[0])
        slope = gridax.gradient(field, grid)[0]
        errors.append(np.abs(slope - 2 * np.pi * np.cos(2 * np.pi * x))[:, 1:-1].max())
    assert np.log2(errors[0] / errors[1]) == pytest.approx(2.0, abs=0.1)


def test_a_linear_field_is_exact_up_to_a_mask() -> None:
    y, x = np.mgrid[0:7, 0:9].astype(float)
    grid = _grid(x, 1.0, 1.0)
    wet = np.ones(x.shape, dtype=bool)
    wet[2:4, 3:5] = False
    mask = gridax.Mask2D.from_grid(grid, wet)
    d_x, d_y = gridax.gradient(2.0 * x + 3.0 * y, grid, mask=mask)
    np.testing.assert_allclose(d_x[wet], 2.0)
    np.testing.assert_allclose(d_y[wet], 3.0)
    assert np.isnan(d_x[~wet]).all()
    # without a mask a NaN spreads to its neighbours
    field = np.where(wet, x, np.nan)
    assert np.isnan(gridax.gradient(field, grid)[0][2, 2])


# ported from geotoolz-ops tests/test_gradient.py


def test_stencils_are_the_textbook_ones() -> None:
    assert difference_offsets(derivative=1, accuracy=1, method="central") == (-1, 0, 1)
    assert difference_coeffs((-1, 0, 1), 1) == (-0.5, 0.0, 0.5)
    assert difference_coeffs((-2, -1, 0, 1, 2), 1) == pytest.approx(
        (1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12)
    )
    assert difference_coeffs((-1, 0, 1), 2) == (1.0, -2.0, 1.0)
    with pytest.raises(ValueError, match="cannot resolve"):
        difference_coeffs((0, 1), 2)


def test_periodic_difference_reads_across_the_seam() -> None:
    lon = np.arange(36) * 10.0
    f = np.sin(np.deg2rad(lon))
    got = difference(f, step=np.deg2rad(10.0), accuracy=3, periodic=True)
    np.testing.assert_allclose(got, np.cos(np.deg2rad(lon)), atol=1e-4)
    assert abs(got[0] - 1.0) < 1e-4  # the seam is a centred stencil too


def test_propagate_spreads_one_halo_and_adaptive_none() -> None:
    f = np.arange(9.0)
    f[4] = np.nan
    plain = difference(f, accuracy=2)
    assert np.isnan(plain).sum() == 3  # the cell and both neighbours
    recovered = adaptive_difference(f, accuracy=2)
    assert np.isnan(recovered).sum() == 1
    np.testing.assert_allclose(recovered[[3, 5]], 1.0)


def test_adaptive_descends_accuracy_for_a_narrow_channel() -> None:
    """A two-cell channel supports a first-order stencil and nothing wider."""
    f = np.array([np.nan, np.nan, 3.0, 4.0, np.nan, np.nan])
    got = adaptive_difference(f, accuracy=3)
    np.testing.assert_allclose(got[2:4], 1.0)
    assert np.isnan(got[[0, 1, 4, 5]]).all()


@needs_jax
def test_adaptive_jits_and_traces_once() -> None:
    traces = []

    @jax.jit
    def step(f, mask):
        traces.append(1)
        return adaptive_difference(f, mask=mask, accuracy=2)

    f = jax.numpy.arange(10.0)
    step(f, jax.numpy.zeros(10, bool))
    out = step(f, jax.numpy.arange(10) == 4)
    assert len(traces) == 1
    np.testing.assert_allclose(np.asarray(out)[[3, 5]], 1.0)


def test_coastline_measurement() -> None:
    """xrgrad's comparison: sin(2λ)cos(φ) on 121 x 91 with a synthetic coast.

    Propagate loses the coastal strip; adaptive loses none of it and stays
    within a small multiple of the interior error there.
    """
    lam = np.deg2rad(np.linspace(0.0, 120.0, 121))
    phi = np.deg2rad(np.linspace(-45.0, 45.0, 91))
    lam2, phi2 = np.meshgrid(lam, phi)
    f = np.sin(2 * lam2) * np.cos(phi2)
    want = 2 * np.cos(2 * lam2) * np.cos(phi2)
    cols = np.arange(121)[None, :]
    rows = np.arange(91)[:, None]
    land = (cols < 10) | ((cols - 70) ** 2 + (rows - 45) ** 2 < 15**2)
    step = lam[1] - lam[0]
    padded = np.pad(land, ((0, 0), (1, 1)))
    coast = ~land & (padded[:, :-2] | padded[:, 2:])  # water next to land along x
    plain = difference(np.where(land, np.nan, f), step=step, accuracy=2)
    adaptive = adaptive_difference(f, mask=land, step=step, accuracy=2)
    assert np.isnan(plain[coast]).all()
    assert np.isfinite(adaptive[~land]).all()
    scale = np.abs(want[~land]).mean()
    interior = ~land & ~coast
    assert np.abs(adaptive - want)[coast].max() / scale < 1e-2
    assert np.abs(adaptive - want)[interior].max() / scale < 1e-3


@pytest.mark.parametrize(
    ("size", "accuracy", "method"),
    [(3, 2, "forward"), (3, 2, "backward"), (5, 4, "central"), (2, 1, "central")],
)
def test_a_short_axis_raises_instead_of_reading_outside_it(
    size: int, accuracy: int, method: str
) -> None:
    with pytest.raises(ValueError, match="short"):
        difference(np.arange(float(size)), accuracy=accuracy, method=method)


def test_adaptive_descends_on_a_short_axis() -> None:
    got = adaptive_difference(np.arange(3.0), accuracy=4)
    np.testing.assert_allclose(got, 1.0)


@pytest.mark.parametrize(("derivative", "accuracy", "method"), CASES)
def test_nonuniform_coeffs_reduce_to_the_uniform_stencil(
    derivative: int, accuracy: int, method: str
) -> None:
    offsets = difference_offsets(
        derivative=derivative, accuracy=accuracy, method=method
    )
    got = nonuniform_coeffs(2.5 * np.asarray(offsets, dtype=float), derivative)
    want = np.asarray(difference_coeffs(offsets, derivative)) / 2.5**derivative
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(("derivative", "accuracy", "method"), CASES)
def test_evenly_spaced_coords_equal_the_step(
    derivative: int, accuracy: int, method: str
) -> None:
    f = np.random.default_rng(accuracy).normal(size=(4, 17))
    coords = -1.5 * np.arange(17.0)
    got = difference(
        f, coords=coords, derivative=derivative, accuracy=accuracy, method=method
    )
    want = difference(
        f, step=-1.5, derivative=derivative, accuracy=accuracy, method=method
    )
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("accuracy", [1, 2, 3])
def test_uneven_levels_are_exact_for_the_stencils_degree(accuracy: int) -> None:
    """A K-point stencil on uneven levels is exact for polynomials of degree K - 1."""
    for derivative in (1, 2):
        degree = derivative + accuracy - 1  # the one-sided stencils' degree
        f = LEVELS**degree
        want = np.polyval(np.polyder(np.eye(degree + 1)[0], derivative), LEVELS)
        got = difference(f, coords=LEVELS, derivative=derivative, accuracy=accuracy)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


@needs_jax
def test_uneven_levels_run_under_jit_with_traced_coords() -> None:
    want = difference(LEVELS**2, coords=LEVELS)
    jitted = jax.jit(lambda f, z: difference(f, coords=z))
    got = jitted(jax.numpy.asarray(LEVELS**2), jax.numpy.asarray(LEVELS))
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12)


def test_adaptive_difference_on_uneven_levels_keeps_the_cells_next_to_a_gap() -> None:
    f = LEVELS.copy()  # f = z: every stencil is exact
    f[4] = np.nan
    got = adaptive_difference(f, coords=LEVELS)
    assert np.isnan(got[4]) and np.isfinite(np.delete(got, 4)).all()
    np.testing.assert_allclose(np.delete(got, 4), 1.0, rtol=1e-12)


def test_coords_must_fit_the_axis_and_not_wrap() -> None:
    with pytest.raises(ValueError, match="entries"):
        difference(np.zeros((3, 9)), coords=LEVELS[:5])
    with pytest.raises(ValueError, match="periodic"):
        difference(np.zeros((3, 9)), coords=LEVELS, periodic=True)


# ported from geotoolz-ops tests/test_calculus.py
