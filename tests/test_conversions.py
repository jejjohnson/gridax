"""Conversions between collocated grids and C-grids (#504)."""

from __future__ import annotations

from typing import Any
import warnings

import gridax as g
from gridax import BC, LossyConversionWarning
from gridax.schemes import WENO, Adaptive, Biased, Centred
import jax
import jax.numpy as jnp
import numpy as np
import pytest

RNG = np.random.default_rng(504)


def irregular(n: int, start: float = -1.0) -> np.ndarray:
    return start + np.cumsum(0.5 + RNG.random(n))


AGRIDS: dict[str, Any] = {
    "1d": g.CartesianAGrid1D.from_interior(9, 3.1, x0=0.3),
    "1d-periodic": g.CartesianAGrid1D.from_interior(9, 3.1, periodic=("x",)),
    "2d": g.CartesianAGrid2D.from_interior(7, 5, 3.1, 2.7, x0=0.3, y0=-1.1),
    "2d-periodic": g.CartesianAGrid2D.from_interior(
        7, 5, 3.1, 2.7, periodic=("x", "y")
    ),
    "raster": g.CartesianAGrid2D.from_affine((30.0, 0, 600.0, 0, 30.0, 900.0), (4, 6)),
    "3d": g.CartesianAGrid3D.from_interior(5, 4, 6, 1.3, 0.7, 2.9, z0=-2.0),
    "sphere": g.SphericalAGrid2D.from_interior(
        8, 6, (0.0, 360.0), (-60.0, 60.0), periodic=("x",)
    ),
    "sphere-region": g.SphericalAGrid2D.from_interior(
        5, 4, (3.0, 11.0), (41.0, 47.0), R=2.0
    ),
    "irregular-1d": g.CartesianAGrid1D.from_coords(irregular(8)),
    "irregular-2d": g.CartesianAGrid2D.from_coords(y=irregular(6), x=irregular(7)),
    "irregular-3d": g.CartesianAGrid3D.from_coords(
        z=irregular(5), y=irregular(6), x=irregular(7)
    ),
    "mixed": g.CartesianAGrid2D(
        **{
            **vars(g.CartesianAGrid2D.from_interior(7, 6, 3.1, 2.7)),
            "_y": irregular(6),
            "_dy": None,
        }
    ),
}
FLAT = {1: BC.no_flux("x"), 2: BC.no_flux(), 3: BC.no_flux("x", "y", "z")}
POINTS = {1: ["T", "U"], 2: ["T", "U", "V", "X"], 3: ["T", "U", "V", "X", "W"]}


def flat(grid: Any) -> BC:
    return FLAT[len(grid.axes)]


def quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LossyConversionWarning)
        return fn(*args, **kwargs)


# ----------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(AGRIDS))
def test_a_grid_round_trips(name: str) -> None:
    grid = AGRIDS[name]
    cgrid = grid.staggered()
    assert cgrid.collocated() == grid
    assert cgrid.axes == grid.axes and cgrid.ghost == 1
    assert cgrid.shape == tuple(n + 2 for n in grid.shape)


@pytest.mark.parametrize("n", [3, 5, 7, 11, 49, 97])
def test_uniform_round_trips_for_any_size(n: int) -> None:
    for _ in range(50):
        length, x0 = RNG.random() * 10.0 ** RNG.integers(-6, 7), RNG.normal()
        grid = g.CartesianAGrid1D.from_interior(n, length, x0=x0)
        assert grid.staggered().collocated() == grid


@pytest.mark.parametrize("name", list(AGRIDS))
def test_points(name: str) -> None:
    grid = AGRIDS[name]
    cgrid = grid.staggered()
    for axis in grid.axes:
        point = {"x": "U", "y": "V", "z": "W"}[axis]
        x = np.asarray(grid.coords(axis))
        t, f = (np.asarray(cgrid.coords(axis, at=p)) for p in ("T", point))
        np.testing.assert_allclose(t[1:-1], x, rtol=1e-13, atol=1e-13)
        # faces at midpoints, ghosts mirrored one spacing out
        np.testing.assert_allclose(f[:-1], (t[:-1] + t[1:]) / 2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(t[0], 2 * x[0] - x[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(t[-1], 2 * x[-1] - x[-2], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", ["irregular-1d", "irregular-2d", "mixed"])
def test_stretched_metric(name: str) -> None:
    cgrid = AGRIDS[name].staggered()
    for axis in cgrid.axes:
        point = {"x": "U", "y": "V"}[axis]
        t, f = (np.asarray(cgrid.coords(axis, at=p)) for p in ("T", point))
        cells = np.asarray(cgrid.spacing(axis)).ravel()
        faces = np.asarray(cgrid.spacing(axis, at=point)).ravel()
        np.testing.assert_allclose(cells[1:-1], np.diff(f)[:-1], rtol=1e-12)
        np.testing.assert_allclose(faces[:-1], np.diff(t), rtol=1e-12)
    # the interior cells tile the domain
    inside = tuple(slice(1, -1) for _ in cgrid.axes)
    total = np.prod(
        [
            np.asarray(cgrid.coords(a, at={"x": "U", "y": "V"}[a]))[[0, -2]] @ [-1, 1]
            for a in cgrid.axes
        ]
    )
    np.testing.assert_allclose(cgrid.area()[inside].sum(), total, rtol=1e-12)


def test_a_c_grid_collocates() -> None:
    for cgrid in (
        g.CartesianCGrid1D.from_interior(6, 3.0),
        g.CartesianCGrid2D.from_interior(6, 5, 3.0, 2.0),
        g.CartesianCGrid3D.from_interior(6, 5, 4, 3.0, 2.0, 1.0),
        g.SphericalCGrid2D.from_interior(6, 5, (0.0, 60.0), (10.0, 40.0)),
    ):
        agrid = cgrid.collocated()
        for axis in cgrid.axes:
            np.testing.assert_array_equal(agrid.coords(axis), cgrid.coords(axis)[1:-1])
            np.testing.assert_allclose(
                np.abs(np.asarray(agrid.spacing(axis)).ravel()[0]),
                np.asarray(cgrid.spacing(axis))[..., 1:2, 1:2].ravel()[0]
                if np.ndim(cgrid.spacing(axis)) == 2
                else cgrid.spacing(axis),
                rtol=1e-12,
            )


def test_a_default_c_grid_has_no_origin() -> None:
    grid = g.CartesianCGrid2D.from_interior(4, 4, 1.0, 1.0)
    assert grid.x0 is None and grid.periodic == ()
    assert len(jax.tree.leaves(grid)) == 2
    assert grid.coords("x").tolist() == [i * 0.25 for i in range(6)]


def test_axes_that_cannot_be_staggered() -> None:
    north_up = g.CartesianAGrid2D.from_affine((30.0, 0, 0, 0, -30.0, 0), (4, 6))
    with pytest.raises(ValueError, match="the y axis decreases"):
        north_up.staggered()
    gaussian = g.SphericalAGrid2D.from_coords(
        lat=[-50.0, -20.0, 0.0, 30.0], lon=[0.0, 10.0, 20.0]
    )
    with pytest.raises(NotImplementedError, match="uniform in longitude"):
        gaussian.staggered()


# ----------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------


@pytest.mark.parametrize("lead", [(), (2,)])
@pytest.mark.parametrize("name", list(AGRIDS))
def test_t_fields_round_trip(name: str, lead: tuple[int, ...]) -> None:
    grid = AGRIDS[name]
    f = RNG.normal(size=(*lead, *grid.shape))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # T to T is silent
        h = g.stagger(f, grid, dst="T", bc=flat(grid))
        back = g.collocate(h, grid.staggered(), src="T")
    np.testing.assert_array_equal(back, f)
    want = g.fill_ghosts(
        g.embed(f, grid.staggered()),
        grid.staggered(),
        flat(grid) if not grid.periodic else BC.periodic(*grid.periodic) | flat(grid),
    )
    if not grid.periodic:
        np.testing.assert_array_equal(h, want)


SCHEMES = [Centred(2), Centred(4), Biased(3, "left")]


@pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
@pytest.mark.parametrize("name", list(AGRIDS))
def test_linear_fields_are_exact(name: str, scheme: Any) -> None:
    grid = AGRIDS[name]
    cgrid = grid.staggered()
    rank = len(grid.axes)

    def linear(of: Any, at: str) -> np.ndarray:
        lines = [np.asarray(of.coords(a, at=at)) for a in of.axes]
        mesh = np.meshgrid(*lines, indexing="ij")
        return 1.5 + sum((k + 0.7) * x for k, x in enumerate(mesh))

    f = linear(grid, "T")
    # the ghosts of a linear field: the wall gradient along each outward normal
    if grid.periodic or name.startswith("sphere"):
        bc = BC(**{a: g.Extrapolation(1) for a in grid.axes})
    else:
        bc = BC(
            **{
                a: (g.Neumann(-(k + 0.7)), g.Neumann(k + 0.7))
                for k, a in enumerate(grid.axes)
            }
        )
        if (scheme != Centred(2) and "irregular" in name) or name == "mixed":
            bc = BC(**{a: g.Extrapolation(1) for a in grid.axes})
    inner = tuple(slice(1, -1) for _ in range(rank))
    for dst in POINTS[rank]:
        if name in ("irregular-1d", "irregular-2d", "irregular-3d", "mixed"):
            # extrapolated ghosts are exact on a uniform axis only
            kept = tuple(slice(3, -4) for _ in range(rank))
            if scheme == Centred(2):
                kept = tuple(slice(1, -2) for _ in range(rank))
        else:
            kept = inner
        got = quiet(g.stagger, f, grid, dst=dst, bc=bc, scheme=scheme)
        assert got.shape == cgrid.shape
        np.testing.assert_allclose(
            got[kept], linear(cgrid, dst)[kept], rtol=1e-9, atol=1e-9
        )
        if dst == "X" or (dst == "W" and rank != 3):
            continue
        back = quiet(g.collocate, linear(cgrid, dst), cgrid, src=dst, scheme=scheme)
        np.testing.assert_allclose(back, f, rtol=1e-9, atol=1e-9)


def test_velocity_round_trip_is_the_binomial_filter() -> None:
    grid = g.CartesianAGrid2D.from_interior(12, 8, 6.0, 4.0, periodic=("x", "y"))
    cgrid = grid.staggered()
    u = RNG.normal(size=cgrid.shape)
    u = g.fill_ghosts(u, cgrid, BC.periodic(), stagger="U")
    back = quiet(
        g.stagger,
        quiet(g.collocate, u, cgrid, src="U"),
        grid,
        dst="U",
        bc=BC.periodic(),
    )
    want = (
        np.roll(u[1:-1, 1:-1], 1, 1) + 2 * u[1:-1, 1:-1] + np.roll(u[1:-1, 1:-1], -1, 1)
    ) / 4
    np.testing.assert_allclose(back[1:-1, 1:-1], want, rtol=1e-13, atol=1e-14)
    # the walls and the ring are there too
    np.testing.assert_allclose(back[:, 0], back[:, -2], rtol=1e-13)


@pytest.mark.parametrize("wave", [1, 2, 4, 6])
@pytest.mark.parametrize(
    ("scheme", "response"),
    [
        (Centred(2), lambda a: np.cos(a / 2) ** 2),
        # (-1, 9, 9, -1)/16, twice
        (Centred(4), lambda a: ((9 * np.cos(a / 2) - np.cos(3 * a / 2)) / 8) ** 2),
    ],
)
def test_response_of_the_round_trip(scheme: Any, response: Any, wave: int) -> None:
    n = 12
    grid = g.CartesianAGrid1D.from_interior(n, float(n), periodic=("x",))
    cgrid = grid.staggered()
    angle = 2.0 * np.pi * wave / n
    u = np.cos(angle * np.arange(-1, n + 1))
    around = BC.periodic("x")
    back = quiet(
        g.stagger,
        quiet(g.collocate, u, cgrid, src="U", scheme=scheme, bc=around),
        grid,
        dst="U",
        bc=around,
        scheme=scheme,
    )
    np.testing.assert_allclose(back[1:-1], response(angle) * u[1:-1], atol=1e-13)


def test_the_shortest_wave_vanishes() -> None:
    n = 12
    cgrid = g.CartesianAGrid1D.from_interior(n, float(n), periodic=("x",)).staggered()
    u = (-1.0) ** np.arange(n + 2)
    np.testing.assert_array_equal(quiet(g.collocate, u, cgrid, src="U"), 0.0)


def test_fourth_order() -> None:
    errors = []
    for n in (16, 32, 64):
        grid = g.CartesianAGrid1D.from_interior(n, 2 * np.pi, periodic=("x",))
        cgrid = grid.staggered()
        f = np.sin(grid.coords("x"))
        u = quiet(g.stagger, f, grid, dst="U", bc=BC.periodic("x"), scheme=Centred(4))
        errors.append(np.abs(u - np.sin(cgrid.coords("x", at="U")))[1:-1].max())
    assert np.polyfit(range(3), -np.log2(errors), 1)[0] > 3.7


def test_fields() -> None:
    grid = AGRIDS["2d"]
    cgrid = grid.staggered()
    data = RNG.normal(size=(*grid.shape, 3))
    h = g.CField(data, ("y", "x", "member"), {"member": ("a", "b", "c")}, loc="T")
    u = quiet(g.stagger, h, grid, dst="U", bc=BC.no_flux())
    assert u.dims == h.dims and u.stagger == (("x", "right"),) and u.labels == h.labels
    plain = quiet(g.stagger, np.moveaxis(data, -1, 0), grid, dst="U", bc=BC.no_flux())
    np.testing.assert_array_equal(np.moveaxis(u.data, -1, 0), plain)
    back = quiet(g.collocate, u, cgrid)
    assert back.dims == h.dims and back.stagger == () and back.labels == h.labels
    np.testing.assert_array_equal(
        np.moveaxis(back.data, -1, 0), quiet(g.collocate, plain, cgrid, src="U")
    )
    with pytest.raises(g.StaggerMismatchError):
        g.collocate(u, cgrid, src="V")
    with pytest.raises(g.StaggerMismatchError):
        g.stagger(g.CField(data, h.dims, loc="U"), grid, dst="U", bc=BC.no_flux())


# ----------------------------------------------------------------------
# The warning
# ----------------------------------------------------------------------


def test_lossy_moves_warn() -> None:
    grid = AGRIDS["2d"]
    cgrid = grid.staggered()
    f = RNG.normal(size=grid.shape)
    for dst in "UVX":
        with pytest.warns(LossyConversionWarning, match=f"stagger to {dst}"):
            u = g.stagger(f, grid, dst=dst, bc=BC.no_flux())
        with pytest.warns(LossyConversionWarning, match=f"collocate from {dst}"):
            g.collocate(u, cgrid, src=dst)
    assert issubclass(LossyConversionWarning, UserWarning)


def test_it_warns_once_per_trace() -> None:
    grid = AGRIDS["2d"]
    f = jnp.asarray(RNG.normal(size=grid.shape))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = jax.jit(lambda f: g.stagger(f, grid, dst="U", bc=BC.no_flux()))
        for _ in range(3):
            run(f)
    lossy = [w for w in caught if issubclass(w.category, LossyConversionWarning)]
    assert len(lossy) == 1


def test_under_jit_with_the_grid_as_an_argument() -> None:
    for name in ("2d", "sphere", "irregular-2d"):
        grid = AGRIDS[name]
        f = jnp.asarray(RNG.normal(size=grid.shape))

        @jax.jit
        def there_and_back(f: Any, grid: Any) -> Any:
            u = g.stagger(f, grid, dst="U", bc=BC.no_flux())
            return u, g.collocate(u, grid.staggered(), src="U")

        got = quiet(there_and_back, f, grid)
        u = quiet(g.stagger, f, grid, dst="U", bc=BC.no_flux())
        np.testing.assert_allclose(got[0], u, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(
            got[1],
            quiet(g.collocate, u, grid.staggered(), src="U"),
            rtol=1e-10,
            atol=1e-11,
        )


# ----------------------------------------------------------------------
# Errors, and the operations of a stretched grid
# ----------------------------------------------------------------------


def test_errors() -> None:
    grid = AGRIDS["2d"]
    cgrid = grid.staggered()
    f = RNG.normal(size=grid.shape)
    with pytest.raises(TypeError, match="is staggered already"):
        g.stagger(f, cgrid, dst="U", bc=BC.no_flux())
    with pytest.raises(TypeError, match="is collocated already"):
        g.collocate(f, grid, src="U")
    with pytest.raises(ValueError, match="no condition for the west face"):
        quiet(g.stagger, f, grid, dst="U", bc=BC.no_flux("y"))
    with pytest.raises(ValueError, match="not available"):
        g.stagger(f, grid, dst="W", bc=BC.no_flux())
    with pytest.raises(TypeError, match="WENO is nonlinear"):
        g.stagger(f, grid, dst="U", bc=BC.no_flux(), scheme=WENO(5))
    with pytest.raises(ValueError, match="is not on"):
        g.stagger(f[1:], grid, dst="T", bc=BC.no_flux())
    with pytest.raises(ValueError, match="src="):
        g.collocate(np.zeros(cgrid.shape), cgrid)


STRETCHED = AGRIDS["irregular-2d"].staggered()


def test_stretched_differences_are_exact_for_linear_fields() -> None:
    for axis, src, dst in (
        ("x", "T", "U"),
        ("x", "U", "T"),
        ("y", "T", "V"),
        ("y", "U", "X"),
    ):
        x = np.asarray(STRETCHED.coords(axis, at=src))
        f = 3.0 * (x if axis == "x" else x[:, None]) * np.ones(STRETCHED.shape)
        for scheme in (Centred(2), Centred(4)):
            got = g.difference(f, STRETCHED, axis=axis, src=src, dst=dst, scheme=scheme)
            np.testing.assert_allclose(got[1:-1, 1:-1], 3.0, rtol=1e-9)


def test_a_wide_stretched_difference_is_exact_for_a_cubic() -> None:
    x = np.asarray(STRETCHED.coords("x"))
    f = np.broadcast_to(x**3, STRETCHED.shape)
    got = g.difference(f, STRETCHED, axis="x", src="T", dst="U", scheme=Centred(4))
    want = 3 * np.asarray(STRETCHED.coords("x", at="U")) ** 2
    np.testing.assert_allclose(got[2, 1:-1], want[1:-1], rtol=1e-9)


def test_stretched_calculus() -> None:
    u, v = RNG.normal(size=(2, *STRETCHED.shape))
    psi = RNG.normal(size=STRETCHED.shape)
    # div grad_perp psi = 0 and curl grad f = 0
    div = g.divergence(*g.grad_perp(psi, STRETCHED), STRETCHED)
    np.testing.assert_allclose(div[2:-1, 2:-1], 0.0, atol=1e-12)
    zeta = g.curl(*g.gradient(psi, STRETCHED), STRETCHED)
    np.testing.assert_allclose(zeta[1:-2, 1:-2], 0.0, atol=1e-12)
    # the divergence theorem: ∑ A ∇·u is the flow through the walls
    u[:, [0, -2]], v[[0, -2]] = 0.0, 0.0
    total = g.area_sum(g.divergence(u, v, STRETCHED), STRETCHED)
    assert abs(float(total)) < 1e-12
    # ∇²f = ∇·∇f
    gx, gy = g.gradient(psi, STRETCHED)
    gx[:, 0] = (psi[:, 1] - psi[:, 0]) / STRETCHED.spacing("x", at="U")[0, 0]
    gy[0] = (psi[1] - psi[0]) / STRETCHED.spacing("y", at="V")[0, 0]
    np.testing.assert_allclose(
        g.laplacian(psi, STRETCHED), g.divergence(gx, gy, STRETCHED), atol=1e-11
    )


def test_stretched_walls() -> None:
    f = RNG.normal(size=STRETCHED.shape)
    out = g.fill_ghosts(f, STRETCHED, BC(x=g.Neumann(0.4), y=g.Dirichlet(0.2)))
    x = np.asarray(STRETCHED.coords("x"))
    slope = (out[2, -1] - out[2, -2]) / (x[-1] - x[-2])
    np.testing.assert_allclose(slope, 0.4)
    slope = (out[2, 0] - out[2, 1]) / (x[1] - x[0])
    np.testing.assert_allclose(slope, 0.4)
    np.testing.assert_allclose(out[0, 2] + out[1, 2], 0.4)


def test_a_mask_on_a_stretched_grid() -> None:
    wet = RNG.random(STRETCHED.shape) > 0.2
    mask = g.Mask2D.from_grid(STRETCHED, wet)
    f = np.where(wet, RNG.normal(size=STRETCHED.shape), np.nan)
    kwargs = {"axis": "x", "src": "T", "dst": "U", "scheme": Centred(4)}
    got = g.difference(f, STRETCHED, mask=mask, **kwargs)
    on = mask.wet(at="U")
    on[[0, -1]], on[:, [0, -1]] = False, False
    assert np.isfinite(got[on]).all()
    with pytest.raises(NotImplementedError, match="mask"):
        g.difference(
            f,
            STRETCHED,
            mask=mask,
            **{**kwargs, "scheme": Adaptive(Centred(4), mask="bias")},
        )


def test_the_operations_a_stretched_grid_has_not() -> None:
    q, u, v = RNG.normal(size=(3, *STRETCHED.shape))
    for call in (
        lambda: g.advect(q, u, v, STRETCHED, scheme=WENO(5)),
        lambda: g.diffuse(q, STRETCHED, kappa=1.0),
        lambda: g.coriolis(u, v, STRETCHED, f=1.0),
        lambda: g.grad_perp(q, STRETCHED, at="T"),
        lambda: g.interpolate(
            q, STRETCHED, src="T", dst="U", scheme=Centred(4, "cell")
        ),
        lambda: g.interpolate(
            q,
            STRETCHED,
            src="T",
            dst="U",
            scheme=Adaptive(Centred(4), edges="periodic"),
        ),
    ):
        with pytest.raises(NotImplementedError):
            call()
