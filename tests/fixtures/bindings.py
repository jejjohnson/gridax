"""Bindings of the canonical functions, under the names gridax had until 0.3.

The golden fixtures and the regression tests were written against one operator
class per family, rank and geometry. #493 removed the classes; these hold a grid
and a mask and call the functions, so the same tests hold the functions to the
same numbers. They add no arithmetic, beyond the few combinations of functions
that were methods (``Vorticity2D.enstrophy``, ``Divergence2D.noflux``, …).
"""

from __future__ import annotations

from typing import Any

import gridax as g
from gridax._src.operators._utils import _sanitize, _where
from gridax._src.pytree import pytree_dataclass
from gridax.schemes import TVD, WENO, Centred, Upwind
import numpy as np

UPWIND = Upwind(1)


def _wet(mask: Any, at: str) -> Any:
    if mask is None:
        return None
    return mask.wet(at=at, kind="strict" if at == "X" else "default")


class _Binding:
    grid: Any
    mask: Any

    @property
    def ops(self) -> g.GridOps:
        return g.GridOps(self.grid, self.mask)

    def _d(self, f: Any, axis: str, src: str, dst: str) -> Any:
        return self.ops.difference(f, axis=axis, src=src, dst=dst)

    def _i(self, f: Any, src: str, dst: str) -> Any:
        return self.ops.interpolate(f, src=src, dst=dst)


def _moves(cls: type) -> type:
    """``diff_x_T_to_U``, ``T_to_U``, … for every move of the function."""
    from gridax._src.operators.triples import PAIRS, TRIPLES

    rank = {"1": ("x",), "2": ("y", "x"), "3": ("z", "y", "x")}[cls.__name__[-2]]
    if "Difference" in cls.__name__:
        names = (
            {"x": "lon", "y": "lat", "z": "z"} if "Spherical" in cls.__name__ else {}
        )
        for axis, src, dst in TRIPLES[rank]:

            def move(
                self: Any, f: Any, a: str = axis, s: str = src, d: str = dst
            ) -> Any:
                return self._d(f, a, s, d)

            setattr(cls, f"diff_{names.get(axis, axis)}_{src}_to_{dst}", move)
    else:
        for src, dst in PAIRS[rank]:

            def to(self: Any, f: Any, s: str = src, d: str = dst) -> Any:
                return self._i(f, s, d)

            setattr(cls, f"{src}_to_{dst}", to)
    return cls


class _Vector:
    """The vector calculus of the Cartesian difference classes."""

    def divergence(self: Any, u: Any, v: Any) -> Any:
        return self.ops.divergence(u, v)

    def curl(self: Any, u: Any, v: Any) -> Any:
        return self.ops.curl(u, v)

    def laplacian(self: Any, h: Any) -> Any:
        return self.ops.laplacian(h)

    def grad_perp(self: Any, psi: Any) -> Any:
        return self.ops.grad_perp(psi, at="T")


@_moves
@pytree_dataclass
class Difference1D(_Vector, _Binding):
    grid: Any
    mask: Any = None


@_moves
@pytree_dataclass
class Difference2D(_Vector, _Binding):
    grid: Any
    mask: Any = None


@_moves
@pytree_dataclass
class Difference3D(_Vector, _Binding):
    grid: Any
    mask: Any = None


@_moves
@pytree_dataclass
class Interpolation1D(_Binding):
    grid: Any
    mask: Any = None


@_moves
@pytree_dataclass
class Interpolation2D(_Binding):
    grid: Any
    mask: Any = None


@_moves
@pytree_dataclass
class Interpolation3D(_Binding):
    grid: Any
    mask: Any = None


class _Spherical:
    def diff2_lon(self: Any, h: Any) -> Any:
        from gridax._src.operators._utils import _metric_like, _spherical_d2_lon

        grid = self.grid
        cos = _metric_like(grid._cos_lat("T"), h)[1:-1, 1:-1]
        c = slice(1, -1)
        values = _spherical_d2_lon(
            h[..., c, c], h[..., c, 2:], h[..., c, :-2], cos, grid.dlon, grid.R
        )
        return self._embedded(values, h)

    def laplacian_merid(self: Any, h: Any) -> Any:
        from gridax._src.operators._utils import (
            _metric_like,
            _spherical_laplacian_merid,
        )

        grid = self.grid
        c = slice(1, -1)
        cos, cos_v = (_metric_like(grid._cos_lat(p), h) for p in "TV")
        values = _spherical_laplacian_merid(
            h[..., c, c],
            h[..., 2:, c],
            h[..., :-2, c],
            cos_v[1:-1, 1:-1],
            cos_v[:-2, 1:-1],
            cos[1:-1, 1:-1],
            grid.dlat,
            grid.R,
        )
        return self._embedded(values, h)

    def _embedded(self: Any, values: Any, h: Any) -> Any:
        from gridax._src.operators._ghost import interior, zero_z_ghosts

        out = interior(values, h)
        if len(self.grid.axes) == 3:
            out = zero_z_ghosts(out)
        return out if self.mask is None else out * self.mask.wet()

    def geostrophic_velocity(
        self: Any, h: Any, f: Any, gravity: float = 9.80665
    ) -> Any:
        return self.ops.geostrophic_velocity(h, f=f, gravity=gravity)


@_moves
@pytree_dataclass
class SphericalDifference2D(_Spherical, _Binding):
    grid: Any
    mask: Any = None


@_moves
@pytree_dataclass
class SphericalDifference3D(_Spherical, _Binding):
    grid: Any
    mask: Any = None


class _Divergence:
    def __call__(self: Any, u: Any, v: Any) -> Any:
        return self.ops.divergence(u, v)


@pytree_dataclass
class Divergence2D(_Divergence, _Binding):
    grid: Any
    mask: Any = None

    def noflux(self, u: Any, v: Any) -> Any:
        u, v = (
            g.fill_ghosts(f, self.grid, g.BC.wall_normal(), stagger=p)
            for f, p in ((u, "U"), (v, "V"))
        )
        return self.ops.divergence(u, v)


@pytree_dataclass
class SphericalDivergence2D(_Divergence, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalDivergence3D(_Divergence, _Binding):
    grid: Any
    mask: Any = None


class _Laplacian:
    def __call__(self: Any, h: Any) -> Any:
        return self.ops.laplacian(h)


@pytree_dataclass
class SphericalLaplacian2D(_Laplacian, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalLaplacian3D(_Laplacian, _Binding):
    grid: Any
    mask: Any = None


class _Vorticity:
    def relative_vorticity(self: Any, u: Any, v: Any) -> Any:
        return self.ops.curl(u, v)

    def potential_vorticity(self: Any, u: Any, v: Any, h: Any, f: Any) -> Any:
        return self.ops.sw_potential_vorticity(u, v, h, f)

    def enstrophy(self: Any, u: Any, v: Any) -> Any:
        if self.mask is not None:
            u, v = (
                _sanitize(_wet(self.mask, "U"), u),
                _sanitize(_wet(self.mask, "V"), v),
            )
        return _where(_wet(self.mask, "X"), g.enstrophy(self.ops.curl(u, v)))

    def potential_enstrophy(self: Any, u: Any, v: Any, h: Any, f: Any) -> Any:
        if self.mask is not None:
            u, v = (
                _sanitize(_wet(self.mask, "U"), u),
                _sanitize(_wet(self.mask, "V"), v),
            )
            f, h = (_sanitize(_wet(self.mask, "T"), x) for x in (f, h))
        q = self.ops.sw_potential_vorticity(u, v, h, f)
        h_on_q = self.ops.interpolate(h, src="T", dst="X")
        return _where(_wet(self.mask, "X"), g.potential_enstrophy(q, h_on_q))

    def pv_flux_energy_conserving(self: Any, q: Any, u: Any, v: Any) -> Any:
        return self.ops.pv_flux(q, u, v, form="energy")

    def pv_flux_enstrophy_conserving(self: Any, q: Any, u: Any, v: Any) -> Any:
        return self.ops.pv_flux(q, u, v, form="enstrophy")

    def pv_flux_blend(self: Any, q: Any, u: Any, v: Any, alpha: float = 1 / 3) -> Any:
        return self.ops.pv_flux(q, u, v, form="blend", alpha=alpha)


@pytree_dataclass
class Vorticity2D(_Vorticity, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class Vorticity3D(_Vorticity, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalVorticity2D(_Vorticity, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalVorticity3D(_Vorticity, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class ArakawaJacobian2D(_Binding):
    grid: Any
    mask: Any = None

    def __call__(self, f: Any, h: Any) -> Any:
        return self.ops.arakawa_jacobian(f, h)


class _Coriolis:
    def __call__(self: Any, u: Any, v: Any, f: Any) -> Any:
        return self.ops.coriolis(u, v, f=f)


@pytree_dataclass
class Coriolis2D(_Coriolis, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class Coriolis3D(_Coriolis, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class Energetics2D(_Binding):
    grid: Any
    mask: Any = None

    def kinetic_energy(self, u: Any, v: Any) -> Any:
        return self.ops.kinetic_energy(u, v)

    def bernoulli_potential(
        self, h: Any, u: Any, v: Any, gravity: float = 9.80665
    ) -> Any:
        return self.ops.bernoulli_potential(h, u, v, gravity=gravity)

    def available_potential_energy(self, h: Any, H: Any, g_prime: Any) -> Any:
        wet = _wet(self.mask, "T")
        h, H = (g.interior(_sanitize(wet, x), self.grid) for x in (h, H))
        ape = g.available_potential_energy(h, H, g_prime)
        return _where(wet, g.embed(ape, self.grid))


@pytree_dataclass
class Strain2D(_Binding):
    grid: Any
    mask: Any = None

    def shear(self, u: Any, v: Any) -> Any:
        return self.ops.shear_strain(u, v)

    def tensor(self, u: Any, v: Any) -> Any:
        return self.ops.tensor_strain(u, v)

    def magnitude_squared(self, u: Any, v: Any) -> Any:
        return self.ops.strain_rate_squared(u, v)

    def okubo_weiss(self, u: Any, v: Any) -> Any:
        return self.ops.okubo_weiss_parameter(u, v)


@pytree_dataclass
class QGPotentialVorticity2D(_Binding):
    grid: Any
    mask: Any = None

    def __call__(self, psi: Any, f0: Any, beta: Any, y: Any, y0: Any) -> Any:
        return self.ops.qg_potential_vorticity(psi, f0=f0, beta=beta, y=y, y0=y0)

    def stretching(self, A: Any, psi: Any) -> Any:
        wet = _wet(self.mask, "T")
        return _where(wet, g.stretching_term(A, _sanitize(wet, psi)))

    def multilayer(self, psi: Any, A: Any, f0: Any, beta: Any, y: Any, y0: Any) -> Any:
        return self.ops.qg_potential_vorticity(psi, f0=f0, beta=beta, y=y, y0=y0, A=A)


class _Advection:
    def __call__(
        self: Any, h: Any, u: Any, v: Any, scheme: Any = UPWIND, wall: str = "closed"
    ) -> Any:
        return self.ops.advect(h, u, v, scheme=scheme, wall=wall)


@pytree_dataclass
class Advection1D(_Binding):
    grid: Any
    mask: Any = None

    def __call__(
        self, h: Any, u: Any, scheme: Any = UPWIND, wall: str = "closed"
    ) -> Any:
        return self.ops.advect(h, u, None, scheme=scheme, wall=wall)


@pytree_dataclass
class Advection2D(_Advection, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class Advection3D(_Advection, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalAdvection2D(_Advection, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalAdvection3D(_Advection, _Binding):
    grid: Any
    mask: Any = None


class _Diffusion:
    def __call__(self: Any, h: Any, kappa: Any, wall: str = "closed") -> Any:
        return self.ops.diffuse(h, kappa=kappa, wall=wall)

    def fluxes(self: Any, h: Any, kappa: Any, wall: str = "closed") -> Any:
        return self.ops.diffusive_fluxes(h, kappa=kappa, wall=wall)


@pytree_dataclass
class Diffusion2D(_Diffusion, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class Diffusion3D(_Diffusion, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalDiffusion2D(_Diffusion, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalDiffusion3D(_Diffusion, _Binding):
    grid: Any
    mask: Any = None


class _Biharmonic:
    def __call__(self: Any, h: Any, kappa: Any) -> Any:
        return self.ops.biharmonic(h, kappa4=kappa)


@pytree_dataclass
class BiharmonicDiffusion2D(_Biharmonic, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class BiharmonicDiffusion3D(_Biharmonic, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalBiharmonicDiffusion2D(_Biharmonic, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalBiharmonicDiffusion3D(_Biharmonic, _Binding):
    grid: Any
    mask: Any = None


class _Momentum:
    def __call__(self: Any, u: Any, v: Any, scheme: str = "energy") -> Any:
        return self.ops.advect_momentum(u, v, form=scheme)


@pytree_dataclass
class MomentumAdvection2D(_Momentum, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class MomentumAdvection3D(_Momentum, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalMomentumAdvection2D(_Momentum, _Binding):
    grid: Any
    mask: Any = None


@pytree_dataclass
class SphericalMomentumAdvection3D(_Momentum, _Binding):
    grid: Any
    mask: Any = None


_RECONSTRUCTIONS = {
    "naive": Centred(),
    "upwind1": Upwind(1),
    "upwind2": Upwind(2),
    "upwind3": Upwind(3),
    "weno3": WENO(3),
    "wenoz3": WENO(3, "z"),
    "weno5": WENO(5),
    "wenoz5": WENO(5, "z"),
    "weno7": WENO(7),
    "weno9": WENO(9),
}


def _reconstructions(cls: type) -> type:
    """``weno5_x``, ``tvd_y_masked``, … for every scheme and axis."""
    for axis in "xy"[: max(1, int(cls.__name__[-2]) - 0)]:
        for name, scheme in _RECONSTRUCTIONS.items():

            def flux(self: Any, h: Any, u: Any, a: str = axis, s: Any = scheme) -> Any:
                return g.reconstruct(h, u, self.grid, axis=a, scheme=s)

            def masked(
                self: Any, h: Any, u: Any, mask: Any, a: str = axis, s: Any = scheme
            ) -> Any:
                return g.reconstruct(h, u, self.grid, axis=a, scheme=s, mask=mask)

            setattr(cls, f"{name}_{axis}", flux)
            setattr(cls, f"{name}_{axis}_masked", masked)

        def tvd(
            self: Any, h: Any, u: Any, limiter: str = "minmod", a: str = axis
        ) -> Any:
            return g.reconstruct(h, u, self.grid, axis=a, scheme=TVD(limiter))

        def tvd_masked(
            self: Any, h: Any, u: Any, mask: Any, limiter: str = "minmod", a: str = axis
        ) -> Any:
            return g.reconstruct(
                h, u, self.grid, axis=a, scheme=TVD(limiter), mask=mask
            )

        setattr(cls, f"tvd_{axis}", tvd)
        setattr(cls, f"tvd_{axis}_masked", tvd_masked)
    return cls


@_reconstructions
@pytree_dataclass
class Reconstruction1D:
    grid: Any


@_reconstructions
@pytree_dataclass
class Reconstruction2D:
    grid: Any


@_reconstructions
@pytree_dataclass
class Reconstruction3D:
    grid: Any


def uv_center_flux(
    h: Any, u: Any, v: Any, grid: Any, scheme: Any = UPWIND, mask: Any = None
) -> tuple[Any, Any]:
    return (
        g.reconstruct(h, u, grid, axis="x", scheme=scheme, mask=mask),
        g.reconstruct(h, v, grid, axis="y", scheme=scheme, mask=mask),
    )


uv_node_flux = uv_center_flux


def pv_flux_blend(q: Any, u: Any, v: Any, *, alpha: float = 1 / 3) -> Any:
    ny, nx = q.shape[-2:]
    grid = g.CartesianCGrid2D.from_interior(nx - 2, ny - 2, 1.0, 1.0)
    return g.pv_flux(q, u, v, grid, form="blend", alpha=alpha)


def _grid(f: Any, dx: Any, dy: Any) -> Any:
    ny, nx = np.shape(f)[-2:]
    return g.CartesianCGrid2D.from_interior(
        nx - 2, ny - 2, (nx - 2) * dx, (ny - 2) * dy
    )


def geostrophic_velocity_sphere(
    h: Any, f: Any, grid: Any, gravity: float = 9.80665
) -> tuple[Any, Any]:
    return g.geostrophic_velocity(h, grid, f=f, gravity=gravity)


def bernoulli_potential(h: Any, u: Any, v: Any, gravity: float = 9.80665) -> Any:
    return g.bernoulli_potential(h, u, v, _grid(h, 1.0, 1.0), gravity=gravity)


def sw_potential_vorticity(u: Any, v: Any, h: Any, f: Any, dx: Any, dy: Any) -> Any:
    return g.sw_potential_vorticity(u, v, h, f, _grid(h, dx, dy))


sw_potential_vorticity_multilayer = sw_potential_vorticity


def qg_potential_vorticity(
    psi: Any, f0: Any, beta: Any, dx: Any, dy: Any, y: Any, y0: Any
) -> Any:
    grid = _grid(psi, dx, dy)
    return g.qg_potential_vorticity(psi, grid, f0=f0, beta=beta, y=y, y0=y0)


def potential_vorticity_multilayer(
    psi: Any, A: Any, f0: Any, beta: Any, dx: Any, dy: Any, y: Any, y0: Any
) -> Any:
    grid = _grid(psi, dx, dy)
    return g.qg_potential_vorticity(psi, grid, f0=f0, beta=beta, y=y, y0=y0, A=A)


def vertical_velocity(
    u: Any, v: Any, dx: Any, dy: Any, dz: Any, mask: Any = None
) -> Any:
    nz, ny, nx = u.shape
    grid = g.CartesianCGrid3D.from_interior(
        nx - 2, ny - 2, nz - 2, (nx - 2) * dx, (ny - 2) * dy, (nz - 2) * dz
    )
    return g.vertical_velocity(u, v, grid, mask=mask)
