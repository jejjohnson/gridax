"""Dispatch of the canonical functions on the grid type.

A canonical function ``op(fields…, grid, *, …)`` (ADR 1) is declared once,
with its documented signature, and calls ``op._impl(grid, fields…, …)``, a
``functools.singledispatch`` on the type of the grid. The implementations
register themselves by grid type, so the rank, the staggering and the geometry
are read off the grid. The type is static in the pytree, so the choice is made
when a function is traced and costs nothing in the compiled program.

``OPERATIONS`` holds every canonical function by name and ``SUPPORT`` the grid
types each one is defined on. ``tests/test_registry_parity.py`` keeps the two
in step.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any, Protocol, cast

#: Every canonical function, by name.
OPERATIONS: dict[str, Any] = {}

_C2 = ("CartesianCGrid2D", "SphericalCGrid2D")
_C3 = ("CartesianCGrid3D", "SphericalCGrid3D")
_C = ("CartesianCGrid1D", *_C2, *_C3)
_S1, _S2, _S3 = (f"StretchedCGrid{rank}D" for rank in (1, 2, 3))
_S = (_S1, _S2, _S3)
_A1, _A3 = "CartesianAGrid1D", "CartesianAGrid3D"
_A2 = ("CartesianAGrid2D", "SphericalAGrid2D")
_A = (_A1, *_A2, _A3)

#: ``operation -> grid types`` it is defined on.
SUPPORT: dict[str, tuple[str, ...]] = {
    "advect": _C,
    "advect_momentum": (*_C2, *_C3),
    "advect_vertical": _C3,
    "arakawa_jacobian": ("CartesianCGrid2D",),
    "area_mean": (*_C2, _S2, *_A2),
    "area_sum": (*_C2, _S2, *_A2),
    "bernoulli_potential": (*_C2, *_C3),
    "biharmonic": (*_C2, *_C3),
    "coriolis": ("CartesianCGrid2D", "CartesianCGrid3D"),
    "curl": (*_C2, *_C3, _S2, _S3, *_A2, _A3),
    "difference": (*_C, *_S, *_A),
    "diffuse": (*_C2, *_C3),
    "diffuse_vertical": (*_C3, _S3),
    "diffusive_fluxes": (*_C2, *_C3),
    "divergence": (*_C2, *_C3, _S2, _S3, *_A2, _A3),
    "geostrophic_velocity": ("SphericalCGrid2D", *_A2),
    "grad_perp": (*_C2, *_C3, _S2, _S3, *_A2, _A3),
    "gradient": (*_C2, *_C3, _S2, _S3, *_A2, _A3),
    "interpolate": (*_C, *_S, *_A),
    "kinetic_energy": (*_C2, *_C3, *_A2, _A3),
    "laplacian": (*_C, *_S, *_A),
    "okubo_weiss_parameter": ("CartesianCGrid2D", *_A2),
    "pv_flux": _C2,
    "qg_potential_vorticity": ("CartesianCGrid2D",),
    "reconstruct": _C,
    "shear_strain": ("CartesianCGrid2D", "CartesianCGrid3D", *_A2, _A3),
    "strain_rate_squared": ("CartesianCGrid2D", *_A2),
    "sw_potential_vorticity": (*_C2, *_A2),
    "tensor_strain": ("CartesianCGrid2D", "CartesianCGrid3D", *_A2, _A3),
    "vertical_velocity": ("CartesianCGrid3D",),
    "volume_mean": (*_C3, _S3, _A3),
    "volume_sum": (*_C3, _S3, _A3),
    "vortex_force": (*_C2, *_C3),
}


def _not_defined(name: str) -> Callable[..., Any]:
    """The default implementation: raise, naming the operation and the grid."""

    def impl(grid: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{name} is not defined on {type(grid).__name__}")

    return impl


class Canonical[**P, R](Protocol):
    """A canonical function: its call, and its dispatch on the grid type."""

    __name__: str
    #: The ``singledispatch`` that the body calls, with the grid first.
    _impl: Any
    #: A decorator that adds the implementation of a grid type.
    register: Any
    #: Grid type to implementation.
    registry: Any

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


def grid_dispatch[**P, R](public: Callable[P, R]) -> Canonical[P, R]:
    """Make a canonical function dispatch on the type of its grid.

    The function gets ``register`` (a decorator, by grid type), ``registry``
    (grid type to implementation) and ``_impl``, the ``singledispatch`` that
    its body calls with the grid first.

    Parameters
    ----------
    public : callable
        The canonical function, with its documented signature.

    Returns
    -------
    callable
        ``public``, listed in ``OPERATIONS``.

    Examples
    --------
    >>> @grid_dispatch
    ... def twice(x, grid):
    ...     return twice._impl(grid, x)
    >>> @twice.register(int)
    ... def _(grid, x):
    ...     return 2 * x
    >>> twice(3.0, 1)
    6.0
    >>> twice(3.0, "sphere")
    Traceback (most recent call last):
        ...
    NotImplementedError: twice is not defined on str
    >>> _ = OPERATIONS.pop("twice")
    """
    op = cast(Any, public)
    impl = functools.singledispatch(_not_defined(op.__name__))
    op.register = impl.register
    op.registry = impl.registry
    op._impl = impl
    OPERATIONS[op.__name__] = op
    return op
