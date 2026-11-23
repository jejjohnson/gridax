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
from typing import Any

#: Every canonical function, by name.
OPERATIONS: dict[str, Any] = {}

_C2 = ("CartesianCGrid2D", "SphericalCGrid2D")
_C3 = ("CartesianCGrid3D", "SphericalCGrid3D")

#: ``operation -> grid types`` it is defined on.
SUPPORT: dict[str, tuple[str, ...]] = {
    "area_mean": _C2,
    "area_sum": _C2,
    "volume_mean": _C3,
    "volume_sum": _C3,
}


def _not_defined(name: str) -> Callable[..., Any]:
    """The default implementation: raise, naming the operation and the grid."""

    def impl(grid: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{name} is not defined on {type(grid).__name__}")

    return impl


def grid_dispatch[F: Callable[..., Any]](public: F) -> F:
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
    impl = functools.singledispatch(_not_defined(public.__name__))
    public.register = impl.register
    public.registry = impl.registry
    public._impl = impl
    OPERATIONS[public.__name__] = public
    return public
