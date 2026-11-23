"""`GridOps`: the canonical functions on one grid and mask."""

from __future__ import annotations

from collections.abc import Callable
import functools
import inspect
from typing import TYPE_CHECKING, Any

from gridax._src.advection import advect, reconstruct
from gridax._src.api import OPERATIONS
from gridax._src.diffusion import diffuse
from gridax._src.operators import (
    calculus,
    coriolis,
    diagnostics,
    jacobian,
    momentum,
    moves,
    reductions,
)
from gridax._src.pytree import pytree_dataclass

#: The modules that declare the canonical functions: importing them fills
#: ``OPERATIONS``.
_DECLARED_IN = (
    advect,
    reconstruct,
    diffuse,
    calculus,
    coriolis,
    diagnostics,
    jacobian,
    momentum,
    moves,
    reductions,
)

__all__ = ["GridOps"]


class _Bound:
    """One canonical function as a method: its grid and mask come from the instance."""

    def __init__(self, op: Callable[..., Any]) -> None:
        self.op = op
        self.__doc__ = op.__doc__
        signature = inspect.signature(op)
        kept = [p for name, p in signature.parameters.items() if name != "grid"]
        self.signature = signature.replace(parameters=kept)

    def __get__(self, ops: Any, owner: type) -> Any:
        if ops is None:
            return self.op
        if not defined_on(self.op, ops.grid):
            raise AttributeError(
                f"{self.op.__name__} is not defined on {type(ops.grid).__name__}"
            )
        bound = functools.partial(self.op, grid=ops.grid, mask=ops.mask)
        functools.update_wrapper(bound, self.op)
        bound.__signature__ = self.signature
        return bound


def defined_on(op: Any, grid: Any) -> bool:
    """Whether a canonical function has an implementation for a grid."""
    return op._impl.dispatch(type(grid)) is not op._impl.dispatch(object)


@pytree_dataclass
class GridOps:
    """The canonical functions with a grid and a mask filled in.

    ``GridOps(grid, mask).curl(u, v)`` is ``gridax.curl(u, v, grid,
    mask=mask)``, and so for every function that is defined on the grid: rank,
    staggering and geometry come from its type. The methods are generated from
    the functions, take the same keywords, and add no arithmetic. A method that
    the grid has no function for is not there (``hasattr`` is false and ``dir``
    does not list it).

    It is a pytree of its grid and its mask, so it crosses ``jax.jit`` as an
    argument.

    Parameters
    ----------
    grid : grid
        Any grid of gridax.
    mask : Mask1D, Mask2D or Mask3D, optional
        The wet cells. A call may give another ``mask=``.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 4.0, 4.0)
    >>> ops = gridax.GridOps(grid)
    >>> y, x = np.meshgrid(grid.coords("y"), grid.coords("x"), indexing="ij")
    >>> ops.curl(-y, x)[2].tolist()  # solid-body rotation
    [0.0, 2.0, 2.0, 2.0, 2.0, 0.0]
    >>> hasattr(gridax.GridOps(gridax.CartesianCGrid1D.from_interior(4, 4.0)), "curl")
    False
    """

    grid: Any
    mask: Any = None

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Callable[..., Any]: ...

    def __dir__(self) -> list[str]:
        """The attributes, with the methods the grid has."""
        absent = {
            name for name, op in OPERATIONS.items() if not defined_on(op, self.grid)
        }
        return [name for name in super().__dir__() if name not in absent]


for _name, _op in OPERATIONS.items():
    setattr(GridOps, _name, _Bound(_op))
