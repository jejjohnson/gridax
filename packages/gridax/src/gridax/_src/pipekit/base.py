"""Base classes and helpers for the `gridax.pipekit` operator layer.

Two tiers, as in geotoolz::

    Tier A  finitevolX primitives        Difference2D(grid).diff_x_T_to_U(h)
            arrays in, arrays out;       eqx.Module / pure function
            no config, no carrier
                    │
                    ▼
    Tier B  pipekit Operators            Difference(grid=G, axis="x", src="T", dst="U")
            JSON config, eqx.Module      composes in pipekit Sequential / Graph,
            leaves, `|` composition      runs under jit / grad / vmap

A Tier-B operator holds its `CartesianGrid2D` as a pytree field: the grid's
sizes are static and its extents are array leaves, so it crosses ``jit`` with
the operator. In the JSON config (``op.state`` / ``Operator.from_state``) it
is the argument list of `CartesianGrid2D.from_interior`.

Fields are ``[*batch, Ny, Nx]``: the 2-D Tier-A operators broadcast over
leading axes themselves (finitevolX #386), so Tier B calls them directly.
`over_leading` lifts a Tier-A function that is still 2-D only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array
from pipekit_jax import JaxOperator

from gridax import CartesianGrid2D


def _as_grid(spec: Sequence[Any] | CartesianGrid2D) -> CartesianGrid2D:
    """The grid of an operator, from a grid or from its JSON config.

    Parameters
    ----------
    spec : CartesianGrid2D or sequence of 4 numbers
        A grid, or ``(nx_interior, ny_interior, Lx, Ly)``, the arguments of
        `CartesianGrid2D.from_interior` (a list after a JSON round trip).

    Returns
    -------
    CartesianGrid2D

    Raises
    ------
    ValueError
        If ``spec`` does not have four entries or a count is not positive.
    """
    if isinstance(spec, CartesianGrid2D):
        return spec
    if len(spec) != 4:
        raise ValueError(
            f"grid must be a CartesianGrid2D or (nx_interior, ny_interior, Lx, Ly),"
            f" got {spec!r}"
        )
    nx, ny, lx, ly = spec
    if int(nx) < 1 or int(ny) < 1:
        raise ValueError(f"grid needs positive cell counts, got {spec!r}")
    return CartesianGrid2D.from_interior(int(nx), int(ny), lx, ly)


def over_leading(fn: Callable[..., Any], core_ndim: int = 2) -> Callable[..., Any]:
    """Lift ``fn`` on ``[*core]`` arrays to ``[*batch, *core]`` arrays.

    Every input shares the same leading shape; it is flattened to one axis,
    ``jax.vmap(fn)`` maps over it, and every output array gets the leading
    shape back::

        x : [T, E, Ny, Nx] ─reshape─► [T·E, Ny, Nx] ─vmap fn─► [T·E, …] ─reshape─► [T, E, …]

    Only needed for Tier-A functions that are still 2-D only; the 2-D
    Cartesian operators take ``[*batch, Ny, Nx]`` natively.

    Parameters
    ----------
    fn : callable
        Function of positional arrays with ``core_ndim`` trailing axes each;
        it may return an array or a pytree of arrays.
    core_ndim : int
        Number of trailing axes ``fn`` consumes (2 for ``[Ny, Nx]``).

    Returns
    -------
    callable
        ``fn`` lifted over any number of leading axes (zero included).

    Raises
    ------
    ValueError
        If the inputs disagree on their leading shape.
    """

    @functools.wraps(fn)
    def lifted(*xs: Array) -> Any:
        leads = {tuple(x.shape[: x.ndim - core_ndim]) for x in xs}
        if len(leads) != 1:
            raise ValueError(f"inputs have different leading shapes: {sorted(leads)}")
        (lead,) = leads
        if not lead:
            return fn(*xs)
        flat = [x.reshape(-1, *x.shape[x.ndim - core_ndim :]) for x in xs]
        out = jax.vmap(fn)(*flat)
        return jax.tree.map(lambda y: y.reshape(*lead, *y.shape[1:]), out)

    return lifted


class FVXOperator(JaxOperator):
    """Base for finitevolX pipekit operators.

    A `pipekit_jax.JaxOperator` (pipekit ``Operator`` + ``eqx.Module``):
    config fields are ``eqx.field(static=True)``, learnable coefficients
    are leaves with ``converter=jnp.asarray``, and ``_apply`` maps fields
    ``[*batch, Ny, Nx]`` to fields.
    """


class GridOperator(FVXOperator):
    """An `FVXOperator` on a 2-D Cartesian C-grid.

    Attributes
    ----------
    grid : CartesianGrid2D
        The grid; ``(nx_interior, ny_interior, Lx, Ly)`` (a list after a JSON
        round trip) is converted on construction.
    """

    grid: CartesianGrid2D = eqx.field(converter=_as_grid)

    def get_config(self) -> dict[str, Any]:
        """The constructor config, with the grid as ``from_interior``'s arguments."""
        grid = self.grid
        ring = 2 * grid.ghost
        spec = (grid.Nx - ring, grid.Ny - ring, float(grid.Lx), float(grid.Ly))
        return {**super().get_config(), "grid": spec}
