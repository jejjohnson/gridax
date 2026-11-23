"""The coordinates of a Cartesian grid's T points, as an (N, D) point array.

A point method (semi-Lagrangian advection in ``integrix.march``, a functax
interpolation plan) takes the nodes of a field as N points in D dimensions.
This is the grid-built half of that call: the T points of the whole array,
ghost ring included, in the order ``field.ravel()`` visits them.
"""

from __future__ import annotations

from jaxtyping import Float
import numpy as np

from gridax._src.grid.cartesian import CartesianCGrid1D, CartesianCGrid2D


def t_point_coordinates(
    grid: CartesianCGrid1D | CartesianCGrid2D,
) -> Float[np.ndarray, "N D"]:
    """Coordinates of every T point of a Cartesian grid, one row per point.

    T[j, i] sits at (y, x) = (j dy, i dx), the same-index convention of
    ``docs/cgrid_discretization.md``, with j and i running over the whole
    array (the ghost ring included, so cell 1 is the first interior cell).
    The columns follow the array's axes, (y, x) for ``[Ny, Nx]``, and the
    rows follow ``field.ravel()``, so a raveled T field and these points
    line up.

    Parameters
    ----------
    grid : CartesianCGrid1D or CartesianCGrid2D
        The grid.

    Returns
    -------
    np.ndarray
        ``[Nx, 1]`` points (x) for a 1-D grid, ``[Ny * Nx, 2]`` points
        (y, x) for a 2-D grid, in float64.

    Raises
    ------
    TypeError
        If ``grid`` is not a 1-D or 2-D Cartesian grid. A spherical grid's
        T points are its latitudes ``grid.coords("y")`` and longitudes
        ``grid.coords("x")``, in radians.

    Examples
    --------
    >>> import gridax
    >>> grid = gridax.CartesianCGrid2D.from_interior(2, 1, 2.0, 0.5)
    >>> points = gridax.t_point_coordinates(grid)
    >>> points.shape
    (12, 2)
    >>> points[:5].tolist()
    [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.5, 0.0]]
    """
    if not isinstance(grid, CartesianCGrid1D | CartesianCGrid2D):
        raise TypeError(
            "t_point_coordinates takes a CartesianCGrid1D or CartesianCGrid2D, got"
            f" {type(grid).__name__}; a spherical grid's T points are its"
            ' latitudes grid.coords("y") and longitudes grid.coords("x").'
        )
    # y_j = j Δy, x_i = i Δx
    points = np.meshgrid(*(grid.coords(axis) for axis in grid.axes), indexing="ij")
    # [N, D], row-major like field.ravel()
    return np.column_stack([axis.ravel() for axis in points])
