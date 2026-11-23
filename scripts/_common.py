"""Helpers shared by the double-gyre example scripts.

The three model scripts (``swm_linear.py``, ``shallow_water.py`` and
``qg_1p5_layer.py``) run beside this module, so each imports it as
``from _common import ...``. It lives here and not in :mod:`gridax` because it
needs ``xarray`` and ``matplotlib``, which the package does not depend on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import jax.numpy as jnp
from jaxtyping import Array, Float
import matplotlib

matplotlib.use("Agg")
from gridax import CartesianCGrid2D
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

__all__ = [
    "FieldBuilder",
    "make_preprocessing_dataset",
    "save_animation_gif",
    "shallow_water_fields",
    "to_wall_field",
]

#: Maps the T-point coordinates ``(x, y)``, each on ``(y, x)``, to named fields.
FieldBuilder = Callable[[xr.DataArray, xr.DataArray], Mapping[str, xr.DataArray]]


def make_preprocessing_dataset(
    grid: CartesianCGrid2D, fields: FieldBuilder, *, model: str
) -> xr.Dataset:
    """Build the coordinate-aware initial condition and forcing of a script.

    Parameters
    ----------
    grid : CartesianCGrid2D
        C-grid of the simulation. The coordinates are its interior T points.
    fields : Callable
        Function of the two coordinates ``x`` and ``y`` [m], each broadcast to
        ``(y, x)``, that returns the fields by name.
    model : str
        Value of the dataset's ``model`` attribute.

    Returns
    -------
    xr.Dataset
        The named fields on the interior grid, with their coordinates and the
        attributes of a closed-basin double gyre.

    Examples
    --------
    Build one field that is linear in ``y``::

        make_preprocessing_dataset(
            grid, lambda x, y: {"coriolis": 1e-4 + 2e-11 * y}, model="example"
        )
    """
    # grid.Nx and grid.Ny count the ghost ring.
    x, y = (
        xr.DataArray(
            (np.arange(size - 2) + 0.5) * grid.spacing(name),
            dims=(name,),
            name=name,
            attrs={"long_name": f"cell_center_{name}", "units": "m"},
        )
        for name, size in (("x", grid.Nx), ("y", grid.Ny))
    )
    x2d, y2d = (c.transpose("y", "x") for c in xr.broadcast(x, y))
    named = fields(x2d, y2d)

    return xr.Dataset(
        data_vars={name: field.rename(name) for name, field in named.items()},
        coords={"x": x, "y": y},
        attrs={
            "model": model,
            "configuration": "double gyre",
            "boundary_conditions": "closed basin (no-normal-flow solid walls)",
        },
    )


def shallow_water_fields(
    *, Lx: float, Ly: float, f0: float, beta: float, wind_acceleration: float
) -> FieldBuilder:
    """Return the fields of the two shallow-water double gyres.

    Parameters
    ----------
    Lx, Ly : float
        Domain lengths [m].
    f0 : float
        Coriolis parameter at the middle of the basin [s⁻¹].
    beta : float
        Meridional Coriolis gradient [m⁻¹ s⁻¹].
    wind_acceleration : float
        Peak zonal body-force acceleration [m s⁻²].

    Returns
    -------
    Callable
        Builder of ``eta0``, ``coriolis`` and ``wind_u``, for
        :func:`make_preprocessing_dataset`.
    """

    def fields(x: xr.DataArray, y: xr.DataArray) -> dict[str, xr.DataArray]:
        # Double-gyre zonal wind body force F_x = -A * cos(2*pi*y/Ly).
        # Westward (trade winds) at y=0, eastward (westerlies) at y=Ly/2.
        # Vorticity source = -d(F_x)/dy = -A*(2*pi/Ly)*sin(2*pi*y/Ly):
        #   y in (0,  Ly/2): < 0 -> anticyclonic (subtropical gyre, south) ✓
        #   y in (Ly/2, Ly): > 0 -> cyclonic    (subpolar    gyre, north) ✓
        return {
            "eta0": 0.01 * np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly),
            "coriolis": f0 + beta * (y - 0.5 * Ly),
            "wind_u": -wind_acceleration * np.cos(2.0 * np.pi * y / Ly),
        }

    return fields


def to_wall_field(field: xr.DataArray) -> Float[Array, "Ny Nx"]:
    """Pad an interior ``xarray`` field with homogeneous Dirichlet (zero) ghost cells.

    All ghost cells are set to zero.  This is appropriate for prognostic
    fields (eta, u, v) where a literal zero boundary value is desired.
    For static fields like Coriolis or wind forcing, consider overwriting
    the ghost cells with edge values after calling this function.

    Parameters
    ----------
    field : xr.DataArray
        Interior field stored on ``(y, x)`` coordinates.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Full field of shape ``[Ny+2, Nx+2]`` with interior values preserved and
        ghost cells set to zero (homogeneous Dirichlet boundary condition).

    Examples
    --------
    Convert an interior forcing field to the gridax storage layout::

        wind_u = to_wall_field(forcing["wind_u"])

    The returned array includes the ghost cells used by the operators::

        wind_u.shape
    """
    interior = jnp.asarray(field.to_numpy())
    return jnp.pad(interior, pad_width=1, mode="constant")


def save_animation_gif(
    dataset: xr.Dataset,
    gif_path: Path,
    variable_name: str,
    title: str,
    cmap: str = "RdBu_r",
    scale_factor: float = 1.0,
    colorbar_label: str | None = None,
    fps: int = 3,
) -> None:
    """Save an animated GIF showing the time evolution of a sampled field.

    Parameters
    ----------
    dataset : xr.Dataset
        Sampled simulation output with ``time``, ``y``, and ``x`` dimensions.
    gif_path : Path
        Output path for the animated GIF.
    variable_name : str
        Name of the field to animate.
    title : str
        Figure title prefix; the simulation time in days is appended per frame.
    cmap : str, optional
        Matplotlib colour map.
    scale_factor : float, optional
        Multiplicative scale applied to the field values before display.
    colorbar_label : str | None, optional
        Colour-bar label.
    fps : int, optional
        Frames per second for the output GIF.

    Examples
    --------
    Save an animated GIF of the free-surface anomaly::

        save_animation_gif(dataset, Path("eta.gif"), "eta", "Linear SWE")

    Save a scaled field at a slower frame rate::

        save_animation_gif(
            dataset,
            Path("zeta.gif"),
            "relative_vorticity",
            "QG vorticity",
            scale_factor=1e5,
            colorbar_label=r"[$10^{-5}\\ \\mathrm{s}^{-1}$]",
            fps=2,
        )
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    if fps < 1:
        msg = f"fps must be >= 1, got {fps}"
        raise ValueError(msg)

    gif_path.parent.mkdir(parents=True, exist_ok=True)

    data = dataset[variable_name] * scale_factor
    vmax = float(np.nanmax(np.abs(data.to_numpy())))
    if vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    frame0 = data.isel(time=0).to_numpy()
    image = ax.pcolormesh(
        dataset["x"].to_numpy(),
        dataset["y"].to_numpy(),
        frame0,
        shading="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    if colorbar_label is not None:
        colorbar.set_label(colorbar_label)
    t_days = float(data["time"].isel(time=0)) / 86400.0
    title_text = ax.set_title(f"{title} | t = {t_days:.1f} d")

    def update(frame: int) -> None:
        field = data.isel(time=frame).to_numpy()
        image.set_array(field.ravel())
        t = float(data["time"].isel(time=frame)) / 86400.0
        title_text.set_text(f"{title} | t = {t:.1f} d")

    n_frames = data.sizes["time"]
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
