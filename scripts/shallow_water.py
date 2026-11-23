from __future__ import annotations

"""Nonlinear shallow-water double-gyre example.

This script modernises the old nonlinear shallow-water example so that it uses the
current :mod:`gridax` API throughout. The model is driven by the same
double-gyre wind stress pattern as the linear case, but the mass equation uses the
full layer thickness and the momentum equation includes the nonlinear advection
term. ``xarray`` handles the coordinate-aware forcing and saved diagnostics, and
the sampled fields are written to Zarr instead of being shown in a live plot.

Time integration uses ``integrix.march.heun_step`` (Heun/RK2 predictor-corrector).

The nonlinear model integrates

- d(eta)/dt = -nabla . ((H + eta) * u_vec) + nu * nabla^2(eta)
- d(u)/dt   = -g * d(eta)/dx - (u_vec . nabla) u + f*v - r*u + nu * nabla^2(u) + F_x
- d(v)/dt   = -g * d(eta)/dy - (u_vec . nabla) v - f*u - r*v + nu * nabla^2(v)

The pressure gradient uses only the linear term ``-g * d(eta)/dx`` (not the full
Bernoulli function ``B = g*eta + 0.5*(u^2 + v^2)``).  Using the full Bernoulli
together with the separate ``-(u_vec . nabla) u`` advection term would double-count
the kinetic energy gradient ``-d(u^2/2)/dx`` that is already included in
``-(u_vec . nabla) u``.

Examples
--------
Run the default experiment and save the outputs::

    uv run python scripts/shallow_water.py

Run a shorter debug case in a temporary directory::

    uv run python scripts/shallow_water.py --steps 300 --output-dir /tmp/nonlinear-swe
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from _common import (
    make_preprocessing_dataset,
    save_animation_gif,
    shallow_water_fields,
    to_wall_field,
)
from gridax import (
    CartesianCGrid2D,
    GridOps,
    embed,
    interior,
)
from gridax.schemes import Upwind
from integrix.march import heun_step
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ShallowWaterConfig:
    """Configuration for the nonlinear shallow-water double-gyre example.

    Parameters
    ----------
    nx, ny : int, optional
        Number of interior grid cells in x and y.
    Lx, Ly : float, optional
        Domain lengths [m].
    gravity : float, optional
        Gravitational acceleration [m s⁻²].
    mean_depth : float, optional
        Reference layer depth ``H`` [m].
    f0 : float, optional
        Reference Coriolis parameter [s⁻¹].
    beta : float, optional
        Meridional Coriolis gradient [m⁻¹ s⁻¹].
    drag : float, optional
        Rayleigh drag coefficient [s⁻¹].
    viscosity : float, optional
        Laplacian viscosity/diffusivity [m² s⁻¹].
    wind_acceleration : float, optional
        Peak zonal body-force acceleration [m s⁻²].
    dt : float, optional
        Time step [s].
    spinup_steps : int, optional
        Number of silent spin-up steps run before snapshot recording begins.
        The spin-up advances the model state but records no output; the first
        saved snapshot starts at ``spinup_steps * dt`` seconds.
    steps : int, optional
        Number of explicit steps to record after the spin-up phase.
    snapshot_interval : int, optional
        Steps between saved snapshots.
    zarr_path, figure_path : Path, optional
        Artifact paths written by the script. ``figure_path`` receives an
        animated GIF of the free-surface anomaly.

    Examples
    --------
    Use the default stable configuration::

        config = ShallowWaterConfig()

    Reduce the runtime for a quick smoke test::

        config = ShallowWaterConfig(nx=24, ny=24, steps=200)
    """

    nx: int = 64
    ny: int = 64
    Lx: float = 5.12e6
    Ly: float = 5.12e6
    gravity: float = 9.81
    mean_depth: float = 500.0
    f0: float = 9.375e-5
    beta: float = 1.754e-11
    drag: float = 5.0e-6
    viscosity: float = 5.0e5
    wind_acceleration: float = 2.0e-7
    dt: float = 200.0
    spinup_steps: int = 150000
    steps: int = 15000
    snapshot_interval: int = 1500
    zarr_path: Path = Path("outputs/shallow_water_double_gyre.zarr")
    figure_path: Path = Path("outputs/shallow_water_double_gyre.gif")


def run_simulation(config: ShallowWaterConfig | None = None) -> xr.Dataset:
    """Run the nonlinear shallow-water double-gyre example.

    Parameters
    ----------
    config : ShallowWaterConfig | None, optional
        Simulation configuration. The default provides a stable reference run.

    Returns
    -------
    xr.Dataset
        Sampled output dataset that is also written to ``config.zarr_path``.

    Examples
    --------
    Run the default example::

        dataset = run_simulation()

    Use a smaller grid when validating the example in tests::

        dataset = run_simulation(ShallowWaterConfig(nx=24, ny=24, steps=160))
    """
    config = config or ShallowWaterConfig()
    grid = CartesianCGrid2D.from_interior(config.nx, config.ny, config.Lx, config.Ly)
    ops = GridOps(grid)
    forcing = make_preprocessing_dataset(
        grid,
        shallow_water_fields(
            Lx=config.Lx,
            Ly=config.Ly,
            f0=config.f0,
            beta=config.beta,
            wind_acceleration=config.wind_acceleration,
        ),
        model="nonlinear shallow water",
    )

    eta = to_wall_field(forcing["eta0"])
    u = jnp.zeros_like(eta)
    v = jnp.zeros_like(eta)
    coriolis = to_wall_field(forcing["coriolis"])
    # Edge-pad Coriolis so interpolations (T->U/V) near walls see physical
    # values instead of artificially vanishing ghost cells.
    coriolis = coriolis.at[0, :].set(coriolis[1, :])
    coriolis = coriolis.at[-1, :].set(coriolis[-2, :])
    coriolis = coriolis.at[:, 0].set(coriolis[:, 1])
    coriolis = coriolis.at[:, -1].set(coriolis[:, -2])
    wind_u = to_wall_field(forcing["wind_u"])
    # Edge-pad wind forcing for the same reason as Coriolis above.
    wind_u = wind_u.at[0, :].set(wind_u[1, :])
    wind_u = wind_u.at[-1, :].set(wind_u[-2, :])
    wind_u = wind_u.at[:, 0].set(wind_u[:, 1])
    wind_u = wind_u.at[:, -1].set(wind_u[:, -2])

    viscosity = config.viscosity
    gravity = config.gravity
    mean_depth = config.mean_depth
    drag = config.drag
    dt = config.dt

    def centered_gradients(
        field: Array,
    ) -> tuple[Array, Array]:
        """Return centre-point gradients by differentiating and re-averaging."""
        dfield_dx = ops.interpolate(
            ops.difference(field, axis="x", src="T", dst="U"), src="U", dst="T"
        )
        dfield_dy = ops.interpolate(
            ops.difference(field, axis="y", src="T", dst="V"), src="V", dst="T"
        )
        return dfield_dx, dfield_dy

    def tendency(
        eta_field: Array,
        u_field: Array,
        v_field: Array,
    ) -> tuple[Array, Array, Array]:
        """Compute the nonlinear shallow-water tendencies on the C-grid.

        Uses the advective form:
          d(u)/dt = -g*d(eta)/dx - (u_vec.nabla)u + f*v - r*u + nu*nabla^2(u) + F_x
          d(v)/dt = -g*d(eta)/dy - (u_vec.nabla)v - f*u - r*v + nu*nabla^2(v)

        The pressure gradient uses only the linear term ``-g * d(eta)/dx``.  The
        kinetic energy is carried entirely by the explicit advection ``-(u.nabla)u``
        to avoid double-counting the KE gradient that would result from using the
        full Bernoulli B = g*eta + 0.5*speed^2 together with the advection term.
        """
        layer_depth = mean_depth + eta_field
        eta_rhs = ops.advect(layer_depth, u_field, v_field, scheme=Upwind(1))
        eta_rhs = eta_rhs + viscosity * ops.laplacian(eta_field)

        u_on_t = ops.interpolate(u_field, src="U", dst="T")
        v_on_t = ops.interpolate(v_field, src="V", dst="T")

        du_dx, du_dy = centered_gradients(u_on_t)
        dv_dx, dv_dy = centered_gradients(v_on_t)
        u_adv = ops.interpolate(u_on_t * du_dx + v_on_t * du_dy, src="T", dst="U")
        v_adv = ops.interpolate(u_on_t * dv_dx + v_on_t * dv_dy, src="T", dst="V")

        # Coriolis: the 4-point cross-face averages, V to U and U to V
        # directly, rather than routing through T-points (V->T->U double-average).
        # Direct cross-face interpolation is more accurate (one averaging step)
        # and consistent with the linear script.
        v_on_u = ops.interpolate(v_field, src="V", dst="U")
        u_on_v = ops.interpolate(u_field, src="U", dst="V")
        coriolis_on_u = ops.interpolate(coriolis, src="T", dst="U")
        coriolis_on_v = ops.interpolate(coriolis, src="T", dst="V")

        # Pressure gradient only (no KE term) — advection handles the rest.
        u_rhs = -gravity * ops.difference(eta_field, axis="x", src="T", dst="U")
        u_rhs = u_rhs - u_adv + coriolis_on_u * v_on_u + wind_u
        u_rhs = u_rhs - drag * u_field + viscosity * ops.laplacian(u_field, at="U")

        v_rhs = -gravity * ops.difference(eta_field, axis="y", src="T", dst="V")
        v_rhs = v_rhs - v_adv - coriolis_on_v * u_on_v
        v_rhs = v_rhs - drag * v_field + viscosity * ops.laplacian(v_field, at="V")
        return eta_rhs, u_rhs, v_rhs

    def apply_bc(state: tuple[Array, ...]) -> tuple[Array, ...]:
        """Re-apply wall (zero ghost-cell) boundary conditions."""
        return tuple(embed(interior(f, grid), grid) for f in state)

    def rhs(state: tuple[Array, ...]) -> tuple[Array, Array, Array]:
        """Tendency with BC enforcement, for use with heun_step."""
        return tendency(*apply_bc(state))

    @jax.jit
    def step(
        eta_field: Array, u_field: Array, v_field: Array
    ) -> tuple[Array, Array, Array]:
        """Advance one Heun step and refill the wall ghost cells."""
        state = heun_step((eta_field, u_field, v_field), rhs, dt)
        return apply_bc(state)

    snapshot_times: list[float] = []
    eta_snapshots: list[np.ndarray] = []
    u_snapshots: list[np.ndarray] = []
    v_snapshots: list[np.ndarray] = []
    speed_snapshots: list[np.ndarray] = []
    relative_vorticity_snapshots: list[np.ndarray] = []
    kinetic_energy: list[float] = []
    min_depth: list[float] = []

    def record_snapshot(
        step_index: int, eta_field: Array, u_field: Array, v_field: Array
    ) -> None:
        """Convert a sampled state into ``xarray``-ready NumPy arrays."""
        eta_np = np.asarray(jax.device_get(eta_field[1:-1, 1:-1]))
        u_center = ops.interpolate(u_field, src="U", dst="T")
        v_center = ops.interpolate(v_field, src="V", dst="T")
        zeta_corner = ops.curl(u_field, v_field)
        zeta_center = ops.interpolate(zeta_corner, src="X", dst="T")
        u_np = np.asarray(jax.device_get(u_center[1:-1, 1:-1]))
        v_np = np.asarray(jax.device_get(v_center[1:-1, 1:-1]))
        zeta_np = np.asarray(jax.device_get(zeta_center[1:-1, 1:-1]))
        speed_np = np.sqrt(u_np**2 + v_np**2)

        snapshot_times.append(step_index * dt)
        eta_snapshots.append(eta_np)
        u_snapshots.append(u_np)
        v_snapshots.append(v_np)
        speed_snapshots.append(speed_np)
        relative_vorticity_snapshots.append(zeta_np)
        kinetic_energy.append(float(0.5 * np.mean(speed_np**2)))
        min_depth.append(float(np.min(mean_depth + eta_np)))

    # Silent spin-up phase: run without recording snapshots.
    for _spinup in range(config.spinup_steps):
        eta, u, v = step(eta, u, v)

    record_snapshot(step_index=config.spinup_steps, eta_field=eta, u_field=u, v_field=v)

    for iteration in range(1, config.steps + 1):
        eta, u, v = step(eta, u, v)
        if iteration % config.snapshot_interval == 0 or iteration == config.steps:
            record_snapshot(
                step_index=config.spinup_steps + iteration,
                eta_field=eta,
                u_field=u,
                v_field=v,
            )

    dataset = xr.Dataset(
        data_vars={
            "eta": (
                ("time", "y", "x"),
                np.stack(eta_snapshots, axis=0),
                {"long_name": "free_surface_anomaly", "units": "m"},
            ),
            "u": (
                ("time", "y", "x"),
                np.stack(u_snapshots, axis=0),
                {"long_name": "zonal_velocity_at_t_points", "units": "m s-1"},
            ),
            "v": (
                ("time", "y", "x"),
                np.stack(v_snapshots, axis=0),
                {"long_name": "meridional_velocity_at_t_points", "units": "m s-1"},
            ),
            "speed": (
                ("time", "y", "x"),
                np.stack(speed_snapshots, axis=0),
                {"long_name": "speed_at_t_points", "units": "m s-1"},
            ),
            "relative_vorticity": (
                ("time", "y", "x"),
                np.stack(relative_vorticity_snapshots, axis=0),
                {"long_name": "relative_vorticity_at_t_points", "units": "s-1"},
            ),
            "wind_u": (
                ("y", "x"),
                forcing["wind_u"].to_numpy(),
                {"long_name": "double_gyre_zonal_forcing", "units": "m s-2"},
            ),
            "kinetic_energy": (
                ("time",),
                np.asarray(kinetic_energy),
                {"long_name": "domain_mean_kinetic_energy", "units": "m2 s-2"},
            ),
            "minimum_depth": (
                ("time",),
                np.asarray(min_depth),
                {"long_name": "minimum_total_depth", "units": "m"},
            ),
        },
        coords={
            "time": xr.DataArray(
                np.asarray(snapshot_times),
                dims=("time",),
                attrs={"long_name": "time", "units": "s"},
            ),
            "x": forcing["x"],
            "y": forcing["y"],
        },
        attrs={
            "model": "nonlinear shallow water",
            "configuration": "double gyre",
            "time_step_seconds": config.dt,
            "spinup_steps": config.spinup_steps,
            "num_steps": config.steps,
            "total_integrated_steps": config.spinup_steps + config.steps,
            "notes": "Finitevolx nonlinear example with xarray preprocessing/postprocessing.",
        },
    )

    config.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(config.zarr_path, mode="w", consolidated=False)
    save_animation_gif(
        dataset=dataset,
        gif_path=config.figure_path,
        variable_name="eta",
        title="Nonlinear shallow-water double gyre: free-surface anomaly",
    )
    return dataset


def parse_args() -> ShallowWaterConfig:
    """Parse the command line for the nonlinear shallow-water example."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = ShallowWaterConfig()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory that will receive the Zarr store and animation GIF.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=defaults.steps,
        help="Number of explicit time steps to integrate.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=defaults.snapshot_interval,
        help="Number of steps between sampled outputs.",
    )
    parser.add_argument(
        "--spinup-steps",
        type=int,
        default=defaults.spinup_steps,
        help="Silent spin-up steps before snapshot recording begins.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        zarr_path = defaults.zarr_path
        figure_path = defaults.figure_path
    else:
        zarr_path = args.output_dir / "shallow_water_double_gyre.zarr"
        figure_path = args.output_dir / "shallow_water_double_gyre.gif"

    return ShallowWaterConfig(
        steps=args.steps,
        snapshot_interval=args.snapshot_interval,
        spinup_steps=args.spinup_steps,
        zarr_path=zarr_path,
        figure_path=figure_path,
    )


def main() -> None:
    """Run the nonlinear shallow-water example from the command line."""
    config = parse_args()
    dataset = run_simulation(config)
    print(f"Saved nonlinear shallow-water fields to {config.zarr_path}")
    print(f"Saved nonlinear shallow-water animation to {config.figure_path}")
    print(
        f"Final minimum depth = {float(dataset['minimum_depth'].isel(time=-1)):.3f} m"
    )


if __name__ == "__main__":
    main()
