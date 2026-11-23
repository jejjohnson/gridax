from __future__ import annotations

"""Wind-driven 1.5-layer quasi-geostrophic closed-basin double-gyre example.

This example shows how to combine the current :mod:`gridax` advection,
interpolation, and spectral elliptic solvers to build a compact 1.5-layer QG
model.  The potential-vorticity (PV) *anomaly* is advected in a closed
rectangular basin with no-normal-flow solid walls, the streamfunction is
recovered from a Helmholtz inversion with homogeneous Dirichlet (psi = 0) wall
boundary conditions using ``solve_helmholtz_dst``, and the output is saved
through ``xarray``/Zarr together with a static before/after figure.

Time integration uses ``integrix.march.heun_step`` (Heun/RK2 predictor-corrector).

**Formulation B -- PV anomaly without the beta background**

The prognostic variable is the PV anomaly

    q_a = zeta - psi / Ld^2

where zeta = nabla^2 psi is the relative vorticity and Ld is the Rossby
deformation radius.  The resting state satisfies q_a = 0 (psi = 0, zeta = 0).
In this example we start from a small sinusoidal PV anomaly perturbation about
this resting state.  The governing equations are

- dq_a/dt = -u_vec . nabla q_a - v*beta + F - r*zeta + nu*nabla^2 q_a
- (nabla^2 - 1/Ld^2) psi = q_a
- u = -dpsi/dy,  v = dpsi/dx

where

- ``F``    = prescribed double-gyre wind-curl PV forcing
- ``r``    = linear drag coefficient acting on relative vorticity zeta
- ``nu``   = horizontal Laplacian viscosity
- ``beta`` = meridional Coriolis gradient (planetary vorticity advection)

The ``-v*beta`` term carries the planetary-vorticity advection explicitly
(rather than absorbing beta*y into q_a, which would require a non-trivial
background state in the initial condition and ghost cells).

Wind forcing
------------
The double-gyre pattern is

    F(y) = -(tau0 / rho0 / H) * (2*pi/Ly) * sin(2*pi*y/Ly)

Peak amplitude ``wind_curl_forcing = tau0 * (2*pi/Ly) / (rho0 * H)``.  For
tau0 = 0.08 N/m^2, rho0 = 1025 kg/m^3, H = 500 m the physical value is
~1.9e-13 s^-2; the default of 2e-12 s^-2 is enhanced ~10x for a faster
demonstration spin-up while keeping velocities in an oceanographically
plausible range (~10 cm/s).

Boundary conditions
-------------------
psi = 0 on all four basin walls (no-normal-flow).  The PV ghost cells are
held at zero (no-slip, consistent with psi = 0 => zeta = 0 on the walls).

Examples
--------
Run the default experiment and save the sampled fields::

    uv run python scripts/qg_1p5_layer.py

Run a shorter debugging case in a temporary directory::

    uv run python scripts/qg_1p5_layer.py --steps 400 --output-dir /tmp/qg-double-gyre
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from _common import (
    FieldBuilder,
    make_preprocessing_dataset,
    save_animation_gif,
    to_wall_field,
)
from gridax import (
    CartesianCGrid2D,
    GridOps,
    embed,
    interior,
)
from gridax.schemes import Upwind
from gridax.solvers import solve_helmholtz_dst
from integrix.march import heun_step
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)


def geostrophic_velocity_from_streamfunction(
    psi_field: Array,
    ops: GridOps,
) -> tuple[Array, Array]:
    """Map a T-point streamfunction to face-centred geostrophic velocities.

    The streamfunction is first averaged to X-points so that the orthogonal
    derivatives land directly on the U and V faces:

    - u[j, i] = -(ψ[j+1/2, i+1/2] - ψ[j-1/2, i+1/2]) / dy
    - v[j, i] =  (ψ[j+1/2, i+1/2] - ψ[j+1/2, i-1/2]) / dx

    Parameters
    ----------
    psi_field : Array
        Streamfunction stored at T-points with one ghost-cell ring.
    ops : GridOps
        The operations on the active grid.

    Returns
    -------
    tuple[Array, Array]
        The zonal and meridional velocities on U and V points.
    """
    psi_on_x = ops.interpolate(psi_field, src="T", dst="X")
    u_field = -ops.difference(psi_on_x, axis="y", src="X", dst="U")
    v_field = ops.difference(psi_on_x, axis="x", src="X", dst="V")
    return u_field, v_field


@dataclass(frozen=True)
class QuasiGeostrophicConfig:
    """Configuration for the 1.5-layer QG double-gyre example.

    Parameters
    ----------
    nx, ny : int, optional
        Number of interior cells in x and y.
    Lx, Ly : float, optional
        Domain lengths [m].
    f0 : float, optional
        Reference Coriolis parameter [s^-1].
    beta : float, optional
        Meridional Coriolis gradient [m^-1 s^-1].
    gravity : float, optional
        Gravitational acceleration [m s^-2].
    rossby_radius : float, optional
        First baroclinic Rossby radius of deformation [m].
    drag : float, optional
        Linear drag coefficient applied to relative vorticity zeta = nabla^2 psi [s^-1].
    viscosity : float, optional
        Laplacian viscosity [m^2 s^-1].
    wind_curl_forcing : float, optional
        Peak PV forcing amplitude [s^-2].  Physically this equals
        ``tau0 * (2*pi/Ly) / (rho0 * H)``; the default 2e-12 s^-2 is ~10x the
        value for tau0 = 0.08 N/m^2, rho0 = 1025 kg/m^3, H = 500 m to allow a
        faster demo spin-up while keeping velocities in a plausible range.
    dt : float, optional
        Explicit time step [s].
    spinup_steps : int, optional
        Number of silent spin-up steps run before snapshot recording begins.
        The spin-up advances the model state but records no output; the first
        saved snapshot starts at ``spinup_steps * dt`` seconds.
    steps : int, optional
        Number of time steps to record after the spin-up phase.
    snapshot_interval : int, optional
        Steps between sampled outputs.
    zarr_path, figure_path : Path, optional
        Artifact paths written by the script. ``figure_path`` receives an
        animated GIF of the free-surface anomaly field.

    Examples
    --------
    Use the default stable setup::

        config = QuasiGeostrophicConfig()

    Use a smaller test case for CI smoke checks::

        config = QuasiGeostrophicConfig(nx=24, ny=24, steps=400)
    """

    nx: int = 64
    ny: int = 64
    Lx: float = 5.12e6
    Ly: float = 5.12e6
    f0: float = 9.375e-5
    beta: float = 1.754e-11
    gravity: float = 9.81
    rossby_radius: float = 4.0e4
    drag: float = 5.0e-8
    viscosity: float = 5.0e4
    wind_curl_forcing: float = 2.0e-12
    dt: float = 4000.0
    spinup_steps: int = 5000
    steps: int = 10000
    snapshot_interval: int = 1000
    zarr_path: Path = Path("outputs/qg_1p5_layer_double_gyre.zarr")
    figure_path: Path = Path("outputs/qg_1p5_layer_double_gyre.gif")


def quasi_geostrophic_fields(config: QuasiGeostrophicConfig) -> FieldBuilder:
    """Return the fields of the QG double gyre.

    Parameters
    ----------
    config : QuasiGeostrophicConfig
        Example configuration.

    Returns
    -------
    Callable
        Builder of the initial PV anomaly ``q0``, the beta-plane term
        ``beta_term`` and the forcing ``wind_curl``, for
        :func:`make_preprocessing_dataset`.
    """

    def fields(x: xr.DataArray, y: xr.DataArray) -> dict[str, xr.DataArray]:
        q0 = (
            5.0e-9 * np.sin(2.0 * np.pi * x / config.Lx) * np.sin(np.pi * y / config.Ly)
        )
        # Double-gyre wind-curl forcing pattern (antisymmetric about basin midline):
        #   F(y) = -(tau0/(rho0*H)) * (2*pi/Ly) * sin(2*pi*y/Ly)
        # Negative in the southern half (anticyclonic, subtropical gyre) and
        # positive in the northern half (cyclonic, subpolar gyre).
        return {
            "q0": q0,
            "beta_term": config.beta * (y - 0.5 * config.Ly),
            "wind_curl": -config.wind_curl_forcing
            * np.sin(2.0 * np.pi * y / config.Ly),
        }

    return fields


def run_simulation(config: QuasiGeostrophicConfig | None = None) -> xr.Dataset:
    """Run the 1.5-layer QG double-gyre example.

    Parameters
    ----------
    config : QuasiGeostrophicConfig | None, optional
        Simulation configuration. When omitted, the default stable setup is used.

    Returns
    -------
    xr.Dataset
        Sampled output dataset written to ``config.zarr_path`` and returned.

    Examples
    --------
    Execute the default QG experiment::

        dataset = run_simulation()

    Run a compact test-sized integration::

        dataset = run_simulation(QuasiGeostrophicConfig(nx=24, ny=24, steps=300))
    """
    config = config or QuasiGeostrophicConfig()
    grid = CartesianCGrid2D.from_interior(config.nx, config.ny, config.Lx, config.Ly)
    ops = GridOps(grid)
    forcing = make_preprocessing_dataset(
        grid, quasi_geostrophic_fields(config), model="1.5-layer quasi-geostrophic"
    )

    q = to_wall_field(forcing["q0"])
    wind_curl = to_wall_field(forcing["wind_curl"])

    beta = config.beta
    viscosity = config.viscosity
    drag = config.drag
    dt = config.dt
    deformation_wavenumber = 1.0 / config.rossby_radius**2

    def invert_streamfunction(q_field: Array) -> Array:
        """Recover the streamfunction from the PV anomaly using Helmholtz DST.

        Formulation B: q_a = zeta - psi/Ld^2, so the Helmholtz equation is

            (nabla^2 - 1/Ld^2) psi = q_a

        with homogeneous Dirichlet BCs (psi = 0 on all four walls), appropriate
        for a closed rectangular basin where q_a = 0 at rest.

        Parameters
        ----------
        q_field : Array
            PV anomaly on the full ``[Ny+2, Nx+2]`` grid (ghost ring included).

        Returns
        -------
        Array
            Streamfunction on the full grid with zero ghost cells (psi = 0 walls).
        """
        # Formulation B: RHS is just q_a -- no beta*(y-y0) subtraction needed.
        # Subtracting the beta background is wrong here because q_a already
        # excludes it; adding it back would drive a large spurious initial psi.
        rhs = q_field[1:-1, 1:-1]
        psi_interior = solve_helmholtz_dst(
            rhs,
            grid.spacing("x"),
            grid.spacing("y"),
            lambda_=deformation_wavenumber,
        )
        psi_field = jnp.zeros_like(q_field)
        # Ghost cells remain zero, encoding psi = 0 on the solid walls.
        return psi_field.at[1:-1, 1:-1].set(psi_interior)

    def tendency(q_field: Array) -> tuple[Array, Array, Array, Array]:
        """Compute the QG PV tendency and the diagnosed balanced state.

        Implements:  dq_a/dt = -div(u*q_a) - v*beta + F - r*zeta + nu*nabla^2 q_a

        The three fixes vs. a naive implementation:
        1. No beta*(y-y0) subtraction in the Helmholtz RHS  (Formulation B).
        2. Explicit ``-v*beta`` planetary-vorticity term in the tendency.
        3. Linear drag acts on relative vorticity zeta = nabla^2 psi, not on q_a.
        """
        psi_field = invert_streamfunction(q_field)
        u_field, v_field = geostrophic_velocity_from_streamfunction(psi_field, ops)
        # -div(u*q_a): advective tendency (writes to [2:-2, 2:-2])
        q_rhs = ops.advect(q_field, u_field, v_field, scheme=Upwind(1))
        # -v*beta: planetary-vorticity advection at T-points.
        # Add directly to q_rhs at the interior to avoid an extra allocation.
        v_center = ops.interpolate(v_field, src="V", dst="T")
        q_rhs = q_rhs.at[1:-1, 1:-1].add(-beta * v_center[1:-1, 1:-1])
        # Linear drag on relative vorticity zeta = nabla^2 psi (not on q_a).
        # Dragging on q_a would add a spurious anti-damping +drag*psi/Ld^2 term.
        zeta = ops.laplacian(psi_field)
        q_rhs = q_rhs + wind_curl - drag * zeta + viscosity * ops.laplacian(q_field)
        return q_rhs, psi_field, u_field, v_field

    def apply_bc(q_field: Array) -> Array:
        """Re-apply wall (zero ghost-cell) boundary conditions."""
        return embed(interior(q_field, grid), grid)

    def pv_tendency(q_field: Array) -> Array:
        """PV tendency with BC enforcement, for use with heun_step."""
        q_rhs, _, _, _ = tendency(apply_bc(q_field))
        return q_rhs

    @jax.jit
    def step(q_field: Array) -> tuple[Array, Array, Array, Array]:
        """Advance one Heun step and diagnose the balanced velocity field."""
        q_next = apply_bc(heun_step(q_field, pv_tendency, dt))
        psi_next = invert_streamfunction(q_next)
        u_next, v_next = geostrophic_velocity_from_streamfunction(psi_next, ops)
        return q_next, psi_next, u_next, v_next

    psi = invert_streamfunction(q)
    u, v = geostrophic_velocity_from_streamfunction(psi, ops)

    eta_scale = config.f0 / config.gravity

    snapshot_times: list[float] = []
    q_snapshots: list[np.ndarray] = []
    psi_snapshots: list[np.ndarray] = []
    eta_snapshots: list[np.ndarray] = []
    u_snapshots: list[np.ndarray] = []
    v_snapshots: list[np.ndarray] = []
    speed_snapshots: list[np.ndarray] = []
    relative_vorticity_snapshots: list[np.ndarray] = []
    pv_enstrophy: list[float] = []

    def record_snapshot(
        step_index: int,
        q_field: Array,
        psi_field: Array,
        u_field: Array,
        v_field: Array,
    ) -> None:
        """Convert a sampled balanced state into ``xarray``-ready arrays."""
        q_np = np.asarray(jax.device_get(q_field[1:-1, 1:-1]))
        psi_np = np.asarray(jax.device_get(psi_field[1:-1, 1:-1]))
        eta_np = eta_scale * psi_np
        u_center = ops.interpolate(u_field, src="U", dst="T")
        v_center = ops.interpolate(v_field, src="V", dst="T")
        zeta_corner = ops.curl(u_field, v_field)
        zeta_center = ops.interpolate(zeta_corner, src="X", dst="T")
        u_np = np.asarray(jax.device_get(u_center[1:-1, 1:-1]))
        v_np = np.asarray(jax.device_get(v_center[1:-1, 1:-1]))
        zeta_np = np.asarray(jax.device_get(zeta_center[1:-1, 1:-1]))
        speed_np = np.sqrt(u_np**2 + v_np**2)

        snapshot_times.append(step_index * dt)
        q_snapshots.append(q_np)
        psi_snapshots.append(psi_np)
        eta_snapshots.append(eta_np)
        u_snapshots.append(u_np)
        v_snapshots.append(v_np)
        speed_snapshots.append(speed_np)
        relative_vorticity_snapshots.append(zeta_np)
        pv_enstrophy.append(float(0.5 * np.mean(q_np**2)))

    # Silent spin-up phase: run without recording snapshots.
    for _spinup in range(config.spinup_steps):
        q, psi, u, v = step(q)

    record_snapshot(
        step_index=config.spinup_steps, q_field=q, psi_field=psi, u_field=u, v_field=v
    )

    for iteration in range(1, config.steps + 1):
        q, psi, u, v = step(q)
        if iteration % config.snapshot_interval == 0 or iteration == config.steps:
            record_snapshot(
                step_index=config.spinup_steps + iteration,
                q_field=q,
                psi_field=psi,
                u_field=u,
                v_field=v,
            )

    dataset = xr.Dataset(
        data_vars={
            "q": (
                ("time", "y", "x"),
                np.stack(q_snapshots, axis=0),
                {"long_name": "potential_vorticity_anomaly", "units": "s-1"},
            ),
            "psi": (
                ("time", "y", "x"),
                np.stack(psi_snapshots, axis=0),
                {"long_name": "streamfunction", "units": "m2 s-1"},
            ),
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
            "wind_curl": (
                ("y", "x"),
                forcing["wind_curl"].to_numpy(),
                {"long_name": "double_gyre_pv_forcing", "units": "s-2"},
            ),
            "pv_enstrophy": (
                ("time",),
                np.asarray(pv_enstrophy),
                {"long_name": "domain_mean_pv_enstrophy", "units": "s-2"},
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
            "model": "1.5-layer quasi-geostrophic",
            "configuration": "double gyre",
            "time_step_seconds": config.dt,
            "spinup_steps": config.spinup_steps,
            "num_steps": config.steps,
            "total_integrated_steps": config.spinup_steps + config.steps,
            "notes": (
                "Finitevolx 1.5-layer QG double-gyre with Formulation B: "
                "q_a = zeta - psi/Ld^2 (no beta background in prognostic PV). "
                "Planetary vorticity advected via explicit -v*beta term. "
                "Drag on relative vorticity zeta=Laplacian(psi). "
                "DST Helmholtz inversion with closed-basin (Dirichlet) wall BCs."
            ),
        },
    )

    config.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(config.zarr_path, mode="w", consolidated=False)
    save_animation_gif(
        dataset=dataset,
        gif_path=config.figure_path,
        variable_name="eta",
        title="1.5-layer QG double gyre: free-surface anomaly",
        colorbar_label="[m]",
    )
    return dataset


def parse_args() -> QuasiGeostrophicConfig:
    """Parse the command line for the QG example."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = QuasiGeostrophicConfig()
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
        zarr_path = args.output_dir / "qg_1p5_layer_double_gyre.zarr"
        figure_path = args.output_dir / "qg_1p5_layer_double_gyre.gif"

    return QuasiGeostrophicConfig(
        steps=args.steps,
        snapshot_interval=args.snapshot_interval,
        spinup_steps=args.spinup_steps,
        zarr_path=zarr_path,
        figure_path=figure_path,
    )


def main() -> None:
    """Run the 1.5-layer QG example from the command line."""
    config = parse_args()
    dataset = run_simulation(config)
    print(f"Saved QG fields to {config.zarr_path}")
    print(f"Saved QG animation to {config.figure_path}")
    print(
        f"Final max |eta| = {float(np.abs(dataset['eta'].isel(time=-1)).max()):.3e} m"
    )


if __name__ == "__main__":
    main()
