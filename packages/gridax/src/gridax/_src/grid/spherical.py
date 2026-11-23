"""Spherical Arakawa C-grid containers (2-D, 3-D).

These grids are concrete subclasses of :class:`CurvilinearGrid2D` /
:class:`CurvilinearGrid3D` with the metric of a sphere: the angular
spacings (``dlon``, ``dlat`` in radians) and the radius ``R`` give

    Δx_P = R cos(lat_P) dlon,    Δy_P = R dlat,

through the same accessor as every other grid, ``grid.spacing(axis, at=P)``,
and ``grid.coords("y")`` / ``grid.coords("x")`` are the latitudes and
longitudes in radians. The inherited extents are the arc lengths
``Lx = R (lon_max - lon_min)`` and ``Ly = R (lat_max - lat_min)``.

The arrays are built with NumPy in float64. The metric (``spacing``,
``min_cell_width``, ...) computes with the namespace of the arrays it reads,
so it runs on the NumPy grid and, inside a JAX transform, on the traced one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from geophysix.earth import SPHERE
from jaxtyping import Float
import numpy as np
from pipekit_array.functions import namespace_of

from gridax._src.grid.base import CurvilinearGrid2D, CurvilinearGrid3D, _as_leaf
from gridax._src.grid.points import point_axes
from gridax._src.pytree import pytree_dataclass

# The coordinate arrays are NumPy when a grid is built and JAX arrays (or
# tracers) once a transform has flattened it. ``jaxtyping.Array`` imports JAX,
# so it names the type for the type checker only.
if TYPE_CHECKING:
    from jaxtyping import Array

    from gridax._src.grid.collocated import SphericalAGrid2D
else:
    Array = np.ndarray


class _SphericalMetric:
    """The metric of a longitude-latitude grid.

    Mixed into :class:`SphericalCGrid2D` and :class:`SphericalCGrid3D`, over
    the uniform metric it overrides along ``x`` and ``y``.
    """

    # Fields and methods of the grids this mixes into.
    axes: tuple[str, ...]
    dlon: Float[Array, ""]
    dlat: Float[Array, ""]
    R: Float[Array, ""]
    _lat: Float[Array, "Ny"]
    _lon: Float[Array, "Nx"]
    _cos_lat_T: Float[Array, "Ny Nx"]
    _cos_lat_V: Float[Array, "Ny Nx"]

    if TYPE_CHECKING:
        _size: Any
        _staggered: Any

    def _cos_lat(self, at: str) -> Float[Array, "Ny Nx"]:
        """cos(latitude) at the points ``at``, ``[Ny, Nx]``.

        U shares the latitude of T, and X that of V.
        """
        return self._cos_lat_V if self._staggered("y", at) else self._cos_lat_T

    def spacing(self, axis: str, *, at: str = "T") -> Float[Array, "..."]:
        """The grid spacing along ``axis`` at the points ``at``.

        ``R cos(lat_P) dlon`` along ``x``, which shrinks towards the poles,
        ``R dlat`` along ``y``, and the uniform ``Δz`` along ``z``.

        Parameters
        ----------
        axis : str
            One of ``grid.axes``.
        at : str
            The point: ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.

        Returns
        -------
        Float[Array, "Ny Nx"] or Float[Array, ""]
            ``[Ny, Nx]`` along ``x`` (shared by the levels of a 3-D grid),
            0-d along the other axes, in the units of ``R``.

        Notes
        -----
        The zonal spacing is the raw metric and is not clamped, so a row at
        or past a pole carries a degenerate width that can be zero or
        slightly negative (in float32 ``cos(pi/2)`` evaluates to
        ``-4.4e-08``). ``min_cell_width`` and ``max_aspect`` do clamp, so
        prefer them when the value feeds a CFL or resolution guard.

        Examples
        --------
        >>> import gridax
        >>> grid = gridax.SphericalCGrid2D.from_interior(
        ...     4, 2, (0.0, 90.0), (-60.0, 60.0), R=1.0
        ... )
        >>> [round(float(d), 4) for d in grid.spacing("x")[1:-1, 0]]
        [0.3401, 0.3401]
        """
        self._staggered(axis, at)
        if axis == "x":
            # Δx_P = R cos(lat_P) dlon
            return self.R * self._cos_lat(at) * self.dlon
        if axis == "y":
            # Δy = R dlat
            return self.R * self.dlat
        return cast(Any, super()).spacing(axis, at=at)

    def coords(self, axis: str, *, at: str = "T") -> Float[Array, "N"]:
        """The coordinate of the points ``at`` along ``axis``, ghost ring included.

        The longitude (``x``) and the latitude (``y``) in radians, shifted by
        half a step at the points staggered along the axis, and ``z`` as on a
        Cartesian grid.

        Parameters
        ----------
        axis : str
            One of ``grid.axes``.
        at : str
            The point: ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.

        Returns
        -------
        Float[Array, "N"]
            One coordinate per index along the axis.

        Examples
        --------
        >>> import gridax, numpy as np
        >>> grid = gridax.SphericalCGrid2D.from_interior(
        ...     4, 2, (0.0, 90.0), (-60.0, 60.0)
        ... )
        >>> np.rad2deg(grid.coords("y")).round(6).tolist()
        [-90.0, -30.0, 30.0, 90.0]
        """
        staggered = self._staggered(axis, at)
        if axis == "x":
            return self._lon + 0.5 * self.dlon if staggered else self._lon
        if axis == "y":
            return self._lat + 0.5 * self.dlat if staggered else self._lat
        return cast(Any, super()).coords(axis, at=at)

    def _area(self, *, at: str = "T", xp: Any = None) -> Float[Array, "Ny Nx"]:
        """The area of the cell around each point ``at``, exact on the sphere.

        ``A = R² dlon (sin(lat_P + dlat/2) − sin(lat_P − dlat/2))
        = 2 R² dlon sin(dlat / 2) cos(lat_P)``.

        Summed over the interior T cells of a grid that spans
        ``[lat_min, lat_max]`` it gives ``R² Lλ (sin lat_max − sin lat_min)``
        exactly (``4πR²`` for the whole sphere), where the midpoint rule
        ``R² cos(lat) dlon dlat`` overestimates by ``dlat / (2 sin(dlat/2))``.

        A cell whose ``cos(lat)`` is negative (a ghost row past a pole) or
        indistinguishable from zero (within ``8 eps`` of the dtype) has area
        zero, so the area is never negative in float32 or float64.
        """
        point_axes(at, self.axes)
        cos = self._cos_lat(at)
        if xp is None:
            xp = namespace_of(cos, self.R, self.dlon, self.dlat)
        cos = xp.asarray(cos)
        base = 2.0 * self.R**2 * self.dlon * xp.sin(xp.asarray(0.5 * self.dlat))
        degenerate = cos <= 8.0 * xp.finfo(cos.dtype).eps
        return xp.where(degenerate, 0.0, base * cos)

    # ------------------------------------------------------------------
    # Physical cell widths (for resolution / CFL guards)
    # ------------------------------------------------------------------

    @property
    def min_cell_width(self) -> Float[Array, ""]:
        """Smallest physical *horizontal* cell width over the interior.

        ``min(min(Δx_T[interior]), Δy)`` — the CFL-limiting length of the
        grid. Only the physical interior ``[1:-1, 1:-1]`` contributes: a
        ghost row can sit past the pole where ``cos(lat) < 0``, which would
        otherwise dominate the minimum with a negative width. The vertical
        spacing is excluded, since the horizontal and vertical CFL
        conditions are governed by different wave speeds.

        A polar row reports exactly zero.  That needs saying because
        ``cos(pi/2)`` is not exactly zero in floating point, and its
        sign is not even consistent: it is ``-4.4e-08`` in float32 and
        ``+6.1e-17`` in float64.  Clamping at zero would therefore fix
        the float32 case — where a negative "width" would silently
        produce a negative CFL time step — while leaving float64 to
        report a positive width of ``4e-10`` metres, which a CFL guard
        would accept and then choose an unusable step from.  Both are
        recognised as degenerate instead: see :func:`_degenerate_width`.

        Returned as a 0-d array rather than a Python ``float`` so the
        property is usable inside ``jax.jit``; call ``float(...)`` on
        it in eager code (for example a preflight assertion).

        Returns
        -------
        Float[Array, ""]
            Smallest cell width, never negative.  Zero if the interior
            reaches a pole, where the zonal width degenerates.
        """
        dx_i = self.spacing("x")[1:-1, 1:-1]
        xp = namespace_of(dx_i)
        dx_i = xp.where(_degenerate_width(dx_i, self.R * self.dlon), 0.0, dx_i)
        return xp.minimum(xp.min(dx_i), xp.asarray(self.spacing("y")))

    @property
    def max_aspect(self) -> Float[Array, ""]:
        """Largest horizontal cell anisotropy ``Δy / Δx_T`` over the interior.

        Grows without bound towards the poles, where cells become
        vanishingly narrow in longitude while keeping their full
        meridional extent.  Useful for warning that an operator or
        solver is being asked to work on highly anisotropic cells.
        Ghost cells are excluded, as for :attr:`min_cell_width`.

        Returned as a 0-d array for the same JIT reason as
        :attr:`min_cell_width`.

        A cell whose zonal width has degenerated at a pole (see
        :attr:`min_cell_width` for why that width is roundoff of
        either sign rather than zero) reports ``inf``, so
        ``max_aspect`` stays monotone in how bad the cell is instead
        of returning a huge negative ratio in float32 or a merely
        large positive one in float64.

        Returns
        -------
        Float[Array, ""]
            Maximum aspect ratio, ``inf`` if any interior cell has a
            degenerate zonal width (interior reaching a pole).
        """
        dx_i = self.spacing("x")[1:-1, 1:-1]
        xp = namespace_of(dx_i)
        degenerate = _degenerate_width(dx_i, self.R * self.dlon)
        safe_dx = xp.where(degenerate, 1.0, dx_i)
        aspect = xp.where(degenerate, np.inf, self.spacing("y") / safe_dx)
        return xp.max(aspect)


@pytree_dataclass
class SphericalCGrid2D(_SphericalMetric, CurvilinearGrid2D):
    """2-D Arakawa C-grid on a sphere.

    Inherits ``Nx``, ``Ny``, ``Lx``, ``Ly`` from
    :class:`CurvilinearGrid2D` and adds the spherical metric.  All
    field arrays have total shape ``[Ny, Nx]``; the physical interior
    is ``(Ny-2) x (Nx-2)`` (slice ``[1:-1, 1:-1]``); one ghost-cell
    ring on each side is reserved for boundary conditions. Build one with
    :meth:`from_interior`; ``Lx = R * (lon_max - lon_min)`` and
    ``Ly = R * (lat_max - lat_min)``.

    Parameters
    ----------
    dlon : float
        Longitudinal grid spacing in radians.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere (metres).

    Notes
    -----
    Colocation convention (same-index rule, in spherical coordinates)::

        T[j, i]  cell centre  at  ( lon_i        ,  lat_j         )
        U[j, i]  east face    at  ( lon_{i+1/2}  ,  lat_j         )
        V[j, i]  north face   at  ( lon_i        ,  lat_{j+1/2}   )
        X[j, i]  NE corner    at  ( lon_{i+1/2}  ,  lat_{j+1/2}   )

    where ``lon_i = lon_min + (i-1) * dlon`` and
    ``lat_j = lat_min + (j - 1/2) * dlat``: latitude is a bounded axis, so
    the T rows are cell centres, the interior rows ``j = 1 .. Ny-2`` span
    ``lat_min + dlat/2 .. lat_max - dlat/2`` symmetrically, and the V faces
    ``lat_{j+1/2} = lat_min + j * dlat`` put ``V[0]`` on ``lat_min`` and
    ``V[Ny-2]`` on ``lat_max``. No T row sits on a pole, and a ghost row
    lies at most ``dlat/2`` past one. (Longitude is periodic: ``lon_i``
    keeps the first interior column on ``lon_min``.) Because U-points
    share the latitude of T-points, and X-points that of V-points.

    As in the Cartesian case, array index ``[j, i]`` encodes the
    **south-west** corner of the stencil neighbourhood.
    """

    dlon: Float[Array, ""]
    dlat: Float[Array, ""]
    R: Float[Array, ""]
    # Latitude and longitude of the T points (radians), and cos(latitude) at
    # the T (and U) and the V (and X) points.
    _lat: Float[Array, "Ny"]
    _lon: Float[Array, "Nx"]
    _cos_lat_T: Float[Array, "Ny Nx"]
    _cos_lat_V: Float[Array, "Ny Nx"]

    @classmethod
    def from_interior(  # intentionally different signature from Curvilinear
        cls,
        nx_interior: int,
        ny_interior: int,
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        R: float = SPHERE.semi_major,
    ) -> SphericalCGrid2D:
        """Construct a spherical grid from interior cell counts and coordinate ranges.

        Parameters
        ----------
        nx_interior : int
            Number of interior (physical) cells in longitude.
        ny_interior : int
            Number of interior (physical) cells in latitude.
        lon_range : tuple[float, float]
            (lon_min, lon_max) in **degrees**.
        lat_range : tuple[float, float]
            (lat_min, lat_max) in **degrees**.
        R : float
            Planet radius in metres (default: the IUGG mean radius of WGS84,
            ``geophysix.earth.SPHERE``, 6,371,008.77 m).

        Returns
        -------
        SphericalCGrid2D
        """
        lon_min_rad = np.deg2rad(lon_range[0])
        lon_max_rad = np.deg2rad(lon_range[1])
        lat_min_rad = np.deg2rad(lat_range[0])
        lat_max_rad = np.deg2rad(lat_range[1])

        dlon = float((lon_max_rad - lon_min_rad) / nx_interior)
        dlat = float((lat_max_rad - lat_min_rad) / ny_interior)

        Nx = nx_interior + 2
        Ny = ny_interior + 2

        # Arc lengths
        Lx = R * float(lon_max_rad - lon_min_rad)
        Ly = R * float(lat_max_rad - lat_min_rad)

        # 1-D coordinate arrays with ghost cells
        # Latitude is bounded: T rows are cell centres,
        #   lat_T[j] = lat_min + (j - 1/2) * dlat,
        # so j=1 is half a cell north of lat_min and j=Ny-2 half a cell south
        # of lat_max; the ghost rows j=0, Ny-1 sit half a cell outside.
        lat_1d = lat_min_rad + (np.arange(Ny) - 0.5) * dlat
        # Longitude is periodic: the first interior column sits on lon_min.
        lon_1d = lon_min_rad + (np.arange(Nx) - 1) * dlon

        # cos(latitude) at the T (and U) rows and, half a cell north, the V
        # (and X) rows: [Ny, Nx]
        lat_2d = np.broadcast_to(lat_1d[:, None], (Ny, Nx))
        cos_lat_T = np.cos(lat_2d)
        cos_lat_V = np.cos(lat_2d + 0.5 * dlat)

        return cls(
            Nx=Nx,
            Ny=Ny,
            Lx=_as_leaf(Lx),
            Ly=_as_leaf(Ly),
            dlon=_as_leaf(dlon),
            dlat=_as_leaf(dlat),
            R=_as_leaf(R),
            _lat=_as_leaf(lat_1d),
            _lon=_as_leaf(lon_1d),
            _cos_lat_T=_as_leaf(cos_lat_T),
            _cos_lat_V=_as_leaf(cos_lat_V),
        )

    def collocated(self) -> SphericalAGrid2D:
        """The collocated grid of the interior T points.

        ``agrid.staggered().collocated()`` is ``agrid``.

        Returns
        -------
        SphericalAGrid2D
        """
        from gridax._src.grid.collocated import SphericalAGrid2D

        ghost = self.ghost
        return SphericalAGrid2D(
            Nx=self.Nx - 2 * ghost,
            Ny=self.Ny - 2 * ghost,
            _x=self._lon[ghost:-ghost],
            _dx=self.dlon,
            _y=self._lat[ghost:-ghost],
            _dy=self.dlat,
            R=self.R,
            periodic=self.periodic,
        )


def _degenerate_width(
    dx: Float[Array, "..."], nominal: Float[Array, ""]
) -> Float[Array, "..."]:
    """Boolean mask of zonal widths that have collapsed at a pole.

    ``dx = R cos(lat) dlon`` vanishes at the poles, but ``cos(pi/2)``
    is not exactly zero in floating point and its sign depends on the
    precision: ``-4.4e-08`` in float32, ``+6.1e-17`` in float64.  A
    test for ``dx <= 0`` therefore catches the float32 case and misses
    the float64 one, letting a CFL guard accept a grid whose narrowest
    cell is nanometres wide.

    The cutoff is the floating-point resolution of the nominal width
    ``R dlon``, with a few ulps of headroom.  Since
    ``cos(pi/2 + delta) ~ -delta``, a row that is genuinely (rather
    than spuriously) close to the pole — by more than an ulp of angle
    — still reports its true width.

    Parameters
    ----------
    dx : Float[Array, "..."]
        Zonal cell widths.
    nominal : Float[Array, ""]
        The equatorial width ``R * dlon`` these are measured against.

    A *genuinely* negative width — a grid whose T rows run past a pole,
    so ``cos(lat)`` is meaningfully below zero rather than roundoff — is
    degenerate too, and by a wider margin. Testing only ``|dx| <= tol``
    would let those through and make ``min_cell_width`` negative, so
    the two conditions are combined.

    Returns
    -------
    Float[Array, "..."]
        True where the width is indistinguishable from zero, or is
        negative.
    """
    tolerance = 8.0 * np.finfo(dx.dtype).eps * abs(nominal)
    return (abs(dx) <= tolerance) | (dx < 0.0)


@pytree_dataclass
class SphericalCGrid3D(_SphericalMetric, CurvilinearGrid3D):
    """3-D Arakawa C-grid on a sphere.

    Inherits ``Nx``, ``Ny``, ``Nz``, ``Lx``, ``Ly``, ``Lz`` from
    :class:`CurvilinearGrid3D` and adds the spherical metric.  Scalar field arrays have total shape
    ``[Nz, Ny, Nx]`` with one ghost-cell ring on every axis; the
    physical interior is ``(Nz-2) x (Ny-2) x (Nx-2)`` (slice
    ``[1:-1, 1:-1, 1:-1]``).  The horizontal metric is 2-D ``[Ny, Nx]``,
    shared across all z-levels because the horizontal staggering is
    independent of depth.

    As with the Cartesian :class:`CartesianCGrid3D`, only the horizontal
    axes are staggered (T, U, V, X share the same ``k * dz``); W-point fields (vertical velocity at z-interfaces) use the same
    ``[Nz, Ny, Nx]`` shape, ``w[k]`` on the top face of ``T[k]`` (as
    ``Mask3D.wet(at="W")`` and ``vertical_velocity``).

    Parameters
    ----------
    dlon : float
        Longitudinal grid spacing in radians.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere (metres).

    Notes
    -----
    Colocation convention (same-index rule, horizontal staggering only)::

        T[k, j, i]  cell centre  at  ( lon_i       ,  lat_j        ,  k * dz )
        U[k, j, i]  east face    at  ( lon_{i+1/2} ,  lat_j        ,  k * dz )
        V[k, j, i]  north face   at  ( lon_i       ,  lat_{j+1/2}  ,  k * dz )
        X[k, j, i]  NE corner    at  ( lon_{i+1/2} ,  lat_{j+1/2}  ,  k * dz )

    All four point types share the same z-coordinate ``k * dz``, so the
    horizontal metric is 2-D and broadcasts over the leading ``Nz``
    axis when used in operators.
    """

    dlon: Float[Array, ""]
    dlat: Float[Array, ""]
    R: Float[Array, ""]
    # Latitude and longitude of the T points (radians), and cos(latitude) at
    # the T (and U) and the V (and X) points.
    _lat: Float[Array, "Ny"]
    _lon: Float[Array, "Nx"]
    _cos_lat_T: Float[Array, "Ny Nx"]
    _cos_lat_V: Float[Array, "Ny Nx"]

    @classmethod
    def from_interior(  # intentionally different signature from Curvilinear
        cls,
        nx_interior: int,
        ny_interior: int,
        nz_interior: int,
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        Lz: float,
        R: float = SPHERE.semi_major,
    ) -> SphericalCGrid3D:
        """Construct a 3-D spherical grid from interior cell counts.

        Parameters
        ----------
        nx_interior : int
            Number of interior cells in longitude.
        ny_interior : int
            Number of interior cells in latitude.
        nz_interior : int
            Number of interior cells in z.
        lon_range : tuple[float, float]
            (lon_min, lon_max) in **degrees**.
        lat_range : tuple[float, float]
            (lat_min, lat_max) in **degrees**.
        Lz : float
            Physical domain depth/height.
        R : float
            Planet radius in metres (default: the IUGG mean radius of WGS84,
            ``geophysix.earth.SPHERE``, 6,371,008.77 m).

        Returns
        -------
        SphericalCGrid3D
        """
        # Build the horizontal grid first
        h_grid = SphericalCGrid2D.from_interior(
            nx_interior, ny_interior, lon_range, lat_range, R
        )

        return cls(
            Nx=h_grid.Nx,
            Ny=h_grid.Ny,
            Nz=nz_interior + 2,
            Lx=h_grid.Lx,
            Ly=h_grid.Ly,
            Lz=_as_leaf(Lz),
            dlon=h_grid.dlon,
            dlat=h_grid.dlat,
            R=h_grid.R,
            _lat=h_grid._lat,
            _lon=h_grid._lon,
            _cos_lat_T=h_grid._cos_lat_T,
            _cos_lat_V=h_grid._cos_lat_V,
        )

    def horizontal_grid(self) -> SphericalCGrid2D:
        """Extract the horizontal 2-D spherical grid.

        Returns
        -------
        SphericalCGrid2D
            A 2-D grid with the same horizontal fields.
        """
        return SphericalCGrid2D(
            Nx=self.Nx,
            Ny=self.Ny,
            Lx=self.Lx,
            Ly=self.Ly,
            dlon=self.dlon,
            dlat=self.dlat,
            R=self.R,
            _lat=self._lat,
            _lon=self._lon,
            _cos_lat_T=self._cos_lat_T,
            _cos_lat_V=self._cos_lat_V,
            ghost=self.ghost,
        )
