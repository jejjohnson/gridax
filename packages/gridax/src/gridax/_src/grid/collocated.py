"""Collocated (Arakawa A) grids: every variable at T, no ghost ring.

The sibling topology of the C-grids, for fields that come with one value per
cell: observations, a raster scene, an analysis product. A collocated grid
holds what the collocated calculus of ``gridax.analysis`` takes as keywords:
the coordinate of each axis (uniform or not), and which axes are periodic.

    CollocatedGrid1D / 2D / 3D    the topology: Nx, Ny, Nz, axes, periodic
        ↓
    CartesianAGrid1D / 2D / 3D    lengths
    SphericalAGrid2D              longitude and latitude, radius R

The steps are signed: the rows of a raster run from north to south, its
``spacing("y")`` is negative, and a derivative along ``y`` still points north.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar

from geophysix.earth import SPHERE
from jaxtyping import Float
import numpy as np

from gridax._src.grid.points import Stagger, point_axes
from gridax._src.pytree import array_namespace, pytree_dataclass, static_field

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


def _leaf(value: Any) -> Any:
    return np.asarray(value, dtype=np.float64)


class _Collocated:
    """The topology and the metric accessor shared by the collocated grids."""

    _collocated: ClassVar[bool] = True

    # Fields of the dataclasses this mixes into.
    axes: tuple[str, ...]
    periodic: tuple[str, ...]

    if TYPE_CHECKING:

        @property
        def shape(self) -> tuple[int, ...]: ...

    def __check_init__(self) -> None:
        unknown = [axis for axis in self.periodic if axis not in self.axes]
        if unknown:
            raise ValueError(f"periodic axes {unknown} are not in {self.axes}")
        for axis in self.axes:
            coords = getattr(self, f"_{axis}")
            if coords.shape != (self._size(axis),):
                raise ValueError(
                    f"{axis} has {coords.shape} coordinates for {self._size(axis)}"
                    " points"
                )
            if axis in self.periodic and self._step(axis) is None:
                raise ValueError(f"a periodic axis is uniform, and {axis} is not")

    def _size(self, axis: str) -> int:
        """Number of points along ``axis``."""
        if axis not in self.axes:
            raise ValueError(
                f"{type(self).__name__} has the axes {self.axes}, not {axis!r}"
            )
        return self.shape[self.axes.index(axis)]

    def _step(self, axis: str) -> Float[Array, ""] | None:
        """The uniform coordinate step along ``axis``, or None if it varies."""
        return getattr(self, f"_d{axis}")

    def _check_point(self, at: str | Stagger) -> None:
        if point_axes(at, self.axes):
            raise ValueError(
                f"at={at!r}: {type(self).__name__} is collocated and has T points only"
            )

    def _coordinate_steps(self, axis: str) -> Float[Array, "N"]:
        """Δ of each cell: the distance between its faces, at the midpoints."""
        x = getattr(self, f"_{axis}")
        xp = array_namespace(x)
        gaps = x[1:] - x[:-1]
        # Δ_i = (x_{i+1} - x_{i-1}) / 2, and the one gap next to an end
        return xp.concat([gaps[:1], 0.5 * (gaps[1:] + gaps[:-1]), gaps[-1:]])

    def coords(self, axis: str, *, at: str | Stagger = "T") -> Float[Array, "N"]:
        """The coordinate of each point along ``axis``.

        Parameters
        ----------
        axis : str
            One of ``grid.axes``.
        at : str
            ``"T"``, the only point of a collocated grid.

        Returns
        -------
        Float[Array, "N"]
            Radians along the axes of a spherical grid.
        """
        self._size(axis)
        self._check_point(at)
        return getattr(self, f"_{axis}")

    def spacing(self, axis: str, *, at: str | Stagger = "T") -> Float[Array, "..."]:
        """The signed grid spacing along ``axis``.

        Parameters
        ----------
        axis : str
            One of ``grid.axes``.
        at : str
            ``"T"``, the only point of a collocated grid.

        Returns
        -------
        Float[Array, "..."]
            0-d on a uniform axis. On an irregular one, the distance between
            the faces of each cell, shaped to broadcast against a field.

        Examples
        --------
        >>> import gridax
        >>> grid = gridax.CartesianAGrid1D.from_coords(x=[0.0, 1.0, 3.0, 6.0])
        >>> grid.spacing("x").tolist()
        [1.0, 1.5, 2.5, 3.0]
        """
        self._size(axis)
        self._check_point(at)
        step = self._step(axis)
        if step is not None:
            return step
        shape = [1] * len(self.axes)
        shape[self.axes.index(axis)] = -1
        return self._coordinate_steps(axis).reshape(shape)

    def area(self, *, at: str | Stagger = "T") -> Float[Array, "..."]:
        """The horizontal area of each cell, ``[Ny, Nx]`` (its length in 1-D).

        Parameters
        ----------
        at : str
            ``"T"``, the only point of a collocated grid.

        Returns
        -------
        Float[Array, "..."]
            Positive, whatever the direction of the axes.
        """
        self._check_point(at)
        return self._area()

    def _area(self, *, at: str | Stagger = "T", xp: Any = None) -> Float[Array, "..."]:
        """:meth:`area` in the namespace ``xp`` (the grid's own by default)."""
        horizontal = [axis for axis in self.axes if axis != "z"]
        cell = abs(self.spacing(horizontal[-1]))
        for axis in horizontal[:-1]:
            cell = cell * abs(self.spacing(axis))
        xp = array_namespace(cell) if xp is None else xp
        shape = tuple(self._size(axis) for axis in horizontal)
        cell = xp.asarray(cell)
        return xp.broadcast_to(cell.reshape(cell.shape[-len(shape) :]), shape)

    def staggered(self) -> Any:
        """The C-grid whose interior T points are the points of this grid.

        One ghost ring more on every side. The faces sit halfway between the
        points, and a ghost point is the mirror of the first one about the wall.

        Returns
        -------
        CartesianCGrid*, SphericalCGrid2D or StretchedCGrid*
            Stretched if a Cartesian axis is irregular.

        Raises
        ------
        ValueError
            If an axis decreases: flip it, and the fields along it, first (a
            flux through a face of a C-grid is positive along the axis).
        NotImplementedError
            If an axis of a spherical grid is irregular.

        Examples
        --------
        >>> import gridax
        >>> grid = gridax.CartesianAGrid1D.from_coords(x=[0.0, 1.0, 3.0])
        >>> cgrid = grid.staggered()
        >>> cgrid.coords("x").tolist(), cgrid.coords("x", at="U").tolist()
        ([-1.0, 0.0, 1.0, 3.0, 5.0], [-0.5, 0.5, 2.0, 4.0, 6.0])
        >>> cgrid.collocated() == grid
        True
        """
        for axis in self.axes:
            if _decreases(getattr(self, f"_{axis}")):
                raise ValueError(
                    f"staggered: the {axis} axis decreases; flip it, and the fields"
                    " along it, first"
                )
        return self._to_staggered()

    def _to_staggered(self) -> Any:
        raise NotImplementedError

    def _cartesian(self, uniform: Any, stretched: Any) -> Any:
        """The uniform or the stretched C-grid of a Cartesian grid."""
        sizes = {f"N{axis}": self._size(axis) + 2 for axis in self.axes}
        if any(self._step(axis) is None for axis in self.axes):
            fields: dict[str, Any] = {}
            for axis in self.axes:
                x = getattr(self, f"_{axis}")
                xp = array_namespace(x)
                low, high = 2.0 * x[:1] - x[1:2], 2.0 * x[-1:] - x[-2:-1]
                fields[f"_{axis}"] = xp.concat([low, x, high])
                fields[f"_d{axis}"] = self._step(axis)
            return stretched(**sizes, **fields, periodic=self.periodic)
        metric: dict[str, Any] = {}
        for axis in self.axes:
            metric[f"L{axis}"] = _length(self._step(axis), self._size(axis))
            metric[f"{axis}0"] = getattr(self, f"_{axis}")[0]
        return uniform(**sizes, **metric, periodic=self.periodic)


def _decreases(x: Any) -> bool:
    """Whether concrete coordinates decrease; a traced one is taken to increase."""
    try:
        return bool(x[1] < x[0]) if x.shape[0] > 1 else False
    except Exception:  # a tracer has no truth value
        return False


def _length(step: Any, n: int) -> Any:
    """The length of ``n`` steps, such that ``length / n`` is ``step`` to the bit."""
    length = step * n
    if not isinstance(step, np.ndarray | np.generic | float):
        return length
    for candidate in (length, np.nextafter(length, np.inf), np.nextafter(length, 0.0)):
        if candidate / n == step:
            return candidate
    return length


@pytree_dataclass
class CollocatedGrid1D:
    """1-D collocated topology: ``Nx`` points, no ghost ring.

    Parameters
    ----------
    Nx : int
        Number of points in x.
    axes : tuple of str
        The names of the array axes (static).
    ghost : int
        Width of the ghost ring: 0 (static).
    periodic : tuple of str
        The axes that wrap (static).
    """

    Nx: int = static_field()
    axes: tuple[str, ...] = static_field(default=("x",), kw_only=True)
    ghost: int = static_field(default=0, kw_only=True)
    periodic: tuple[str, ...] = static_field(default=(), kw_only=True)

    @property
    def shape(self) -> tuple[int, ...]:
        """The array shape of a field: ``(Nx,)``."""
        return (self.Nx,)


@pytree_dataclass
class CollocatedGrid2D:
    """2-D collocated topology: ``[Ny, Nx]`` points, no ghost ring.

    Parameters
    ----------
    Nx, Ny : int
        Number of points in x and y.
    axes : tuple of str
        The names of the array axes, in array order (static).
    ghost : int
        Width of the ghost ring: 0 (static).
    periodic : tuple of str
        The axes that wrap (static).
    """

    Nx: int = static_field()
    Ny: int = static_field()
    axes: tuple[str, ...] = static_field(default=("y", "x"), kw_only=True)
    ghost: int = static_field(default=0, kw_only=True)
    periodic: tuple[str, ...] = static_field(default=(), kw_only=True)

    @property
    def shape(self) -> tuple[int, ...]:
        """The array shape of a field: ``(Ny, Nx)``."""
        return (self.Ny, self.Nx)


@pytree_dataclass
class CollocatedGrid3D:
    """3-D collocated topology: ``[Nz, Ny, Nx]`` points, no ghost ring.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of points in x, y and z.
    axes : tuple of str
        The names of the array axes, in array order (static).
    ghost : int
        Width of the ghost ring: 0 (static).
    periodic : tuple of str
        The axes that wrap (static).
    """

    Nx: int = static_field()
    Ny: int = static_field()
    Nz: int = static_field()
    axes: tuple[str, ...] = static_field(default=("z", "y", "x"), kw_only=True)
    ghost: int = static_field(default=0, kw_only=True)
    periodic: tuple[str, ...] = static_field(default=(), kw_only=True)

    @property
    def shape(self) -> tuple[int, ...]:
        """The array shape of a field: ``(Nz, Ny, Nx)``."""
        return (self.Nz, self.Ny, self.Nx)


def _uniform(n: int, step: float, origin: float) -> dict[str, Any]:
    """The coordinates ``x_i = origin + i step`` and their step."""
    return {"coords": _leaf(origin + np.arange(n) * step), "step": _leaf(step)}


def _irregular(coords: Any) -> dict[str, Any]:
    """Given coordinates, with their step where it is the same everywhere."""
    x = _leaf(coords)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"coordinates must be 1-D with 2 points or more: {x.shape}")
    gaps = np.diff(x)
    if not (np.all(gaps > 0) or np.all(gaps < 0)):
        raise ValueError("coordinates must increase or decrease strictly")
    return {"coords": x, "step": None}


def _fields(**axes: dict[str, Any]) -> dict[str, Any]:
    """``_x``, ``_dx``, ``Nx``, … from the coordinates of each axis."""
    out: dict[str, Any] = {}
    for axis, given in axes.items():
        out[f"N{axis}"] = given["coords"].size
        out[f"_{axis}"] = given["coords"]
        out[f"_d{axis}"] = given["step"]
    return out


def _affine_axes(
    transform: Sequence[float], shape: tuple[int, int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The x and y axes of an unrotated affine transform, at the cell centres."""
    a, b, c, d, e, f = (float(v) for v in tuple(transform)[:6])
    if b != 0.0 or d != 0.0:
        raise ValueError(
            f"the transform is rotated (b={b}, d={d}): a collocated grid has its axes"
            " along the rows and the columns"
        )
    ny, nx = shape
    # x = a (i + 1/2) + c, y = e (j + 1/2) + f
    return _uniform(nx, a, c + 0.5 * a), _uniform(ny, e, f + 0.5 * e)


@pytree_dataclass
class CartesianAGrid1D(_Collocated, CollocatedGrid1D):
    """1-D Cartesian collocated grid.

    Build one with :meth:`from_interior` (uniform) or :meth:`from_coords`
    (any spacing).

    Examples
    --------
    >>> import gridax
    >>> gridax.CartesianAGrid1D.from_interior(4, 2.0).coords("x").tolist()
    [0.0, 0.5, 1.0, 1.5]
    """

    _x: Float[Array, "Nx"]
    _dx: Float[Array, ""] | None

    def _to_staggered(self) -> Any:
        from gridax._src.grid.cartesian import CartesianCGrid1D
        from gridax._src.grid.stretched import StretchedCGrid1D

        return self._cartesian(CartesianCGrid1D, StretchedCGrid1D)

    @classmethod
    def from_interior(
        cls, nx: int, Lx: float, *, x0: float = 0.0, periodic: Sequence[str] = ()
    ) -> CartesianAGrid1D:
        """A uniform grid, ``x_i = x0 + i Lx / nx``.

        Parameters
        ----------
        nx : int
            Number of points.
        Lx : float
            Length of the domain.
        x0 : float
            Coordinate of the first point.
        periodic : sequence of str
            The axes that wrap.

        Returns
        -------
        CartesianAGrid1D
        """
        axes = _fields(x=_uniform(nx, Lx / nx, x0))
        return cls(**axes, periodic=tuple(periodic))

    @classmethod
    def from_coords(cls, x: Any) -> CartesianAGrid1D:
        """A grid on given coordinates, one per point.

        Parameters
        ----------
        x : array-like [Nx]
            Strictly increasing or decreasing.

        Returns
        -------
        CartesianAGrid1D
        """
        return cls(**_fields(x=_irregular(x)))


@pytree_dataclass
class CartesianAGrid2D(_Collocated, CollocatedGrid2D):
    """2-D Cartesian collocated grid.

    Build one with :meth:`from_interior` (uniform), :meth:`from_affine` (a
    raster's transform) or :meth:`from_coords` (any spacing per axis).

    Examples
    --------
    >>> import gridax
    >>> t = (30.0, 0.0, 600.0, 0.0, -30.0, 900.0)  # a raster, north up
    >>> grid = gridax.CartesianAGrid2D.from_affine(t, shape=(2, 3))
    >>> grid.coords("x").tolist(), grid.coords("y").tolist()
    ([615.0, 645.0, 675.0], [885.0, 855.0])
    >>> float(grid.spacing("y")), float(grid.area()[0, 0])
    (-30.0, 900.0)
    """

    _x: Float[Array, "Nx"]
    _dx: Float[Array, ""] | None
    _y: Float[Array, "Ny"]
    _dy: Float[Array, ""] | None

    def _to_staggered(self) -> Any:
        from gridax._src.grid.cartesian import CartesianCGrid2D
        from gridax._src.grid.stretched import StretchedCGrid2D

        return self._cartesian(CartesianCGrid2D, StretchedCGrid2D)

    @classmethod
    def from_interior(
        cls,
        nx: int,
        ny: int,
        Lx: float,
        Ly: float,
        *,
        x0: float = 0.0,
        y0: float = 0.0,
        periodic: Sequence[str] = (),
    ) -> CartesianAGrid2D:
        """A uniform grid, ``x_i = x0 + i Lx / nx`` and the same along y.

        The arguments are in the order of ``CartesianCGrid2D.from_interior``.

        Parameters
        ----------
        nx, ny : int
            Number of points.
        Lx, Ly : float
            Lengths of the domain.
        x0, y0 : float
            Coordinates of the first point.
        periodic : sequence of str
            The axes that wrap.

        Returns
        -------
        CartesianAGrid2D
        """
        axes = _fields(x=_uniform(nx, Lx / nx, x0), y=_uniform(ny, Ly / ny, y0))
        return cls(**axes, periodic=tuple(periodic))

    @classmethod
    def from_affine(
        cls, transform: Sequence[float], shape: tuple[int, int]
    ) -> CartesianAGrid2D:
        """The cell centres of an affine transform ``(a, b, c, d, e, f)``.

        ``x = a (i + 1/2) + c`` and ``y = e (j + 1/2) + f``, for column ``i``
        and row ``j``.

        Parameters
        ----------
        transform : sequence of float
            The first six numbers of the transform, with ``b = d = 0``.
        shape : tuple of int
            ``(Ny, Nx)``.

        Returns
        -------
        CartesianAGrid2D

        Raises
        ------
        ValueError
            If the transform is rotated.
        """
        x, y = _affine_axes(transform, shape)
        return cls(**_fields(x=x, y=y))

    @classmethod
    def from_coords(cls, *, y: Any, x: Any) -> CartesianAGrid2D:
        """A grid on given coordinates along each axis.

        Parameters
        ----------
        y, x : array-like [Ny], [Nx]
            Strictly increasing or decreasing.

        Returns
        -------
        CartesianAGrid2D
        """
        return cls(**_fields(x=_irregular(x), y=_irregular(y)))


@pytree_dataclass
class CartesianAGrid3D(_Collocated, CollocatedGrid3D):
    """3-D Cartesian collocated grid.

    Build one with :meth:`from_interior` (uniform) or :meth:`from_coords`
    (any spacing per axis, such as vertical levels).
    """

    _x: Float[Array, "Nx"]
    _dx: Float[Array, ""] | None
    _y: Float[Array, "Ny"]
    _dy: Float[Array, ""] | None
    _z: Float[Array, "Nz"]
    _dz: Float[Array, ""] | None

    def _to_staggered(self) -> Any:
        from gridax._src.grid.cartesian import CartesianCGrid3D
        from gridax._src.grid.stretched import StretchedCGrid3D

        return self._cartesian(CartesianCGrid3D, StretchedCGrid3D)

    @classmethod
    def from_interior(
        cls,
        nx: int,
        ny: int,
        nz: int,
        Lx: float,
        Ly: float,
        Lz: float,
        *,
        x0: float = 0.0,
        y0: float = 0.0,
        z0: float = 0.0,
        periodic: Sequence[str] = (),
    ) -> CartesianAGrid3D:
        """A uniform grid, ``x_i = x0 + i Lx / nx`` and the same along y and z.

        Parameters
        ----------
        nx, ny, nz : int
            Number of points.
        Lx, Ly, Lz : float
            Lengths of the domain.
        x0, y0, z0 : float
            Coordinates of the first point.
        periodic : sequence of str
            The axes that wrap.

        Returns
        -------
        CartesianAGrid3D
        """
        axes = _fields(
            x=_uniform(nx, Lx / nx, x0),
            y=_uniform(ny, Ly / ny, y0),
            z=_uniform(nz, Lz / nz, z0),
        )
        return cls(**axes, periodic=tuple(periodic))

    @classmethod
    def from_coords(cls, *, z: Any, y: Any, x: Any) -> CartesianAGrid3D:
        """A grid on given coordinates along each axis.

        Parameters
        ----------
        z, y, x : array-like [Nz], [Ny], [Nx]
            Strictly increasing or decreasing.

        Returns
        -------
        CartesianAGrid3D
        """
        return cls(**_fields(x=_irregular(x), y=_irregular(y), z=_irregular(z)))


@pytree_dataclass
class SphericalAGrid2D(_Collocated, CollocatedGrid2D):
    """2-D collocated grid on a sphere.

    ``coords`` are the longitude and latitude in radians, and ``spacing`` the
    arc lengths ``R cos(lat) dlon`` and ``R dlat``. The constructors take
    degrees, as ``SphericalCGrid2D.from_interior`` does.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.SphericalAGrid2D.from_coords(
    ...     lat=[60.0, 0.0, -60.0], lon=[0.0, 90.0, 180.0, 270.0], R=1.0
    ... )
    >>> grid.spacing("x")[:, 0].round(4).tolist()
    [0.7854, 1.5708, 0.7854]
    """

    _x: Float[Array, "Nx"]
    _dx: Float[Array, ""] | None
    _y: Float[Array, "Ny"]
    _dy: Float[Array, ""] | None
    R: Float[Array, ""]

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, int | float) and not field.metadata.get("static"):
                object.__setattr__(self, field.name, _leaf(value))
        self.__check_init__()

    def _to_staggered(self) -> Any:
        from gridax._src.grid.spherical import SphericalCGrid2D

        dlon, dlat = self._dx, self._dy
        if dlon is None or dlat is None:
            raise NotImplementedError(
                "staggered: a spherical C-grid is uniform in longitude and latitude"
            )
        xp = array_namespace(self._x, self._y, dlon, dlat)
        lon = xp.concat([self._x[:1] - dlon, self._x, self._x[-1:] + dlon])
        lat = xp.concat([self._y[:1] - dlat, self._y, self._y[-1:] + dlat])
        shape = (self.Ny + 2, self.Nx + 2)
        rows = xp.broadcast_to(lat[:, None], shape)
        return SphericalCGrid2D(
            Nx=self.Nx + 2,
            Ny=self.Ny + 2,
            Lx=self.R * dlon * self.Nx,
            Ly=self.R * dlat * self.Ny,
            dlon=dlon,
            dlat=dlat,
            R=self.R,
            _lat=lat,
            _lon=lon,
            _cos_lat_T=xp.cos(rows),
            _cos_lat_V=xp.cos(rows + 0.5 * dlat),
            periodic=self.periodic,
        )

    @classmethod
    def from_interior(
        cls,
        nx: int,
        ny: int,
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        R: float = SPHERE.semi_major,
        *,
        periodic: Sequence[str] = (),
    ) -> SphericalAGrid2D:
        """A uniform grid, with the points of ``SphericalCGrid2D``'s interior T cells.

        ``lon_i = lon_min + i dlon`` and ``lat_j = lat_min + (j + 1/2) dlat``.

        Parameters
        ----------
        nx, ny : int
            Number of points in longitude and latitude.
        lon_range, lat_range : tuple of float
            Degrees.
        R : float
            Radius (default: the mean radius of WGS84, in metres).
        periodic : sequence of str
            The axes that wrap: ``("x",)`` for a global longitude.

        Returns
        -------
        SphericalAGrid2D
        """
        lon, lat = np.deg2rad(lon_range), np.deg2rad(lat_range)
        dlon, dlat = (lon[1] - lon[0]) / nx, (lat[1] - lat[0]) / ny
        axes = _fields(
            x=_uniform(nx, dlon, lon[0]), y=_uniform(ny, dlat, lat[0] + 0.5 * dlat)
        )
        return cls(**axes, R=_leaf(R), periodic=tuple(periodic))

    @classmethod
    def from_affine(
        cls,
        transform: Sequence[float],
        shape: tuple[int, int],
        R: float = SPHERE.semi_major,
        *,
        periodic: Sequence[str] = (),
    ) -> SphericalAGrid2D:
        """The cell centres of an affine transform in degrees.

        Parameters
        ----------
        transform : sequence of float
            The first six numbers of the transform, with ``b = d = 0``.
        shape : tuple of int
            ``(Ny, Nx)``.
        R : float
            Radius.
        periodic : sequence of str
            The axes that wrap.

        Returns
        -------
        SphericalAGrid2D

        Raises
        ------
        ValueError
            If the transform is rotated.
        """
        x, y = _affine_axes(transform, shape)
        axes = {
            name: {
                "coords": np.deg2rad(axis["coords"]),
                "step": np.deg2rad(axis["step"]),
            }
            for name, axis in (("x", x), ("y", y))
        }
        return cls(**_fields(**axes), R=_leaf(R), periodic=tuple(periodic))

    @classmethod
    def from_coords(
        cls, *, lat: Any, lon: Any, R: float = SPHERE.semi_major
    ) -> SphericalAGrid2D:
        """A grid on given latitudes and longitudes, in degrees.

        Parameters
        ----------
        lat, lon : array-like [Ny], [Nx]
            Strictly increasing or decreasing.
        R : float
            Radius.

        Returns
        -------
        SphericalAGrid2D
        """
        axes = _fields(x=_irregular(np.deg2rad(lon)), y=_irregular(np.deg2rad(lat)))
        return cls(**axes, R=_leaf(R))

    def _cos_lat(self) -> Float[Array, "Ny 1"]:
        xp = array_namespace(self._y)
        return xp.cos(self._y)[:, None]

    def spacing(self, axis: str, *, at: str | Stagger = "T") -> Float[Array, "..."]:
        """The signed arc length along ``axis``.

        Parameters
        ----------
        axis : str
            ``"x"`` (longitude) or ``"y"`` (latitude).
        at : str
            ``"T"``, the only point of a collocated grid.

        Returns
        -------
        Float[Array, "Ny Nx"] or Float[Array, "..."]
            ``R cos(lat) dlon``, ``[Ny, Nx]``, or ``R dlat``.
        """
        step = super().spacing(axis, at=at)
        if axis == "y":
            return self.R * step
        xp = array_namespace(self._y)
        return xp.broadcast_to(self.R * self._cos_lat() * step, self.shape)


def agrid_from(grid: Any) -> CartesianAGrid2D | SphericalAGrid2D:
    """The collocated grid of a georeferenced raster.

    Reads ``grid.transform``, ``grid.shape`` and, if present,
    ``grid.crs.is_geographic``, which a ``geoarray.Grid`` and a rasterio
    dataset have; gridax imports neither.

    Parameters
    ----------
    grid : object
        With an affine ``transform`` and a ``shape`` that ends in
        ``(Ny, Nx)``.

    Returns
    -------
    SphericalAGrid2D or CartesianAGrid2D
        Spherical if the CRS is geographic (degrees).

    Raises
    ------
    ValueError
        If the transform is rotated.

    Examples
    --------
    >>> from types import SimpleNamespace as NS
    >>> import gridax
    >>> scene = NS(
    ...     transform=(0.1, 0, 10, 0, -0.1, 50),
    ...     shape=(4, 5),
    ...     crs=NS(is_geographic=True),
    ... )
    >>> type(gridax.agrid_from(scene)).__name__
    'SphericalAGrid2D'
    """
    ny, nx = tuple(grid.shape)[-2:]
    crs = getattr(grid, "crs", None)
    cls = SphericalAGrid2D if getattr(crs, "is_geographic", False) else CartesianAGrid2D
    return cls.from_affine(grid.transform, (int(ny), int(nx)))
