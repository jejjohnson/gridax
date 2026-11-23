"""One-face boundary conditions: what the ghost layers of a face hold.

An atom knows neither its face nor its point: `gridax.BC` says which face it
is on and `gridax.fill_ghosts` which point the field sits on. The point
decides where the wall is relative to the field's nodes (same-index
staggering, AGENTS.md contract 1):

- **Cell-centred along the face normal** (T everywhere; U on the y faces, V
  on the x faces): the wall is half a cell outside the first interior node.
  Layer ``k`` outwards sits ``(k - 1/2) Δn`` outside it, as the interior node
  ``φ_k`` sits inside, and the atom fills it.
- **Node on the wall** (U and X on the x faces, V and X on the y faces, W on
  the z faces): the field's own node ``φ_0`` is the wall. The atom writes it,
  and the layers outside it are mirrors about it.

The gradient ``g`` of `Neumann` and `Robin` is along the outward normal, and
``Δn`` is the grid's spacing on the face, which varies by row on a sphere.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.backend import set_at
from gridax._src.grid.points import Stagger, point_axes
from gridax._src.operators._utils import _metric_like
from gridax._src.pytree import array_namespace, pytree_dataclass, static_field
from gridax._src.stencils.coefficients import difference_coeffs

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

#: ``φ(k)``: the k-th interior node from the wall, with the face's axis dropped.
type Interior = Callable[[int], Float[Array, "*line"]]


def _check_unit(name: str, value: Any) -> None:
    """Raise unless a concrete Python number lies in [0, 1] (arrays pass)."""
    if isinstance(value, int | float) and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")


class _Atom:
    """The mirror rules shared by the atoms; subclasses give the formulas."""

    def at(self, axis: str, point: str | Stagger, axes: tuple[str, ...]) -> Any:
        """The atom a face of ``axis`` has for a field at ``point``, or ``None``."""
        return self

    def _ghost(self, phi: Interior, k: int, step: Any) -> Any:
        """Layer ``k`` outside a wall half a cell from the first node."""
        raise NotImplementedError

    def _wall(self, phi: Interior, step: Any) -> Any:
        """The node on the wall."""
        raise NotImplementedError

    def _outside(self, wall: Any, phi: Interior, k: int, step: Any) -> Any:
        """Layer ``k`` outside a node on the wall: the odd mirror about it."""
        return 2.0 * wall - phi(k)

    def fill(
        self,
        f: Float[Array, "..."],
        grid: Any,
        axis: str,
        side: int,
        point: str | Stagger,
        width: int,
    ) -> Float[Array, "..."]:
        """Write this face's ghost layers (and the wall node, where there is one).

        Parameters
        ----------
        f : array
            A field whose last axes are ``grid.axes``, with ``width`` ghost
            layers on each side.
        grid : grid
            Gives the axes and the spacing on the face.
        axis : str
            The axis normal to the face.
        side : {0, 1}
            The low (south, west, bottom) or the high side.
        point : str or Stagger
            The point of the field.
        width : int
            Number of ghost layers.

        Returns
        -------
        array
            ``f`` with the face filled.
        """
        pos = grid.axes.index(axis) - len(grid.axes)  # from the end
        n = f.shape[pos]

        def at(i: int) -> Any:
            # line i along the axis, from this side's end of the array
            index = i if side == 0 else n - 1 - i
            return (..., index, *(slice(None),) * (-pos - 1))

        on_wall = axis in point_axes(point, grid.axes)
        # A node on the wall is `width - 1` from the low end and `width` from
        # the high one (same-index staggering: U[i] is the east face of T[i]).
        wall = width - 1 if side == 0 else width
        first = width if side == 0 or not on_wall else width + 1

        def phi(k: int) -> Any:
            return f[at(first + k - 1)]

        step = _face_spacing(grid, axis, side, point, f)
        if not on_wall:
            layers = {width - k: self._ghost(phi, k, step) for k in range(1, width + 1)}
        else:
            node = self._wall(phi, step)
            layers = {wall: node}
            for k in range(1, wall + 1):
                layers[wall - k] = self._outside(node, phi, k, step)
        for i, values in layers.items():
            f = set_at(f, at(i), values)
        return f

    def coast(
        self,
        f: Float[Array, "..."],
        grid: Any,
        axis: str,
        side: int,
        point: str | Stagger,
        layer: int,
    ) -> Float[Array, "..."]:
        """What every point would hold as layer ``layer`` behind a coast.

        The rules of :meth:`fill`, with the wall wherever the water ends
        instead of at the edge of the array.

        Parameters
        ----------
        f : array
            A field whose last axes are ``grid.axes``.
        grid : grid
            Gives the axes and the spacing at the coast.
        axis : str
            The axis normal to the coast.
        side : {0, 1}
            The side of the water the coast is on, low or high.
        point : str or Stagger
            The point of the field.
        layer : int
            The dry points between the point and the water, plus one. Where
            the field has a node on the wall, that is layer 1.

        Returns
        -------
        array
            The shape of ``f``; it means something on the dry points of the
            layer only.
        """
        xp = array_namespace(f)
        pos = grid.axes.index(axis) - len(grid.axes)
        inwards = 1 if side == 0 else -1
        # the axis normal to the coast first, so face values and spacings
        # broadcast along the coast as they do along a face
        lines = xp.moveaxis(f, pos, 0)

        def phi(k: int) -> Any:
            return xp.roll(lines, -inwards * (layer + k - 1), axis=0)

        step = _face_spacing(grid, axis, side, point, f)
        if axis not in point_axes(point, grid.axes):
            values = self._ghost(phi, layer, step)
        else:
            values = self._wall(phi, step)
            if layer > 1:
                values = self._outside(values, phi, layer - 1, step)
        return xp.moveaxis(xp.broadcast_to(values, lines.shape), 0, pos)


def _face_spacing(
    grid: Any, axis: str, side: int, point: str | Stagger, f: Float[Array, "..."]
) -> Float[Array, "..."]:
    """The normal spacing on the faces of ``axis``, one value per face point.

    0-d where it is uniform. A spacing that varies along the face (Δx with
    latitude on a sphere) has ``f``'s size along it: where ``f`` has more or
    fewer rows than the grid, those beyond it repeat the last one.
    """
    wall_step = getattr(grid, "_wall_step", None)
    step: Any = None
    if wall_step is None:
        step = grid.spacing(axis, at=point)
    else:
        step = wall_step(axis, side, point)  # an irregular axis
    if np.ndim(step) == 0:
        return _metric_like(step, f)
    # [Ny, Nx] on the sphere: the same at every longitude, so one column
    line = step[..., grid.ghost]
    xp = array_namespace(line)
    extra = (f.shape[-2] - line.shape[0]) // 2
    if extra > 0:
        line = xp.concat([line[:1]] * extra + [line] + [line[-1:]] * extra)
    elif extra < 0:
        line = line[-extra:extra]
    return _metric_like(line, f)


def _value_like(value: Any, like: Float[Array, "..."]) -> Any:
    """A face value (number, array or tracer) in the dtype of the field."""
    return _metric_like(value, like)


@pytree_dataclass
class Dirichlet(_Atom):
    """A value at the wall.

    Cell-centred: ``φ_{-k} = 2 v - φ_k``. On the wall: ``φ_0 = v``, with odd
    mirrors outside. Zero is a solid wall for the normal velocity.

    Parameters
    ----------
    value : float or array
        The wall value ``v``, broadcast to the face (``[*batch, N_face]``).
        A pytree leaf, so a loss can be differentiated with respect to it.

    Examples
    --------
    >>> import gridax, numpy as np
    >>> grid = gridax.CartesianCGrid2D.from_interior(4, 4, 1.0, 1.0)
    >>> walls = gridax.BC(x=gridax.Dirichlet(0.0))
    >>> u = gridax.fill_ghosts(np.ones(grid.shape), grid, walls, stagger="U")
    >>> u[2, 0], u[2, -2]  # the two wall faces
    (np.float64(0.0), np.float64(0.0))
    """

    value: Any

    def _ghost(self, phi: Interior, k: int, step: Any) -> Any:
        # φ_wall = (φ_{-k} + φ_k) / 2 = v
        return 2.0 * _value_like(self.value, phi(k)) - phi(k)

    def _wall(self, phi: Interior, step: Any) -> Any:
        xp = array_namespace(phi(1))
        return xp.zeros_like(phi(1)) + _value_like(self.value, phi(1))


@pytree_dataclass
class Neumann(_Atom):
    """A gradient ``∂φ/∂n`` along the outward normal.

    Cell-centred: ``φ_{-k} = φ_k + (2k - 1) Δn g``, so the difference across
    the wall is ``g``. On the wall: ``φ_0 = φ_1 + Δn g`` (one-sided) and
    ``φ_{-k} = φ_k + 2k Δn g`` outside.

    Parameters
    ----------
    gradient : float or array
        The gradient ``g``, broadcast to the face. Zero by default.
    """

    gradient: Any = 0.0

    def _ghost(self, phi: Interior, k: int, step: Any) -> Any:
        return phi(k) + (2 * k - 1) * (_value_like(self.gradient, phi(k)) * step)

    def _wall(self, phi: Interior, step: Any) -> Any:
        return phi(1) + _value_like(self.gradient, phi(1)) * step

    def _outside(self, wall: Any, phi: Interior, k: int, step: Any) -> Any:
        if k == 1:
            # 2 φ_0 - φ_1 = φ_1 + 2 Δn g
            return 2.0 * wall - phi(1)
        return phi(k) + 2 * k * (_value_like(self.gradient, phi(k)) * step)


@pytree_dataclass
class Periodic(_Atom):
    """Wrap around: ``φ_{-k} = φ_{N+1-k}``, the node one period away.

    Both faces of an axis must be periodic. The rule is the same at every
    point: ``U[:, 0]`` is the same face as ``U[:, Nx - 2]``.
    """

    def fill(
        self,
        f: Float[Array, "..."],
        grid: Any,
        axis: str,
        side: int,
        point: str | Stagger,
        width: int,
    ) -> Float[Array, "..."]:
        """Write this face's ghost layers with the nodes one period away."""
        point_axes(point, grid.axes)
        pos = grid.axes.index(axis) - len(grid.axes)
        n = f.shape[pos]
        rest = (slice(None),) * (-pos - 1)
        period = n - 2 * width
        ghosts = slice(0, width) if side == 0 else slice(n - width, n)
        shift = period if side == 0 else -period
        source = slice(ghosts.start + shift, ghosts.stop + shift)
        return set_at(f, (..., ghosts, *rest), f[(..., source, *rest)])


@pytree_dataclass
class Outflow(_Atom):
    """Zero gradient by copy: every layer holds the nearest interior node."""

    def _ghost(self, phi: Interior, k: int, step: Any) -> Any:
        return phi(1)

    def _wall(self, phi: Interior, step: Any) -> Any:
        return phi(1)

    def _outside(self, wall: Any, phi: Interior, k: int, step: Any) -> Any:
        return phi(1)


@pytree_dataclass
class Reflective(_Atom):
    """Even symmetry about the wall: ``φ_{-k} = φ_k``."""

    def _ghost(self, phi: Interior, k: int, step: Any) -> Any:
        return phi(k)

    def _wall(self, phi: Interior, step: Any) -> Any:
        return phi(1)

    def _outside(self, wall: Any, phi: Interior, k: int, step: Any) -> Any:
        return phi(k)


@pytree_dataclass
class Sponge(_Atom):
    """Relax the ghost layers towards a background value.

    ``φ_{-k} = (1 - w) φ_k + w b``. It acts on the ghost ring only; an
    interior sponge layer is a tendency of the model.

    Parameters
    ----------
    background : float or array
        The background ``b``, broadcast to the face.
    weight : float or array
        The weight ``w`` in ``[0, 1]``.
    """

    background: Any
    weight: Any

    def __check_init__(self) -> None:
        _check_unit("Sponge weight", self.weight)

    def _blend(self, inner: Any) -> Any:
        w = _value_like(self.weight, inner)
        return (1.0 - w) * inner + w * _value_like(self.background, inner)

    def _ghost(self, phi: Interior, k: int, step: Any) -> Any:
        return self._blend(phi(k))

    def _wall(self, phi: Interior, step: Any) -> Any:
        return self._blend(phi(1))


@pytree_dataclass
class Robin(_Atom):
    """A mixed condition at the wall, ``a φ + b ∂φ/∂n = c``.

    Cell-centred, with ``φ_wall = (φ_{-k} + φ_k) / 2``,
    ``∂φ/∂n = (φ_{-k} - φ_k) / ((2k - 1) Δn)`` and ``s = b / ((2k - 1) Δn)``::

        φ_{-k} = (c - φ_k (a/2 - s)) / (a/2 + s)

    On the wall, with the one-sided gradient and ``s = b / Δn``::

        φ_0 = (c + s φ_1) / (a + s)

    ``b = 0`` is `Dirichlet` with ``c / a`` and ``a = 0`` is `Neumann` with
    ``c / b``. The condition is singular where a denominator is zero.

    Parameters
    ----------
    a, b, c : float or array
        The coefficients, broadcast to the face.
    """

    a: Any
    b: Any
    c: Any

    def __check_init__(self) -> None:
        if all(isinstance(x, int | float) and x == 0.0 for x in (self.a, self.b)):
            raise ValueError("Robin needs a or b to be non-zero")

    def _ghost(self, phi: Interior, k: int, step: Any) -> Any:
        a, b, c = (_value_like(x, phi(k)) for x in (self.a, self.b, self.c))
        s = b / ((2 * k - 1) * step)
        return (c - phi(k) * (a / 2.0 - s)) / (a / 2.0 + s)

    def _wall(self, phi: Interior, step: Any) -> Any:
        a, b, c = (_value_like(x, phi(1)) for x in (self.a, self.b, self.c))
        s = b / step
        return (c + s * phi(1)) / (a + s)


@pytree_dataclass
class Extrapolation(_Atom):
    """Extrapolate the polynomial through the first interior nodes.

    ``φ_{-k} = ∑_m ℓ_m(-k) φ_m`` over ``order + 1`` nodes, with the exact
    Lagrange weights of `gridax.stencils.difference_coeffs`. The axis needs
    that many interior nodes. On the wall the node itself is extrapolated,
    with odd mirrors outside.

    Parameters
    ----------
    order : int
        Degree of the polynomial, 1 to 5. Order 0 is `Outflow`.
    """

    order: int = static_field(default=1)

    def __check_init__(self) -> None:
        if not 1 <= self.order <= 5:
            raise ValueError(f"Extrapolation order must be in [1, 5], got {self.order}")

    def _ghost(self, phi: Interior, k: int, step: Any) -> Any:
        # node m sits m + k - 1 steps inside layer k
        offsets = tuple(range(k, k + self.order + 1))
        weights = difference_coeffs(offsets, 0)
        out = phi(1) * 0.0
        for m, weight in enumerate(weights, start=1):
            out = out + weight * phi(m)
        return out

    def _wall(self, phi: Interior, step: Any) -> Any:
        return self._ghost(phi, 1, step)


@pytree_dataclass
class Slip(_Atom):
    """The tangential velocity at a solid wall, from no-slip to free-slip.

    The wall value is ``a`` times the interior one: cell-centred
    ``φ_{-k} = (2a - 1) φ_k``; on the wall ``φ_0 = a φ_1``.

    Parameters
    ----------
    coefficient : float or array
        ``a`` in ``[0, 1]``: 1 (the default) is free-slip, 0 no-slip.
    """

    coefficient: Any = 1.0

    def __check_init__(self) -> None:
        _check_unit("Slip coefficient", self.coefficient)

    def _ghost(self, phi: Interior, k: int, step: Any) -> Any:
        # φ_wall = (φ_{-k} + φ_k) / 2 = a φ_k
        return (2.0 * _value_like(self.coefficient, phi(k)) - 1.0) * phi(k)

    def _wall(self, phi: Interior, step: Any) -> Any:
        return _value_like(self.coefficient, phi(1)) * phi(1)


@pytree_dataclass
class Wall(_Atom):
    """A solid wall: one condition for the normal component, one for the others.

    The normal component of a vector has its nodes on the wall (U across x, V
    across y, W across z), and the fields that have not take the tangential
    one. ``None`` leaves such a field alone. `BC.wall_normal` and
    `BC.wall_tangential` build it.

    Parameters
    ----------
    normal, tangential : atom, optional
        The conditions.
    """

    normal: Any = None
    tangential: Any = None

    def at(self, axis: str, point: str | Stagger, axes: tuple[str, ...]) -> Any:
        """The condition of the component that a field at ``point`` is."""
        return self.normal if axis in point_axes(point, axes) else self.tangential

    def fill(
        self,
        f: Float[Array, "..."],
        grid: Any,
        axis: str,
        side: int,
        point: str | Stagger,
        width: int,
    ) -> Float[Array, "..."]:
        """Write this face's ghost layers with the condition of the component."""
        atom = self.at(axis, point, grid.axes)
        return f if atom is None else atom.fill(f, grid, axis, side, point, width)


type Atom = (
    Dirichlet
    | Neumann
    | Periodic
    | Outflow
    | Reflective
    | Sponge
    | Robin
    | Extrapolation
    | Slip
    | Wall
)
