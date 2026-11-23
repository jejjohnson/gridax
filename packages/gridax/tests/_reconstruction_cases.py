"""Every reconstruction call of gridax, on fixed random data (#464).

``freeze()`` records what each call returns; ``test_reconstruct.py`` checks the
axis-generic rewrite against the record, bit for bit. Rerun this module as a
script only to re-freeze on purpose:

    uv run python packages/gridax/tests/_reconstruction_cases.py
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import itertools
from pathlib import Path

import gridax
from gridax.schemes import TVD, WENO, Centred, Scheme, Upwind
import numpy as np

FROZEN = Path(__file__).parent / "data" / "reconstruction_frozen.npz"
LIMITERS = ("minmod", "van_leer", "superbee", "mc")


def _fields(shape: tuple[int, ...], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape), rng.standard_normal(shape)


def _mask_2d(ny: int, nx: int) -> gridax.Mask2D:
    wet = np.ones((ny, nx), dtype=bool)
    wet[0, :] = wet[-1, :] = wet[:, 0] = wet[:, -1] = False
    wet[5:8, 6:10] = False  # an island
    wet[1:3, nx - 5 : nx - 2] = False  # a coastal notch
    return gridax.Mask2D.from_mask(wet)


def flux(q, u, *, axis: int, scheme, mask=None):
    """`gridax.reconstruct` on a grid of the arrays' shape, the axis by its index."""
    sizes = [n - 2 for n in reversed(q.shape)]
    cls = getattr(gridax, f"CartesianCGrid{q.ndim}D")
    grid = cls.from_interior(*sizes, *([1.0] * q.ndim))
    name = grid.axes[axis % q.ndim]
    return gridax.reconstruct(q, u, grid, axis=name, scheme=scheme, mask=mask)


#: ``name -> scheme`` of the methods the Reconstruction classes had.
METHODS: dict[str, Scheme] = {
    "naive": Centred(),
    "upwind1": Upwind(1),
    "upwind2": Upwind(2),
    "upwind3": Upwind(3),
    "weno3": WENO(3),
    "wenoz3": WENO(3, "z"),
    "weno5": WENO(5),
    "wenoz5": WENO(5, "z"),
    "weno7": WENO(7),
    "weno9": WENO(9),
    **{f"tvd.{limiter}": TVD(limiter) for limiter in LIMITERS},
}


def cases() -> Iterator[tuple[str, Callable[[], np.ndarray]]]:
    """``(name, call)`` of every recorded call, named for the method it was.

    The records of the masked 3-D methods are left out: those took a 2-D mask,
    and `gridax.reconstruct` takes a mask of the grid's rank.
    """
    shapes = {1: (20,), 2: (14, 18), 3: (6, 12, 14)}
    for ndim, shape in shapes.items():
        h, u = _fields(shape, ndim)
        mask = _mask_2d(14, 18) if ndim == 2 else None
        for (method, scheme), k in itertools.product(
            METHODS.items(), range(min(ndim, 2))
        ):
            family, _, limiter = method.partition(".")
            end = f".{limiter}" if limiter else ""
            for suffix, given in (("", None), ("_masked", mask)):
                if suffix and mask is None:
                    continue
                yield (
                    f"{ndim}d.{family}_{'xy'[k]}{suffix}{end}",
                    lambda h=h, u=u, axis=-1 - k, scheme=scheme, given=given: flux(
                        h, u, axis=axis, scheme=scheme, mask=given
                    ),
                )
    h, u = _fields((14, 18), 9)
    for dim, (method, variant) in itertools.product(
        (0, 1), {"weno": "js", "wenoz": "z", "linear": None}.items()
    ):
        for num_pts in (1, 3, 5):
            scheme = (
                Upwind(num_pts)
                if variant is None or num_pts == 1
                else WENO(num_pts, variant)
            )
            # named for the old call reconstruct(q, u, dim, method, num_pts)
            yield (
                f"reconstruct.dim{dim}.{method}.{num_pts}",
                lambda axis=dim - 2, scheme=scheme, h=h, u=u: flux(
                    h, u, axis=axis, scheme=scheme
                ),
            )


def freeze() -> None:
    np.savez(FROZEN, **{name: np.asarray(call()) for name, call in cases()})


if __name__ == "__main__":
    freeze()
    print(f"froze {len(np.load(FROZEN).files)} calls into {FROZEN}")
