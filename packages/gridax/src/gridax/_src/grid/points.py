"""The points of the Arakawa C-grid, as the axes each one is staggered along."""

from __future__ import annotations

#: ``point -> axes`` it sits half a cell along (the ``+`` face of its T cell).
POINT_AXES: dict[str, tuple[str, ...]] = {
    "T": (),
    "U": ("x",),
    "V": ("y",),
    "X": ("x", "y"),
    "W": ("z",),
}


def point_axes(at: str, axes: tuple[str, ...]) -> tuple[str, ...]:
    """The axes the point ``at`` is staggered along, on a grid with ``axes``.

    Parameters
    ----------
    at : str
        ``"T"``, ``"U"``, ``"V"``, ``"X"`` or ``"W"``.
    axes : tuple of str
        The grid's axis names.

    Returns
    -------
    tuple of str
        The staggered axes, empty at T.

    Raises
    ------
    ValueError
        If ``at`` is not a point of a grid with those axes.

    Examples
    --------
    >>> point_axes("X", ("y", "x"))
    ('x', 'y')
    """
    staggered = POINT_AXES.get(at)
    if staggered is None or not set(staggered) <= set(axes):
        valid = [p for p, along in POINT_AXES.items() if set(along) <= set(axes)]
        raise ValueError(f"at={at!r} is not one of {valid} on a grid with axes {axes}")
    return staggered
