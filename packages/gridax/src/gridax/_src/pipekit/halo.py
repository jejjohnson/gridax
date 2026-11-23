"""How far a graph of operators reads: the halo of a patch."""

from __future__ import annotations

from typing import Any

from pipekit_ops import Graph, Operator, Sequential

__all__ = ["graph_halo"]


def graph_halo(graph: Operator) -> int:
    """The halo of a graph: how many cells around a point its result reads.

    ``H(path) = ∑ₖ Hₖ`` along a path from an input to an output, and the
    halo of the graph is the largest of them. A patch cut with that many cells
    on every side (the ghost ring included) gives the values the whole domain
    gives, on the patch without them.

    Parameters
    ----------
    graph : Operator
        A ``Graph``, a ``Sequential``, or any operator with a ``halo``; they
        may be nested.

    Returns
    -------
    int

    Raises
    ------
    TypeError
        If an operator has no ``halo``: a reduction reads the whole domain, and
        nothing is known of an operator from elsewhere.

    Examples
    --------
    >>> from gridax.pipekit import Difference, Interp, graph_halo
    >>> from gridax.schemes import Centred
    >>> G = (8, 8, 1.0, 1.0)
    >>> to_u = Interp(grid=G, src="T", dst="U", scheme=Centred(4))
    >>> back = Difference(grid=G, axis="x", src="U", dst="T", scheme=Centred(4))
    >>> graph_halo(to_u | back)
    4
    """
    if isinstance(graph, Sequential):
        return sum(graph_halo(op) for op in graph.operators)
    if isinstance(graph, Graph):
        depth: dict[int, int] = {id(node): 0 for node in graph.inputs.values()}
        for node in graph._order:
            before = max((depth[id(p)] for p in node.parents), default=0)
            depth[id(node)] = before + graph_halo(_operator(node))
        return max((depth[id(node)] for node in graph.outputs.values()), default=0)
    halo = getattr(graph, "halo", None)
    if not isinstance(halo, int):
        raise TypeError(
            f"{type(graph).__name__} has no halo: it reads the whole domain, or is not"
            " an operator of gridax"
        )
    return halo


def _operator(node: Any) -> Operator:
    if node.operator is None:
        raise TypeError("a graph node without an operator")
    return node.operator
