from __future__ import annotations

from collections.abc import Callable
import functools
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from pipekit_array.functions import namespace_of

from gridax._src.backend import stop_gradient
from gridax._src.stencils.coefficients import (
    difference_coeffs,
    optimal_weights,
    smoothness_matrix,
    weno_stencils,
)

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

# Relative WENO floor (#243).  The weights are the textbook
# ``g_k / (beta_k + eps)²`` with ``eps = eps_rel · Σβ``, a floor relative to
# the stencil's own smoothness, so rescaling the field rescales the
# reconstruction exactly.  They are evaluated on the normalised indicators
# ``b_k = beta_k / Σβ`` (O(1), so nothing over- or underflows at any
# amplitude, float32 included), with ``Σβ`` held out of the gradient: the
# weights depend on the ratios ``beta_j / beta_k`` only (homogeneous of
# degree 0), so by Euler's theorem that is exact.  A flat stencil
# (``Σβ = 0``) gives the linear weights.
_WENO_EPS_REL = 1e-6


def _normalised(betas: tuple[Array, ...], *extra: Array) -> list[Array]:
    """``(*betas, *extra) / Σβ`` with ``Σβ`` held out of the gradient.

    Exact (see above); ``extra`` carries WENO-Z's ``tau``, a difference of
    indicators that scales like them.
    """
    xp = namespace_of(*betas)
    total = stop_gradient(sum(betas))
    tiny = xp.finfo(xp.result_type(total)).tiny
    inv = 1.0 / xp.where(total > tiny, total, 1.0)
    return [v * inv for v in (*betas, *extra)]


class _WenoTables(NamedTuple):
    """The linear parts of one WENO scheme, from Layer 0 (``gridax.stencils``)."""

    coeffs: tuple[tuple[float, ...], ...]  # one cell-average stencil per sub-stencil
    betas: tuple[np.ndarray, ...]  # Jiang-Shu matrices, beta_k = q^T B_k q
    weights: tuple[float, ...]  # the optimal (linear) weights g_k


@functools.cache
def _weno_tables(order: int, bias: Literal["left", "right"]) -> _WenoTables:
    """Sub-stencil coefficients, smoothness matrices and optimal weights, exact.

    Built on first use from the Layer 0 solver and cached for the process, so
    ``import gridax`` computes none of them. Every entry is the float nearest
    the exact rational, bitwise equal to the tables typed here before #463.
    """
    full, subs = weno_stencils(order, bias=bias)
    # the reconstructed (upwind) cell: left of the face for left bias
    cell = -0.5 if bias == "left" else 0.5
    return _WenoTables(
        coeffs=tuple(difference_coeffs(sub, 0, data="cell") for sub in subs),
        betas=tuple(np.asarray(smoothness_matrix(sub, cell=cell)) for sub in subs),
        weights=optimal_weights(full, subs),
    )


def _linear_combo(coeffs: tuple[float, ...], stencil: tuple[Array, ...]) -> Any:
    return sum(coeff * value for coeff, value in zip(coeffs, stencil, strict=True))


def _smoothness_indicator(beta_matrix: np.ndarray, stencil: tuple[Array, ...]) -> Any:
    # beta = q^T B q = sum_i q_i * (sum_j B[i, j] * q_j)
    #
    # Same quadratic form as
    #   jnp.einsum("i...,ij,j...->...", jnp.stack(stencil), B, jnp.stack(stencil))
    # but unrolled over the (at most 5x5) constant matrix, so it lowers to one
    # fused elementwise pass instead of a stack, a dot and transposes of the
    # stacked field (WENO-7 advection forward ~11x, WENO-9 ~14x on CPU).  The
    # coefficients are Python floats (weakly typed), so the result keeps the
    # input dtype under float32 and float64.
    return sum(
        q_i * sum(float(beta_matrix[i, j]) * q_j for j, q_j in enumerate(stencil))
        for i, q_i in enumerate(stencil)
    )


def _weno_reconstruct(
    stencils: tuple[tuple[Array, ...], ...],
    coeffs: tuple[tuple[float, ...], ...],
    beta_matrices: tuple[np.ndarray, ...],
    linear_weights: tuple[float, ...],
) -> Array:
    candidates = [
        _linear_combo(stencil_coeffs, stencil)
        for stencil_coeffs, stencil in zip(coeffs, stencils, strict=True)
    ]
    betas = [
        _smoothness_indicator(beta_matrix, stencil)
        for beta_matrix, stencil in zip(beta_matrices, stencils, strict=True)
    ]
    # alpha_k = g_k / (b_k + eps_rel)², b_k = beta_k / Σβ
    alphas = [
        weight / (b + _WENO_EPS_REL) ** 2
        for weight, b in zip(linear_weights, _normalised(tuple(betas)), strict=True)
    ]
    alpha_sum = sum(alphas)
    return (
        sum(
            alpha * candidate
            for alpha, candidate in zip(alphas, candidates, strict=True)
        )
        / alpha_sum
    )


#: The weights of WENO-Z's global indicator ``tau = |sum_k t_k beta_k|``, by
#: the number of sub-stencils.
_TAU: dict[int, tuple[int, ...]] = {
    2: (1, -1),
    3: (1, 0, -1),
    4: (1, 3, -3, -1),
    5: (1, 2, -6, 2, 1),
}


@functools.lru_cache(maxsize=64)
def weno_kernel(
    order: int, bias: Literal["left", "right"], variant: Literal["js", "z"] = "js"
) -> Callable[..., Array]:
    """The WENO face value of any odd order, generated from Layer 0.

    ``r = (order + 1) / 2`` sub-stencils of ``r`` cells, with the face values
    ``q_k`` of their cell averages, the smoothness ``beta_k = q^T B_k q`` and the
    weights ``g_k`` whose combination is the upwind value of order ``2 r - 1``
    (``_weno_tables``). With ``b_k = beta_k / sum beta`` and the floor ``eps``::

        "js"    alpha_k = g_k / (eps + b_k)^2
        "z"     alpha_k = g_k (1 + (tau / (eps + b_k))^2)

    ``tau`` is the modulus of the combination of the ``beta_k`` that vanishes
    to the highest order on smooth data (``_TAU``).

    Parameters
    ----------
    order : int
        Odd, at least 3; at most 9 for ``"z"``.
    bias : {"left", "right"}
        The upwind side.
    variant : {"js", "z"}
        Jiang-Shu or WENO-Z.

    Returns
    -------
    Callable
        ``order`` cell values, in order → the face value.
    """
    r = (order + 1) // 2
    # The matrices annihilate constants, so the reconstructed cell may be taken
    # off, and a large mean costs no digits. The orders that have a hand-written
    # kernel are evaluated as it is, without.
    upwind = r - 1
    centred = variant == "z" or order > 9

    def kernel(*q: Array) -> Array:
        tables = _weno_tables(order, bias)
        xp = namespace_of(*q)
        departures = [value - q[upwind] for value in q] if centred else q
        face, betas = [], []
        for k in range(r):
            face.append(_linear_combo(tables.coeffs[k], q[k : k + r]))
            betas.append(
                _smoothness_indicator(tables.betas[k], tuple(departures[k : k + r]))
            )
        # a sum of squares, up to rounding
        betas = [xp.maximum(beta, xp.zeros_like(beta)) for beta in betas]
        if variant == "js":
            alphas = [
                g / (b + _WENO_EPS_REL) ** 2
                for g, b in zip(tables.weights, _normalised(tuple(betas)), strict=True)
            ]
        else:
            tau = xp.abs(
                sum(t * beta for t, beta in zip(_TAU[r], betas, strict=True) if t)
            )
            *normalised, t = _normalised(tuple(betas), tau)
            alphas = [
                g * (1.0 + (t / (b + _WENO_EPS_REL)) ** 2)
                for g, b in zip(tables.weights, normalised, strict=True)
            ]
        total = sum(alphas)
        return sum(a * v for a, v in zip(alphas, face, strict=True)) / total

    return kernel


def weno_3pts(qm: Array, q0: Array, qp: Array) -> Array:
    """
    3-points non-linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    Efficient Implementation of Weighted ENO Schemes, Jiang and Shu,
    Journal of Computation Physics 126, 202–228 (1996)
    """

    qi1 = -1.0 / 2.0 * qm + 3.0 / 2.0 * q0
    qi2 = 1.0 / 2.0 * (q0 + qp)

    beta1 = (q0 - qm) ** 2
    beta2 = (qp - q0) ** 2

    g1, g2 = 1.0 / 3.0, 2.0 / 3.0
    b1, b2 = _normalised((beta1, beta2))
    w1 = g1 / (b1 + _WENO_EPS_REL) ** 2
    w2 = g2 / (b2 + _WENO_EPS_REL) ** 2

    qi_weno3 = (w1 * qi1 + w2 * qi2) / (w1 + w2)

    return qi_weno3


def weno_3pts_improved(qm: Array, q0: Array, qp: Array) -> Array:
    """
    3-points non-linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008).
    """
    xp = namespace_of(qm, q0, qp)

    qi1 = -1.0 / 2.0 * qm + 3.0 / 2.0 * q0
    qi2 = 1.0 / 2.0 * (q0 + qp)

    beta1 = (q0 - qm) ** 2
    beta2 = (qp - q0) ** 2
    tau = xp.abs(beta2 - beta1)

    g1, g2 = 1.0 / 3.0, 2.0 / 3.0
    b1, b2, t = _normalised((beta1, beta2), tau)
    w1 = g1 * (1.0 + t / (b1 + _WENO_EPS_REL))
    w2 = g2 * (1.0 + t / (b2 + _WENO_EPS_REL))

    qi_weno3 = (w1 * qi1 + w2 * qi2) / (w1 + w2)

    return qi_weno3


def weno_5pts(qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array) -> Array:
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    Efficient Implementation of Weighted ENO Schemes, Jiang and Shu,
    Journal of Computation Physics 126, 202–228 (1996)
    """
    qi1 = 1.0 / 3.0 * qmm - 7.0 / 6.0 * qm + 11.0 / 6.0 * q0
    qi2 = -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp
    qi3 = 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp

    k1, k2 = 13.0 / 12.0, 0.25
    beta1 = k1 * (qmm - 2 * qm + q0) ** 2 + k2 * (qmm - 4 * qm + 3 * q0) ** 2
    beta2 = k1 * (qm - 2 * q0 + qp) ** 2 + k2 * (qm - qp) ** 2
    beta3 = k1 * (q0 - 2 * qp + qpp) ** 2 + k2 * (3 * q0 - 4 * qp + qpp) ** 2

    g1, g2, g3 = 0.1, 0.6, 0.3
    b1, b2, b3 = _normalised((beta1, beta2, beta3))
    w1 = g1 / (b1 + _WENO_EPS_REL) ** 2
    w2 = g2 / (b2 + _WENO_EPS_REL) ** 2
    w3 = g3 / (b3 + _WENO_EPS_REL) ** 2

    qi_weno5 = (w1 * qi1 + w2 * qi2 + w3 * qi3) / (w1 + w2 + w3)

    return qi_weno5


def weno_5pts_improved(
    qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array
) -> Array:
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008)
    """
    xp = namespace_of(qmm, qm, q0, qp, qpp)

    qi1 = 1.0 / 3.0 * qmm - 7.0 / 6.0 * qm + 11.0 / 6.0 * q0
    qi2 = -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp
    qi3 = 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp

    k1, k2 = 13.0 / 12.0, 0.25
    beta1 = k1 * (qmm - 2 * qm + q0) ** 2 + k2 * (qmm - 4 * qm + 3 * q0) ** 2
    beta2 = k1 * (qm - 2 * q0 + qp) ** 2 + k2 * (qm - qp) ** 2
    beta3 = k1 * (q0 - 2 * qp + qpp) ** 2 + k2 * (3 * q0 - 4 * qp + qpp) ** 2

    tau5 = xp.abs(beta1 - beta3)

    g1, g2, g3 = 0.1, 0.6, 0.3
    b1, b2, b3, t = _normalised((beta1, beta2, beta3), tau5)
    w1 = g1 * (1.0 + t / (b1 + _WENO_EPS_REL))
    w2 = g2 * (1.0 + t / (b2 + _WENO_EPS_REL))
    w3 = g3 * (1.0 + t / (b3 + _WENO_EPS_REL))

    qi_weno5 = (w1 * qi1 + w2 * qi2 + w3 * qi3) / (w1 + w2 + w3)

    return qi_weno5


def weno_3pts_right(q0: Array, qp: Array, qpp: Array) -> Array:
    """
    3-points non-linear right-biased stencil reconstruction at face q0+1/2:

    q0--x--qp-----qpp

    Uses sub-stencils {q0, qp} and {qp, qpp} with reversed optimal weights.

    Jiang and Shu, J. Comput. Phys. 126, 202–228 (1996), Section 2.3.
    """

    qi1 = 0.5 * (q0 + qp)
    qi2 = 1.5 * qp - 0.5 * qpp

    beta1 = (qp - q0) ** 2
    beta2 = (qpp - qp) ** 2

    g1, g2 = 2.0 / 3.0, 1.0 / 3.0
    b1, b2 = _normalised((beta1, beta2))
    w1 = g1 / (b1 + _WENO_EPS_REL) ** 2
    w2 = g2 / (b2 + _WENO_EPS_REL) ** 2

    return (w1 * qi1 + w2 * qi2) / (w1 + w2)


def weno_3pts_improved_right(q0: Array, qp: Array, qpp: Array) -> Array:
    """
    3-points non-linear right-biased WENO-Z stencil reconstruction at face q0+1/2:

    q0--x--qp-----qpp

    Borges et al, J. Comput. Phys. 227 (2008).
    """
    xp = namespace_of(q0, qp, qpp)

    qi1 = 0.5 * (q0 + qp)
    qi2 = 1.5 * qp - 0.5 * qpp

    beta1 = (qp - q0) ** 2
    beta2 = (qpp - qp) ** 2
    tau = xp.abs(beta2 - beta1)

    g1, g2 = 2.0 / 3.0, 1.0 / 3.0
    b1, b2, t = _normalised((beta1, beta2), tau)
    w1 = g1 * (1.0 + t / (b1 + _WENO_EPS_REL))
    w2 = g2 * (1.0 + t / (b2 + _WENO_EPS_REL))

    return (w1 * qi1 + w2 * qi2) / (w1 + w2)


def weno_5pts_right(qm: Array, q0: Array, qp: Array, qpp: Array, qppp: Array) -> Array:
    """
    5-points non-linear right-biased stencil reconstruction at face q0+1/2:

    qm-----q0--x--qp----qpp---qppp

    Uses cells {qm, q0, qp, qpp, qppp} to reconstruct the face value from
    the right (upwind for negative velocity).

    Jiang and Shu, J. Comput. Phys. 126, 202–228 (1996), Section 2.3.
    """

    # Candidate reconstructions at face q0+1/2
    qi1 = 1.0 / 3.0 * qppp - 7.0 / 6.0 * qpp + 11.0 / 6.0 * qp
    qi2 = -1.0 / 6.0 * qpp + 5.0 / 6.0 * qp + 1.0 / 3.0 * q0
    qi3 = 1.0 / 3.0 * qp + 5.0 / 6.0 * q0 - 1.0 / 6.0 * qm

    # Smoothness indicators
    k1, k2 = 13.0 / 12.0, 0.25
    beta1 = k1 * (qp - 2 * qpp + qppp) ** 2 + k2 * (3 * qp - 4 * qpp + qppp) ** 2
    beta2 = k1 * (q0 - 2 * qp + qpp) ** 2 + k2 * (q0 - qpp) ** 2
    beta3 = k1 * (qm - 2 * q0 + qp) ** 2 + k2 * (qm - 4 * q0 + 3 * qp) ** 2

    g1, g2, g3 = 0.1, 0.6, 0.3
    b1, b2, b3 = _normalised((beta1, beta2, beta3))
    w1 = g1 / (b1 + _WENO_EPS_REL) ** 2
    w2 = g2 / (b2 + _WENO_EPS_REL) ** 2
    w3 = g3 / (b3 + _WENO_EPS_REL) ** 2

    return (w1 * qi1 + w2 * qi2 + w3 * qi3) / (w1 + w2 + w3)


def weno_5pts_improved_right(
    qm: Array, q0: Array, qp: Array, qpp: Array, qppp: Array
) -> Array:
    """
    5-points non-linear right-biased WENO-Z stencil reconstruction at face q0+1/2:

    qm-----q0--x--qp----qpp---qppp

    Borges et al, J. Comput. Phys. 227 (2008).
    """
    xp = namespace_of(qm, q0, qp, qpp, qppp)

    qi1 = 1.0 / 3.0 * qppp - 7.0 / 6.0 * qpp + 11.0 / 6.0 * qp
    qi2 = -1.0 / 6.0 * qpp + 5.0 / 6.0 * qp + 1.0 / 3.0 * q0
    qi3 = 1.0 / 3.0 * qp + 5.0 / 6.0 * q0 - 1.0 / 6.0 * qm

    k1, k2 = 13.0 / 12.0, 0.25
    beta1 = k1 * (qp - 2 * qpp + qppp) ** 2 + k2 * (3 * qp - 4 * qpp + qppp) ** 2
    beta2 = k1 * (q0 - 2 * qp + qpp) ** 2 + k2 * (q0 - qpp) ** 2
    beta3 = k1 * (qm - 2 * q0 + qp) ** 2 + k2 * (qm - 4 * q0 + 3 * qp) ** 2

    tau5 = xp.abs(beta1 - beta3)

    g1, g2, g3 = 0.1, 0.6, 0.3
    b1, b2, b3, t = _normalised((beta1, beta2, beta3), tau5)
    w1 = g1 * (1.0 + t / (b1 + _WENO_EPS_REL))
    w2 = g2 * (1.0 + t / (b2 + _WENO_EPS_REL))
    w3 = g3 * (1.0 + t / (b3 + _WENO_EPS_REL))

    return (w1 * qi1 + w2 * qi2 + w3 * qi3) / (w1 + w2 + w3)


def weno_7pts(
    qmmm: Array,
    qmm: Array,
    qm: Array,
    q0: Array,
    qp: Array,
    qpp: Array,
    qppp: Array,
) -> Array:
    """
    7-points non-linear left-biased stencil reconstruction

    qmmm---qmm----qm-----q0--x--qp----qpp---qppp

    Efficient finite-volume WENO-7 reconstruction using Jiang-Shu
    smoothness indicators.
    """
    stencils = (
        (qmmm, qmm, qm, q0),
        (qmm, qm, q0, qp),
        (qm, q0, qp, qpp),
        (q0, qp, qpp, qppp),
    )
    return _weno_reconstruct(stencils, *_weno_tables(7, "left"))


def weno_9pts(
    qmmmm: Array,
    qmmm: Array,
    qmm: Array,
    qm: Array,
    q0: Array,
    qp: Array,
    qpp: Array,
    qppp: Array,
    qpppp: Array,
) -> Array:
    """
    9-points non-linear left-biased stencil reconstruction

    qmmmm--qmmm---qmm----qm-----q0--x--qp----qpp---qppp--qpppp

    Efficient finite-volume WENO-9 reconstruction using Jiang-Shu
    smoothness indicators.
    """
    stencils = (
        (qmmmm, qmmm, qmm, qm, q0),
        (qmmm, qmm, qm, q0, qp),
        (qmm, qm, q0, qp, qpp),
        (qm, q0, qp, qpp, qppp),
        (q0, qp, qpp, qppp, qpppp),
    )
    return _weno_reconstruct(stencils, *_weno_tables(9, "left"))


def weno_7pts_right(
    qmm: Array,
    qm: Array,
    q0: Array,
    qp: Array,
    qpp: Array,
    qppp: Array,
    qpppp: Array,
) -> Array:
    """
    7-points non-linear right-biased stencil reconstruction at face q0+1/2:

    qmm----qm-----q0--x--qp----qpp---qppp--qpppp

    Uses cells {q_{i-2}..q_{i+4}} for right-biased (negative-flow) WENO-7
    reconstruction.  Sub-stencil coefficients, optimal weights, and smoothness
    indicators are the mirror of the left-biased formulas.

    Jiang and Shu, J. Comput. Phys. 126, 202–228 (1996), Section 2.3.
    """
    stencils = (
        (qmm, qm, q0, qp),
        (qm, q0, qp, qpp),
        (q0, qp, qpp, qppp),
        (qp, qpp, qppp, qpppp),
    )
    return _weno_reconstruct(stencils, *_weno_tables(7, "right"))


def weno_9pts_right(
    qmmm: Array,
    qmm: Array,
    qm: Array,
    q0: Array,
    qp: Array,
    qpp: Array,
    qppp: Array,
    qpppp: Array,
    qppppp: Array,
) -> Array:
    """
    9-points non-linear right-biased stencil reconstruction at face q0+1/2:

    qmmm---qmm----qm-----q0--x--qp----qpp---qppp--qpppp-qppppp

    Uses cells {q_{i-3}..q_{i+5}} for right-biased (negative-flow) WENO-9
    reconstruction.  Sub-stencil coefficients, optimal weights, and smoothness
    indicators are the mirror of the left-biased formulas.

    Jiang and Shu, J. Comput. Phys. 126, 202–228 (1996), Section 2.3.
    """
    stencils = (
        (qmmm, qmm, qm, q0, qp),
        (qmm, qm, q0, qp, qpp),
        (qm, q0, qp, qpp, qppp),
        (q0, qp, qpp, qppp, qpppp),
        (qp, qpp, qppp, qpppp, qppppp),
    )
    return _weno_reconstruct(stencils, *_weno_tables(9, "right"))
