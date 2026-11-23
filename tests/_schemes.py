"""The schemes of the parametrised advection tests, by the names of their cases."""

from __future__ import annotations

from gridax.schemes import TVD, WENO, Centred, Scheme, Upwind

SCHEMES: dict[str, Scheme] = {
    "naive": Centred(),
    "upwind1": Upwind(1),
    "upwind2": Upwind(2),
    "upwind3": Upwind(3),
    "weno3": WENO(3),
    "weno5": WENO(5),
    "weno7": WENO(7),
    "weno9": WENO(9),
    "wenoz3": WENO(3, "z"),
    "wenoz5": WENO(5, "z"),
    "minmod": TVD("minmod"),
    "van_leer": TVD("van_leer"),
    "superbee": TVD("superbee"),
    "mc": TVD("mc"),
}
