"""The array namespace of mixed inputs, with and without ``array-api-compat``."""

from __future__ import annotations

import sys

from gridax._src.pytree import array_namespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(params=["with compat", "without compat"], autouse=True)
def compat(request, monkeypatch):
    if request.param == "without compat":
        monkeypatch.setitem(sys.modules, "array_api_compat", None)
    else:
        pytest.importorskip("array_api_compat")


@pytest.mark.parametrize("host", [np.ones(2), np.float64(1.0), 1.0, None, [1.0], "x"])
def test_host_values_join_jax_and_resolve_to_numpy_alone(host):
    assert array_namespace(host) is np
    assert array_namespace(host, jnp.ones(2)) is jnp
    assert array_namespace(jnp.ones(2), host) is jnp


def test_a_numpy_array_joins_a_tracer():
    seen = []

    @jax.jit
    def probe(x):
        seen.append(array_namespace(np.ones(2), x))
        return x

    probe(jnp.ones(2))
    assert seen == [jnp]


def test_no_values_resolve_to_numpy():
    assert array_namespace() is np
