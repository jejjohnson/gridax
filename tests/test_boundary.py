"""Tests for the standalone ghost-ring fillers (the atoms: ``test_fill_ghosts.py``)."""

from gridax._src.boundary.boundary import enforce_periodic
import jax.numpy as jnp
import numpy as np


class TestEnforcePeriodic:
    def test_output_shape(self):
        f = jnp.ones((6, 6))
        result = enforce_periodic(f)
        assert result.shape == (6, 6)

    def test_south_ghost_equals_north_interior(self):
        # row 0 should equal row Ny-2
        f = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[0, :], result[-2, :])

    def test_north_ghost_equals_south_interior(self):
        f = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[-1, :], result[1, :])

    def test_west_ghost_equals_east_interior(self):
        f = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[:, 0], result[:, -2])

    def test_east_ghost_equals_west_interior(self):
        f = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[:, -1], result[:, 1])

    def test_constant_field_stays_constant(self):
        f = 5.0 * jnp.ones((8, 8))
        result = enforce_periodic(f)
        np.testing.assert_allclose(result, 5.0)
