# Analysis kernels (gridax)

Differences and remapping along one axis of a collocated array, from `gridax.analysis`. They compute in the namespace of their input, so the same call runs on NumPy, on JAX (also under `jax.jit`), or on any backend `array-api-compat` knows, and returns the input's dtype. Their stencil coefficients come from `gridax.stencils` (Layer 0); the gradient, divergence, curl and Laplacian of a collocated field are the [functions of the grid](divergence.md) on a collocated grid (see [the derivative stack](https://jejjohnson.github.io/gridax/stencils/#derivative-stack)).

::: gridax.analysis
