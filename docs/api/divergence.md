# Divergence Operators

The canonical vector calculus, on any grid: the type of the grid gives the rank, the staggering and the geometry.

::: gridax.divergence

::: gridax.curl

::: gridax.laplacian

::: gridax.gradient

::: gridax.grad_perp

Without a grid object, write `interior(diff_x_bwd_2d(u) / dx + diff_y_bwd_2d(v) / dy, u)`.
