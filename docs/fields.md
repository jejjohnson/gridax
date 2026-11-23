# Located fields

On a C-grid a plain array does not record whether it holds T, U, V or X values: all four have the shape `[Ny, Nx]`. Passing a U array where T is meant gives a wrong answer and no error. [`CField`](xref:api#gridax.CField) closes that gap: an array with named dims and the point of the grid it sits on. It is opt-in, and plain arrays stay first-class.

```python
import gridax
import jax.numpy as jnp
from gridax import CField, combine

grid = gridax.CartesianCGrid2D.from_interior(16, 16, 1.0, 1.0)
shape = (3, *grid.shape)  # three ensemble members

# [E, Ny, Nx] at U and at T
u = CField(jnp.ones(shape), dims=("member", "y", "x"), loc="U")
h = CField(jnp.ones(shape), dims=("member", "y", "x"), loc="T")
u.loc, u.sizes  # ("U", {"member": 3, "y": 18, "x": 18})
```

## The stagger

A field's point is its *stagger*: the position of its values along each axis where they are off the cell centre, in xgcm's vocabulary. `loc=` is a shorthand for the five points the operators compute on.

| `loc` | `stagger` |
|---|---|
| `"T"` | `()` |
| `"U"` | `(("x", "right"),)` |
| `"V"` | `(("y", "right"),)` |
| `"X"` | `(("x", "right"), ("y", "right"))` |
| `"W"` | `(("z", "right"),)` |

Other positions (`"left"`, `"outer"`, `"inner"`) describe fields from other models, such as a velocity on the west face. The grid takes a stagger wherever it takes a point: `grid.spacing("x", at=u.stagger)`.

A field holds its array, its dims, the static labels of any unstaggered dim and its stagger, and never the grid. The array is its only pytree leaf, so a compiled program equals the one on bare arrays, and a new point is the only thing that recompiles. Under a bare `jax.vmap` the mapped dim is the first one: it is dropped inside and comes back as `"batch"`.

## Element-wise maths

A field has no arithmetic operators, because `u * h` multiplies values half a cell apart. [`combine`](xref:api#gridax.combine) applies an element-wise function, and its check is that every field sits on the same point.

```python
# 2u − u, at U
v = combine(lambda a, b: 2.0 * a - b, u, u)

combine(jnp.multiply, u, h)  # raises StaggerMismatchError, under jit too
```

Fields are aligned by dim name. A plain array has no point, so it may only broadcast: it must have size 1 along every axis the fields are staggered along, and along at least one grid axis. A scalar, a per-layer column `[nl, 1, 1]` and, next to a U field, a profile `[Ny, 1]` are fine. Wrap a full array with `like`, which asserts the point it sits on:

```python
forcing = u.like(jnp.zeros(u.shape))
combine(jnp.add, u, forcing)
```

`gridax.fields` holds `combine` with the array namespace's functions: `abs`, `clip`, `cos`, `exp`, `log`, `maximum`, `minimum`, `sign`, `sin`, `sqrt`, `tanh` and `where`.
