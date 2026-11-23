# Located fields

An array with named dims and the point of the grid it sits on, and element-wise maths that checks the point. See the [guide](../fields/).

::: gridax.CField

::: gridax.combine

::: gridax.StaggerMismatchError

## Conversions

`CField.to_xarray` and `CField.from_xarray` above cross to xarray.

::: gridax.to_right

::: gridax.to_dimarray

## Element-wise functions

`gridax.fields` holds `combine` with the array namespace's function of the same name: `abs`, `clip`, `cos`, `exp`, `log`, `maximum`, `minimum`, `sign`, `sin`, `sqrt`, `tanh` and `where`. Keyword arguments pass through.
