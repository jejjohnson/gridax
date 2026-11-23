"""Located fields: the `CField` carrier and element-wise functions on it.

`CField` is an array with named dims and the point of the grid it sits on;
`combine` applies an element-wise function to fields on one point and raises
`StaggerMismatchError` otherwise. The functions below are `combine` with the
array namespace's function of the same name.
"""

from gridax._src.fields.cfield import CField, StaggerMismatchError
from gridax._src.fields.combine import FUNCTIONS, combine

abs = FUNCTIONS["abs"]
clip = FUNCTIONS["clip"]
cos = FUNCTIONS["cos"]
exp = FUNCTIONS["exp"]
log = FUNCTIONS["log"]
maximum = FUNCTIONS["maximum"]
minimum = FUNCTIONS["minimum"]
sign = FUNCTIONS["sign"]
sin = FUNCTIONS["sin"]
sqrt = FUNCTIONS["sqrt"]
tanh = FUNCTIONS["tanh"]
where = FUNCTIONS["where"]

__all__ = [
    "CField",
    "StaggerMismatchError",
    "abs",
    "clip",
    "combine",
    "cos",
    "exp",
    "log",
    "maximum",
    "minimum",
    "sign",
    "sin",
    "sqrt",
    "tanh",
    "where",
]
