"""Register `CFieldAdapter` once ``pipekit_ops`` is imported, without importing it.

``import gridax`` never imports pipekit, but a `CField` has to be a
registered carrier from the moment pipekit is loaded, whichever of the two
comes first. (``pipekit_ops.on_import`` does this for libraries that depend
on pipekit; gridax only has it as an extra.)
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

_WAITING_FOR = "pipekit_ops"
_ADAPTER = "gridax._src.fields.adapter"


class _Loader(importlib.abc.Loader):
    """The module's own loader, then the adapter's registration."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Any:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        importlib.import_module(_ADAPTER)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _Finder(importlib.abc.MetaPathFinder):
    """Finds nothing itself: wraps the loader of ``pipekit_ops``, once."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != _WAITING_FOR:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def register_when_pipekit_loads() -> None:
    """Load the adapter now if ``pipekit_ops`` is imported, else once it is."""
    if _WAITING_FOR in sys.modules:
        importlib.import_module(_ADAPTER)
    elif not any(type(finder).__module__ == __name__ for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
