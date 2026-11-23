"""Every public name has an API page, and its annotations resolve (#326, #328)."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import re
import typing
from typing import Any

import pytest

MODULES = (
    "gridax",
    "gridax.stencils",
    "gridax.schemes",
    "gridax.fields",
    "gridax.analysis",
    "gridax.advection",
    "gridax.advection.kernels",
    "gridax.solvers",
    "gridax.pipekit",
)
PAGES = sorted((Path(__file__).resolve().parents[1] / "docs" / "api").glob("*.md"))


def public(module: str) -> list[tuple[str, Any]]:
    """The functions and classes of gridax that a module exports."""
    owner = importlib.import_module(module)
    objects = [(name, getattr(owner, name)) for name in owner.__all__]
    return [
        (name, obj)
        for name, obj in objects
        if (inspect.isfunction(obj) or inspect.isclass(obj))
        # a re-export of spectraldiffx or gaussx is documented there
        and obj.__module__.startswith("gridax")
    ]


def documented() -> tuple[set[int], set[str]]:
    """The objects with a directive, by identity, and the modules with one."""
    objects, modules = set(), set()
    for page in PAGES:
        for path in re.findall(r"^::: ([\w.]+)", page.read_text(), re.MULTILINE):
            owner, _, name = path.rpartition(".")
            if path in MODULES:
                modules.add(path)
                continue
            # a directive names something that exists
            objects.add(id(getattr(importlib.import_module(owner), name)))
    return objects, modules


@pytest.mark.parametrize("module", MODULES)
def test_every_name_has_an_api_page(module: str) -> None:
    objects, modules = documented()
    if module in modules:
        return
    missing = [name for name, obj in public(module) if id(obj) not in objects]
    assert not missing


@pytest.mark.parametrize("module", MODULES)
def test_the_annotations_resolve(module: str) -> None:
    for name, obj in public(module):
        if inspect.isfunction(obj):
            assert typing.get_type_hints(obj) is not None, name


def test_the_pages_are_in_the_navigation() -> None:
    nav = (PAGES[0].parents[2] / "mkdocs.yml").read_text()
    assert not [page.name for page in PAGES if f" {page.name}\n" not in nav]
