# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate ``docs/api/*.rst`` and ``docs/api/_toctree.rst`` from public module ``__all__`` exports.

Module pages are committed to git so CI can detect stale API docs; per-symbol stubs land in git-ignored
``docs/api/_generated/``. Run ``uv run --group docs python docs/generate_api.py`` to refresh standalone.
"""

from __future__ import annotations

import importlib
import inspect
import shutil
import sys
from pathlib import Path
from types import GenericAlias, ModuleType

from sphinx.pycode import ModuleAnalyzer
from sphinx.util.typing import stringify_annotation

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "api"

PACKAGE = "newton_neural"

TOCTREE_DIR = "_generated"
"""Output directory for autosummary per-symbol stub pages, relative to ``docs/api``."""

HEADER = [
    ".. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers",
    ".. SPDX-License-Identifier: CC-BY-4.0",
    "",
]


def public_symbols(module: ModuleType) -> list[str]:
    """Return ``module.__all__`` or raise if missing/invalid."""
    names = getattr(module, "__all__", None)
    if not isinstance(names, list | tuple) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{module.__name__} must define __all__ as a list or tuple of names")
    return list(names)


def public_modules() -> dict[str, ModuleType]:
    """Recursively collect ``newton_neural`` and all submodules exported via ``__all__``."""
    modules: dict[str, ModuleType] = {}

    def add(name: str, module: ModuleType) -> None:
        modules[name] = module
        for symbol in public_symbols(module):
            child = getattr(module, symbol)
            if inspect.ismodule(child) and child.__name__.startswith(f"{PACKAGE}."):
                add(f"{name}.{symbol}", child)

    add(PACKAGE, importlib.import_module(PACKAGE))
    return modules


def page_name(module_name: str) -> str:
    """Convert dotted module name to flat ``.rst`` stem."""
    return module_name.replace(".", "_")


def alias_description(symbol: str, alias: GenericAlias) -> list[str]:
    """Extract the attribute docstring following a module-level ``GenericAlias`` assignment."""
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith(f"{PACKAGE}.") and vars(module).get(symbol) is alias:
            description = ModuleAnalyzer.for_module(module_name).find_attr_docs().get(("", symbol))
            if description:
                return description
    raise ValueError(f"the type alias {symbol} needs a docstring after its assignment")


def write_module_page(name: str, module: ModuleType) -> None:
    """Generate the ``docs/api/<page>.rst`` reference page for one public module."""
    modules, classes, functions, aliases, constants = [], [], [], [], []
    for symbol in sorted(public_symbols(module)):
        value = getattr(module, symbol)
        if inspect.ismodule(value):
            modules.append(symbol)
        elif isinstance(value, GenericAlias):  # Check before isclass/callable (GenericAlias matches both)
            aliases.append(symbol)
        elif inspect.isclass(value):
            classes.append(symbol)
        elif callable(value):
            functions.append(symbol)
        else:
            constants.append(symbol)

    lines = [*HEADER, name, "=" * len(name), ""]
    if module.__doc__:
        lines += [inspect.cleandoc(module.__doc__), ""]
    lines += [f".. py:module:: {name}", f".. currentmodule:: {name}", ""]
    if modules:
        lines += [".. rubric:: Submodules", ""]
        lines += [f"- :doc:`{name}.{symbol} <{page_name(f'{name}.{symbol}')}>`" for symbol in modules]
        lines.append("")
    if classes:
        # Omit signatures in summary tables since each class page documents its constructor.
        lines += [".. rubric:: Classes", "", ".. autosummary::", f"   :toctree: {TOCTREE_DIR}", "   :nosignatures:", ""]
        lines += [f"   {symbol}" for symbol in classes]
        lines.append("")
    if functions:
        lines += [".. rubric:: Functions", "", ".. autosummary::", f"   :toctree: {TOCTREE_DIR}", ""]
        lines += [f"   {symbol}" for symbol in functions]
        lines.append("")
    if aliases:
        lines += [".. rubric:: Type aliases", ""]
        for symbol in aliases:
            alias = getattr(module, symbol)
            canonical = stringify_annotation(alias, "fully-qualified-except-typing")
            lines += [f".. py:type:: {symbol}", f"   :canonical: {canonical}", ""]
            lines += [f"   {line}".rstrip() for line in alias_description(symbol, alias)]
            lines.append("")
    if constants:
        lines += [".. rubric:: Constants", "", ".. list-table::", "   :header-rows: 1", ""]
        lines += ["   * - Name", "     - Value"]
        for symbol in constants:
            lines += [f"   * - ``{symbol}``", f"     - ``{getattr(module, symbol)}``"]
        lines.append("")

    (OUTPUT_DIR / f"{page_name(name)}.rst").write_text("\n".join(lines), encoding="utf-8")


def write_toctree(module_names: list[str]) -> None:
    """Write ``docs/api/_toctree.rst`` included by ``docs/index.rst``."""
    lines = [*HEADER, ".. toctree::", "   :maxdepth: 1", "   :hidden:", "   :caption: API Reference", ""]
    lines += [f"   api/{page_name(name)}" for name in module_names]
    lines.append("")
    (OUTPUT_DIR / "_toctree.rst").write_text("\n".join(lines), encoding="utf-8")


def generate_all() -> None:
    """Regenerate all ``docs/api/*.rst`` files cleanly from current ``__all__`` exports."""
    modules = public_modules()
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)
    for name, module in modules.items():
        write_module_page(name, module)
    write_toctree(list(modules))


if __name__ == "__main__":
    generate_all()
