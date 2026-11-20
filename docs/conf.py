# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for the ``newton-neural`` documentation.

Regenerates ``docs/api/*.rst`` before every build via :func:`generate_api.generate_all` and runs guide code blocks
as isolated doctests (each group in a temporary directory).
"""

import datetime
import inspect
import os
import pkgutil
import sys
import tomllib
from pathlib import Path
from typing import Any

DOCS = Path(__file__).parent
REPO_ROOT = DOCS.parent

# Import generate_api from docs/ and the directives of docs/_ext.
sys.path.append(str(DOCS))
sys.path.append(str(DOCS / "_ext"))

from generate_api import generate_all  # noqa: E402

# Branch/tag for GitHub source links (GitHub Actions sets GITHUB_REF_NAME; local builds link to main).
github_version = os.environ.get("GITHUB_REF_NAME", "main")
github_url = "https://github.com/newton-physics/newton-neural"

# -- Project information -----------------------------------------------------

project = "newton-neural"
author = "The Newton Developers"
copyright = f"{datetime.date.today().year}, {author}. Documentation licensed under CC-BY-4.0"
with open(REPO_ROOT / "pyproject.toml", "rb") as file:
    release = tomllib.load(file)["project"]["version"]

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",  # runs the code of the guide pages
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",  # links every documented object to its source on GitHub
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_design",
    "experimental",  # marks API that may change without prior notice
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    # Included from index.rst, not a page of its own.
    "api/_toctree.rst",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "warp": ("https://nvidia.github.io/warp/stable", None),
    "newton": ("https://newton-physics.github.io/newton/stable", None),
}

extlinks = {
    "github": (f"{github_url}/blob/{github_version}/%s", "%s"),
}

# -- Tested code -------------------------------------------------------------

# Run each doctest group in its own temporary directory; underscore-prefixed names force explicit imports in guide code.
doctest_global_setup = """
import os as _os
import tempfile as _tempfile

import warp as _wp

_wp.config.log_level = _wp.LOG_WARNING  # Suppress Warp banner/kernel output during doctest comparisons
_start_folder = _os.getcwd()
_scratch = _tempfile.TemporaryDirectory()
_os.chdir(_scratch.name)
"""
doctest_global_cleanup = """
_os.chdir(_start_folder)
_scratch.cleanup()
"""

# -- Autodoc -----------------------------------------------------------------

# Render type hints inside parameter descriptions rather than cluttering signatures.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
# Preserve public type alias names in signatures so they cross-reference cleanly.
autodoc_type_aliases = {"Frames": "Frames"}
autodoc_preserve_defaults = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    # Exclude import-time subclass registries from API docs.
    "exclude-members": "registry",
}
toc_object_entries_show_parents = "hide"
# Omit attribute type annotations that expose private _src module paths.
napoleon_attr_annotations = False

autosummary_generate = True

# -- HTML output -------------------------------------------------------------

html_title = "newton-neural"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_sidebars = {"**": ["sidebar-nav-bs.html"]}
html_theme_options = {
    "logo": {
        "image_light": "_static/newton-neural-logo-light.svg",
        "image_dark": "_static/newton-neural-logo-dark.svg",
        "text": f"newton-neural <span style='font-size: 0.8em; color: #888;'>({release})</span>",
        "alt_text": "newton-neural logo",
    },
    "icon_links": [{"name": "GitHub", "url": github_url, "icon": "fa-brands fa-github"}],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navigation_depth": 1,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "collapse_navigation": False,
    "show_prev_next": False,
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
}

# -- Source links ------------------------------------------------------------


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Resolve GitHub source URLs for ``sphinx.ext.linkcode``."""
    if domain != "py" or not info["module"].startswith("newton_neural"):
        return None
    try:
        obj = pkgutil.resolve_name(f"{info['module']}:{info['fullname']}")
        obj = inspect.unwrap(obj.fget if isinstance(obj, property) else obj)
        path = Path(inspect.getsourcefile(obj)).relative_to(REPO_ROOT)
        line = inspect.getsourcelines(obj)[1]
    except (AttributeError, TypeError, OSError, ValueError):
        # Skip attributes without standalone source definitions and external classes.
        return None
    return f"{github_url}/blob/{github_version}/{path.as_posix()}#L{line}"


def link_type_alias(app: Any, env: Any, node: Any, contnode: Any) -> Any:
    """Resolve missing ``py:class`` references against ``py:type`` aliases (e.g., ``Frames``)."""
    if node.get("refdomain") != "py" or node.get("reftype") != "class":
        return None
    return env.get_domain("py").resolve_xref(
        env, node["refdoc"], app.builder, "type", node["reftarget"], node, contnode
    )


def setup(app: Any) -> None:
    """Regenerate ``docs/api`` before every Sphinx build and hook type-alias resolution."""
    generate_all()
    app.connect("missing-reference", link_type_alias)
