# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sphinx directive that marks experimental API, with the syntax and the notice of Newton's own directive.

A docstring or a guide page writes ``.. experimental::`` where the API it describes may change without the normal
deprecation period. Without a body the directive renders the standard notice. A body names the exact scope.
"""

from __future__ import annotations

from typing import Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import Directive, directives

DEFAULT_NOTICE = (
    "Experimental feature. API, behavior, defaults, and supported use cases may change without prior notice."
)


class ExperimentalDirective(Directive):
    """Render an admonition titled ``Experimental`` around the body, or around the standard notice."""

    has_content = True
    option_spec: ClassVar[dict[str, Any]] = {
        "name": directives.unchanged,
        "class": directives.class_option,
    }

    def run(self) -> list[nodes.Node]:
        node = nodes.admonition(classes=["experimental", *self.options.get("class", [])])
        self.add_name(node)
        node += nodes.title(text="Experimental")
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, node)
        else:
            node += nodes.paragraph(text=DEFAULT_NOTICE)
        return [node]


def setup(app: Any) -> dict[str, bool]:
    """Register the ``experimental`` directive with Sphinx."""
    app.add_directive("experimental", ExperimentalDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
