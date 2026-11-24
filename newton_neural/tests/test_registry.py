# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``Registered`` and ``Part`` class-name registries, collision checks, and reload semantics."""

from __future__ import annotations

import unittest
from typing import Any, ClassVar

from newton_neural._src.core.registry import Part


class _PartKind(Part):
    """A kind of part of its own, so that the tests leave the registries of the library alone."""

    registry: ClassVar[dict[str, type]] = {}

    def config(self) -> dict[str, Any]:
        return {}


def _define(name: str, module: str) -> type[_PartKind]:
    """A part of the kind named ``name``, as the top level of the module ``module`` would define it."""
    return type(name, (_PartKind,), {"__module__": module})


class TestRegistry(unittest.TestCase):
    def test_base_class_is_no_entry(self) -> None:
        """Expect the base class of a kind to own the registry without being an entry of it."""
        self.assertNotIn("_PartKind", _PartKind.registry)

    def test_unknown_name_lists_the_registered(self) -> None:
        """Ask for a name that no class registered and expect a ValueError that lists the names that are."""
        named = _define("PartNamed", "package.parts")
        self.assertIs(_PartKind.registered("PartNamed"), named)
        with self.assertRaisesRegex(ValueError, "unknown _PartKind 'PartMissing'.*PartNamed.*Import the module"):
            _PartKind.registered("PartMissing")
        with self.assertRaisesRegex(ValueError, "unknown _PartKind 'PartMissing'"):
            _PartKind.create("PartMissing", {})

    def test_part_must_describe_itself(self) -> None:
        """Leave the config out of a part and expect that it cannot be built, since no file could name it."""

        class PartUndescribed(_PartKind):
            config = Part.config

        with self.assertRaisesRegex(TypeError, "abstract.*PartUndescribed.*config"):
            PartUndescribed()

    def test_definition_at_the_same_place_replaces(self) -> None:
        """Define a part twice at one place, as a notebook cell that runs again does, and get the later class."""
        classes = []
        for _ in range(2):

            class PartCell(_PartKind):
                pass

            classes.append(PartCell)
        self.assertIsNot(classes[0], classes[1])
        self.assertIs(_PartKind.registry["PartCell"], classes[1])
        self.assertIs(type(_PartKind.create("PartCell", {})), classes[1])

        first, second = _define("PartReloaded", "package.parts"), _define("PartReloaded", "package.parts")
        self.assertIsNot(first, second)
        self.assertIs(_PartKind.registry["PartReloaded"], second)

    def test_same_name_from_another_place_is_refused(self) -> None:
        """Define a part under a name that another module, or another scope, has taken and expect a TypeError."""
        first = _define("PartShared", "package.parts")
        with self.assertRaisesRegex(TypeError, r"'PartShared' is already taken by package\.parts\.PartShared"):
            _define("PartShared", "other_package.parts")
        with self.assertRaisesRegex(TypeError, "already taken"):

            class PartShared(_PartKind):  # the qualified name of a class in a function differs
                pass

        self.assertIs(_PartKind.registry["PartShared"], first)


if __name__ == "__main__":
    unittest.main()
