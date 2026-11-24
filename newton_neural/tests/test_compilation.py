# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``with_own_code`` function cloning and code-object isolation."""

from __future__ import annotations

import unittest

from newton_neural._src.core.compilation import with_own_code


class TestWithOwnCode(unittest.TestCase):
    def test_same_function_with_its_own_code(self) -> None:
        """Verify clones preserve defaults, kw-only defaults, closures, and names with distinct code objects."""
        scale = 3.0

        def function(value: float, offset: float = 1.0, *, power: float = 2.0) -> float:
            return scale * value**power + offset

        copy = with_own_code(function)
        self.assertEqual(copy(2.0), function(2.0))
        self.assertEqual(copy(2.0, 0.5, power=3.0), function(2.0, 0.5, power=3.0))
        self.assertEqual(copy.__name__, function.__name__)
        codes = {id(code) for code in (function.__code__, copy.__code__, with_own_code(function).__code__)}
        self.assertEqual(len(codes), 3)


if __name__ == "__main__":
    unittest.main()
