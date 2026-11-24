# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for ``python -m newton_neural.tests`` (forwards extra args like ``-k`` or ``-v``)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    tests = Path(__file__).parent
    top = tests.parent.parent  # Repository/site-packages root containing newton_neural
    unittest.main(module=None, argv=[sys.argv[0], "discover", "-s", str(tests), "-t", str(top), *sys.argv[1:]])
