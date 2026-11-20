# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit test ensuring every Warp kernel in ``newton_neural._src`` sets ``enable_backward=False``."""

from __future__ import annotations

import importlib
import pkgutil
import unittest

import warp as wp

import newton_neural._src


class TestKernels(unittest.TestCase):
    def test_no_kernel_compiles_an_adjoint(self) -> None:
        """Import every module of the package and expect each of its kernels to be compiled without an adjoint."""
        kernels = []
        for module in pkgutil.walk_packages(newton_neural._src.__path__, "newton_neural._src."):
            members = vars(importlib.import_module(module.name)).values()
            kernels += [member for member in members if isinstance(member, wp.Kernel)]
        self.assertTrue(kernels)
        for kernel in kernels:
            options = kernel.module.options | kernel.options  # a kernel overrides the options of its module
            self.assertFalse(options["enable_backward"], kernel.key)


if __name__ == "__main__":
    unittest.main()
