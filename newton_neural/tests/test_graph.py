# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-runnable unit tests for ``GraphedWork`` (CUDA graph replay is tested in solver/simulation suites)."""

from __future__ import annotations

import inspect
import re
import unittest

import torch
import warp as wp

from newton_neural._src.core.graph import GraphedWork


class TestGraphedWork(unittest.TestCase):
    def test_private_torch_functions_exist(self) -> None:
        """Verify PyTorch still exposes the private CUDA memory-pool C++ helpers used during graph capture."""
        names = re.findall(r"torch\._C\.(\w+)", inspect.getsource(GraphedWork._allocating_from_pool))
        self.assertEqual(len(set(names)), 3)
        for name in names:
            self.assertTrue(hasattr(torch._C, name), f"torch._C.{name} is gone, which GraphedWork calls")

    def test_cpu_runs_the_work_as_it_is(self) -> None:
        """Verify CPU execution runs eagerly, allows arbitrary new argument objects, and rejects CPU capture."""
        calls = []
        work = GraphedWork(lambda *arguments: calls.append(arguments), wp.get_device("cpu"))
        work.run(1, "state")
        work.run(2)
        self.assertEqual(calls, [(1, "state"), (2,)])
        with self.assertRaisesRegex(RuntimeError, "CPU"), wp.ScopedCapture(device="cpu"):
            work.run(3)
        self.assertEqual(len(calls), 2)
        # CPU path does not cache graphs, so distinct objects per call never hit _GRAPH_LIMIT.
        for _ in range(20):
            work.run(object())
        self.assertEqual(len(calls), 22)


if __name__ == "__main__":
    unittest.main()
