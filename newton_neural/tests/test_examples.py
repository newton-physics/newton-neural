# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Subprocess smoke tests running each ``newton_neural.examples`` script with ``--viewer null --test``."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from newton_neural import SignalRole, SignalSpec
from newton_neural.data import TrajectoryDataset, TrajectoryWriter
from newton_neural.examples import cached_dataset, get_examples
from newton_neural.tests.unittest_utils import add_function_test, get_cuda_test_devices

OPTIONS = {
    "quickstart": {"train-steps": 300, "num-frames": 200},
    "contact_set": {"train-steps": 1000, "num-frames": 100, "world-count": 64},
    "contact_probes": {"train-steps": 1000, "num-frames": 100, "world-count": 64},
    "coupling": {"train-steps": 2000, "num-frames": 100, "world-count": 16},
}
"""Arguments of each example in the tests. Training is as short as the criteria of the example allow."""


class TestExamples(unittest.TestCase):
    def test_every_example_is_tested(self) -> None:
        """The examples the package lists are exactly the ones this module runs."""
        self.assertEqual(sorted(get_examples()), sorted(OPTIONS))

    def test_descriptions(self) -> None:
        """Every example describes itself, its success criteria and its run commands in its module docstring."""
        for name, module in get_examples().items():
            with open(importlib.util.find_spec(module).origin, encoding="utf-8") as source:
                description = ast.get_docstring(ast.parse(source.read()))
            self.assertIsNotNone(description, f"the example {name} has no module docstring")
            self.assertIn("\nSuccess criteria:\n- ", description)
            self.assertIn(f"\nRuns:\n- `python -m newton_neural.examples {name}`", description)

    def test_cached_dataset(self) -> None:
        """Open the dataset of an earlier run, find none where no run recorded, and refuse a recording that broke."""
        signals = {"joint_q": SignalSpec("joint_q", (2,), SignalRole.STATE)}
        frames = {"joint_q": torch.zeros(3, 4, 2)}
        flags = torch.ones(3, 4, dtype=torch.bool)
        with tempfile.TemporaryDirectory() as cache:
            folder = Path(cache)
            self.assertIsNone(cached_dataset(folder / "none"))
            with TrajectoryWriter(folder / "whole", signals, 0.02) as writer:
                writer.write(frames, flags, flags)
            self.assertEqual(cached_dataset(folder / "whole").frame_count, 4)
            with self.assertRaises(KeyboardInterrupt), TrajectoryWriter(folder / "broken", signals, 0.02) as writer:
                writer.write(frames, flags, flags)
                raise KeyboardInterrupt
            self.assertEqual(len(TrajectoryDataset(folder / "broken").shards), 1)  # it opens, with a part
            with self.assertRaisesRegex(RuntimeError, "did not run to its end.*Delete the folder"):
                cached_dataset(folder / "broken")


def test_example(test: TestExamples, device: torch.device, example: str) -> None:
    """Run an example in test mode and expect its success criteria to hold."""
    with tempfile.TemporaryDirectory() as cache:
        command = [sys.executable, "-m", "newton_neural.examples", example, "--device", str(device)]
        command += ["--viewer", "null", "--test", "--quiet", "--cache-dir", cache]
        for option, value in OPTIONS[example].items():
            command += [f"--{option}", str(value)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    test.assertEqual(result.returncode, 0, f"{' '.join(command)} failed\n{result.stdout}\n{result.stderr}")


for _example in OPTIONS:
    # The examples train their models, which takes a CUDA device.
    add_function_test(TestExamples, f"test_{_example}", test_example, devices=get_cuda_test_devices(), example=_example)


if __name__ == "__main__":
    unittest.main()
