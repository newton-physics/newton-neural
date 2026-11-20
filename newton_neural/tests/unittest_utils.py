# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-device ``unittest`` registration helpers and temporary-directory test case."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def get_test_devices() -> list[torch.device]:
    """Return CPU plus ``cuda:0`` if CUDA is available."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    return devices


def get_cuda_test_devices() -> list[torch.device]:
    """Return available CUDA test devices (empty list on CPU-only machines)."""
    return [device for device in get_test_devices() if device.type == "cuda"]


def add_function_test(
    cls: type[unittest.TestCase],
    name: str,
    func: Callable[..., Any],
    *,
    devices: list[torch.device] | None = None,
    **kwargs: Any,
) -> None:
    """Register ``func(test, device, **kwargs)`` on ``cls`` as ``<name>_<device.type>`` per device.

    If ``devices`` is empty (e.g., CUDA-only test on CPU), registers a single skipped test under ``name``.
    Raises ``ValueError`` if ``cls`` already defines that attribute.
    """

    def register(test_name: str, test: Callable[..., None]) -> None:
        if test_name in vars(cls):
            raise ValueError(f"{cls.__name__} has a test {test_name} already. Give {func.__name__} another name")
        test.__doc__ = func.__doc__
        setattr(cls, test_name, test)

    devices = get_test_devices() if devices is None else devices
    if not devices:
        register(name, unittest.skip("this machine has none of the devices of the test")(lambda self: None))
    for device in devices:

        def test(self: unittest.TestCase, device: torch.device = device) -> None:
            func(self, device, **kwargs)

        register(f"{name}_{device.type}", test)


class FolderTestCase(unittest.TestCase):
    """``unittest.TestCase`` providing a fresh temporary directory ``self.folder`` cleaned up after each test."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
