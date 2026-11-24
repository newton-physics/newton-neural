# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inference precision modes and scoped CUDA TF32 matmul flag management."""

from __future__ import annotations

import threading
from enum import StrEnum

import torch


class Precision(StrEnum):
    """Numeric precision mode for deployed inference.

    :attr:`FP16` and :attr:`BF16` store ``nn.Linear`` weights in 16-bit and run under ``torch.autocast``. Input
    frames, history buffers, normalization, attention scores/softmax, residual streams, predictions, and state
    integration stay in FP32. Every mode except :attr:`FP32` allows TF32 for remaining FP32 CUDA matmuls.
    """

    FP32 = "fp32"
    """Strict IEEE FP32 throughout (TF32 disabled regardless of process-global settings)."""

    TF32 = "tf32"
    """FP32 tensors throughout with TF32 CUDA tensor-core matmuls (resolves to :attr:`FP32` on CPU). Default."""

    FP16 = "fp16"
    """16-bit half-precision linear weights and autocast matmuls (fastest on GPU at large world counts)."""

    BF16 = "bf16"
    """16-bit bfloat16 linear weights and autocast matmuls (FP32 dynamic range, 7-bit mantissa)."""


def tensor_float_products(device: torch.device) -> bool:
    """Whether float32 matrix products on ``device`` run in TF32 under PyTorch's process-global setting."""
    return device.type == "cuda" and torch.backends.cuda.matmul.fp32_precision == "tf32"


SWITCH_LOCK = threading.RLock()
"""Reentrant lock serializing access to PyTorch's process-global CUDA ``fp32_precision`` flag."""


class Float32Products:
    """Context manager toggling PyTorch's process-global CUDA TF32 matmul flag and restoring it on exit.

    Enter while holding :data:`SWITCH_LOCK` so concurrent threads do not clobber each other's flag state. Note that
    ``torch.get_float32_matmul_precision()`` can raise inside this block if its legacy enum disagrees with
    ``torch.backends.cuda.matmul.fp32_precision`` (which affects ``torch.compile`` ``max_autotune``).
    """

    def __init__(self, *, tensor_float: bool) -> None:
        self._tensor_float = tensor_float

    def __enter__(self) -> None:
        switch = torch.backends.cuda.matmul
        self._found = switch.fp32_precision
        self._wrote = (self._found == "tf32") != self._tensor_float  # "none" and "ieee" both use IEEE FP32
        if self._wrote:
            switch.fp32_precision = "tf32" if self._tensor_float else "ieee"

    def __exit__(self, *raised: object) -> None:
        if self._wrote:
            switch = torch.backends.cuda.matmul
            switch.fp32_precision = "none"  # Clear local override to check whether inheritance matches _found
            if switch.fp32_precision != self._found:
                switch.fp32_precision = self._found
