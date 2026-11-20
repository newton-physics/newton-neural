# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Safe checkpoint serialization for trained models.

Checkpoints store only tensors and JSON-compatible metadata so they load safely with ``torch.load(weights_only=True)``
without executing arbitrary code. This module handles file I/O and format-version validation; manifest semantics are
owned by :class:`~newton_neural.models.NeRDBase`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import newton
import torch
from torch import Tensor

from ..._version import __version__

FORMAT = 1
"""Checkpoint schema version. Readers reject checkpoints written with any other format version."""


def write(path: str | Path, manifest: dict[str, Any], state: dict[str, Tensor]) -> None:
    """Write a checkpoint file containing ``manifest`` and ``state``.

    Automatically prepends ``format`` (:data:`FORMAT`), ``library`` (``newton_neural`` version), and ``newton``
    (Newton version) to ``manifest``.

    Args:
        path: Output file path (parent directories are created if needed).
        manifest: JSON-serializable plain metadata dict.
        state: Named tensors (e.g., a ``state_dict``), detached and saved on CPU.

    Raises:
        TypeError: If ``manifest`` contains non-JSON-serializable objects.
    """
    manifest = {"format": FORMAT, "library": __version__, "newton": newton.__version__, **manifest}
    try:
        json.dumps(manifest)
    except TypeError as error:
        raise TypeError(f"a checkpoint manifest holds plain data only, but {error}") from error
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in state.items()}
    torch.save({"manifest": manifest, "state": state}, path)


def read(path: str | Path) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Read a checkpoint file written by :func:`write`.

    Args:
        path: Checkpoint file path.

    Returns:
        Tuple of ``(manifest, state)`` with tensors loaded on CPU.

    Raises:
        ValueError: If the file is not a valid checkpoint or uses an incompatible :data:`FORMAT`.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.keys() != {"manifest", "state"}:
        raise ValueError(f"{path} is not a newton_neural checkpoint")
    manifest = payload["manifest"]
    written = manifest.get("format") if isinstance(manifest, dict) else None
    if not isinstance(written, int):
        raise ValueError(f"{path} is not a newton_neural checkpoint")
    if written != FORMAT:
        raise ValueError(
            f"{path} has checkpoint format {written}, written by newton_neural {manifest.get('library')}. "
            f"This version reads format {FORMAT}. Load the file with the version that wrote it"
        )
    return manifest, payload["state"]
