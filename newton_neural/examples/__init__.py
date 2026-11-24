# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI parser, cache helpers, and training loop for ``python -m newton_neural.examples <name>``."""

import argparse
import itertools
import os
import runpy
import sys
import time
from pathlib import Path

import newton.examples
import torch

from newton_neural.data import TrajectoryDataset, TrajectoryWindows
from newton_neural.models import NeRDBase
from newton_neural.training import LossBase, create_optimizer


def get_examples() -> dict[str, str]:
    """Map each example short name to its importable module path."""
    modules = sorted(Path(__file__).parent.glob("example_*.py"))
    return {module.stem.removeprefix("example_"): f"{__name__}.{module.stem}" for module in modules}


def create_parser(*, world_count: int, train_steps: int) -> argparse.ArgumentParser:
    """Create the example CLI parser combining Newton's flags with ``--train-steps`` and ``--cache-dir``."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "newton-neural" / "examples"
    parser = newton.examples.create_parser()
    newton.examples.add_world_count_arg(parser)
    parser.set_defaults(world_count=world_count)
    parser.add_argument("--train-steps", type=int, default=train_steps, help="Optimizer steps of the training.")
    parser.add_argument("--cache-dir", type=Path, default=cache, help="Folder that keeps datasets and trained models.")
    return parser


def cache_folder(args: argparse.Namespace, name: str) -> Path:
    """Create and print the cache directory for example ``name``."""
    folder = args.cache_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"The dataset and the trained model are kept in {folder}. Delete the folder to record and train again.")
    return folder


def cached_dataset(path: Path) -> TrajectoryDataset | None:
    """Return a cached :class:`TrajectoryDataset` at ``path``, or ``None`` if absent.

    Raises ``RuntimeError`` if a previous recording was interrupted (``complete=False``).
    """
    if not (path / "meta.json").exists():
        return None
    dataset = TrajectoryDataset(path)
    if not dataset.generation["complete"]:
        raise RuntimeError(
            f"the recording in {path} did not run to its end, so it holds only a part of the frames of the "
            "example. Delete the folder and run the example again"
        )
    return dataset


def train(
    nerd: NeRDBase,
    windows: TrajectoryWindows,
    loss: LossBase,
    step_count: int,
    *,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Fit normalizers and run a standard cosine-annealed PyTorch training loop for ``step_count`` steps."""
    # Enable TF32 matmuls on tensor-core GPUs for faster training.
    torch.set_float32_matmul_precision("high")
    nerd.fit_normalization(windows.loader(batch_size))
    # Keep loss eager here: compile startup overhead outweighs per-step gains for short example runs.
    optimizer = create_optimizer(nerd, lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step_count)
    start = time.perf_counter()
    for step, batch in enumerate(itertools.islice(windows.loader(batch_size, epochs=None), step_count), start=1):
        optimizer.zero_grad()
        error = loss(nerd, batch)
        error.backward()
        optimizer.step()
        schedule.step()
        if step % 500 == 0 or step == step_count:
            print(f"Step {step} of {step_count}, loss {error.item():.4f}, {time.perf_counter() - start:.0f} s.")


def main() -> None:
    """Dispatch ``python -m newton_neural.examples <name> [options]`` or list available examples."""
    examples = get_examples()
    if len(sys.argv) < 2 or sys.argv[1] not in examples:
        print("Run python -m newton_neural.examples <name> [options] with one of these names.")
        for name in examples:
            print(f"  {name}")
        sys.exit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "--list") else 1)
    module = examples[sys.argv[1]]
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")
