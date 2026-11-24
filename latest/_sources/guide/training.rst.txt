.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton_neural.models

Training
========

Because every model is a standard ``torch.nn.Module``, you write and own your PyTorch training loop.
``newton-neural`` provides GPU-resident batching (:class:`~newton_neural.data.TrajectoryWindows`), the training
objective (:class:`~newton_neural.training.LossOneStep`), and optimizers
(:func:`~newton_neural.training.create_optimizer`).

This page continues from the cartpole dataset recorded in :doc:`recording`.

.. testsetup::

    from guide_setup import cartpole_recording

    cartpole_recording(frame_count=128)  # Cartpole recording from the previous page (fewer worlds)

1. Load the untrained model
---------------------------

Load the untrained checkpoint saved alongside the dataset so all model parts (system, backbone, contact input,
prediction) stay in sync with the recorded signals:

.. testcode::

    import itertools

    import torch

    from newton_neural.data import TrajectoryWindows
    from newton_neural.models import FlashNeRD
    from newton_neural.training import LossOneStep, create_optimizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nerd = FlashNeRD.load("data/cartpole/flashnerd.pt", device=device)
    print(nerd.receptive_field)

.. testoutput::

    13

Use :meth:`NeRDBase.load` instead of ``FlashNeRD.load`` if your training script should work with any registered
model family.

2. Build training windows and fit normalization
-----------------------------------------------

:class:`~newton_neural.data.TrajectoryWindows` indexes valid contiguous slices within single episodes. A window of
``length`` steps contains ``length + 1`` consecutive frames.

.. testcode::

    windows = TrajectoryWindows("data/cartpole", length=16, device=device)
    train, held_out = windows.split(fraction=0.1)
    nerd.fit_normalization(train.loader(256))

Key behaviors to know:

- **Window length vs. receptive field:** Choose ``length >= nerd.receptive_field``. :class:`~newton_neural.training.LossOneStep`
  only scores predictions whose full temporal context fits in the window (frames ``receptive_field - 1`` onward) or
  whose window begins at an episode start. With ``length=16`` and ``receptive_field=13``, the final 4 transitions of
  every non-start window contribute to the loss. Passing ``length < receptive_field`` raises unless every window in
  the batch starts an episode.
- **Unbounded history (e.g., RNNs):** If ``nerd.receptive_field is None``, only episode-start windows are scored, and
  batches containing no episode starts raise. Sample them with ``train.loader(256, weights=train.episode_start)``.
- **Device-resident batching:** Loaded worlds stay on ``device`` and batches are indexed directly on-device, so no
  ``DataLoader`` worker processes are needed.
- **Deterministic world split:** :meth:`~newton_neural.data.TrajectoryWindows.split` holds out whole worlds
  deterministically from global world IDs, ensuring distributed ranks and offline evaluation agree on the exact
  train/validation split.
- **Normalization:** :meth:`NeRDBase.fit_normalization` estimates input/output mean and standard deviation from up to
  ``batch_limit=64`` batches and infers ``frame_dt`` from the dataset. Call it once on the training device before
  evaluating the loss (otherwise the loss raises).

3. Run the training loop
------------------------

.. testcode::

    step_count = 20  # Use thousands of steps for a real model
    loss = LossOneStep(noise=0.1)
    optimizer = create_optimizer(nerd, lr=2e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step_count)

    for batch in itertools.islice(train.loader(256, epochs=None), step_count):
        optimizer.zero_grad()
        error = loss(nerd, batch)
        error.backward()
        optimizer.step()
        schedule.step()

The shipped examples share an identical loop in ``newton_neural.examples.train``.

.. tip::

    On tensor-core GPUs, call ``torch.set_float32_matmul_precision("high")`` before training to run FP32 matmuls in
    TF32, which is typically much faster with negligible impact on loss. Deployed inference precision is independent
    of this global flag (see :doc:`performance`).

Loss options (:class:`~newton_neural.training.LossOneStep`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``LossOneStep`` computes the mean normalized one-step prediction error over valid frames:

- ``error``: ``"absolute"`` (L1, default; less sensitive to heavy-tailed contact impulses) or ``"squared"`` (L2).
- ``noise``: Adds zero-mean Gaussian state noise scaled by ``noise * output_normalizer.std`` to every input frame and
  supervises against the change from the *perturbed* state to the recorded next state, teaching the model to recover
  from compounding drift. Recomputable contact inputs (such as :class:`~newton_neural.contacts.ContactInputProbes`)
  are automatically re-evaluated on the perturbed states. Models with recorded labels
  (:class:`~newton_neural.predictions.PredictionFreeMotion`) do not support state noise and raise if ``noise > 0``.
- ``loss.compile()``: Compiles the tensor loss computation with ``torch.compile`` (see below).

Optimizer (:func:`~newton_neural.training.create_optimizer`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``name="muon"`` *(default)*: Uses :class:`~newton_neural.training.OptimizerMuon` for 2D+ hidden weight matrices
  declared by module ``hidden_weights()`` methods (implemented by all built-in backbones and ``FlashNeRD``), and
  ``AdamW`` for all remaining parameters (embeddings, readouts, biases, norms, contact encoders).
- ``name="adamw"``: Uses ``torch.optim.AdamW`` for all parameters. Required if a custom network declares no
  ``hidden_weights()``.

Both options share the same ``lr`` and ``weight_decay`` scaling, so switching requires changing only ``name``.

4. Compile the loss (optional)
------------------------------

For long training runs on CUDA, call :meth:`~newton_neural.training.LossOneStep.compile` to compile the forward and
backward loss computation with ``torch.compile`` (noise sampling and contact recomputation stay outside the compiled
graph). Compilation adds one-time startup overhead on the first step, so it pays off for thousands of steps rather
than short experiments.

.. testcode::

    loss = LossOneStep(noise=0.1)
    if nerd.device.type == "cuda":
        loss.compile()

    for batch in itertools.islice(train.loader(256, epochs=None), 2):
        optimizer.zero_grad()
        loss(nerd, batch).backward()  # First call compiles and verifies against eager loss
        optimizer.step()

CPU compilation requires a local C++ toolchain, so the guard above enables it only on CUDA.

By default (``check=True``), the first gradient-tracking call and first no-grad call also compute the eager loss and
raise a ``RuntimeError`` if the compiled value differs by more than ``1e-5`` relative to ``max(|eager|, 1)``, or by
more than ``1e-3`` where FP32 matmuls run in TF32, whose rounding depends on the order of the products and on the
GPU. Calls under ``torch.autocast`` and later recompilations (e.g., for different batch sizes) are not re-checked.
If your custom network is stochastic during training (e.g., dropout), pass ``loss.compile(check=False)``.

5. Save and load checkpoints
----------------------------

.. testcode::

    nerd.metadata["dataset"] = "data/cartpole"
    nerd.save("cartpole.pt")

    loaded = FlashNeRD.load("cartpole.pt", device=device)

A checkpoint is a single ``.pt`` file containing model weights, normalizer statistics, and a plain-data manifest
(model class, part classes and configs, family settings, ``frame_dt``, ``precision``, signal specs, and ``metadata``).
Because it stores only tensors and plain JSON-compatible data, loading uses ``torch.load(..., weights_only=True)``
and never executes pickled code.

Advanced batching patterns
--------------------------

- **Upweighting rare events:** Compute per-window scalar weights with
  :meth:`~newton_neural.data.TrajectoryWindows.measure` and pass ``weights=...`` to ``loader()`` to sample windows
  with replacement proportional to their weight.
- **Several GPUs:** Pass ``worlds=slice(rank, None, rank_count)`` to ``TrajectoryWindows`` so each process loads a
  disjoint shard of worlds while preserving the identical global ``split()`` held-out set, and train through
  ``DistributedDataParallel`` (see :ref:`training-several-gpus`).
- **Single-frame supervision:** Models whose labels are stored on each frame
  (:class:`~newton_neural.predictions.PredictionFreeMotion`) train with ``length=0`` (:doc:`coupling`).

.. _training-several-gpus:

Training on several GPUs
------------------------

Several processes train one model with PyTorch's ``DistributedDataParallel``, one process per GPU, started with
``torchrun``. Each process loads its own shard of worlds, and the loss takes the wrapper in place of the model. The
forward pass then runs through the wrapper, which averages the gradients across the processes.

.. code-block:: python

    import os

    import torch.distributed
    from torch.nn.parallel import DistributedDataParallel

    torch.distributed.init_process_group("nccl")
    rank, rank_count = torch.distributed.get_rank(), torch.distributed.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))  # Set by torchrun, the GPU of this process

    nerd = FlashNeRD.load("data/cartpole/flashnerd.pt", device=device)
    windows = TrajectoryWindows("data/cartpole", length=16, device=device, worlds=slice(rank, None, rank_count))
    train, held_out = windows.split(fraction=0.1)
    nerd.fit_normalization(train.loader(256))
    distributed = DistributedDataParallel(nerd)

    loss, optimizer = LossOneStep(noise=0.1), create_optimizer(nerd, lr=2e-3)
    for batch in itertools.islice(train.loader(256, epochs=None), step_count):
        optimizer.zero_grad()
        loss(distributed, batch).backward()  # The wrapper, so that the gradients are averaged
        optimizer.step()

    if rank == 0:
        nerd.save("cartpole.pt")

- **Normalization.** Call ``fit_normalization`` in every process before wrapping the model. The wrapper copies the
  statistics of the first process to all of them, so every process normalizes alike.
- **The wrapper in training steps, the model elsewhere.** Give the loss the wrapper in every training step. Given
  the model itself, the loss computes the gradient of one process and averages nothing. Give it the model
  (``nerd``) for what one process does alone, such as a held-out loss, since a forward pass of the wrapper waits
  for the other processes. Evaluation and saving use the model as well.
- **Equal step counts.** Every process takes the same number of steps. Draw batches with ``epochs=None`` and a step
  count, because the shards hold different numbers of windows.
- **Compiled loss.** ``loss.compile()`` works with the wrapper. Its first call checks itself on the model, as it
  does in one process.
- **Parameters without a gradient.** The wrapper refuses a model with a parameter that the loss does not reach,
  such as the lag bias of an attention block whose span is ``1``. Build the wrapper of such a model with
  ``DistributedDataParallel(nerd, find_unused_parameters=True)``.
