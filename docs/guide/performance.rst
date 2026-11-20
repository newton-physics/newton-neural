.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton_neural.models

Performance
===========

This page covers why learned dynamics run fast, the runtime options that control throughput and numerical
precision, how to validate reduced precision on your own model, and how to benchmark accurately.

Why inference is fast
---------------------

Two architectural properties account for most of the speedup over a physics solver:

- **One step per frame.** Physics solvers split a control period into multiple substeps (often 4–8) for numerical
  stability. A learned model is trained on full frames and advances the entire frame in one forward pass.
- **Fixed cost independent of contact complexity.** Runtime scales with world count and network size, not with
  contact count, joint stiffness, or actuator complexity.

Accordingly, speedups are largest on contact-rich robots with many substeps and complex drives (such as legged
locomotion), and smallest on tiny free-space systems where a physics solver is already cheap.

At the implementation level, every deployed step combines six optimizations (streaming and parallel blocks apply to
:class:`FlashNeRD` backbones; the rest apply to any :class:`NeRDBase` family):

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Optimization
     - How it works
   * - Streaming history
     - Each step embeds only the newest frame. Attention backbones store past keys and values in a fixed-size
       per-world ring buffer updated in place, keeping step cost :math:`O(1)` with episode length.
   * - Parallel Flash blocks
     - :class:`~newton_neural.backbones.BackboneFlash` computes attention and SwiGLU FFN in parallel from one packed
       input projection and merges them with one output projection (2 linear layers per block instead of 4), with
       multi-query attention (1 shared KV head) to minimize ring-buffer traffic.
   * - Tensor cores
     - On CUDA, linear-layer matmuls execute on tensor cores (TF32 by default, or FP16/BF16 when selected). Pure
       IEEE FP32 disables tensor cores and is typically several times slower.
   * - ``torch.compile``
     - Compiles the streaming step for fixed static shapes (``fullgraph=True, dynamic=False``) on CUDA.
   * - CUDA graph replay
     - Captures the entire solver step (contact sensing, network inference, state write, forward kinematics) into one
       CUDA graph launch.
   * - Zero host sync / copies
     - After the first warm-up step, stepping performs zero allocations and zero device-to-host syncs. For replicated
       worlds, PyTorch accesses Newton's Warp state/control arrays directly as zero-copy views.

On CPU, tensor cores, compilation, and CUDA graphs are disabled automatically and steps run as eager PyTorch and
Warp.

.. testsetup::

    from guide_setup import cartpole_recording
    from newton_neural.data import TrajectoryWindows

    # Cartpole recording from the previous pages, fitted and saved without training.
    model, system, simulation, nerd, reach, dataset = cartpole_recording(frame_count=64)
    train, held_out = TrajectoryWindows(dataset, length=16, device=nerd.device).split(fraction=0.1)
    nerd.fit_normalization(train.loader(256))
    nerd.save("cartpole.pt")

Runtime and training options
----------------------------

Three options govern deployed inference and one governs training. With all defaults enabled on CUDA, a solver runs
compiled, CUDA-graphed, and tensor-core accelerated.

.. list-table::
   :widths: 15 30 16 25 14
   :header-rows: 1

   * - Option
     - Controls
     - Default
     - Where set
     - Disable with
   * - ``precision``
     - Inference numeric mode (see `Precision modes`_).
     - Checkpoint precision (``"tf32"`` for new models)
     - :class:`NeRDStepper`, :class:`~newton_neural.solvers.SolverNeRD`,
       :class:`~newton_neural.solvers.SolverNeRDMuJoCo`,
       :func:`~newton_neural.evaluation.rollout_error`, and :attr:`NeRDBase.precision`
     - ``precision="fp32"``
   * - ``compile``
     - Compile model step via ``torch.compile``.
     - ``True`` on CUDA
     - Same calls as ``precision``
     - ``compile=False``
   * - ``use_cuda_graph``
     - Replay solver step from a CUDA graph (for :class:`~newton_neural.solvers.SolverNeRDMuJoCo`, the neural
       prediction stage only).
     - ``True`` on CUDA
     - :class:`~newton_neural.solvers.SolverNeRD`, :class:`~newton_neural.solvers.SolverNeRDMuJoCo`
     - ``use_cuda_graph=False``
   * - ``loss.compile()``
     - Compile training loss with ``torch.compile`` (:doc:`training`).
     - Off
     - :class:`~newton_neural.training.LossOneStep`
     - Omit call

Notes on CUDA graph boundaries:

- :class:`NeRDStepper` does not manage its own CUDA graph, making it safe to capture inside your own outer graph.
- :class:`~newton_neural.solvers.SolverNeRDMuJoCo` graphs only its neural prediction kernel; MuJoCo Warp stages
  launch normally unless you capture the surrounding loop with ``wp.ScopedCapture()`` (:doc:`solvers`).

Override options as keyword arguments at construction:

.. testcode::

    from newton_neural.solvers import SolverNeRD

    solver = SolverNeRD(model, "cartpole.pt")  # All defaults
    plain = SolverNeRD(model, "cartpole.pt", precision="fp32", compile=False, use_cuda_graph=False)

Precision is saved inside the checkpoint. Set ``nerd.precision`` before saving so all downstream solvers, steppers,
and rollout evaluations inherit it automatically:

.. testcode::

    from newton_neural.models import NeRDBase

    saved = NeRDBase.load("cartpole.pt")
    saved.precision = "fp16"
    saved.save("cartpole_fp16.pt")
    print(SolverNeRD(model, "cartpole_fp16.pt").stepper.precision)

.. testoutput::

    fp16

For debugging, you can force eager execution process-wide via ``torch.compiler.set_stance("force_eager")`` (note
that ``stepper.compiled`` still reflects how the stepper was constructed).

Inspecting active settings
--------------------------

Printing a solver or stepper reports its resolved runtime configuration:

.. testcode::
    :hide:

    print(solver)

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    SolverNeRD(NeRDStepper(FlashNeRD, world_count=32, device=...), use_cuda_graph=...)

.. code-block:: text

    SolverNeRD(NeRDStepper(FlashNeRD, world_count=32, device='cuda:0', precision='tf32', compiled=True), use_cuda_graph=True)

On CPU, this resolves to ``precision='fp32'``, ``compiled=False``, and ``use_cuda_graph=False``. The individual
fields are also exposed as ``solver.stepper.precision``, ``solver.stepper.compiled``, and ``solver.use_cuda_graph``.

Precision modes
---------------

``precision`` accepts a :class:`Precision` enum value or its lowercase string:

.. list-table::
   :widths: 12 88
   :header-rows: 1

   * - Mode
     - Behavior
   * - ``"fp32"``
     - Strict IEEE FP32 throughout (TF32 disabled regardless of global PyTorch settings). Accuracy reference; slowest
       on GPU because matmuls cannot use tensor cores.
   * - ``"tf32"``
     - FP32 tensors throughout; CUDA matmuls round inputs to TF32 (10-bit mantissa, 8-bit exponent) on tensor cores.
       Resolves to ``"fp32"`` on CPU. **Default for new models.**
   * - ``"fp16"``
     - Linear layers store weights and compute in FP16 under ``torch.autocast``. Fastest option at large world
       counts.
   * - ``"bf16"``
     - Same as ``"fp16"`` using BF16 (FP32 dynamic range, 7-bit mantissa).

Start with the default ``"tf32"``. Switch to ``"fp16"`` when stepping thousands of worlds and the checks in
`Validating precision on your model`_ confirm equivalent accuracy to ``"fp32"``.

In every mode except ``"fp32"``, remaining FP32 matmuls on CUDA (such as :class:`~newton_neural.contacts.ContactInputSet`
pooling) may execute in TF32.

What runs in 16-bit vs. FP32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **``"tf32"``** keeps all tensors in FP32 and accumulates matmuls in FP32; only matmul inputs round to a 10-bit
  mantissa, so overflow behavior matches IEEE FP32.
- **``"fp16"`` and ``"bf16"``** create a deployment copy with 16-bit ``nn.Linear`` weights and run under
  ``torch.autocast``. For :class:`FlashNeRD` with built-in backbones:

  - **16-bit:** ``nn.Linear`` weights and outputs, activations and SwiGLU products between linear layers, injected
    contact sums, readout/skip additions, and attention output values.
  - **FP32:** input frames, output state, per-world history buffers, feature extraction, contact codes,
    normalization, attention scores/softmax, residual stream, denormalized prediction, and state integration.

Because the network predicts normalized *increments* rather than absolute coordinates, 16-bit rounding applies to
the one-frame delta rather than the full state magnitude. Custom model families follow standard ``torch.autocast``
rules while keeping system feature extraction, normalization, and integration in FP32.

Empirical accuracy notes
^^^^^^^^^^^^^^^^^^^^^^^^

Across trained models of a robot arm, a walking quadruped, and a dexterous hand:

- **``"tf32"`` and ``"fp16"``** behaved comparably (both use a 10-bit mantissa): one-frame divergence from ``"fp32"``
  was smaller than the models' own prediction error, with ``"tf32"`` slightly closer.
- **``"bf16"``** degraded noticeably on the highest-precision models (7-bit mantissa), exceeding single-step model
  error on two of three robots.
- **Downstream RL:** Quadruped and hand policies achieved matching returns across all four precisions.
- **FP16 overflow risk:** Values exceeding FP16 range (``~65504``) become ``inf`` and permanently corrupt that
  world. Although no overflows occurred in internal benchmarks, always validate ``"fp16"`` on your model before
  enabling it by default.

Validating precision on your model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verify a candidate precision against ``"fp32"`` at three levels:

1. **Automatic first-step sanity check:** On its first call, every :class:`NeRDStepper` also executes the eager
   model in IEEE FP32 and raises ``RuntimeError`` if the max relative normalized output difference exceeds the mode
   tolerance (``1e-4`` for ``fp32``, ``2e-2`` for ``tf32``/``fp16``, ``1e-1`` for ``bf16``). This catches compiler
   bugs or severe quantization failures immediately, though it does not test multi-step rollouts.
2. **Open-loop rollout error:** Run :func:`~newton_neural.evaluation.rollout_error` across precisions on held-out
   data and compare ``nrmse`` and percentiles against ``"fp32"``.
3. **Live audits:** Run :func:`~newton_neural.evaluation.audit_rest` and
   :func:`~newton_neural.evaluation.audit_paired` (:doc:`evaluation`) using a solver constructed with the candidate
   ``precision``.

.. testcode::

    from newton_neural.evaluation import rollout_error

    for precision in ("fp32", "tf32", "fp16"):
        error = rollout_error(nerd, held_out, horizons=(16,), precision=precision)[16]
        print(precision, error.nrmse, error.p50["joint_q"], error.p99["joint_q"])

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    fp32 ...
    tf32 ...
    fp16 ...

Interaction with PyTorch's global TF32 flag
-------------------------------------------

On CUDA, PyTorch controls TF32 via a process-global flag (``torch.backends.cuda.matmul.fp32_precision``, also set by
``torch.set_float32_matmul_precision``). Deployed steps isolate themselves from that flag:

- During a step, :class:`NeRDStepper` acquires a lock, sets the flag required by its ``precision``, and restores the
  previous value on exit (even if an exception occurs).
- Captured CUDA graphs bake in the precision recorded during capture and do not touch the global flag at replay.
- If multiple threads run neural steps concurrently with other PyTorch FP32 matmuls, the other thread's matmuls
  issued during an ungraphed step will observe the temporarily active flag.
- **``max_autotune`` caveat:** If you enable ``torch.compile`` autotuning (``mode="max-autotune"``), call
  ``torch.set_float32_matmul_precision("high")`` before the first step (or pass ``compile=False``) so PyTorch's
  autotuner does not clash with the temporarily toggled flag.

Training losses (:class:`~newton_neural.training.LossOneStep`) are *not* deployed steps and follow your process's
global matmul precision setting (see :doc:`training`).

Benchmarking guidelines
-----------------------

When measuring speedups:

- **Compare frames, not solver steps:** Time 1 neural step against all physics substeps (plus collision detection)
  required to advance the same ``frame_dt``.
- **Benchmark realistic setups:** Match your production world count, precision, and contact input. Random weights
  have identical step latency to trained weights.
- **Measure end-to-end training:** Once physics stepping is fast, RL wall-clock speedup is bounded by observations,
  rewards, resets, and policy updates.
- **Warm up and synchronize:** Run several warm-up steps to complete compilation and CUDA graph capture, then call
  ``wp.synchronize()`` before starting and stopping the timer:

.. testcode::

    import time

    import warp as wp

    state, control = model.state(), model.control()
    frame_dt = solver.nerd.frame_dt

    for _ in range(5):  # Warm up: compile, validate, and capture CUDA graph
        solver.step(state, state, control, None, frame_dt)

    step_count = 100
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(step_count):
        solver.step(state, state, control, None, frame_dt)
    wp.synchronize()
    step_time = (time.perf_counter() - start) / step_count

    world_count = solver.binding.world_count
    print(f"{1e6 * step_time:.0f} us per step, {world_count / step_time:,.0f} world steps per second")

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    ... us per step, ... world steps per second

See :doc:`development` for the repository's built-in benchmark suite under ``asv/benchmarks``.
