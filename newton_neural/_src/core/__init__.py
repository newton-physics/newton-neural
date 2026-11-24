# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Core foundation layer shared across the library.

Contains model-agnostic primitives: signal specifications, running normalizers, quaternion/rotation-vector math,
checkpoint I/O, class registries, deployment precision modes, CUDA graph replay for mixed PyTorch/Warp steps, and
``torch.compile`` isolation helpers. Imports no higher layers.
"""
