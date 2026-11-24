# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Model families (:class:`NeRDBase`, :class:`FlashNeRD`) and compiled single-frame stepping (:class:`NeRDStepper`).

Higher layers (training, evaluation, solvers) depend only on :class:`NeRDBase` and never import specific families.
"""
