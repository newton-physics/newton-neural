# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""System specifications and Newton array bindings.

``SystemArticulation`` describes an articulated robot (fixed or floating base), maps raw Newton signals to
network features, and integrates predicted increments; ``BindingArticulation`` binds those signals to Newton's Warp
arrays. ``robot_frame`` and ``robot_frame_floating`` compute the same robot coordinate frame inside Warp kernels.
Subclass ``SystemBase`` and ``BindingBase`` for non-articulated systems.
"""

from ._src.systems.articulation import SystemArticulation, robot_frame, robot_frame_floating
from ._src.systems.articulation_binding import BindingArticulation
from ._src.systems.system import BindingBase, SystemBase

__all__ = [
    "BindingArticulation",
    "BindingBase",
    "SystemArticulation",
    "SystemBase",
    "robot_frame",
    "robot_frame_floating",
]
