# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""External contact sensing and encoding layer.

Defines :class:`ContactInputBase` and :class:`ContactSensorBase` plus built-in implementations
(:class:`ContactInputProbes`, :class:`ContactInputSet`) that sense contact in Warp kernels and encode it into
fixed-size vectors for any model family. Depends on ``core`` and ``systems``.
"""
