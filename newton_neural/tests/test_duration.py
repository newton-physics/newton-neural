# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for floating-point tolerant ``same_duration`` comparisons."""

from __future__ import annotations

import unittest

import numpy as np

from newton_neural._src.core.duration import same_duration


class TestSameDuration(unittest.TestCase):
    def test_rounding_is_no_difference(self) -> None:
        """Verify ``same_duration`` accepts rounding-level differences and rejects >= 0.1% differences."""
        frame_dt = 0.02
        self.assertTrue(same_duration(frame_dt, frame_dt))
        self.assertTrue(same_duration(1.0 / 50.0, frame_dt))
        self.assertTrue(same_duration(0.005 * 4, frame_dt))  # 4 substeps per frame
        self.assertTrue(same_duration(frame_dt * (1.0 + 5e-7), frame_dt))
        self.assertTrue(same_duration(frame_dt * (1.0 - 5e-7), frame_dt))
        self.assertTrue(same_duration(float(np.float32(frame_dt)), frame_dt))  # float32 round-trip
        for other in (frame_dt * 1.001, frame_dt * 0.999, 0.005, 0.04, 0.0):
            self.assertFalse(same_duration(other, frame_dt), other)


if __name__ == "__main__":
    unittest.main()
