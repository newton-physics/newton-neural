# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Package version read from installed distribution metadata (defined in ``pyproject.toml``)."""

from importlib import metadata

__version__ = metadata.version("newton-neural")
