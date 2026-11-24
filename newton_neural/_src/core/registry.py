# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Class-name registries for reconstructing models and composable parts from checkpoints."""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Self


class Registered:
    """Mixin that auto-registers subclasses by ``__name__`` on each base class's ``registry``.

    Each abstract base class defines ``registry: ClassVar[dict[str, type]] = {}``. Re-defining a class at the same
    ``(__module__, __qualname__)`` updates the entry (supporting module reloads and notebook re-runs); registering a
    colliding name from another location raises ``TypeError``. External subclasses are discoverable once their module
    is imported.
    """

    registry: ClassVar[dict[str, type]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip abstract base classes defining a new registry and mixins without one.
        if "registry" in cls.__dict__ or not hasattr(cls, "registry"):
            return
        registered = cls.registry.get(cls.__name__)
        place = (cls.__module__, cls.__qualname__)
        if registered is not None and (registered.__module__, registered.__qualname__) != place:
            raise TypeError(
                f"the name {cls.__name__!r} is already taken by {registered.__module__}.{registered.__qualname__}"
            )
        cls.registry[cls.__name__] = cls

    @classmethod
    def registered(cls, name: str) -> type[Self]:
        """Look up the subclass registered under ``name``.

        Raises:
            ValueError: If ``name`` is not in ``cls.registry``.
        """
        if name not in cls.registry:
            known = ", ".join(sorted(cls.registry))
            raise ValueError(
                f"unknown {cls.__name__} {name!r}. The registered ones are {known}. Import the module that defines "
                "it before loading"
            )
        return cls.registry[name]


class Part(Registered, abc.ABC):
    """Base mixin for composable model parts reconstructed from a class name and JSON-compatible config."""

    @abc.abstractmethod
    def config(self) -> dict[str, Any]:
        """Return JSON-serializable plain data passed to :meth:`from_config` during checkpoint loading."""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Reconstruct an instance from :meth:`config` (defaults to ``cls(**config)``)."""
        return cls(**config)

    @classmethod
    def create(cls, name: str, config: dict[str, Any]) -> Self:
        """Look up ``name`` in ``cls.registry`` and reconstruct it via :meth:`from_config`."""
        return cls.registered(name).from_config(config)
