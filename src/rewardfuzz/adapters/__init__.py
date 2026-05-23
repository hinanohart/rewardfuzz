"""Adapter registry.

Adapters wrap a target reward function into the canonical :class:`~rewardfuzz.types.RewardResult`
interface. Register custom adapters with :func:`register_adapter`.
"""

from __future__ import annotations

from .base import BaseAdapter
from .callable_ import CallableAdapter
from .openevolve import OpenEvolveAdapter
from .rlvr import RLVRAdapter

_REGISTRY: dict[str, type[BaseAdapter]] = {}


def register_adapter(name: str, cls: type[BaseAdapter]) -> None:
    """Register an adapter class under ``name`` (overwrites any existing entry)."""

    if not issubclass(cls, BaseAdapter):
        raise TypeError(f"{cls!r} is not a BaseAdapter subclass")
    _REGISTRY[name] = cls


def load_adapter(name: str) -> type[BaseAdapter]:
    """Look up a registered adapter class by name."""

    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown adapter {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)


for _cls in (CallableAdapter, OpenEvolveAdapter, RLVRAdapter):
    register_adapter(_cls.name, _cls)

__all__ = [
    "BaseAdapter",
    "CallableAdapter",
    "OpenEvolveAdapter",
    "RLVRAdapter",
    "register_adapter",
    "load_adapter",
    "list_adapters",
]
