"""Attack-strategy registry.

The four rule-based strategies are deterministic and form the default, CI-gated attack set.
``llm_specgaming`` is registered but excluded from the default set because it is non-deterministic
and requires the ``[hf]`` extra.
"""

from __future__ import annotations

from collections.abc import Callable

from ..types import Strategy
from .degenerate import DegenerateStrategy
from .llm_specgaming import LLMSpecGamingStrategy
from .numeric_exploit import NumericExploitStrategy
from .side_effect import SideEffectStrategy
from .test_tamper import TestTamperStrategy

StrategyFactory = Callable[[], Strategy]

_REGISTRY: dict[str, StrategyFactory] = {}

#: The deterministic strategies that run by default and gate CI.
DEFAULT_STRATEGIES: tuple[str, ...] = (
    "degenerate",
    "numeric_exploit",
    "test_tamper",
    "side_effect",
)


def register_strategy(name: str, factory: StrategyFactory) -> None:
    """Register a strategy factory (a zero-arg callable returning a strategy instance)."""

    _REGISTRY[name] = factory


def get_strategy(name: str) -> Strategy:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(
            f"unknown strategy {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)


register_strategy("degenerate", DegenerateStrategy)
register_strategy("numeric_exploit", NumericExploitStrategy)
register_strategy("test_tamper", TestTamperStrategy)
register_strategy("side_effect", SideEffectStrategy)
register_strategy("llm_specgaming", LLMSpecGamingStrategy)

__all__ = [
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "DEFAULT_STRATEGIES",
    "DegenerateStrategy",
    "NumericExploitStrategy",
    "TestTamperStrategy",
    "SideEffectStrategy",
    "LLMSpecGamingStrategy",
]
